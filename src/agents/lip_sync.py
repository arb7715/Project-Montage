"""
Lip Sync Agent - Phase 2 convergence node.

Visual Novel PiP composition (primary mode)
---------------------------------------------
Per dialogue line:
  1. Pick the speaking character portrait from outputs/images/.
  2. Send (portrait + line_audio) to the Colab Wav2Lip API.
  3. Keep the raw lip-synced talking-head clip (no full-canvas normalization).

Per scene (PiP composition):
  1. Build a full-duration Ken Burns background from the wide B-roll,
     lasting the entire scene audio (AudioTrack.duration_seconds).
  2. Apply action banner to background (first ~2.5 s, fade in/out).
  3. For each dialogue line:
     a. Resize Wav2Lip portrait to PiP height (~40 % canvas height = ~192 px).
     b. Position: char #0 -> bottom-left, char #1 -> bottom-right (alternating).
     c. Assign start time from character_tracks.start_ms (already set by VoiceSynthesizer).
  4. CompositeVideoClip([background, pip1, pip2, ...]).
  5. Apply timed subtitle strip (character name + text shown when each char speaks).
  6. Attach master scene WAV as audio.
  7. Write outputs/raw_scenes/scene_{id}_final.mp4

Fallbacks (graceful degradation, no exceptions):
  - Wav2Lip fails for a line -> still-portrait PiP with audio.
  - No portrait for character -> wide B-roll still as PiP.
  - Timing data missing       -> estimated from cumulative WAV durations.
  - moviepy not available     -> shutil.copy wide swapped video as final.
"""
from __future__ import annotations

import json
import logging
import shutil
import wave
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from src.agents.base_agent import BaseAgent
from src.schema import AudioTrack, SceneTask, SyncedScene, VideoFrame
from src.utils.memory import MemoryStore
from src.utils.prompts import LIP_SYNC_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional video/image deps
# ---------------------------------------------------------------------------
try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

_MOVIEPY_V2 = False
_MOVIEPY_AVAILABLE = False
try:
    from moviepy import (
        AudioFileClip,
        ColorClip,
        CompositeVideoClip,
        ImageClip,
        VideoFileClip,
        concatenate_videoclips,
    )
    _MOVIEPY_AVAILABLE = True
    _MOVIEPY_V2 = True
except ImportError:
    try:
        from moviepy.editor import (
            AudioFileClip,
            ColorClip,
            CompositeVideoClip,
            ImageClip,
            VideoFileClip,
            concatenate_videoclips,
        )
        _MOVIEPY_AVAILABLE = True
    except ImportError:
        _MOVIEPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
FPS = 24
RESOLUTION = (854, 480)          # final canvas  W x H

PIP_HEIGHT_RATIO   = 0.40        # PiP portrait: 40 % of canvas height  (~192 px)
SUBTITLE_BAR_H     = 46          # bottom subtitle strip height in px
PIP_PADDING        = 18          # distance from canvas edge to PiP box in px
ACTION_BANNER_SEC  = 2.5         # seconds the action banner is visible at scene start
GAP_BETWEEN_LINES  = 0.15        # seconds of silence between lines (used for timing estimates)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class LipSyncAgent(BaseAgent):
    """Convergence agent: per-line Wav2Lip + Visual Novel PiP scene composition."""

    def __init__(self, memory_store: Optional[MemoryStore] = None):
        super().__init__(name="Lip Sync", role="lip_sync", memory_store=memory_store)
        self.system_prompt = LIP_SYNC_SYSTEM_PROMPT
        self.output_dir = Path("outputs/raw_scenes")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._char_db: Dict[str, Dict] = {}

    # ------------------------------------------------------------------ #
    # Public entry                                                         #
    # ------------------------------------------------------------------ #

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        audio_track = self._coerce_audio_track(input_data)
        video_frame = self._coerce_video_frame(input_data)
        scene_task  = self._coerce_scene_task(input_data)

        if audio_track is None or video_frame is None:
            return {"success": False,
                    "error": "audio_track and video_frame are required",
                    "synced_scene": None}

        scene_id   = audio_track.scene_id
        out_root   = Path(input_data.get("output_dir", "outputs"))
        scenes_dir = out_root / "raw_scenes"
        scenes_dir.mkdir(parents=True, exist_ok=True)
        final_path = scenes_dir / f"scene_{scene_id}_final.mp4"

        self._load_char_db(input_data.get("character_db_path", "outputs/character_db.json"))
        wide_static = self._wide_static_path(scene_id, scenes_dir)
        action_text = (scene_task.actions[0] if scene_task and scene_task.actions else None)

        logger.info(f"Lip Sync: scene {scene_id} -> {final_path.name}  (PiP mode)")

        # ── 1. Build per-line tracks with timing ──────────────────────────
        raw_tracks = self._per_line_tracks(audio_track, scene_task)
        enriched   = self._fill_timing(raw_tracks)

        # ── 2. Build per-line PiP clips via Wav2Lip (no canvas normalization)
        pip_clips_info: List[Tuple[Path, str, float, float, str]] = []
        tmp_paths: List[Path] = []

        for idx, (char_name, line_audio_path, text, start_sec, end_sec) in enumerate(enriched):
            line_audio = Path(line_audio_path)
            if not line_audio.exists():
                logger.warning(f"  Missing audio for line {idx}: {line_audio_path}")
                continue
            tmp_clip = scenes_dir / f"_tmp_scene_{scene_id}_pip_{idx:02d}.mp4"
            tmp_paths.append(tmp_clip)
            ok = self._build_pip_clip(scene_id, char_name, line_audio, tmp_clip, wide_static)
            if ok and tmp_clip.exists():
                pip_clips_info.append((tmp_clip, char_name, start_sec, end_sec, text))
                logger.info(f"  PiP [{idx}] {char_name}  {start_sec:.2f}s -> {end_sec:.2f}s  ✓")
            else:
                logger.warning(f"  PiP clip failed for scene {scene_id} line {idx} ({char_name})")

        # ── 3. Master duration ────────────────────────────────────────────
        master_duration = audio_track.duration_seconds
        if master_duration <= 0:
            master_duration = self._wav_seconds(Path(audio_track.audio_path))
        if master_duration <= 0 and enriched:
            master_duration = enriched[-1][4]   # last end_sec

        # ── 4. PiP composition ────────────────────────────────────────────
        dialogues_for_sub = [
            (char, text, start_sec, end_sec)
            for (_, char, start_sec, end_sec, text) in pip_clips_info
        ]
        ok, duration, lip_synced = self._compose_pip_scene(
            scene_id        = scene_id,
            wide_static     = wide_static,
            master_audio    = Path(audio_track.audio_path),
            master_duration = master_duration,
            pip_clips_info  = pip_clips_info,
            action_text     = action_text,
            dialogues       = dialogues_for_sub,
            final_path      = final_path,
        )

        # ── 5. Cleanup tmp clips ──────────────────────────────────────────
        for p in tmp_paths:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass

        synced = SyncedScene(
            scene_id         = scene_id,
            video_path       = str(final_path),
            audio_path       = audio_track.audio_path,
            lip_synced       = lip_synced,
            duration_seconds = duration,
        )
        self.commit_to_memory("synced_scene", {
            "scene_id":        scene_id,
            "video_path":      str(final_path),
            "audio_path":      audio_track.audio_path,
            "duration_seconds": duration,
            "lip_synced":      lip_synced,
            "lines_processed": len(pip_clips_info),
            "completed_at":    datetime.now().isoformat(),
        })
        return {"success": ok, "synced_scene": synced, "scene_id": scene_id}

    # ------------------------------------------------------------------ #
    # Input coercion                                                       #
    # ------------------------------------------------------------------ #

    def _coerce_audio_track(self, data: Dict[str, Any]) -> Optional[AudioTrack]:
        at = data.get("audio_track")
        if isinstance(at, AudioTrack):
            return at
        raw = data.get("audio_track_dict") or (at if isinstance(at, dict) else None)
        return AudioTrack(**raw) if raw else None

    def _coerce_video_frame(self, data: Dict[str, Any]) -> Optional[VideoFrame]:
        vf = data.get("video_frame")
        if isinstance(vf, VideoFrame):
            return vf
        raw = data.get("video_frame_dict") or (vf if isinstance(vf, dict) else None)
        return VideoFrame(**raw) if raw else None

    def _coerce_scene_task(self, data: Dict[str, Any]) -> Optional[SceneTask]:
        st = data.get("scene_task")
        if isinstance(st, SceneTask):
            return st
        raw = data.get("scene_task_dict") or (st if isinstance(st, dict) else None)
        return SceneTask(**raw) if raw else None

    def _load_char_db(self, path: str):
        p = Path(path)
        if not p.exists():
            return
        try:
            text = p.read_text(encoding="utf-8")
            if text.startswith("\ufeff"):
                text = text[1:]
            db = json.loads(text)
            for char in db.get("characters", []):
                name = (char.get("name") or "").strip().lower()
                if name:
                    self._char_db[name] = char
        except Exception as exc:
            logger.warning(f"Could not load character_db: {exc}")

    def _portrait_for(self, character_name: str) -> Optional[Path]:
        if not character_name:
            return None
        char = self._char_db.get(character_name.lower(), {})
        ref  = char.get("image_reference")
        if ref and Path(ref).exists():
            return Path(ref)
        safe = "".join(c if c.isalnum() else "_" for c in character_name).strip("_")
        for stem in (safe, safe.lower()):
            for ext in ("png", "jpg", "jpeg"):
                p = Path("outputs/images") / f"{stem}.{ext}"
                if p.exists():
                    return p
        return None

    def _wide_static_path(self, scene_id: int, scenes_dir: Path) -> Optional[Path]:
        candidates = [
            scenes_dir / f"scene_{scene_id}_frames" / f"scene_{scene_id}_static_swapped.jpg",
            scenes_dir / f"scene_{scene_id}_frames" / f"scene_{scene_id}_static.jpg",
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    # ------------------------------------------------------------------ #
    # Timing helpers                                                       #
    # ------------------------------------------------------------------ #

    def _per_line_tracks(
        self,
        audio_track: AudioTrack,
        scene_task: Optional[SceneTask],
    ) -> List[Tuple[str, str, str, Optional[float], Optional[float]]]:
        """Return [(char, audio_path, text, start_ms|None, end_ms|None)]."""
        out       = []
        dialogues = scene_task.dialogues if scene_task else []

        for idx, ct in enumerate(audio_track.character_tracks or []):
            char     = ct.get("character") or "Narrator"
            path     = ct.get("path")
            text     = dialogues[idx].dialogue if idx < len(dialogues) else ""
            start_ms = ct.get("start_ms")
            end_ms   = ct.get("end_ms")
            if path:
                out.append((char, path, text, start_ms, end_ms))
        if out:
            return out

        # Fallback: conventional file naming, no timing data
        if scene_task is None:
            return []
        scene_audio_dir = Path(audio_track.audio_path).parent
        for idx, dlg in enumerate(scene_task.dialogues):
            p = scene_audio_dir / f"scene_{scene_task.scene_id}_line_{idx:02d}.wav"
            out.append((dlg.character or "Narrator", str(p), dlg.dialogue, None, None))
        return out

    def _fill_timing(
        self,
        tracks: List[Tuple[str, str, str, Optional[float], Optional[float]]],
    ) -> List[Tuple[str, str, str, float, float]]:
        """Convert start_ms/end_ms (may be None) to float seconds; estimate if missing."""
        result  = []
        cursor  = 0.0
        for char, path, text, start_ms, end_ms in tracks:
            dur = self._wav_seconds(Path(path)) if Path(path).exists() else 1.0
            s   = (start_ms / 1000.0) if start_ms is not None else cursor
            e   = (end_ms   / 1000.0) if end_ms   is not None else (s + dur)
            result.append((char, path, text, s, e))
            if start_ms is None:
                cursor = e + GAP_BETWEEN_LINES
        return result

    # ------------------------------------------------------------------ #
    # Wav2Lip per-line (PiP clip, no canvas normalization)                #
    # ------------------------------------------------------------------ #

    def _build_pip_clip(
        self,
        scene_id: int,
        character: str,
        line_audio: Path,
        out_path: Path,
        wide_static: Optional[Path],
    ) -> bool:
        """Build a raw Wav2Lip clip for PiP use. No full-canvas blurred backdrop."""
        portrait = self._portrait_for(character)
        if portrait is None:
            logger.info(f"  No portrait for '{character}' -> wide static as PiP source")
            if wide_static and wide_static.exists():
                return self._still_with_audio(wide_static, line_audio, out_path)
            return False

        if self._call_colab_wav2lip(portrait, line_audio, out_path):
            return out_path.exists() and out_path.stat().st_size > 0

        logger.info(f"  Wav2Lip failed for '{character}' -> still portrait PiP")
        return self._still_with_audio(portrait, line_audio, out_path)

    def _call_colab_wav2lip(self, face: Path, audio: Path, out_path: Path) -> bool:
        url_file = Path("config/colab_api.txt")
        if not url_file.exists():
            return False
        url = url_file.read_text().strip()
        if not url:
            return False
        try:
            logger.info(f"  Wav2Lip request: face={face.name} audio={audio.name}")
            with open(face, "rb") as fp, open(audio, "rb") as ap:
                files = {
                    "face": (face.name, fp,
                             "image/png" if face.suffix.lower() != ".mp4" else "video/mp4"),
                    "audio": (audio.name, ap, "audio/wav"),
                }
                resp = requests.post(
                    f"{url}/lip_sync", files=files,
                    headers={"ngrok-skip-browser-warning": "true"},
                    timeout=600,
                )
            if resp.status_code != 200:
                logger.warning(f"  Wav2Lip API {resp.status_code}: {resp.text[:300]}")
                return False
            out_path.write_bytes(resp.content)
            return out_path.exists() and out_path.stat().st_size > 0
        except Exception as exc:
            logger.warning(f"  Wav2Lip call failed: {exc}")
            return False

    def _still_with_audio(self, image: Path, audio: Path, out_path: Path) -> bool:
        if not _MOVIEPY_AVAILABLE:
            return False
        try:
            ac = AudioFileClip(str(audio))
            dur = max(0.6, float(ac.duration or 1.0))
            ic  = ImageClip(str(image))
            ic  = ic.with_duration(dur) if _MOVIEPY_V2 else ic.set_duration(dur)
            ic  = ic.with_fps(FPS)      if _MOVIEPY_V2 else ic.set_fps(FPS)
            out = ic.with_audio(ac)     if _MOVIEPY_V2 else ic.set_audio(ac)
            self._write_video(out, out_path)
            ic.close(); ac.close(); out.close()
            return out_path.exists()
        except Exception as exc:
            logger.warning(f"  still_with_audio failed: {exc}")
            return False

    # ------------------------------------------------------------------ #
    # PiP scene composition (primary render path)                         #
    # ------------------------------------------------------------------ #

    def _compose_pip_scene(
        self,
        scene_id: int,
        wide_static: Optional[Path],
        master_audio: Path,
        master_duration: float,
        pip_clips_info: List[Tuple[Path, str, float, float, str]],
        action_text: Optional[str],
        dialogues: List[Tuple[str, str, float, float]],
        final_path: Path,
    ) -> Tuple[bool, float, bool]:
        """
        Compose the final PiP scene:
          background (Ken Burns, full duration)
          + per-line PiP portrait overlays at their dialogue timestamps
          + timed subtitle strip
          + master audio
        """
        if not _MOVIEPY_AVAILABLE:
            return self._raw_fallback(final_path, wide_static, master_audio)

        canvas_w, canvas_h = RESOLUTION
        pip_h       = int(canvas_h * PIP_HEIGHT_RATIO)          # ~192 px
        pip_bottom  = canvas_h - SUBTITLE_BAR_H - PIP_PADDING - pip_h

        # ── Background: Ken Burns for entire scene duration ────────────
        bg_tmp = final_path.with_name(f"_tmp_scene_{scene_id}_bg.mp4")
        background = None
        try:
            if wide_static and wide_static.exists():
                bg_ok = self._render_ken_burns(wide_static, bg_tmp, master_duration,
                                               action_text=None)
                if bg_ok and bg_tmp.exists():
                    background = VideoFileClip(str(bg_tmp))
        except Exception as exc:
            logger.warning(f"Ken Burns background failed: {exc}")

        if background is None:
            background = ColorClip(size=(canvas_w, canvas_h), color=(10, 10, 12),
                                   duration=master_duration)
            background = (background.with_fps(FPS) if _MOVIEPY_V2
                          else background.set_fps(FPS))

        # ── Action banner (first ACTION_BANNER_SEC seconds only) ───────
        if action_text:
            background = self._apply_action_banner_timed(background, action_text)

        # ── Character -> left/right side assignment ────────────────────
        char_to_side: Dict[str, str] = {}
        for _, char, *_ in pip_clips_info:
            if char not in char_to_side:
                char_to_side[char] = "left" if len(char_to_side) % 2 == 0 else "right"

        # ── PiP overlay clips ──────────────────────────────────────────
        overlay_clips = [background]
        any_pip = False

        for pip_path, char, start_sec, end_sec, _ in pip_clips_info:
            if not pip_path.exists():
                continue
            try:
                pip = VideoFileClip(str(pip_path))
                # Strip embedded audio — master audio handles it
                pip = (pip.without_audio() if _MOVIEPY_V2 else pip.set_audio(None))

                # Resize to PiP height, preserving aspect ratio
                pip_src_w, pip_src_h = pip.size
                pip_w = max(1, int(round(pip_src_w * (pip_h / max(pip_src_h, 1)))))
                pip   = (pip.resized(height=pip_h) if _MOVIEPY_V2
                         else pip.resize(height=pip_h))

                # Position: bottom-left or bottom-right
                side  = char_to_side.get(char, "left")
                x_pos = (PIP_PADDING if side == "left"
                         else canvas_w - pip_w - PIP_PADDING)
                y_pos = pip_bottom

                if _MOVIEPY_V2:
                    pip = pip.with_position((x_pos, y_pos)).with_start(start_sec)
                else:
                    pip = pip.set_position((x_pos, y_pos)).set_start(start_sec)

                overlay_clips.append(pip)
                any_pip = True
                logger.info(f"  PiP overlay '{char}' @ {start_sec:.2f}s  side={side}  "
                            f"size={pip_w}x{pip_h}  y={y_pos}")
            except Exception as exc:
                logger.warning(f"  PiP overlay failed for '{char}': {exc}")

        # ── Composite ─────────────────────────────────────────────────
        try:
            composite = CompositeVideoClip(overlay_clips, size=(canvas_w, canvas_h))

            # Timed subtitle strip
            if dialogues and _PIL_AVAILABLE and np is not None:
                composite = self._apply_timed_subtitles(composite, dialogues)

            # Master audio
            if master_audio.exists():
                master_ac = AudioFileClip(str(master_audio))
                composite = (composite.with_audio(master_ac) if _MOVIEPY_V2
                             else composite.set_audio(master_ac))

            self._write_video(composite, final_path)
            duration = float(composite.duration or master_duration)

            # Cleanup
            composite.close()
            background.close()
            for c in overlay_clips[1:]:
                try: c.close()
                except Exception: pass
            if bg_tmp.exists():
                bg_tmp.unlink()

            return True, duration, any_pip

        except Exception as exc:
            logger.warning(f"PiP CompositeVideoClip failed: {exc}")
            try: background.close()
            except Exception: pass
            if bg_tmp.exists():
                bg_tmp.unlink()
            return self._raw_fallback(final_path, wide_static, master_audio)

    # ------------------------------------------------------------------ #
    # Visual overlays                                                      #
    # ------------------------------------------------------------------ #

    def _apply_action_banner_timed(self, clip, action_text: str,
                                    show_until: float = ACTION_BANNER_SEC):
        """Semi-transparent action description visible only for the first N seconds."""
        if not _PIL_AVAILABLE or np is None:
            return clip
        from PIL import ImageDraw, ImageFont, Image as _PilImage

        disp  = (action_text[:60] + "...") if len(action_text) > 60 else action_text
        box_w = max(400, 40 + len(disp) * 9)
        FADE  = 0.3

        def add_banner(get_frame, t):
            frame = get_frame(t)
            if t > show_until:
                return frame
            alpha = 255
            if t < FADE:
                alpha = int((t / FADE) * 255)
            elif t > show_until - FADE:
                alpha = int(((show_until - t) / FADE) * 255)
            alpha = max(0, min(255, alpha))
            img  = _PilImage.fromarray(frame)
            draw = ImageDraw.Draw(img, "RGBA")
            draw.rectangle([(20, 20), (box_w, 62)], fill=(0, 0, 0, int(alpha * 0.7)))
            try:
                font = ImageFont.truetype("arial.ttf", 18)
            except IOError:
                font = ImageFont.load_default()
            draw.text((30, 30), disp, font=font, fill=(255, 255, 255, alpha))
            return np.array(img)

        return clip.transform(add_banner) if _MOVIEPY_V2 else clip.fl(add_banner)

    def _apply_timed_subtitles(self, clip,
                                dialogues: List[Tuple[str, str, float, float]]):
        """Bottom subtitle strip: shows 'Character: line' when that char is speaking."""
        if not _PIL_AVAILABLE or np is None:
            return clip
        from PIL import ImageDraw, ImageFont, Image as _PilImage

        def add_subtitles(get_frame, t):
            frame  = get_frame(t)
            active = next(
                ((c, tx) for c, tx, s, e in dialogues if s <= t < e), None
            )
            if active is None:
                return frame

            char, text = active
            img  = _PilImage.fromarray(frame)
            draw = ImageDraw.Draw(img, "RGBA")
            w, h = img.width, img.height

            draw.rectangle([(0, h - SUBTITLE_BAR_H), (w, h)], fill=(0, 0, 0, 190))
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except IOError:
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", 20)
                except IOError:
                    font = ImageFont.load_default()

            text_str = f"{char}: {text}"
            if len(text_str) > 82:
                text_str = text_str[:79] + "..."

            cx, cy = w // 2, h - SUBTITLE_BAR_H // 2
            try:
                draw.text((cx, cy), text_str, font=font,
                          fill=(255, 255, 255, 255), anchor="mm")
            except TypeError:
                try:
                    bb = draw.textbbox((0, 0), text_str, font=font)
                    tw = bb[2] - bb[0]
                except AttributeError:
                    tw = len(text_str) * 10
                draw.text(((w - tw) // 2, h - SUBTITLE_BAR_H + 13),
                          text_str, font=font, fill=(255, 255, 255, 255))
            return np.array(img)

        return clip.transform(add_subtitles) if _MOVIEPY_V2 else clip.fl(add_subtitles)

    def _apply_action_banner(self, clip, action_text: str):
        """Legacy full-duration action banner (used if called directly from render_ken_burns)."""
        if not _PIL_AVAILABLE or np is None:
            return clip
        from PIL import ImageDraw, ImageFont, Image as _PilImage

        def add_banner(get_frame, t):
            frame = get_frame(t)
            alpha = 255
            if t < 0.3:
                alpha = int((t / 0.3) * 255)
            elif t > (clip.duration - 0.3):
                alpha = int(((clip.duration - t) / 0.3) * 255)
            alpha = max(0, min(255, alpha))
            img  = _PilImage.fromarray(frame)
            draw = ImageDraw.Draw(img, "RGBA")
            disp = (action_text[:60] + "...") if len(action_text) > 60 else action_text
            draw.rectangle([(20, 20), (max(400, 40 + len(disp) * 8), 60)],
                           fill=(0, 0, 0, int(alpha * 0.7)))
            try:
                font = ImageFont.truetype("arial.ttf", 18)
            except IOError:
                font = ImageFont.load_default()
            draw.text((30, 30), disp, font=font, fill=(255, 255, 255, alpha))
            return np.array(img)

        return clip.transform(add_banner) if _MOVIEPY_V2 else clip.fl(add_banner)

    def _apply_subtitle(self, clip, character: str, text: str):
        """Per-clip subtitle overlay (legacy single-clip version, kept for fallback)."""
        if not _PIL_AVAILABLE or np is None:
            return clip
        from PIL import ImageDraw, ImageFont, Image as _PilImage

        def add_sub(get_frame, t):
            frame = get_frame(t)
            img  = _PilImage.fromarray(frame)
            draw = ImageDraw.Draw(img, "RGBA")
            h, w = img.height, img.width
            text_str = f"{character}: {text}"
            draw.rectangle([(0, h - 50), (w, h)], fill=(0, 0, 0, 180))
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except IOError:
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", 20)
                except IOError:
                    font = ImageFont.load_default()
            draw.text((20, h - 40), text_str, font=font, fill=(255, 255, 255, 255))
            return np.array(img)

        return clip.transform(add_sub) if _MOVIEPY_V2 else clip.fl(add_sub)

    def _blurred_backdrop_clip(self, src_clip, duration):
        """Blurred + darkened still of first frame, canvas-sized. Used by _fit_to_canvas."""
        canvas_w, canvas_h = RESOLUTION
        try:
            frame = src_clip.get_frame(0)
            if cv2 is not None:
                bg = cv2.resize(frame, (canvas_w, canvas_h), interpolation=cv2.INTER_AREA)
                bg = cv2.GaussianBlur(bg, (0, 0), sigmaX=18)
                bg = (bg * 0.55).astype(bg.dtype)
            else:
                from PIL import Image as _PilImage, ImageFilter
                im = (_PilImage.fromarray(frame)
                      .resize((canvas_w, canvas_h))
                      .filter(ImageFilter.GaussianBlur(18)))
                bg = np.array(im) if np is not None else None
                if bg is not None:
                    bg = (bg * 0.55).astype("uint8")
            if bg is None:
                raise RuntimeError("no numpy/cv2 backend")
            ic = ImageClip(bg)
            return (ic.with_duration(duration) if _MOVIEPY_V2 else ic.set_duration(duration))
        except Exception:
            cc = ColorClip(size=(canvas_w, canvas_h), color=(10, 10, 12), duration=duration)
            return cc

    def _fit_to_canvas(self, clip):
        """Fit clip to RESOLUTION with blurred backdrop. Used in legacy normalize path."""
        canvas_w, canvas_h = RESOLUTION
        cw, ch = clip.size
        scale  = canvas_h / ch
        new_w  = int(round(cw * scale))
        scaled = (clip.resized(height=canvas_h) if _MOVIEPY_V2 else clip.resize(height=canvas_h))
        if new_w > canvas_w:
            scaled = (clip.resized(width=canvas_w) if _MOVIEPY_V2 else clip.resize(width=canvas_w))
            new_w, _ = scaled.size
        x_off = (canvas_w - new_w) // 2
        y_off = (canvas_h - scaled.size[1]) // 2
        backdrop = self._blurred_backdrop_clip(clip, scaled.duration)
        scaled   = (scaled.with_position((x_off, y_off)) if _MOVIEPY_V2
                    else scaled.set_position((x_off, y_off)))
        composite = CompositeVideoClip([backdrop, scaled], size=(canvas_w, canvas_h))
        if scaled.audio is not None:
            composite = (composite.with_audio(scaled.audio) if _MOVIEPY_V2
                         else composite.set_audio(scaled.audio))
        return composite

    def _normalize_clip(self, clip_path: Path,
                        character: Optional[str] = None,
                        text: Optional[str] = None) -> bool:
        """
        Legacy full-canvas normalization (concatenation mode).
        NOT called in the PiP path; kept for fallback.
        """
        if not _MOVIEPY_AVAILABLE or not clip_path.exists():
            return clip_path.exists()
        try:
            src    = VideoFileClip(str(clip_path))
            fitted = self._fit_to_canvas(src)
            if character and text:
                fitted = self._apply_subtitle(fitted, character, text)
            tmp = clip_path.with_name(clip_path.stem + "_norm.mp4")
            self._write_video(fitted, tmp)
            src.close(); fitted.close()
            shutil.move(str(tmp), str(clip_path))
            return clip_path.exists()
        except Exception as exc:
            logger.warning(f"normalize_clip failed: {exc}")
            return clip_path.exists()

    # ------------------------------------------------------------------ #
    # Ken Burns background                                                 #
    # ------------------------------------------------------------------ #

    def _render_ken_burns(self, image_path: Path, out_path: Path,
                           duration: float,
                           action_text: Optional[str] = None) -> bool:
        if not _MOVIEPY_AVAILABLE:
            return False
        try:
            ic = ImageClip(str(image_path))
            ic = ic.with_duration(duration) if _MOVIEPY_V2 else ic.set_duration(duration)
            ic = ic.with_fps(FPS)           if _MOVIEPY_V2 else ic.set_fps(FPS)

            def zoom(get_frame, t):
                f = get_frame(t)
                if cv2 is None:
                    return f
                # Fixed rate: reach max zoom (1.10x) at 3 s, then hold.
                scale = min(1.10, 1.0 + (0.10 / 3.0) * t)
                h, w  = f.shape[:2]
                nw    = int(round(w / scale))
                nh    = int(round(h / scale))
                x1    = (w - nw) // 2
                y1    = (h - nh) // 2
                crop  = f[y1:y1 + nh, x1:x1 + nw]
                return cv2.resize(crop, (RESOLUTION[0], RESOLUTION[1]),
                                  interpolation=cv2.INTER_LINEAR)

            zoomed = ic.transform(zoom) if _MOVIEPY_V2 else ic.fl(zoom)
            if action_text:
                zoomed = self._apply_action_banner(zoomed, action_text)
            self._write_video(zoomed, out_path)
            ic.close(); zoomed.close()
            return out_path.exists()
        except Exception as exc:
            logger.warning(f"Ken Burns failed: {exc}")
            return False

    # ------------------------------------------------------------------ #
    # Video I/O                                                            #
    # ------------------------------------------------------------------ #

    def _write_video(self, clip, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        kwargs = dict(codec="libx264", audio_codec="aac", fps=FPS, logger=None)
        try:
            clip.write_videofile(str(out_path), **kwargs)
        except TypeError:
            clip.write_videofile(str(out_path), codec="libx264", audio_codec="aac", fps=FPS)

    def _wav_seconds(self, path: Path) -> float:
        if not path.exists():
            return 0.0
        try:
            with wave.open(str(path), "r") as wf:
                return wf.getnframes() / float(wf.getframerate())
        except Exception:
            return 0.0

    # Alias used by workflow_phase2.py's MCP tool binding
    def _get_wav_duration(self, path: Path) -> float:
        return self._wav_seconds(path)

    def _merge_with_moviepy(
        self, video_path: Path, audio_path: Path,
        output_path: Path, duration: float,
    ) -> Tuple[bool, float]:
        """Mux a video and audio file; used by the MCP lip_sync_aligner tool."""
        if not _MOVIEPY_AVAILABLE:
            return False, duration
        try:
            vc = VideoFileClip(str(video_path))
            ac = AudioFileClip(str(audio_path))
            out = vc.with_audio(ac) if _MOVIEPY_V2 else vc.set_audio(ac)
            self._write_video(out, output_path)
            d = float(out.duration or duration)
            vc.close(); ac.close(); out.close()
            return output_path.exists(), d
        except Exception as exc:
            logger.warning(f"merge_with_moviepy failed: {exc}")
            return False, duration

    # ------------------------------------------------------------------ #
    # Last-resort fallback                                                 #
    # ------------------------------------------------------------------ #

    def _raw_fallback(self, final_path: Path, wide: Optional[Path],
                      audio: Path) -> Tuple[bool, float, bool]:
        """Write a still-with-audio clip or just copy the wide image."""
        if wide and audio.exists() and self._still_with_audio(wide, audio, final_path):
            return True, self._wav_seconds(audio), False
        if wide and wide.exists():
            shutil.copy2(str(wide), str(final_path))
            return True, 0.0, False
        return False, 0.0, False

    # ------------------------------------------------------------------ #
    # Legacy concatenation compose (kept as fallback, not primary path)   #
    # ------------------------------------------------------------------ #

    def _compose_scene(
        self,
        scene_id: int,
        establishing_path: Optional[Path],
        line_clips: List[Path],
        final_path: Path,
        fallback_audio: Path,
        fallback_wide: Optional[Path],
    ) -> Tuple[bool, float, bool]:
        """Legacy concatenation mode. Not called by execute(); preserved as fallback."""
        if not _MOVIEPY_AVAILABLE:
            return self._raw_fallback(final_path, fallback_wide, fallback_audio)

        clips = []
        if establishing_path and establishing_path.exists():
            try:
                clips.append(VideoFileClip(str(establishing_path)))
            except Exception as exc:
                logger.warning(f"establishing clip load failed: {exc}")
        for p in line_clips:
            try:
                clips.append(VideoFileClip(str(p)))
            except Exception as exc:
                logger.warning(f"line clip load failed: {exc}")

        if not clips:
            return self._raw_fallback(final_path, fallback_wide, fallback_audio)

        try:
            joined = concatenate_videoclips(clips, method="compose")
            self._write_video(joined, final_path)
            duration = float(joined.duration or 0.0)
            joined.close()
            for c in clips:
                try: c.close()
                except Exception: pass
            return True, duration, len(line_clips) > 0
        except Exception as exc:
            logger.warning(f"concatenate failed: {exc}")
            for c in clips:
                try: c.close()
                except Exception: pass
            return self._raw_fallback(final_path, fallback_wide, fallback_audio)
