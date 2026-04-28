"""
Face Swap Agent - Phase 2 video branch.

Responsibility (post-refactor)
------------------------------
Produce a single high-quality face-injected wide establishing image for the
scene.  The LipSyncAgent owns ALL video assembly (Ken Burns establishing clip
+ per-line Wav2Lip talking-heads), so this agent only needs to deliver the
swapped JPEG.

It runs Stable-Diffusion + IP-Adapter (Plus-Face) on the Colab backend to
inject the primary speaking character's identity into the wide B-roll. If the
backend is unavailable, the original wide static is reused (the pipeline is
designed to degrade gracefully).
"""
from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from src.agents.base_agent import BaseAgent
from src.schema import SceneTask, VideoFrame
from src.utils.memory import MemoryStore
from src.utils.prompts import FACE_SWAP_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class FaceSwapAgent(BaseAgent):
    def __init__(self, memory_store: Optional[MemoryStore] = None):
        super().__init__(name="Face Swap", role="face_swap", memory_store=memory_store)
        self.system_prompt = FACE_SWAP_SYSTEM_PROMPT
        self.char_db: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        video_frame = self._coerce_video_frame(input_data)
        if video_frame is None:
            return {"success": False, "error": "No video_frame provided"}

        scene_task = self._coerce_scene_task(input_data)
        char_db_path = input_data.get("character_db_path", "outputs/character_db.json")
        self._load_char_db(char_db_path)

        wide_static = Path(video_frame.raw_video_path)
        if not wide_static.exists():
            return {"success": False, "error": f"wide static not found: {wide_static}"}

        swapped_path = wide_static.with_name(f"{wide_static.stem}_swapped.jpg")

        # Identity validation gate (MCP face_swap stage 1)
        primary_char = self._primary_speaker(scene_task)
        portrait_path = self._portrait_for(primary_char) if primary_char else None

        if not portrait_path or not portrait_path.exists():
            logger.info(
                f"FaceSwap scene {video_frame.scene_id}: no usable portrait for "
                f"'{primary_char}', keeping raw wide."
            )
            shutil.copy2(str(wide_static), str(swapped_path))
            self._commit(video_frame, swapped_path, swapped=False)
            return {
                "success": True,
                "swapped_image_path": str(swapped_path),
                "scene_id": video_frame.scene_id,
                "frames_processed": 1,
            }

        prompt = self._build_prompt(scene_task)
        ok = self._call_colab_faceswap(
            prompt=prompt,
            portrait_path=portrait_path,
            base_path=wide_static,
            out_path=swapped_path,
        )

        if not ok:
            logger.warning(
                f"FaceSwap scene {video_frame.scene_id}: IP-Adapter failed, "
                "falling back to raw wide."
            )
            shutil.copy2(str(wide_static), str(swapped_path))
            self._commit(video_frame, swapped_path, swapped=False)
        else:
            self._commit(video_frame, swapped_path, swapped=True)

        return {
            "success": True,
            "swapped_image_path": str(swapped_path),
            "scene_id": video_frame.scene_id,
            "frames_processed": 1,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _coerce_video_frame(self, data: Dict[str, Any]) -> Optional[VideoFrame]:
        vf = data.get("video_frame")
        if isinstance(vf, VideoFrame):
            return vf
        raw = data.get("video_frame_dict") or (vf if isinstance(vf, dict) else None)
        if raw:
            return VideoFrame(**raw)
        return None

    def _coerce_scene_task(self, data: Dict[str, Any]) -> Optional[SceneTask]:
        st = data.get("scene_task")
        if isinstance(st, SceneTask):
            return st
        raw = data.get("scene_task_dict") or (st if isinstance(st, dict) else None)
        if raw:
            return SceneTask(**raw)
        return None

    def _primary_speaker(self, scene_task: Optional[SceneTask]) -> Optional[str]:
        if not scene_task or not scene_task.dialogues:
            return None
        for d in scene_task.dialogues:
            if d.character and d.character.strip().lower() != "narrator":
                return d.character.strip().lower()
        return None

    def _build_prompt(self, scene_task: Optional[SceneTask]) -> str:
        if scene_task is None:
            return "cinematic photograph, soft lighting, a person standing in scene"
        prompt = (
            f"{scene_task.heading}, cinematic photograph, soft lighting, "
            "a person standing in the scene, facing the camera, photorealistic, sharp focus"
        )
        if scene_task.actions:
            prompt += f", {scene_task.actions[0]}"
        elif scene_task.visual_cues:
            prompt += f", {scene_task.visual_cues[0]}"
        return prompt

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
                    self.char_db[name] = char
        except Exception:
            pass

    def _portrait_for(self, char_name: str) -> Optional[Path]:
        char = self.char_db.get(char_name, {})
        ref = char.get("image_reference")
        if ref and Path(ref).exists():
            return Path(ref)
        safe = "".join(c if c.isalnum() else "_" for c in char_name).strip("_")
        for stem in (safe, safe.lower()):
            for ext in ("png", "jpg", "jpeg"):
                p = Path("outputs/images") / f"{stem}.{ext}"
                if p.exists():
                    return p
        return None

    def _call_colab_faceswap(self, prompt: str, portrait_path: Path,
                             base_path: Path, out_path: Path) -> bool:
        url_file = Path("config/colab_api.txt")
        if not url_file.exists():
            return False
        url = url_file.read_text().strip()
        if not url:
            return False

        try:
            logger.info("FaceSwap: calling Colab IP-Adapter (this may take ~30s)...")
            with open(portrait_path, "rb") as p_file, open(base_path, "rb") as b_file:
                files = {
                    "swap_image": ("portrait.png", p_file, "image/png"),
                    "base_image": ("base.jpg", b_file, "image/jpeg"),
                }
                data = {"prompt": prompt, "ip_scale": "0.85", "strength": "0.45", "steps": "30"}
                headers = {"ngrok-skip-browser-warning": "true"}
                resp = requests.post(
                    f"{url}/face_swap", files=files, data=data, headers=headers, timeout=300
                )
            if resp.status_code != 200:
                logger.error(f"IP-Adapter API status {resp.status_code}: {resp.text[:200]}")
                return False
            out_path.write_bytes(resp.content)
            return out_path.exists() and out_path.stat().st_size > 0
        except Exception as exc:
            logger.error(f"IP-Adapter call failed: {exc}")
            return False

    def _commit(self, video_frame: VideoFrame, swapped_path: Path, swapped: bool):
        self.commit_to_memory(
            "face_swap_result",
            {
                "scene_id": video_frame.scene_id,
                "swapped_image_path": str(swapped_path),
                "ip_adapter_applied": swapped,
                "completed_at": datetime.now().isoformat(),
            },
        )
