"""
Video Generator Agent - Phase 2 video branch.

Responsibility (post-refactor)
------------------------------
Produce a single high-quality wide *establishing* B-roll image for the scene.
The image is sized to match the final 16:9 canvas so downstream Ken Burns
zooms remain crisp. Returns a VideoFrame whose `raw_video_path` points at the
image (NOT a video) - that's intentional; LipSyncAgent renders all video.
"""
from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import requests

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False

from PIL import Image

from src.agents.base_agent import BaseAgent
from src.schema import SceneTask, VideoFrame
from src.utils.memory import MemoryStore
from src.utils.prompts import VIDEO_GENERATOR_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


# 16:9 canvas at standard 480p (matches RESOLUTION in lip_sync.py).
TARGET_W, TARGET_H = 854, 480
SD_W, SD_H = 896, 512  # SD 1.5 friendly multiples of 8 close to 16:9 480p


class VideoGeneratorAgent(BaseAgent):
    def __init__(self, memory_store: Optional[MemoryStore] = None):
        super().__init__(name="Video Generator", role="video_generator", memory_store=memory_store)
        self.system_prompt = VIDEO_GENERATOR_SYSTEM_PROMPT
        self._char_db: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        scene_task = self._coerce_scene_task(input_data)
        if scene_task is None:
            return {"success": False, "error": "No scene_task provided"}

        self._load_char_db(input_data.get("character_db_path", "outputs/character_db.json"))

        out_root = Path(input_data.get("output_dir", "outputs"))
        scenes_dir = out_root / "raw_scenes"
        frame_dir = scenes_dir / f"scene_{scene_task.scene_id}_frames"
        frame_dir.mkdir(parents=True, exist_ok=True)

        prompt = self._build_prompt(scene_task)
        logger.info(f"Video Generator: scene {scene_task.scene_id} prompt: {prompt}")

        bg = self._generate_broll(prompt)
        wide_path = frame_dir / f"scene_{scene_task.scene_id}_static.jpg"
        if _CV2:
            cv2.imwrite(str(wide_path), bg, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        else:
            Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB) if _CV2 else bg).save(
                str(wide_path), quality=92
            )

        vf = VideoFrame(
            scene_id=scene_task.scene_id,
            raw_video_path=str(wide_path),
            frame_dir=str(frame_dir),
            frame_count=1,
        )
        return {"success": True, "video_frame": vf}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _coerce_scene_task(self, data: Dict[str, Any]) -> Optional[SceneTask]:
        st = data.get("scene_task")
        if isinstance(st, SceneTask):
            return st
        raw = data.get("scene_task_dict") or (st if isinstance(st, dict) else None)
        if raw:
            return SceneTask(**raw)
        return None

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
                self._char_db[(char.get("name") or "").strip().lower()] = char
        except Exception as exc:
            logger.warning(f"Could not load character_db: {exc}")

    def _build_prompt(self, scene_task: SceneTask) -> str:
        """
        Build a scene-specific SD prompt for the wide establishing B-roll.

        Priority order (most scene-differentiating first):
          1. Visual cues — the art-director note, usually the most specific.
          2. Action description — gives mood and staging.
          3. Heading context — location / time of day.
          4. Cinematic boilerplate — quality anchor.

        Keeps the background *empty of recognisable faces/people* so that the
        IP-Adapter face-swap can inject the correct character later without
        fighting an existing face.
        """
        parts: list[str] = []

        # 1. Visual cues lead (most differentiating)
        if scene_task.visual_cues:
            parts.extend(scene_task.visual_cues[:2])

        # 2. Action mood — extract environment/mood words, skip character names
        if scene_task.actions:
            action_text = scene_task.actions[0]
            # Strip known character names so they don't anchor the BG prompt
            char_names = {c.lower() for c in (scene_task.character_names or [])}
            filtered = " ".join(
                w for w in action_text.split()
                if w.lower().rstrip(".,") not in char_names
            )
            if filtered.strip():
                parts.append(filtered.strip())

        # 3. Heading provides location / time-of-day context
        if scene_task.heading:
            parts.append(scene_task.heading)

        # 4. Cinematic quality boilerplate
        parts += [
            "cinematic photograph",
            "establishing shot",
            "wide angle",
            "soft natural lighting",
            "shallow depth of field",
            "rich environmental detail",
            "8k",
        ]

        return ", ".join(parts)

    def _get_colab_url(self) -> str:
        path = Path("config/colab_api.txt")
        if path.exists():
            return path.read_text().strip()
        return ""

    def _generate_broll(self, prompt: str) -> np.ndarray:
        url = self._get_colab_url()
        # Default placeholder (BGR, dark gray)
        placeholder = np.full((TARGET_H, TARGET_W, 3), 32, dtype=np.uint8)
        if not url:
            return placeholder

        try:
            payload = {
                "prompt": prompt,
                "negative_prompt": (
                    "people, person, faces, characters, portraits, humans, crowd, "
                    "text, watermark, logo, blurry, distorted, low quality, deformed, "
                    "mutated, ugly, extra limbs, bad anatomy, multiple heads"
                ),
                "width": SD_W,
                "height": SD_H,
                "steps": 28,
            }
            headers = {"ngrok-skip-browser-warning": "true"}
            resp = requests.post(f"{url}/generate", data=payload, headers=headers, timeout=180)
            if resp.status_code != 200:
                logger.warning(f"Colab /generate returned {resp.status_code}: {resp.text[:200]}")
                return placeholder

            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            arr = np.array(img)
            if _CV2:
                bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                return cv2.resize(bgr, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
            return arr
        except Exception as exc:
            logger.warning(f"Failed to generate B-roll: {exc}")
            return placeholder
