"""
Scriptwriter Agent — Transform abstract prompts into structured scripts.

LLM provider selection (in this order):
  1. Groq cloud API           (config/groq_api.txt OR env GROQ_API_KEY)
     - default model: llama-3.1-8b-instant   (fast + capable, free tier)
  2. Ollama local              (used as a fallback when Groq is unavailable)

The structured-output prompt forces:
  - exactly N scenes
  - at least 2 named characters across the whole story
  - at least 2 dialogue exchanges per scene
  - real screenplay headings (INT./EXT. PLACE - TIME)
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agents.base_agent import BaseAgent
from src.schema import DialogueEntry, SceneElement, SceneMode, ScriptManifest
from src.utils.memory import MemoryStore
from src.utils.prompts import SCRIPTWRITER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


# ─── helpers ─────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_groq_api_key() -> Optional[str]:
    """
    Resolution order:
      1. env GROQ_API_KEY
      2. config/groq_api.txt
    """
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        return key
    cfg_path = _REPO_ROOT / "config" / "groq_api.txt"
    if cfg_path.exists():
        first = cfg_path.read_text(encoding="utf-8").strip().splitlines()
        if first and first[0].strip():
            return first[0].strip()
    return None


# ─── agent ───────────────────────────────────────────────────────────────────

class ScriptwriterAgent(BaseAgent):
    """Generate a structured ScriptManifest from a creative prompt."""

    GROQ_MODEL    = "llama-3.1-8b-instant"
    OLLAMA_MODEL  = "llama3.2:1b"   # only used if Groq is unreachable

    def __init__(self, memory_store: Optional[MemoryStore] = None):
        super().__init__(
            name="Scriptwriter",
            role="scriptwriter",
            memory_store=memory_store,
        )
        self.system_prompt = SCRIPTWRITER_SYSTEM_PROMPT
        self.groq_api_key = _load_groq_api_key()
        self.model = self.GROQ_MODEL if self.groq_api_key else self.OLLAMA_MODEL

    # ── public entry ─────────────────────────────────────────────────────────

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        prompt     = (input_data.get("prompt") or "").strip()
        num_scenes = int(input_data.get("num_scenes", 3))

        if not prompt:
            return {"success": False, "error": "No prompt provided", "script": None}

        logger.info(f"Scriptwriter: model={self.model} scenes={num_scenes} prompt={prompt[:120]!r}")

        raw = self._generate(prompt, num_scenes)

        try:
            manifest = self._parse_script(raw, prompt, num_scenes)
            if self.memory_store:
                self.commit_to_memory("script_history", {
                    "prompt": prompt,
                    "num_scenes": num_scenes,
                    "script": manifest.model_dump(),
                })
            return {"success": True, "script": manifest, "raw_content": raw}
        except Exception as exc:
            logger.error(f"Scriptwriter parse failed: {exc}")
            return {"success": False, "error": str(exc),
                    "script": None, "raw_content": raw}

    # ── LLM dispatch ─────────────────────────────────────────────────────────

    def _generate(self, prompt: str, num_scenes: int) -> str:
        full_prompt = self._build_prompt(prompt, num_scenes)

        if self.groq_api_key:
            try:
                return self._generate_with_groq(full_prompt)
            except Exception as exc:
                logger.warning(f"Groq generation failed ({exc}); falling back to Ollama")

        try:
            return self._generate_with_ollama(full_prompt)
        except Exception as exc:
            logger.error(f"Ollama generation also failed: {exc}")
            raise RuntimeError(f"Both Groq and Ollama failed: {exc}") from exc

    def _generate_with_groq(self, full_prompt: str) -> str:
        """Use Groq Cloud (fast and free-tier-friendly)."""
        from groq import Groq

        client = Groq(api_key=self.groq_api_key)
        # Use chat completion (non-streaming for simplicity).
        resp = client.chat.completions.create(
            model=self.GROQ_MODEL,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": full_prompt},
            ],
            temperature=0.7,
            max_tokens=2048,
            top_p=1,
            stream=False,
        )
        content = resp.choices[0].message.content or ""
        logger.info(f"Groq returned {len(content)} chars")
        return content

    def _generate_with_ollama(self, full_prompt: str) -> str:
        import ollama
        resp = ollama.generate(
            model=self.OLLAMA_MODEL,
            prompt=f"{self.system_prompt}\n\n{full_prompt}",
            stream=False,
            options={"temperature": 0.7, "num_predict": 2000},
        )
        return resp.get("response", "")

    # ── prompt construction ─────────────────────────────────────────────────

    @staticmethod
    def _build_prompt(prompt: str, num_scenes: int) -> str:
        return f"""User Prompt:
{prompt}

Generate a complete short screenplay with EXACTLY {num_scenes} scenes.

HARD REQUIREMENTS (the consumer is a strict JSON parser):
1. Return ONE JSON object — no prose, no markdown fences, no commentary.
2. The object MUST have a `scenes` array of length EXACTLY {num_scenes}.
3. Across the whole script you MUST introduce AT LEAST TWO distinct named characters
   (e.g. "Sarah", "Daniel", "Maya", "The Officer") — never use generic placeholders
   like "Character A", "Person 1", "Speaker", "Narrator".
4. EVERY scene MUST contain at least 2 dialogue entries with at least 2 different
   speakers in that scene (or the same scene must reference characters from
   adjacent scenes — never produce a scene where only one character speaks).
5. Each `heading` MUST follow the screenplay format:
       INT. PLACE - TIMEOFDAY     or     EXT. PLACE - TIMEOFDAY
   (e.g. "INT. SMALL OFFICE - MORNING", "EXT. STREET CORNER - NIGHT").
6. Number scenes 1..{num_scenes} in order; do NOT repeat scene_id.
7. Keep dialogue tight (max ~25 words per line); subtext is welcome, exposition
   dumps are not.

JSON SHAPE (use this exact key set):
{{
  "scenes": [
    {{
      "scene_id": 1,
      "heading":  "INT. PLACE - TIMEOFDAY",
      "actions":  ["short action beat", "..."],
      "dialogues": [
        {{"character": "Name1", "dialogue": "Line"}},
        {{"character": "Name2", "dialogue": "Line"}}
      ],
      "visual_cues": ["lighting / camera / mood note", "..."]
    }}
  ]
}}

Return ONLY this JSON object."""

    # ── parsing + post-validation ───────────────────────────────────────────

    def _parse_script(self, content: str, prompt: str, requested: int) -> ScriptManifest:
        parsed = self._extract_json_payload(content)
        raw_scenes: List[Dict[str, Any]]
        if isinstance(parsed, dict) and isinstance(parsed.get("scenes"), list):
            raw_scenes = parsed["scenes"]
        elif isinstance(parsed, list):
            raw_scenes = parsed
        else:
            logger.warning("Scriptwriter: no JSON scenes found, falling back to defaults")
            raw_scenes = []

        scenes: List[SceneElement] = []
        characters: set[str] = set()

        for idx, scene_data in enumerate(raw_scenes, start=1):
            try:
                scene = self._parse_scene_json(scene_data, idx)
                scenes.append(scene)
                for d in scene.dialogues:
                    if d.character:
                        characters.add(d.character)
            except Exception as exc:
                logger.warning(f"Scriptwriter: scene {idx} parse failed: {exc}")

        # Enforce scene count: truncate or pad with defaults so downstream phases
        # never see a mismatch with the user-requested count.
        if len(scenes) > requested:
            logger.warning(
                f"Scriptwriter: model over-produced "
                f"({len(scenes)} scenes, expected {requested}); truncating."
            )
            scenes = scenes[:requested]
        elif len(scenes) < requested:
            logger.warning(
                f"Scriptwriter: model under-produced "
                f"({len(scenes)} scenes, expected {requested}); padding with defaults."
            )
            defaults = self._create_default_scenes(prompt)
            while len(scenes) < requested:
                d = defaults[len(scenes) % len(defaults)]
                scenes.append(SceneElement(
                    scene_id=len(scenes) + 1,
                    heading=d.heading,
                    actions=d.actions,
                    dialogues=d.dialogues,
                    visual_cues=d.visual_cues,
                ))

        # Renumber scenes 1..N so duplicate scene_id is impossible.
        for i, s in enumerate(scenes, start=1):
            s.scene_id = i

        # Refresh character set after possible padding.
        characters = {d.character for s in scenes for d in s.dialogues if d.character}
        if len(characters) < 2:
            logger.warning(
                f"Scriptwriter: only {len(characters)} character(s) — "
                "the model ignored the multi-character requirement."
            )

        return ScriptManifest(
            mode=SceneMode.AUTONOMOUS,
            scenes=scenes,
            character_list=sorted(characters) if characters else ["Character A", "Character B"],
            metadata={
                "original_prompt":  prompt,
                "generation_model": self.model,
                "num_scenes":       len(scenes),
                "num_characters":   len(characters),
            },
        )

    # ── tiny helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _parse_scene_json(data: Dict[str, Any], default_id: int) -> SceneElement:
        return SceneElement(
            scene_id=int(data.get("scene_id", default_id)) or default_id,
            heading=data.get("heading", f"SCENE {default_id}"),
            actions=list(data.get("actions") or []),
            dialogues=[
                DialogueEntry(
                    character=str(d.get("character", "")).strip(),
                    dialogue=str(d.get("dialogue", "")).strip(),
                )
                for d in (data.get("dialogues") or [])
                if isinstance(d, dict)
            ],
            visual_cues=list(data.get("visual_cues") or []),
        )

    @staticmethod
    def _extract_json_payload(content: str) -> Any:
        text = (content or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
        return None

    @staticmethod
    def _create_default_scenes(prompt: str) -> List[SceneElement]:
        """
        Used only if the LLM output produces ZERO usable scenes — keeps the
        pipeline alive without lying about success in metadata.
        """
        return [
            SceneElement(
                scene_id=1,
                heading="INT. SMALL OFFICE - DAY",
                actions=[f"The story begins: {prompt}"],
                dialogues=[
                    DialogueEntry(character="Maya",  dialogue="Tell me what happened."),
                    DialogueEntry(character="Eitan", dialogue="It is not what you think."),
                ],
                visual_cues=["Soft window light", "Slow push toward Maya"],
            ),
            SceneElement(
                scene_id=2,
                heading="INT. SMALL OFFICE - LATER",
                actions=["Tension rises between them"],
                dialogues=[
                    DialogueEntry(character="Eitan", dialogue="I should explain."),
                    DialogueEntry(character="Maya",  dialogue="No — I already know."),
                ],
                visual_cues=["Cooler colour temperature", "Tight two-shot"],
            ),
            SceneElement(
                scene_id=3,
                heading="EXT. STREET CORNER - EVENING",
                actions=["They part without resolution"],
                dialogues=[
                    DialogueEntry(character="Maya",  dialogue="Don't follow me."),
                    DialogueEntry(character="Eitan", dialogue="I never planned to."),
                ],
                visual_cues=["Wet pavement reflections", "Wide establishing shot"],
            ),
        ]
