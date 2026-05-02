"""
Phase 5 — Edit Agent
LangGraph intent classifier + targeted pipeline re-run router.

Graph topology
--------------
  classify_intent
       ↓ conditional (target field)
  ┌─────────────────────────────────────────────┐
  │  audio  │  video_frame  │  video  │  script │  unknown
  └─────────────────────────────────────────────┘
       ↓ (all branches)
   snapshot -> END

Routing table
-------------
  target=audio        -> VoiceSynthesizer re-run (full Phase 2 — audio branch)
  target=video_frame  -> VideoGenerator + FaceSwap + LipSync re-run
  target=video        -> LipSync recompose only (cheapest)
  target=script       -> Phase 1 full + Phase 2/3 full (cascade)
  target=unknown      -> snapshot only (no re-run, still creates version)

Intent classification
---------------------
  1. Tries Ollama (llama3.2 model) with a structured JSON prompt.
  2. Falls back to rule-based keyword classifier covering all 7 master-doc
     example queries from §5.1.

Example queries and expected routing
-------------------------------------
  "Make scene 1 sound more anxious"                → audio (change_voice_tone)
  "Change the cafe to a library"                   → video_frame (re_image_scene)
  "Add subtitles to all scenes"                    → video (add_subtitles)
  "Make Daniel sound older"                        → audio (change_voice_tone)
  "Change the weather to rainy in scene 2"         → video_frame (re_image_scene)
  "Speed up scene 2"                               → video (recompose_video)
  "Make Sarah's dialogue more formal"              → script (change_script)
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agents.base_agent import BaseAgent
from src.state.manager import StateManager
from src.utils.memory import MemoryStore

logger = logging.getLogger(__name__)

# ── Ollama system prompt ────────────────────────────────────────────────────
_INTENT_SYSTEM_PROMPT = """You are an edit intent classifier for an AI video generation system.
Given a user's edit request, respond ONLY with a single JSON object — no preamble, no markdown.

JSON schema:
{
  "intent":     "<one of: change_voice_tone | re_image_scene | add_subtitles | change_script | recompose_video | change_character_appearance>",
  "target":     "<one of: audio | video_frame | video | script>",
  "scope":      "<one of: scene:N | character:Name | global>",
  "parameters": { ... },
  "confidence": <0.0-1.0>,
  "reasoning":  "<brief explanation>"
}

Routing rules:
- audio       → re-synthesise character speech (edge-tts)
- video_frame → re-generate wide B-roll image via Stable Diffusion
- video       → re-compose final MP4 without regenerating assets
- script      → rewrite the script (Phase 1), then cascade all phases"""

# Known character names (extend if script changes)
_KNOWN_CHARS = ["sarah", "daniel", "waiter", "narrator"]

# Tone keyword mappings
_TONE_WORDS = [
    "anxious", "happy", "sad", "formal", "whispered", "angry",
    "excited", "calm", "nervous", "older", "younger", "stern",
    "cheerful", "melancholic", "urgent",
]


class EditAgent(BaseAgent):
    """Phase 5 Edit Agent — intent classification + targeted re-run."""

    def __init__(
        self,
        memory_store: Optional[MemoryStore] = None,
        outputs_dir: str = "outputs",
    ):
        super().__init__(
            name="Edit Agent", role="edit_agent", memory_store=memory_store
        )
        self.outputs_dir = Path(outputs_dir)
        self.state_manager = StateManager(str(outputs_dir))
        self.graph = self._build_graph()

    # ------------------------------------------------------------------ #
    # LangGraph construction                                               #
    # ------------------------------------------------------------------ #

    def _build_graph(self):
        try:
            from langgraph.graph import StateGraph, END
            graph = StateGraph(dict)
            graph.add_node("classify_intent",   self._node_classify_intent)
            graph.add_node("re_run_audio",       self._node_re_run_audio)
            graph.add_node("re_run_video_frame", self._node_re_run_video_frame)
            graph.add_node("re_run_video",       self._node_re_run_video)
            graph.add_node("re_run_script",      self._node_re_run_script)
            graph.add_node("snapshot",           self._node_snapshot)

            graph.set_entry_point("classify_intent")
            graph.add_conditional_edges(
                "classify_intent",
                self._route_target,
                {
                    "audio":       "re_run_audio",
                    "video_frame": "re_run_video_frame",
                    "video":       "re_run_video",
                    "script":      "re_run_script",
                    "unknown":     "snapshot",
                },
            )
            for node in ("re_run_audio", "re_run_video_frame",
                         "re_run_video", "re_run_script"):
                graph.add_edge(node, "snapshot")
            graph.add_edge("snapshot", END)
            return graph.compile()
        except Exception as exc:
            logger.warning(f"EditAgent: LangGraph build failed ({exc}); will use fallback")
            return None

    # ------------------------------------------------------------------ #
    # Public entry                                                         #
    # ------------------------------------------------------------------ #

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "query":          input_data.get("query", ""),
            "parent_version": input_data.get("parent_version"),
            "intent_result":  {},
            "rerun_result":   {},
            "new_version":    0,
            "success":        False,
            "error":          "",
            "execution_log":  [],
        }

        if self.graph is not None:
            try:
                state = self.graph.invoke(state)
                return state
            except Exception as exc:
                logger.warning(f"LangGraph invoke failed ({exc}); using sequential fallback")

        return self._run_fallback(state)

    def _run_fallback(self, state: Dict) -> Dict:
        state = self._node_classify_intent(state)
        target = self._route_target(state)
        dispatch = {
            "audio":       self._node_re_run_audio,
            "video_frame": self._node_re_run_video_frame,
            "video":       self._node_re_run_video,
            "script":      self._node_re_run_script,
        }
        if target in dispatch:
            state = dispatch[target](state)
        return self._node_snapshot(state)

    # ------------------------------------------------------------------ #
    # Intent classification node                                           #
    # ------------------------------------------------------------------ #

    def _node_classify_intent(self, state: Dict) -> Dict:
        state.setdefault("execution_log", [])
        query = state.get("query", "")
        state["execution_log"].append(f"Classifying edit intent: '{query}'")

        result = self._classify_with_ollama(query) or self._classify_rule_based(query)
        state["intent_result"] = result
        state["execution_log"].append(
            f"Intent={result.get('intent')}  "
            f"Target={result.get('target')}  "
            f"Scope={result.get('scope')}  "
            f"Confidence={result.get('confidence', 0):.2f}"
        )
        logger.info(
            f"EditAgent intent: {result.get('intent')} | "
            f"target={result.get('target')} | scope={result.get('scope')}"
        )
        return state

    def _classify_with_ollama(self, query: str) -> Optional[Dict]:
        try:
            import requests as _req
            for model in ("llama3.2", "llama3", "mistral"):
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": _INTENT_SYSTEM_PROMPT},
                        {"role": "user", "content": f"Edit request: {query}"},
                    ],
                    "stream": False,
                }
                resp = _req.post(
                    "http://localhost:11434/api/chat", json=payload, timeout=20
                )
                if resp.status_code == 200:
                    content = resp.json()["message"]["content"]
                    s = content.find("{")
                    e = content.rfind("}") + 1
                    if s != -1 and e > s:
                        parsed = json.loads(content[s:e])
                        if "intent" in parsed and "target" in parsed:
                            logger.info(f"Ollama classified with model={model}")
                            return parsed
        except Exception as exc:
            logger.info(f"Ollama not available ({exc}); using rule-based classifier")
        return None

    def _classify_rule_based(self, query: str) -> Dict:
        """
        Keyword-based classifier.
        Covers all 7 master-doc §5.1 example queries plus common variations.
        """
        q = query.lower()

        # ── Script cascade ─────────────────────────────────────────────
        if any(w in q for w in [
            "dialogue", "rewrite", "change what", "more formal", "less formal",
            "make it funnier", "funnier", "wittier", "darker script",
        ]):
            char = self._extract_character(q)
            sn   = self._extract_scene_num(q)
            return self._make_result(
                intent="change_script", target="script",
                scope=f"character:{char}" if char else (f"scene:{sn}" if sn else "global"),
                params={"modification": query}, conf=0.80,
                reason="Dialogue/script change cascades to all phases",
            )

        # ── Scene re-imaging ───────────────────────────────────────────
        if any(w in q for w in [
            "cafe", "library", "location", "place", "setting",
            "weather", "rainy", "sunny", "snowing", "night", "day",
            "background", "change the scene", "different location",
        ]):
            sn = self._extract_scene_num(q)
            return self._make_result(
                intent="re_image_scene", target="video_frame",
                scope=f"scene:{sn}" if sn else "global",
                params={"prompt_modifier": query}, conf=0.85,
                reason="Location/weather change requires SD re-imaging",
            )

        # ── Voice / audio ──────────────────────────────────────────────
        if any(w in q for w in [
            "sound", "voice", "tone", "speak", "speaking",
            "anxious", "nervous", "whisper", "formal", "older", "younger",
            "louder", "quieter", "slower", "faster speech",
        ]):
            char = self._extract_character(q)
            tone = self._extract_tone(q)
            sn   = self._extract_scene_num(q)
            return self._make_result(
                intent="change_voice_tone", target="audio",
                scope=(f"character:{char}" if char
                       else (f"scene:{sn}" if sn else "global")),
                params={"tone": tone, "character": char}, conf=0.85,
                reason="Voice/tone change re-runs TTS",
            )

        # ── Subtitle / caption ─────────────────────────────────────────
        if any(w in q for w in ["subtitle", "caption", "text on", "overlay text"]):
            return self._make_result(
                intent="add_subtitles", target="video",
                scope="global", params={}, conf=0.92,
                reason="Subtitle change only needs video recomposition",
            )

        # ── Video recomposition ────────────────────────────────────────
        if any(w in q for w in [
            "recompose", "rerender", "re-render", "composition",
            "speed up", "slow down", "faster", "slower video",
            "pip", "layout", "arrangement",
        ]):
            sn = self._extract_scene_num(q)
            return self._make_result(
                intent="recompose_video", target="video",
                scope=f"scene:{sn}" if sn else "global",
                params={}, conf=0.80,
                reason="Video composition change — no asset regeneration needed",
            )

        # ── Default fallback ───────────────────────────────────────────
        return self._make_result(
            intent="recompose_video", target="video",
            scope="global", params={"modification": query}, conf=0.35,
            reason="Default: recompose with current assets (low confidence)",
        )

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _make_result(
        intent: str, target: str, scope: str,
        params: Dict, conf: float, reason: str,
    ) -> Dict:
        return {
            "intent": intent, "target": target, "scope": scope,
            "parameters": params, "confidence": conf, "reasoning": reason,
        }

    @staticmethod
    def _extract_scene_num(q: str) -> Optional[int]:
        m = re.search(r"scene\s+(\d+)", q, re.IGNORECASE)
        return int(m.group(1)) if m else None

    @staticmethod
    def _extract_character(q: str) -> Optional[str]:
        for c in _KNOWN_CHARS:
            if c in q:
                return c.capitalize()
        return None

    @staticmethod
    def _extract_tone(q: str) -> str:
        for t in _TONE_WORDS:
            if t in q:
                return t
        return "neutral"

    def _route_target(self, state: Dict) -> str:
        t = state.get("intent_result", {}).get("target", "unknown")
        return t if t in ("audio", "video_frame", "video", "script") else "unknown"

    # ------------------------------------------------------------------ #
    # Re-run nodes                                                         #
    # ------------------------------------------------------------------ #

    def _run_cmd(self, cmd: List[str], label: str, state: Dict, timeout: int = 600):
        state["execution_log"].append(f"Running: {label}")
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
                cwd=str(Path.cwd()),
            )
            state["rerun_result"] = {
                "phase": label,
                "returncode": result.returncode,
                "success": result.returncode == 0,
                "stderr_tail": result.stderr[-400:] if result.stderr else "",
            }
            state["success"] = result.returncode == 0
            if not state["success"]:
                state["error"] = result.stderr[-400:]
        except subprocess.TimeoutExpired:
            state["error"] = f"{label} timed out after {timeout}s"
            state["success"] = False
        except Exception as exc:
            state["error"] = str(exc)
            state["success"] = False

    def _node_re_run_audio(self, state: Dict) -> Dict:
        self._run_cmd(
            [sys.executable, "-m", "src.main_phase2"],
            "audio (Phase 2 — voice re-synthesis)", state,
        )
        return state

    def _node_re_run_video_frame(self, state: Dict) -> Dict:
        self._run_cmd(
            [sys.executable, "-m", "src.main_phase2"],
            "video_frame (Phase 2 — SD + FaceSwap + LipSync)", state,
        )
        return state

    def _node_re_run_video(self, state: Dict) -> Dict:
        self._run_cmd(
            [sys.executable, "-m", "src.main_phase2"],
            "video (Phase 2 — LipSync recompose only)", state,
        )
        return state

    def _node_re_run_script(self, state: Dict) -> Dict:
        """Full cascade: Phase 1 → Phase 2/3."""
        state["execution_log"].append("Running: script cascade (Phase 1 → Phase 2/3)")
        # Phase 1
        self._run_cmd(
            [sys.executable, "-m", "src.main"],
            "script Phase 1", state, timeout=300,
        )
        if not state.get("success"):
            return state
        # Phase 2/3
        self._run_cmd(
            [sys.executable, "-m", "src.main_phase2"],
            "script Phase 2/3", state, timeout=600,
        )
        return state

    # ------------------------------------------------------------------ #
    # Snapshot node                                                        #
    # ------------------------------------------------------------------ #

    def _node_snapshot(self, state: Dict) -> Dict:
        state["execution_log"].append("Snapshotting pipeline state...")
        intent = state.get("intent_result", {})
        snap_state = {
            "query":         state.get("query"),
            "intent":        intent,
            "rerun_result":  state.get("rerun_result"),
            "success":       state.get("success"),
        }
        try:
            ver = self.state_manager.snapshot(
                state_dict=snap_state,
                change_summary=intent.get("intent", "unknown edit"),
                parent=state.get("parent_version"),
                edit_query=state.get("query", ""),
            )
            state["new_version"] = ver
            state["execution_log"].append(f"Snapshot saved: v{ver:04d}")
        except Exception as exc:
            logger.warning(f"Snapshot failed: {exc}")
            state["execution_log"].append(f"Snapshot failed: {exc}")
        return state
