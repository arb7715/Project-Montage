"""
Scene Parser Agent - Transform scene_manifest.json into parallelisable SceneTask units.
Phase 2 entry agent: reads Phase 1 outputs and fans out work to audio/video branches.
"""
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

from src.agents.base_agent import BaseAgent
from src.schema import SceneTask, DialogueEntry
from src.utils.prompts import SCENE_PARSER_SYSTEM_PROMPT
from src.utils.memory import MemoryStore

logger = logging.getLogger(__name__)


class SceneParserAgent(BaseAgent):
    """
    Scene Parser Agent
    -----------------
    Reads scene_manifest.json produced by Phase 1 and converts each scene into
    an independent SceneTask that can be processed in parallel by the audio and
    video branches.

    MCP Tools used: get_task_graph, commit_memory
    """

    def __init__(self, memory_store: Optional[MemoryStore] = None):
        super().__init__(
            name="Scene Parser",
            role="scene_parser",
            memory_store=memory_store,
        )
        self.system_prompt = SCENE_PARSER_SYSTEM_PROMPT

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse scene manifest into task units.

        Input:
            {
                "scene_manifest_path": "outputs/scene_manifest.json",
                "output_dir": "outputs"
            }
        Output:
            {
                "success": bool,
                "tasks": List[SceneTask],
                "task_graph_path": str,
                "total_scenes": int
            }
        """
        manifest_path = input_data.get("scene_manifest_path", "outputs/scene_manifest.json")
        output_dir = Path(input_data.get("output_dir", "outputs"))

        logger.info(f"Scene Parser: loading manifest from {manifest_path}")

        # -- Load manifest --------------------------------------------------
        manifest = self._load_manifest(manifest_path)
        if manifest is None:
            return {"success": False, "error": f"Cannot read {manifest_path}", "tasks": []}

        # -- Build SceneTasks -----------------------------------------------
        tasks = self._build_tasks(manifest)
        logger.info(f"Scene Parser: built {len(tasks)} SceneTasks")

        # -- Persist task graph log -----------------------------------------
        task_graph_path = self._save_task_graph(tasks, output_dir)

        # -- Commit to memory (fault-tolerance / resumability) --------------
        self.commit_to_memory("task_graph", {
            "total_scenes": len(tasks),
            "task_graph_path": str(task_graph_path),
            "tasks": [t.model_dump() for t in tasks],
            "created_at": datetime.now().isoformat(),
        })

        logger.info(f"Scene Parser: task graph saved to {task_graph_path}")

        return {
            "success": True,
            "tasks": tasks,
            "task_graph_path": str(task_graph_path),
            "total_scenes": len(tasks),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_manifest(self, path: str) -> Optional[Dict]:
        """Load and parse scene_manifest.json."""
        p = Path(path)
        if not p.exists():
            logger.error(f"Scene manifest not found: {path}")
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.error(f"Failed to parse manifest: {exc}")
            return None

    def _build_tasks(self, manifest: Dict) -> List[SceneTask]:
        """Convert raw manifest scenes into SceneTask objects."""
        tasks: List[SceneTask] = []
        scenes = manifest.get("scenes", [])

        for raw_scene in scenes:
            # Collect unique character names from dialogues
            char_names: List[str] = list({
                d.get("character", "") for d in raw_scene.get("dialogues", [])
                if d.get("character")
            })

            # Build DialogueEntry list
            dialogues = [
                DialogueEntry(
                    character=d.get("character", ""),
                    dialogue=d.get("dialogue", ""),
                )
                for d in raw_scene.get("dialogues", [])
            ]

            task = SceneTask(
                scene_id=raw_scene.get("scene_id", len(tasks) + 1),
                heading=raw_scene.get("heading", f"SCENE {len(tasks)+1}"),
                actions=raw_scene.get("actions", []),
                dialogues=dialogues,
                visual_cues=raw_scene.get("visual_cues", []),
                character_names=char_names,
            )
            tasks.append(task)

        return tasks

    def _save_task_graph(self, tasks: List[SceneTask], output_dir: Path) -> Path:
        """Persist task graph as JSON for logging / resumability."""
        output_dir.mkdir(parents=True, exist_ok=True)
        graph_path = output_dir / "task_graph_log.json"

        graph_data = {
            "generated_at": datetime.now().isoformat(),
            "total_scenes": len(tasks),
            "agent": self.name,
            "mcp_tool_used": "get_task_graph",
            "tasks": [
                {
                    "scene_id": t.scene_id,
                    "heading": t.heading,
                    "character_names": t.character_names,
                    "dialogue_count": len(t.dialogues),
                    "visual_cue_count": len(t.visual_cues),
                    "action_count": len(t.actions),
                    "status": "queued",
                }
                for t in tasks
            ],
        }

        with open(graph_path, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=2)

        return graph_path
