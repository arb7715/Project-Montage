"""
Scriptwriter Agent - Transform abstract prompts into structured scripts
"""
import json
import logging
from typing import Dict, Any, Optional, List
import ollama

from src.agents.base_agent import BaseAgent
from src.schema import SceneElement, DialogueEntry, ScriptManifest, SceneMode
from src.utils.prompts import SCRIPTWRITER_SYSTEM_PROMPT
from src.utils.memory import MemoryStore

logger = logging.getLogger(__name__)

class ScriptwriterAgent(BaseAgent):
    def __init__(self, memory_store: Optional[MemoryStore] = None):
        super().__init__(
            name="Scriptwriter",
            role="scriptwriter",
            memory_store=memory_store
        )
        self.model = "llama3.2:1b"
        self.system_prompt = SCRIPTWRITER_SYSTEM_PROMPT
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute scriptwriting process.
        Input: {"prompt": "user creative prompt", "num_scenes": 3}
        Output: {"script": ScriptManifest, "success": bool}
        """
        prompt = input_data.get("prompt", "")
        num_scenes = input_data.get("num_scenes", 3)
        
        if not prompt:
            return {
                "success": False,
                "error": "No prompt provided",
                "script": None
            }
        
        logger.info(f"Scriptwriter generating script from prompt: {prompt[:100]}...")
        
        # Generate script using Ollama
        script_content = self._generate_script_with_ollama(prompt, num_scenes)
        
        # Parse into structured format
        try:
            script_manifest = self._parse_script(script_content, prompt)
            
            # Commit to memory
            if self.memory_store:
                self.commit_to_memory("script_history", {
                    "prompt": prompt,
                    "num_scenes": num_scenes,
                    "script": script_manifest.model_dump()
                })
            
            return {
                "success": True,
                "script": script_manifest,
                "raw_content": script_content
            }
        except Exception as e:
            logger.error(f"Failed to parse script: {e}")
            return {
                "success": False,
                "error": str(e),
                "script": None,
                "raw_content": script_content
            }
    
    def _generate_script_with_ollama(self, prompt: str, num_scenes: int) -> str:
        """Call Ollama to generate script content."""
        refined_prompt = f"""{self.system_prompt}

User Prompt: {prompt}

Generate exactly {num_scenes} scenes for a screenplay based on the above prompt.

Return ONLY valid JSON with this exact shape:
{{
    "scenes": [
        {{
            "scene_id": 1,
            "heading": "INT. LOCATION - TIME",
            "actions": ["..."],
            "dialogues": [
                {{"character": "Name", "dialogue": "Line"}}
            ],
            "visual_cues": ["..."]
        }}
    ]
}}

Rules:
- No markdown fences.
- No explanation text.
- Exactly {num_scenes} scene objects in the scenes array.
- Use screenplay-friendly headings like INT./EXT. with time of day.
- Keep dialogue natural and story-relevant.
"""
        
        try:
            response = ollama.generate(
                model=self.model,
                prompt=refined_prompt,
                stream=False,
                options={
                    "temperature": 0.7,
                    "num_predict": 2000
                }
            )
            return response.get("response", "")
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise RuntimeError(f"Failed to generate script: {e}")
    
    def _parse_script(self, content: str, prompt: str) -> ScriptManifest:
        """Parse generated content into structured ScriptManifest."""
        scenes = []
        characters = set()

        parsed = self._extract_json_payload(content)
        if isinstance(parsed, dict) and isinstance(parsed.get("scenes"), list):
            raw_scenes = parsed["scenes"]
        elif isinstance(parsed, list):
            raw_scenes = parsed
        else:
            logger.warning("No JSON scenes found, creating default scenes")
            raw_scenes = []

        for index, scene_data in enumerate(raw_scenes, start=1):
            try:
                scene = self._parse_scene_json(scene_data, index)
                scenes.append(scene)

                for dialogue in scene.dialogues:
                    if dialogue.character:
                        characters.add(dialogue.character)
            except Exception as exc:
                logger.warning(f"Could not parse scene JSON at index {index}: {exc}")
        
        if not scenes:
            scenes = self._create_default_scenes(prompt)
        
        return ScriptManifest(
            mode=SceneMode.AUTONOMOUS,
            scenes=scenes,
            character_list=list(characters) if characters else ["Character A", "Character B"],
            metadata={
                "original_prompt": prompt,
                "generation_model": self.model,
                "num_scenes": len(scenes)
            }
        )
    
    def _parse_scene_json(self, data: Dict[str, Any], default_id: int) -> SceneElement:
        """Parse a single scene JSON object."""
        return SceneElement(
            scene_id=data.get("scene_id", default_id),
            heading=data.get("heading", f"SCENE {default_id}"),
            actions=data.get("actions", []),
            dialogues=[
                DialogueEntry(character=d.get("character", ""), dialogue=d.get("dialogue", ""))
                for d in data.get("dialogues", [])
            ],
            visual_cues=data.get("visual_cues", [])
        )

    def _extract_json_payload(self, content: str) -> Any:
        """Extract JSON from raw model output, tolerating fences and extra text."""
        import re

        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None

        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None

        return None
    
    def _create_default_scenes(self, prompt: str) -> List[SceneElement]:
        """Create default scenes if parsing fails."""
        return [
            SceneElement(
                scene_id=1,
                heading="INT. LOCATION - DAY",
                actions=[f"Scene begins with: {prompt}"],
                dialogues=[
                    DialogueEntry(character="Character A", dialogue="Tell me more about what's happening."),
                    DialogueEntry(character="Character B", dialogue="I'll explain everything.")
                ],
                visual_cues=["Establishing shot", "Close-up on Character A's reaction"]
            ),
            SceneElement(
                scene_id=2,
                heading="INT. LOCATION - LATER",
                actions=["The situation develops"],
                dialogues=[
                    DialogueEntry(character="Character B", dialogue="This is quite important."),
                    DialogueEntry(character="Character A", dialogue="I understand now.")
                ],
                visual_cues=["Tension builds", "Camera slow push toward Character B"]
            ),
            SceneElement(
                scene_id=3,
                heading="INT. LOCATION - EVENING",
                actions=["Resolution approaches"],
                dialogues=[
                    DialogueEntry(character="Character A", dialogue="What happens next?"),
                    DialogueEntry(character="Character B", dialogue="Let's continue our story.")
                ],
                visual_cues=["Golden hour lighting", "Wide establishing shot"]
            )
        ]
