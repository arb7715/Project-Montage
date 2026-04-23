"""
Script Validator Agent - Ensures correctness of manually provided scripts
"""
import re
import logging
from typing import Dict, Any, Optional, List, Tuple

from src.agents.base_agent import BaseAgent
from src.schema import ValidationResult, SceneElement, DialogueEntry, ScriptManifest, SceneMode
from src.utils.prompts import VALIDATOR_SYSTEM_PROMPT
from src.utils.memory import MemoryStore

logger = logging.getLogger(__name__)

class ValidatorAgent(BaseAgent):
    def __init__(self, memory_store: Optional[MemoryStore] = None):
        super().__init__(
            name="Script Validator",
            role="validator",
            memory_store=memory_store
        )
        self.system_prompt = VALIDATOR_SYSTEM_PROMPT
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a script as JSON or parse it from text.
        Input: {"script_text": "screenplay text", "script_id": "optional"}
        Output: {"validation_result": ValidationResult, "parsed_script": ScriptManifest}
        """
        script_text = input_data.get("script_text", "")
        
        if not script_text:
            return {
                "validation_result": ValidationResult(
                    is_valid=False,
                    errors=["No script text provided"]
                ),
                "parsed_script": None
            }
        
        logger.info(f"Validator checking script...")

        raw_errors = self._validate_raw_structure(script_text)
        if raw_errors:
            return {
                "validation_result": ValidationResult(
                    is_valid=False,
                    errors=raw_errors,
                    warnings=[]
                ),
                "parsed_script": None,
                "raw_input": script_text
            }
        
        # Parse the script text
        parsed_script = self._parse_screenplay_text(script_text)
        
        # Validate the parsed script
        validation_result = self._validate_script_structure(parsed_script)
        
        return {
            "validation_result": validation_result,
            "parsed_script": parsed_script if validation_result.is_valid else None,
            "raw_input": script_text
        }
    
    def _parse_screenplay_text(self, text: str) -> ScriptManifest:
        """
        Parse screenplay text into structured ScriptManifest.
        Expected format:
        SCENE: 1
        HEADING: INT. COFEE SHOP - MORNING
        ACTION: Description
        CHARACTER: Name
        Dialogue text
        ---
        """
        scenes = []
        characters = set()
        
        # Split by scene separator
        scene_blocks = re.split(r'---+', text)
        
        for block_idx, block in enumerate(scene_blocks):
            if not block.strip():
                continue
            
            scene = self._parse_scene_block(block, block_idx + 1)
            if scene:
                scenes.append(scene)
                for dialogue in scene.dialogues:
                    characters.add(dialogue.character)
        
        if not scenes:
            logger.warning("No scenes parsed from screenplay text")
        
        return ScriptManifest(
            mode=SceneMode.MANUAL,
            scenes=scenes,
            character_list=list(characters),
            metadata={
                "input_format": "screenplay_text",
                "num_scenes": len(scenes)
            }
        )
    
    def _parse_scene_block(self, block: str, default_id: int) -> Optional[SceneElement]:
        """Parse a single scene block."""
        lines = block.strip().split('\n')
        if not lines:
            return None
        
        scene_data = {
            "scene_id": None,
            "heading": None,
            "actions": [],
            "dialogues": [],
            "visual_cues": []
        }
        
        current_section = None
        current_character = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Parse metadata lines
            if line.startswith("SCENE:"):
                try:
                    scene_data["scene_id"] = int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith("HEADING:"):
                scene_data["heading"] = line.split(":", 1)[1].strip()
            elif line.startswith("ACTION:"):
                scene_data["actions"].append(line.split(":", 1)[1].strip())
            elif line.startswith("CHARACTER:"):
                current_character = line.split(":", 1)[1].strip()
            elif line.startswith("VISUAL:"):
                scene_data["visual_cues"].append(line.split(":", 1)[1].strip())
            elif current_character and not line.startswith(("SCENE:", "HEADING:", "ACTION:", "CHARACTER:", "VISUAL:")):
                # This is dialogue
                scene_data["dialogues"].append({
                    "character": current_character,
                    "dialogue": line
                })
                current_character = None
        
        # Set defaults if not provided
        if scene_data["scene_id"] is None:
            scene_data["scene_id"] = default_id
        if scene_data["heading"] is None:
            scene_data["heading"] = f"SCENE {default_id}"
        
        # Convert to SceneElement
        try:
            return SceneElement(
                scene_id=scene_data["scene_id"],
                heading=scene_data["heading"],
                actions=scene_data["actions"],
                dialogues=[
                    DialogueEntry(character=d["character"], dialogue=d["dialogue"])
                    for d in scene_data["dialogues"]
                ],
                visual_cues=scene_data["visual_cues"]
            )
        except Exception as e:
            logger.error(f"Failed to create SceneElement: {e}")
            return None
    
    def _validate_script_structure(self, script: ScriptManifest) -> ValidationResult:
        """Validate the script structure."""
        errors = []
        warnings = []
        
        if not script.scenes:
            errors.append("Script must contain at least one scene")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
        
        # Validate each scene
        for scene in script.scenes:
            # Check scene heading
            if not scene.heading or not self._is_valid_heading(scene.heading):
                errors.append(f"Scene {scene.scene_id}: Invalid heading format. Expected 'INT/EXT LOCATION - TIME'")
            
            # Check dialogues have character labels
            for dialogue in scene.dialogues:
                if not dialogue.character:
                    errors.append(f"Scene {scene.scene_id}: Dialogue missing character label")
                if not dialogue.dialogue:
                    warnings.append(f"Scene {scene.scene_id}: Empty dialogue for {dialogue.character}")
            
            # Check for actions
            if not scene.actions:
                warnings.append(f"Scene {scene.scene_id}: No action descriptions")
        
        # Check character consistency
        all_characters = set()
        for scene in script.scenes:
            for dialogue in scene.dialogues:
                all_characters.add(dialogue.character)
        
        # Warn if character appears in scene but listed as different
        if script.character_list:
            unlisted_chars = all_characters - set(script.character_list)
            for char in unlisted_chars:
                warnings.append(f"Character '{char}' appears in dialogue but not in character_list")
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            corrected_script=script if is_valid else None
        )

    def _validate_raw_structure(self, text: str) -> List[str]:
        """Validate raw manual screenplay structure before parsing."""
        errors = []
        
        if not re.search(r'^\s*HEADING\s*:\s*.+$', text, re.IGNORECASE | re.MULTILINE):
            errors.append("Missing HEADING line(s). Manual scripts must define scene headings.")
        
        if not re.search(r'^\s*CHARACTER\s*:\s*.+$', text, re.IGNORECASE | re.MULTILINE):
            errors.append("Missing CHARACTER line(s). Manual scripts must label dialogue speakers.")
        
        if not re.search(r'^\s*ACTION\s*:\s*.+$', text, re.IGNORECASE | re.MULTILINE):
            errors.append("Missing ACTION line(s). Manual scripts must include action descriptions.")
        
        if not re.search(r'^\s*---\s*$', text, re.MULTILINE):
            errors.append("Missing scene separator '---'. Manual scripts must separate scenes.")
        
        return errors
    
    def _is_valid_heading(self, heading: str) -> bool:
        """Check if heading matches screenplay format: INT/EXT LOCATION - TIME"""
        pattern = r'^(INT|EXT)\.\s+.+\s*-\s*(MORNING|DAY|NIGHT|EVENING|LATER|DAWN|DUSK)$'
        return bool(re.search(pattern, heading, re.IGNORECASE))
