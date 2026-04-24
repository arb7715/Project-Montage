"""
Character Designer Agent - Extract and formalize character identities from scripts
Includes gender inference for accurate portrait generation and voice synthesis.
"""
import logging
import json
import re
from typing import Dict, Any, Optional, List
import ollama

from src.agents.base_agent import BaseAgent
from src.schema import CharacterProfile, ScriptManifest
from src.utils.prompts import CHARACTER_DESIGNER_SYSTEM_PROMPT
from src.utils.memory import MemoryStore

logger = logging.getLogger(__name__)

# ── Gender inference from character names ──────────────────────────────────────
# Curated lists for common names.  Falls back to "unknown" if not matched.
_FEMALE_NAMES = {
    "sarah", "sara", "emma", "olivia", "sophia", "ava", "isabella", "mia",
    "charlotte", "amelia", "harper", "evelyn", "abigail", "emily", "elizabeth",
    "ella", "grace", "chloe", "victoria", "lily", "hannah", "nora", "zoe",
    "riley", "aria", "ellie", "aubrey", "stella", "natalie", "lucy", "anna",
    "samantha", "caroline", "claire", "maya", "elena", "alice", "julia",
    "maria", "rose", "katherine", "kate", "jessica", "jennifer", "laura",
    "rachel", "rebecca", "mary", "margaret", "linda", "susan", "dorothy",
    "betty", "nancy", "karen", "helen", "sandra", "donna", "carol",
    "ruth", "sharon", "michelle", "diana", "catherine", "christine",
    "waitress", "hostess", "queen", "princess", "duchess", "baroness",
    "mother", "grandmother", "sister", "daughter", "wife", "girlfriend",
    "nurse", "actress", "heroine",
}

_MALE_NAMES = {
    "daniel", "dan", "james", "john", "robert", "michael", "william", "david",
    "richard", "joseph", "thomas", "charles", "christopher", "matthew", "andrew",
    "mark", "donald", "steven", "paul", "kevin", "jason", "brian", "george",
    "edward", "peter", "henry", "jack", "ryan", "tyler", "jacob", "ethan",
    "noah", "logan", "lucas", "mason", "oliver", "liam", "benjamin", "alex",
    "alexander", "samuel", "nathan", "adam", "luke", "owen", "isaac", "dylan",
    "caleb", "max", "leo", "oscar", "arthur", "felix", "hugo", "harry",
    "waiter", "bartender", "king", "prince", "duke", "baron",
    "father", "grandfather", "brother", "son", "husband", "boyfriend",
    "detective", "officer", "sergeant", "captain", "doctor",
}


def infer_gender(name: str) -> str:
    """Infer gender from a character name using curated name lists."""
    # Normalize: take the first word of the name, lowercase
    tokens = name.lower().strip().split()
    for token in tokens:
        if token in _FEMALE_NAMES:
            return "female"
        if token in _MALE_NAMES:
            return "male"
    # Check if the full lowered name matches
    full = name.lower().strip()
    if full in _FEMALE_NAMES:
        return "female"
    if full in _MALE_NAMES:
        return "male"
    return "unknown"


class CharacterDesignerAgent(BaseAgent):
    def __init__(self, memory_store: Optional[MemoryStore] = None):
        super().__init__(
            name="Character Designer",
            role="character_designer",
            memory_store=memory_store
        )
        self.model = "llama3.2:1b"
        self.system_prompt = CHARACTER_DESIGNER_SYSTEM_PROMPT
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract character profiles from script.
        Input: {"script": ScriptManifest, "character_names": ["optional list"]}
        Output: {"characters": [CharacterProfile], "success": bool}
        """
        script = input_data.get("script")
        character_names = input_data.get("character_names")
        
        if not script:
            return {
                "success": False,
                "error": "No script provided",
                "characters": []
            }
        
        logger.info(f"Character Designer extracting profiles from script...")
        
        # Get character list from script
        if not character_names:
            character_names = script.character_list
        
        profiles = []
        
        for char_name in character_names:
            try:
                # Infer gender from name
                gender = infer_gender(char_name)
                logger.info(f"  {char_name} -> gender: {gender}")

                # Extract character dialogue and actions from script
                char_context = self._extract_character_context(script, char_name)
                
                # Generate profile using Ollama with gender hint
                profile = self._generate_character_profile(char_name, char_context, gender)
                
                if profile:
                    profiles.append(profile)
                    
                    # Commit to memory
                    if self.memory_store:
                        self.commit_to_memory("character_metadata", {
                            "character_name": profile.name,
                            "profile": profile.model_dump()
                        })
            except Exception as e:
                logger.error(f"Failed to generate profile for {char_name}: {e}")
        
        return {
            "success": len(profiles) > 0,
            "characters": profiles,
            "total_processed": len(character_names),
            "successful": len(profiles)
        }
    
    def _extract_character_context(self, script: ScriptManifest, character_name: str) -> str:
        """Extract dialogue and action context for a character."""
        context_parts = []
        
        for scene in script.scenes:
            scene_dialogues = [d for d in scene.dialogues if d.character == character_name]
            if scene_dialogues:
                context_parts.append(f"Scene {scene.scene_id} ({scene.heading}):")
                for d in scene_dialogues:
                    context_parts.append(f"  {character_name}: {d.dialogue}")
        
        return "\n".join(context_parts) if context_parts else f"Character {character_name} appears in the script."
    
    def _generate_character_profile(
        self, character_name: str, context: str, gender: str
    ) -> Optional[CharacterProfile]:
        """Generate character profile using Ollama with explicit gender hint."""

        # Build gender-specific instruction
        if gender == "female":
            gender_instruction = (
                f"{character_name} is a FEMALE character. "
                "Describe her with feminine features: e.g. long or styled hair, "
                "soft facial features, feminine clothing. Do NOT describe masculine traits."
            )
        elif gender == "male":
            gender_instruction = (
                f"{character_name} is a MALE character. "
                "Describe him with masculine features: e.g. short hair, strong jaw, "
                "masculine build and clothing."
            )
        else:
            gender_instruction = (
                f"Determine the likely gender of {character_name} from context and "
                "describe appropriate physical features."
            )

        prompt = f"""{self.system_prompt}

Character Name: {character_name}
Gender: {gender.upper()}

{gender_instruction}

Script Context:
{context}

Based on the script context, create a detailed character profile for {character_name}.
Respond in ONLY valid JSON (no markdown, no explanation) with these fields:
{{
    "name": "{character_name}",
    "personality_traits": ["trait1", "trait2", "trait3"],
    "appearance_description": "Detailed physical appearance description matching the character's gender",
    "reference_style": "realistic"
}}
"""
        
        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                stream=False,
                options={
                    "temperature": 0.7,
                    "num_predict": 500
                }
            )
            
            content = response.get("response", "")
            return self._parse_profile_response(character_name, content, gender)
        except Exception as e:
            logger.error(f"Ollama profile generation failed: {e}")
            return self._create_default_profile(character_name, gender)
    
    def _parse_profile_response(
        self, character_name: str, content: str, gender: str
    ) -> Optional[CharacterProfile]:
        """Parse Ollama response into CharacterProfile."""
        # Try to extract JSON
        json_pattern = r'\{.*\}'
        match = re.search(json_pattern, content, re.DOTALL)
        
        if match:
            try:
                data = json.loads(match.group())
                appearance = data.get("appearance_description", "")

                # Safety check: if gender is female but description has male traits, fix it
                if gender == "female" and any(w in appearance.lower() for w in ["buzz cut", "stubble", "beard", "muscular build"]):
                    logger.warning(f"LLM generated male traits for female character {character_name}, using default")
                    return self._create_default_profile(character_name, gender)

                return CharacterProfile(
                    name=data.get("name", character_name),
                    personality_traits=data.get("personality_traits", ["Undetermined"]),
                    appearance_description=appearance,
                    reference_style=data.get("reference_style", "realistic"),
                    gender=gender,
                    metadata={"auto_generated": True}
                )
            except json.JSONDecodeError:
                pass
        
        return self._create_default_profile(character_name, gender)
    
    def _create_default_profile(self, character_name: str, gender: str = "unknown") -> CharacterProfile:
        """Create a gender-appropriate default profile when LLM parsing fails."""

        if gender == "female":
            appearance = (
                f"Age: late 20s, Build: slender, Hair: long flowing brown hair, "
                f"Eyes: warm brown eyes, Skin: fair complexion, "
                f"Clothing: casual elegant blouse and skirt. "
                f"Feminine features, gentle expression. Style: realistic."
            )
        elif gender == "male":
            appearance = (
                f"Age: early 30s, Build: athletic, Hair: short dark hair, "
                f"Eyes: sharp blue eyes, strong jawline, light stubble, "
                f"Clothing: tailored dark jacket with button-up shirt. "
                f"Masculine features, confident expression. Style: realistic."
            )
        else:
            appearance = (
                f"{character_name} has a distinctive presence with compelling features. "
                f"Style: realistic."
            )

        return CharacterProfile(
            name=character_name,
            personality_traits=["Determined", "Intelligent"],
            appearance_description=appearance,
            reference_style="realistic",
            gender=gender,
            metadata={"auto_generated": True, "source": "default"}
        )
