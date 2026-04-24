"""
Image Synthesizer Agent - Generate character reference images.
Uses the Colab FastAPI backend to rapidly generate Stable Diffusion images.
"""
import logging
import json
import os
import re
import requests
import io
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from PIL import Image

from src.agents.base_agent import BaseAgent
from src.schema import CharacterProfile
from src.utils.prompts import IMAGE_SYNTHESIZER_SYSTEM_PROMPT
from src.utils.memory import MemoryStore

logger = logging.getLogger(__name__)

class ImageSynthesizerAgent(BaseAgent):
    def __init__(self, memory_store: Optional[MemoryStore] = None):
        super().__init__(
            name="Image Synthesizer",
            role="image_synthesizer",
            memory_store=memory_store
        )
        self.output_dir = Path("outputs/images")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.system_prompt = IMAGE_SYNTHESIZER_SYSTEM_PROMPT
        self.db_path = Path("outputs/character_db.json")
        
        self.colab_url = self._get_colab_url()

    def _get_colab_url(self) -> str:
        path = Path("config/colab_api.txt")
        if path.exists():
            return path.read_text().strip()
        return ""

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate character images using Colab API and update character_db.json."""
        logger.info("Image Synthesizer starting...")
        
        if not self.colab_url:
            return {
                "success": False,
                "error": "Colab API URL not found in config/colab_api.txt",
                "images": []
            }
        
        if not self.db_path.exists():
            return {"success": False, "error": f"character_db.json not found", "images": []}
        
        try:
            with open(self.db_path, 'r', encoding='utf-8') as f:
                db_content = f.read()
                if db_content.startswith('\ufeff'):
                    db_content = db_content[1:]
                character_db = json.loads(db_content)
        except Exception as e:
            return {"success": False, "error": f"Failed to read character_db.json: {e}", "images": []}
        
        characters = character_db.get("characters", [])
        if not characters:
            return {"success": False, "error": "No characters found", "images": []}
        
        images = []
        updated_characters = []
        
        for character_data in characters:
            try:
                char_name = character_data.get("name", "Unknown")
                appearance = character_data.get("appearance_description", "")
                reference_style = character_data.get("reference_style", "realistic")
                gender = character_data.get("gender", "unknown")
                
                logger.info(f"Generating image for {char_name} (gender: {gender}) via Colab API...")
                
                prompt = self._construct_prompt(appearance, reference_style, gender)
                negative_prompt = self._construct_negative_prompt(gender)
                
                image = self._generate_image(prompt, negative_prompt)
                
                if image is None:
                    logger.error(f"Failed to generate image for {char_name}")
                    updated_characters.append(character_data)
                    continue
                
                safe_name = self._sanitize_filename(char_name).lower()
                image_path = self.output_dir / f"{safe_name}.png"
                image.save(image_path)
                logger.info(f"Saved image for {char_name} to {image_path}")
                
                character_data["image_reference"] = str(image_path)
                updated_characters.append(character_data)
                
                images.append({
                    "character": char_name,
                    "path": str(image_path),
                    "reference_style": reference_style
                })
                
                if self.memory_store:
                    self.commit_to_memory("image_reference", {
                        "character_name": char_name,
                        "image_path": str(image_path),
                        "reference_style": reference_style,
                        "gender": gender,
                        "generated_at": datetime.now().isoformat()
                    })
                
            except Exception as e:
                logger.error(f"Failed to generate image for {character_data.get('name', 'Unknown')}: {e}")
                updated_characters.append(character_data)
                continue
        
        try:
            character_db["characters"] = updated_characters
            character_db["updated_at"] = datetime.now().isoformat()
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(character_db, f, indent=2)
        except Exception as e:
            return {"success": False, "error": f"Failed to update character_db.json: {e}", "images": images}
        
        return {
            "success": len(images) > 0,
            "images": images,
            "total_processed": len(characters),
            "successful": len(images)
        }
    
    def _construct_prompt(self, appearance_description: str, reference_style: str, gender: str = "unknown") -> str:
        if gender == "female":
            gender_prefix = "beautiful woman, female portrait"
        elif gender == "male":
            gender_prefix = "handsome man, male portrait"
        else:
            gender_prefix = "character portrait"

        style_modifiers = {
            "realistic": "photorealistic",
            "stylized": "stylized character portrait",
            "anime": "anime portrait",
            "cartoon": "cartoon portrait"
        }

        style_prompt = style_modifiers.get(reference_style, "portrait")
        appearance_summary = self._summarize_appearance(appearance_description)
        return f"{gender_prefix}, {style_prompt}, {appearance_summary}, high quality, detailed face, studio lighting"

    def _construct_negative_prompt(self, gender: str = "unknown") -> str:
        base_negative = "text, watermark, blurry, low quality, deformed, distorted, extra limbs, disfigured, bad anatomy"
        if gender == "female":
            return f"{base_negative}, masculine features, beard, stubble, muscular, male"
        elif gender == "male":
            return f"{base_negative}, feminine features, breasts, long eyelashes, lipstick, female"
        return base_negative

    def _summarize_appearance(self, appearance_description: str) -> str:
        cleaned = re.sub(r"\s+", " ", appearance_description.strip())
        parts = [part.strip() for part in cleaned.split(",") if part.strip()]
        keep_parts = []
        for part in parts:
            lower = part.lower()
            if any(keyword in lower for keyword in ("age", "build", "hair", "eyes", "beard", "clothing", "style", "feature", "skin", "complexion")):
                keep_parts.append(part)
            if len(keep_parts) == 4: break
        if not keep_parts: keep_parts = parts[:3]
        return ", ".join(keep_parts)
    
    def _generate_image(self, prompt: str, negative_prompt: str = None) -> Optional[Image.Image]:
        if negative_prompt is None:
            negative_prompt = "text, watermark, blurry, low quality, deformed, distorted, extra limbs"

        url = f"{self.colab_url}/generate"
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": 512,
            "height": 512
        }
        try:
            headers = {"ngrok-skip-browser-warning": "true"}
            response = requests.post(url, data=payload, headers=headers, timeout=120)
            if response.status_code == 200:
                image = Image.open(io.BytesIO(response.content))
                return image
            else:
                logger.error(f"Colab API returned status {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Failed to generate image via API: {e}")
            return None
    
    def _sanitize_filename(self, name: str) -> str:
        sanitized = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in name)
        return sanitized.strip().replace(' ', '_')
