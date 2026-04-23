"""
Human-in-the-Loop Agent - Provides control checkpoint before execution continues
"""
import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime

from src.agents.base_agent import BaseAgent
from src.schema import ScriptManifest
from src.utils.prompts import HITL_SYSTEM_PROMPT
from src.utils.memory import MemoryStore

logger = logging.getLogger(__name__)

class HITLAgent(BaseAgent):
    def __init__(self, memory_store: Optional[MemoryStore] = None):
        super().__init__(
            name="Human-in-the-Loop",
            role="hitl",
            memory_store=memory_store
        )
        self.system_prompt = HITL_SYSTEM_PROMPT
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Checkpoint for human approval before proceeding.
        Input: {"script": ScriptManifest, "context": "what's being approved"}
        Output: {"user_decision": "approve|revise|reject", "feedback": str}
        """
        script = input_data.get("script")
        context = input_data.get("context", "Script Generation")
        
        if not script:
            return {
                "user_decision": "reject",
                "feedback": "No script provided",
                "timestamp": datetime.now().isoformat()
            }
        
        logger.info("\n" + "="*80)
        logger.info("HUMAN-IN-THE-LOOP CHECKPOINT")
        logger.info("="*80)
        logger.info(f"Context: {context}\n")
        
        # Display the script
        self._display_script(script)
        
        # Request user approval
        decision = self._request_approval()
        
        result = {
            "user_decision": decision,
            "feedback": "",
            "timestamp": datetime.now().isoformat(),
            "script": script if decision == "approve" else None
        }
        
        if decision in ["revise", "reject"]:
            feedback = input("\nProvide feedback (press Enter to skip): ").strip()
            result["feedback"] = feedback
        
        logger.info("="*80 + "\n")
        
        return result
    
    def _display_script(self, script: ScriptManifest):
        """Display the script in readable format."""
        print(f"\nGenerated Script ({script.mode.value} mode):")
        print(f"Total Scenes: {len(script.scenes)}")
        print(f"Characters: {', '.join(script.character_list)}\n")
        
        for scene in script.scenes:
            print(f"\n{'='*70}")
            print(f"SCENE {scene.scene_id}")
            print(f"Heading: {scene.heading}")
            print(f"{'='*70}")
            
            if scene.actions:
                print("\nACTION:")
                for action in scene.actions:
                    print(f"  {action}")
            
            if scene.dialogues:
                print("\nDIALOGUE:")
                for dialogue in scene.dialogues:
                    print(f"  {dialogue.character}")
                    print(f"    {dialogue.dialogue}")
            
            if scene.visual_cues:
                print("\nVISUAL CUES:")
                for cue in scene.visual_cues:
                    print(f"  • {cue}")
        
        print(f"\n{'='*70}\n")
    
    def _request_approval(self) -> str:
        """Request user decision via terminal input."""
        while True:
            try:
                decision = input("Do you approve this script? (approve/revise/reject): ").strip().lower()
            except EOFError:
                logger.warning("No stdin available for HITL approval; defaulting to reject")
                return "reject"
            
            if decision in ["approve", "revise", "reject"]:
                return decision
            else:
                print("Invalid input. Please enter 'approve', 'revise', or 'reject'.")
