"""
LangGraph Workflow - Multi-agent state machine orchestration
Implements the Supervisor-Worker hierarchical model with LangGraph StateGraph
"""
import logging
from typing import Dict, Any, Optional, Literal
from langgraph.graph import StateGraph, END
from dataclasses import dataclass, field, fields, fields

from src.agents.scriptwriter import ScriptwriterAgent
from src.agents.validator import ValidatorAgent
from src.agents.character_designer import CharacterDesignerAgent
from src.agents.image_synthesizer import ImageSynthesizerAgent
from src.agents.hitl_agent import HITLAgent
from src.schema import ScriptManifest, CharacterDatabase, CharacterProfile
from src.utils.memory import MemoryStore
from src.utils.mcp_registry import get_mcp_registry

logger = logging.getLogger(__name__)

@dataclass
class WorkflowState:
    """Shared state across all agents in the workflow."""
    mode: Literal["manual", "autonomous"] = "autonomous"
    user_input: Optional[str] = None
    num_scenes: int = 3
    
    # Scriptwriter outputs
    script: Optional[ScriptManifest] = None
    script_raw: Optional[str] = None
    scriptwriter_success: bool = False
    
    # Validator outputs
    validation_passed: bool = False
    validation_errors: list = field(default_factory=list)
    validation_warnings: list = field(default_factory=list)
    
    # HITL outputs
    user_approved: bool = False
    user_feedback: str = ""
    hitl_decision: Literal["approve", "revise", "reject"] = "approve"
    revision_count: int = 0
    
    # Character Designer outputs
    characters: list = field(default_factory=list)
    character_db: Optional[CharacterDatabase] = None
    characters_success: bool = False
    
    # Image Synthesizer outputs
    generated_images: list = field(default_factory=list)
    images_success: bool = False
    
    # Metadata
    execution_log: list = field(default_factory=list)
    errors: list = field(default_factory=list)

class WritersRoomWorkflow:
    """The Writer's Room - Multi-agent story generation workflow."""
    
    def __init__(self, memory_store: Optional[MemoryStore] = None):
        self.memory_store = memory_store
        self._bind_mcp_tools()
        self.graph = self._build_graph()
        
        # Initialize agents
        self.scriptwriter = ScriptwriterAgent(memory_store)
        self.validator = ValidatorAgent(memory_store)
        self.character_designer = CharacterDesignerAgent(memory_store)
        self.image_synthesizer = ImageSynthesizerAgent(memory_store)
        self.hitl = HITLAgent(memory_store)

    def _coerce_state(self, state: Any) -> WorkflowState:
        """Normalize graph state to WorkflowState object."""
        if isinstance(state, WorkflowState):
            return state
        if isinstance(state, dict):
            names = {f.name for f in fields(WorkflowState)}
            filtered = {k: v for k, v in state.items() if k in names}
            return WorkflowState(**filtered)
        raise TypeError(f"Unsupported state type: {type(state)}")

    def _state_out(self, state: WorkflowState) -> Dict[str, Any]:
        """Return graph-compatible dict state."""
        return state.__dict__

    def _bind_mcp_tools(self):
        """Bind runtime implementations for MCP-discovered tools."""
        registry = get_mcp_registry()

        def commit_memory_impl(entry_type: str, content: Dict[str, Any]):
            if not self.memory_store:
                return {"success": False, "entry_id": ""}
            entry_id = self.memory_store.commit(entry_type, content)
            return {"success": True, "entry_id": entry_id}

        registry.bind_implementation("commit_memory", commit_memory_impl)
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        
        graph = StateGraph(dict)
        
        # Add nodes
        graph.add_node("mode_selector", self._node_mode_selector)
        graph.add_node("scriptwriter", self._node_scriptwriter)
        graph.add_node("validator", self._node_validator)
        graph.add_node("hitl", self._node_hitl)
        graph.add_node("character_designer", self._node_character_designer)
        graph.add_node("image_synthesizer", self._node_image_synthesizer)
        graph.add_node("memory_commit", self._node_memory_commit)
        graph.add_node("finalize", self._node_finalize)
        
        # Add edges
        graph.set_entry_point("mode_selector")
        
        # Mode selector branches
        graph.add_conditional_edges(
            "mode_selector",
            self._route_mode,
            {
                "manual": "validator",
                "autonomous": "scriptwriter"
            }
        )
        
        # Scriptwriter path
        graph.add_edge("scriptwriter", "hitl")
        
        # HITL decision routing
        graph.add_conditional_edges(
            "hitl",
            self._route_hitl_decision,
            {
                "approve": "character_designer",
                "revise": "scriptwriter",
                "reject": "finalize"
            }
        )
        
        # Validator path (for manual scripts)
        graph.add_conditional_edges(
            "validator",
            self._route_validation_result,
            {
                "approve": "hitl",
                "reject": "finalize"
            }
        )
        
        # Character and image generation
        graph.add_edge("character_designer", "image_synthesizer")
        graph.add_edge("image_synthesizer", "memory_commit")
        
        # Finalization
        graph.add_edge("memory_commit", "finalize")
        graph.add_edge("finalize", END)
        
        return graph.compile()
    
    def _route_mode(self, state: WorkflowState) -> str:
        """Route to manual or autonomous mode."""
        state = self._coerce_state(state)
        return state.mode
    
    def _route_hitl_decision(self, state: WorkflowState) -> str:
        """Route based on HITL decision."""
        state = self._coerce_state(state)
        if state.revision_count >= 3:
            logger.warning("Max revisions reached, moving to character design")
            return "character_designer"
        return state.hitl_decision

    def _route_validation_result(self, state: WorkflowState) -> str:
        """Route manual scripts based on validation outcome."""
        state = self._coerce_state(state)
        return "approve" if state.validation_passed else "reject"
    
    def _node_mode_selector(self, state: WorkflowState) -> WorkflowState:
        """Select between manual and autonomous modes."""
        state = self._coerce_state(state)
        state.execution_log.append("Mode Selector: Determining execution path")
        
        if state.user_input and state.mode == "manual":
            state.execution_log.append("Mode Selector: Using manual script input")
        else:
            state.mode = "autonomous"
            state.execution_log.append("Mode Selector: Using autonomous generation")
        
        return self._state_out(state)
    
    def _node_scriptwriter(self, state: WorkflowState) -> WorkflowState:
        """Execute Scriptwriter Agent."""
        state = self._coerce_state(state)
        state.execution_log.append("Scriptwriter: Generating script from prompt")
        
        result = self.scriptwriter.execute({
            "prompt": state.user_input or "A compelling story unfolds",
            "num_scenes": getattr(state, "num_scenes", 3),
        })
        
        if result["success"]:
            state.script = result["script"]
            state.script_raw = result.get("raw_content")
            state.scriptwriter_success = True
            state.execution_log.append("Scriptwriter: Script generated successfully")
        else:
            state.errors.append(f"Scriptwriter failed: {result.get('error')}")
            state.execution_log.append(f"Scriptwriter: Generation failed - {result.get('error')}")
        
        return self._state_out(state)
    
    def _node_validator(self, state: WorkflowState) -> WorkflowState:
        """Execute Validator Agent."""
        state = self._coerce_state(state)
        state.execution_log.append("Validator: Validating script structure")
        
        if not state.user_input:
            state.errors.append("Validator: No script text provided")
            return self._state_out(state)
        
        result = self.validator.execute({
            "script_text": state.user_input
        })
        
        validation_result = result["validation_result"]
        state.validation_passed = validation_result.is_valid
        state.validation_errors = validation_result.errors
        state.validation_warnings = validation_result.warnings
        
        if validation_result.corrected_script:
            state.script = validation_result.corrected_script

        if not validation_result.is_valid:
            state.errors.extend(validation_result.errors)
        
        state.execution_log.append(
            f"Validator: {'Script valid' if validation_result.is_valid else 'Script invalid'} "
            f"({len(validation_result.errors)} errors, {len(validation_result.warnings)} warnings)"
        )
        
        return self._state_out(state)
    
    def _node_hitl(self, state: WorkflowState) -> WorkflowState:
        """Execute Human-in-the-Loop Agent."""
        state = self._coerce_state(state)
        state.execution_log.append("HITL: Requesting user approval")
        
        if not state.script:
            state.errors.append("HITL: No script to review")
            state.hitl_decision = "reject"
            return self._state_out(state)
        
        result = self.hitl.execute({
            "script": state.script,
            "context": f"Script Review (Mode: {state.mode})"
        })
        
        state.hitl_decision = result["user_decision"]
        state.user_feedback = result.get("feedback", "")
        
        if result["user_decision"] == "approve":
            state.user_approved = True
            state.execution_log.append("HITL: Script approved by user")
        elif result["user_decision"] == "revise":
            state.revision_count += 1
            state.execution_log.append(f"HITL: User requested revision (count: {state.revision_count})")
        else:
            state.execution_log.append("HITL: Script rejected by user")
        
        return self._state_out(state)
    
    def _node_character_designer(self, state: WorkflowState) -> WorkflowState:
        """Execute Character Designer Agent."""
        state = self._coerce_state(state)
        state.execution_log.append("Character Designer: Extracting character profiles")
        
        if not state.script:
            state.errors.append("Character Designer: No script available")
            return self._state_out(state)
        
        result = self.character_designer.execute({
            "script": state.script,
            "character_names": state.script.character_list
        })
        
        if result["success"]:
            state.characters = result["characters"]
            state.characters_success = True
            state.execution_log.append(f"Character Designer: Generated {len(result['characters'])} character profiles")

            # Save character_db.json to disk NOW so image_synthesizer can read it
            import json
            from pathlib import Path
            from datetime import datetime
            char_db = {
                "characters": [c.model_dump() for c in state.characters],
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "total_count": len(state.characters)
            }
            db_path = Path("outputs/character_db.json")
            db_path.parent.mkdir(parents=True, exist_ok=True)
            with open(db_path, "w", encoding="utf-8") as f:
                json.dump(char_db, f, indent=2)
            logger.info(f"Character Designer: saved character_db.json with {len(state.characters)} characters")
        else:
            state.errors.append(f"Character Designer failed: {result.get('error')}")
            state.execution_log.append(f"Character Designer: Failed - {result.get('error')}")
        
        return self._state_out(state)
    
    def _node_image_synthesizer(self, state: WorkflowState) -> WorkflowState:
        """Execute Image Synthesizer Agent."""
        state = self._coerce_state(state)
        state.execution_log.append("Image Synthesizer: Generating character visuals")
        
        if not state.characters:
            state.errors.append("Image Synthesizer: No characters to visualize")
            return self._state_out(state)
        
        result = self.image_synthesizer.execute({
            "characters": state.characters
        })
        
        if result["success"]:
            state.generated_images = result["images"]
            state.images_success = True
            state.execution_log.append(f"Image Synthesizer: Generated {len(result['images'])} images")
        else:
            state.errors.append(f"Image Synthesizer failed: {result.get('error')}")
            state.execution_log.append(f"Image Synthesizer: Failed - {result.get('error')}")
        
        return self._state_out(state)
    
    def _node_memory_commit(self, state: WorkflowState) -> WorkflowState:
        """Commit final outputs to memory."""
        state = self._coerce_state(state)
        state.execution_log.append("Memory: Committing final outputs")
        
        if self.memory_store:
            if state.script:
                self.memory_store.commit("script_history", {
                    "script": state.script.model_dump(),
                    "user_approved": state.user_approved
                })
            
            if state.characters:
                for char in state.characters:
                    self.memory_store.commit("character_metadata", {
                        "character": char.model_dump()
                    })
        
        return self._state_out(state)
    
    def _node_finalize(self, state: WorkflowState) -> WorkflowState:
        """Finalize workflow execution."""
        state = self._coerce_state(state)
        state.execution_log.append("Finalize: Workflow complete")
        return self._state_out(state)
    
    def run(self, initial_state: WorkflowState) -> Dict[str, Any]:
        """Execute the workflow."""
        logger.info("Starting The Writer's Room Workflow")
        logger.info(f"Initial mode: {initial_state.mode}")
        
        try:
            # Execute the compiled graph
            final_state = self.graph.invoke(initial_state.__dict__)
        except KeyError as e:
            # Some langgraph builds can raise KeyError('__start__') during invoke.
            # Fallback keeps the assignment workflow functional end-to-end.
            if str(e) != "'__start__'":
                raise
            logger.warning("LangGraph invoke failed with __start__; using sequential fallback execution")
            final_state = self._run_fallback(initial_state)
        
        logger.info("Workflow completed")
        return self._coerce_state(final_state)

    def _run_fallback(self, initial_state: WorkflowState) -> Dict[str, Any]:
        """Sequential fallback execution that mirrors graph routing behavior."""
        state = self._coerce_state(initial_state)

        state = self._coerce_state(self._node_mode_selector(state))

        if self._route_mode(state) == "manual":
            state = self._coerce_state(self._node_validator(state))
            if self._route_validation_result(state) == "reject":
                return self._node_finalize(state)
            state = self._coerce_state(self._node_hitl(state))
        else:
            state = self._coerce_state(self._node_scriptwriter(state))
            state = self._coerce_state(self._node_hitl(state))

        if state.hitl_decision == "reject":
            return self._node_finalize(state)

        if state.hitl_decision == "revise":
            state.revision_count += 1
            state = self._coerce_state(self._node_scriptwriter(state))

        state = self._coerce_state(self._node_character_designer(state))
        state = self._coerce_state(self._node_image_synthesizer(state))
        state = self._coerce_state(self._node_memory_commit(state))
        return self._node_finalize(state)
