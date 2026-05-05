"""
Main Entry Point - The Writer's Room Phase 1
Autonomous Story and Image Generation Layer
"""
import json
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from src.workflow import WritersRoomWorkflow, WorkflowState
from src.schema import CharacterDatabase, SceneMode
from src.utils.memory import MemoryStore
from src.utils.mcp_registry import get_mcp_registry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="The Writer's Room Phase 1 - Autonomous Story and Image Generation"
    )
    
    parser.add_argument(
        "--mode",
        choices=["autonomous", "manual"],
        default="autonomous",
        help="Execution mode: autonomous (LLM generation) or manual (script input)"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Creative prompt for autonomous generation (required for autonomous mode)"
    )
    
    parser.add_argument(
        "--script-file",
        type=str,
        default=None,
        help="Path to screenplay file for manual mode"
    )
    
    parser.add_argument(
        "--num-scenes",
        type=int,
        default=3,
        help="Number of scenes to generate (autonomous mode)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for artifacts"
    )
    
    parser.add_argument(
        "--export-mcp-registry",
        action="store_true",
        help="Export MCP registry metadata"
    )
    
    args = parser.parse_args()
    
    # Initialize output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export MCP registry if requested
    if args.export_mcp_registry:
        registry = get_mcp_registry()
        registry_path = output_dir / "mcp_registry.json"
        registry.export_registry(str(registry_path))
        logger.info(f"MCP registry exported to {registry_path}")
        return
    
    # Initialize memory store
    memory_store = MemoryStore(str(output_dir / "memory"))
    logger.info(f"Memory store initialized at {output_dir / 'memory'}")
    
    # Initialize workflow
    workflow = WritersRoomWorkflow(memory_store)
    logger.info("WritersRoom Workflow initialized")
    
    # Prepare initial state based on mode
    if args.mode == "manual":
        if not args.script_file:
            logger.error("Manual mode requires --script-file argument")
            return
        
        script_path = Path(args.script_file)
        if not script_path.exists():
            logger.error(f"Script file not found: {args.script_file}")
            return
        
        with open(script_path, 'r') as f:
            script_content = f.read()
        
        initial_state = WorkflowState(
            mode="manual",
            user_input=script_content,
            num_scenes=args.num_scenes,
        )
        logger.info(f"Manual mode: Loaded script from {args.script_file}")
    
    else:  # autonomous mode
        if not args.prompt:
            # Use default prompt
            args.prompt = "A mysterious stranger arrives in a quiet town, bringing secrets that will change everything."
            logger.info(f"Using default prompt: {args.prompt}")
        
        initial_state = WorkflowState(
            mode="autonomous",
            user_input=args.prompt,
            num_scenes=args.num_scenes,
        )
        logger.info(f"Autonomous mode: Using prompt - {args.prompt}")
    
    # Run the workflow
    print("\n" + "="*80)
    print("THE WRITER'S ROOM - PHASE 1")
    print("Autonomous Story and Image Generation Layer")
    print("="*80 + "\n")
    
    try:
        final_state = workflow.run(initial_state)
        
        # Generate output artifacts
        success = _generate_artifacts(final_state, output_dir)
        
        # Print execution summary
        _print_summary(final_state)
        
        if success:
            logger.info("Phase 1 execution completed successfully")
        else:
            logger.warning("Phase 1 completed with warnings or errors")
    
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}", exc_info=True)
        return 1
    
    return 0

def _generate_artifacts(state: WorkflowState, output_dir: Path):
    """Generate required output artifacts."""
    logger.info("\nGenerating output artifacts...")
    
    success = True
    
    # 1. scene_manifest.json
    if state.script:
        manifest_path = output_dir / "scene_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(state.script.model_dump(), f, indent=2)
        logger.info(f"✓ Generated scene_manifest.json")
    else:
        logger.warning("✗ Could not generate scene_manifest.json (no script)")
        success = False
    
    # 2. character_db.json
    if state.characters:
        char_db = {
            "characters": [c.model_dump() for c in state.characters],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "total_count": len(state.characters)
        }
        char_db_path = output_dir / "character_db.json"
        with open(char_db_path, 'w') as f:
            json.dump(char_db, f, indent=2)
        logger.info(f"✓ Generated character_db.json with {len(state.characters)} characters")
    else:
        logger.warning("✗ Could not generate character_db.json (no characters)")
        success = False
    
    # 3. Images are already saved in outputs/images/ by Image Synthesizer
    if state.generated_images:
        logger.info(f"✓ Generated {len(state.generated_images)} character reference images")
    else:
        logger.warning("✗ No images generated")
    
    # 4. execution_log.json
    log_path = output_dir / "execution_log.json"
    with open(log_path, 'w') as f:
        json.dump({
            "mode": state.mode,
            "execution_log": state.execution_log,
            "errors": state.errors,
            "user_approved": state.user_approved,
            "revision_count": state.revision_count,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    logger.info(f"✓ Generated execution_log.json")
    
    # 5. workflow_graph.json (export LangGraph structure)
    graph_json = output_dir / "workflow_graph.json"
    try:
        # Convert graph to readable format
        graph_info = {
            "workflow": "WritersRoom",
            "nodes": [
                "mode_selector",
                "scriptwriter",
                "validator",
                "hitl",
                "character_designer",
                "image_synthesizer",
                "memory_commit",
                "finalize"
            ],
            "description": "Multi-agent Supervisor-Worker hierarchical workflow",
            "mcp_integration": "Dynamic tool discovery via MCP Registry",
            "memory_layer": "FAISS + JSON persistence"
        }
        with open(graph_json, 'w') as f:
            json.dump(graph_info, f, indent=2)
        logger.info(f"✓ Generated workflow_graph.json")
    except Exception as e:
        logger.warning(f"Could not generate workflow_graph.json: {e}")
    
    return success

def _print_summary(state: WorkflowState):
    """Print execution summary."""
    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    
    print(f"\nMode: {state.mode}")
    print(f"Script Generated: {'Yes' if state.script else 'No'}")
    if state.script:
        print(f"  - Scenes: {len(state.script.scenes)}")
        print(f"  - Characters: {len(state.script.character_list)}")
    
    print(f"\nHITL Decision: {state.hitl_decision}")
    print(f"User Approved: {state.user_approved}")
    print(f"Revision Count: {state.revision_count}")
    
    print(f"\nCharacters Extracted: {len(state.characters)}")
    if state.characters:
        print("  - " + ", ".join([c.name for c in state.characters]))
    
    print(f"\nImages Generated: {len(state.generated_images)}")
    
    if state.errors:
        print(f"\nErrors ({len(state.errors)}):")
        for error in state.errors:
            print(f"  - {error}")
    
    print(f"\nExecution Log ({len(state.execution_log)} steps):")
    for log_entry in state.execution_log:
        print(f"  - {log_entry}")
    
    print("\n" + "="*80)
    print("Output artifacts saved to: outputs/")
    print("  - scene_manifest.json")
    print("  - character_db.json")
    print("  - images/ (character reference cards)")
    print("  - execution_log.json")
    print("  - workflow_graph.json")
    print("  - memory/ (persistent memory store)")
    print("="*80 + "\n")

if __name__ == "__main__":
    exit(main())
