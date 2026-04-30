"""
Main Entry Point - The Studio Floor Phase 2
Video and Audio Synthesis Layer
"""
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Optional
from datetime import datetime

# Force UTF-8 output so Unicode arrow/checkmark chars in log messages don't crash
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.workflow_phase2 import StudioFloorWorkflow
from src.schema import Phase2State
from src.utils.memory import MemoryStore
from src.state.manager import StateManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)



def main():
    parser = argparse.ArgumentParser(
        description="The Studio Floor - Phase 2: Video and Audio Synthesis Layer"
    )
    parser.add_argument(
        "--scene-manifest",
        type=str,
        default="outputs/scene_manifest.json",
        help="Path to scene_manifest.json produced by Phase 1",
    )
    parser.add_argument(
        "--character-db",
        type=str,
        default="outputs/character_db.json",
        help="Path to character_db.json produced by Phase 1",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Root output directory",
    )
    args = parser.parse_args()

    # Validate inputs
    if not Path(args.scene_manifest).exists():
        logger.error(
            f"scene_manifest.json not found at '{args.scene_manifest}'. "
            "Run Phase 1 first (python -m src.main)."
        )
        return 1

    # Ensure output directories exist
    output_dir = Path(args.output_dir)
    (output_dir / "raw_scenes").mkdir(parents=True, exist_ok=True)
    (output_dir / "audio").mkdir(parents=True, exist_ok=True)

    # Initialise memory store
    memory_store = MemoryStore(str(output_dir / "memory"))
    logger.info(f"Memory store initialised at {output_dir / 'memory'}")

    # Banner
    print("\n" + "=" * 80)
    print("THE STUDIO FLOOR - PHASE 2")
    print("Video and Audio Synthesis Layer")
    print("=" * 80 + "\n")

    # Build initial state
    initial_state = Phase2State(
        scene_manifest_path=args.scene_manifest,
        character_db_path=args.character_db,
        output_dir=args.output_dir,
    )

    # Run workflow
    workflow = StudioFloorWorkflow(memory_store)
    logger.info("StudioFloor Workflow initialised")

    try:
        final_state = workflow.run(initial_state)
        _generate_artifacts(final_state, output_dir)
        _print_summary(final_state)

        # ── Phase 5: auto-snapshot after every successful run ──────────
        try:
            mgr = StateManager(str(output_dir))
            snap_state = {
                "phase": 2,
                "scene_tasks":    len(final_state.get("scene_tasks", [])),
                "synced_scenes":  len(final_state.get("synced_scenes", [])),
                "errors":         final_state.get("errors", []),
            }
            ver = mgr.snapshot(snap_state, change_summary="Phase 2/3 pipeline run")
            logger.info(f"StateManager: auto-snapshot v{ver:04d} created")
        except Exception as snap_exc:
            logger.warning(f"StateManager snapshot failed (non-fatal): {snap_exc}")

        logger.info("Phase 2 execution completed successfully")
    except Exception as exc:
        logger.error(f"Phase 2 workflow failed: {exc}", exc_info=True)
        return 1

    return 0


# ---------------------------------------------------------------------------
# Artifact generation
# ---------------------------------------------------------------------------

def _generate_artifacts(state: dict, output_dir: Path):
    """Write Phase 2 summary artifacts."""
    logger.info("\nGenerating Phase 2 output artifacts...")

    # execution_log.json (Phase 2 version)
    log_path = output_dir / "execution_log_phase2.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "phase": 2,
                "execution_log": state.get("execution_log", []),
                "errors": state.get("errors", []),
                "audio_tracks": state.get("audio_tracks", []),
                "synced_scenes": state.get("synced_scenes", []),
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
            default=str,
        )
    logger.info("[OK] execution_log_phase2.json")

    # Verify deliverables
    raw_scenes_dir = output_dir / "raw_scenes"
    audio_dir = output_dir / "audio"

    mp4s = list(raw_scenes_dir.glob("scene_*_final.mp4")) if raw_scenes_dir.exists() else []
    wavs = list(audio_dir.glob("scene_*.wav")) if audio_dir.exists() else []
    task_graph = output_dir / "task_graph_log.json"

    logger.info(
        f"[OK] {len(mp4s)} final MP4(s) in raw_scenes/  |  "
        f"{len(wavs)} WAV(s) in audio/  |  "
        f"task_graph: {'[OK]' if task_graph.exists() else '[X]'}"
    )


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(state: dict):
    print("\n" + "=" * 80)
    print("PHASE 2 EXECUTION SUMMARY")
    print("=" * 80)

    audio_tracks = state.get("audio_tracks", [])
    video_frames = state.get("video_frames", [])
    synced_scenes = state.get("synced_scenes", [])
    errors = state.get("errors", [])

    print(f"\nScene Tasks Processed : {len(state.get('scene_tasks', []))}")
    print(f"Audio Tracks Generated: {len(audio_tracks)}")
    print(f"Video Frames Generated: {len(video_frames)}")
    print(f"Lip-Synced Scenes     : {len(synced_scenes)}")

    if synced_scenes:
        print("\nFinal Videos:")
        for sc in synced_scenes:
            sc_dict = sc if isinstance(sc, dict) else sc.model_dump()
            synced_flag = "[lip-synced]" if sc_dict.get("lip_synced") else "[audio-merged]"
            print(
                f"  Scene {sc_dict['scene_id']:>2}: "
                f"{Path(sc_dict['video_path']).name}  "
                f"({sc_dict.get('duration_seconds', 0):.1f}s)  {synced_flag}"
            )

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for err in errors:
            print(f"  - {err}")

    log = state.get("execution_log", [])
    print(f"\nExecution Log ({len(log)} steps):")
    for entry in log:
        print(f"  - {entry}")

    print("\n" + "=" * 80)
    print("Output artifacts:")
    print("  outputs/raw_scenes/scene_*_final.mp4  <- lip-synced videos")
    print("  outputs/audio/scene_*.wav              <- audio tracks")
    print("  outputs/task_graph_log.json            <- task graph log")
    print("  outputs/execution_log_phase2.json      <- execution trace")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    exit(main())
