"""
Phase 5 — StateManager
Append-only versioned snapshot store for the full pipeline state.

Directory layout
----------------
outputs/versions/
  v0001/
    pipeline_state.json   # serialised workflow state dict
    scene_manifest.json
    character_db.json
    timing_manifest.json  (if exists)
    execution_log.json    (if exists)
    execution_log_phase2.json (if exists)
    audio/                # copy of outputs/audio/
    images/               # copy of outputs/images/
    raw_scenes/           # copy of outputs/raw_scenes/
    meta.json             # {version, parent_version, change_summary, edit_query, timestamp}

Public API
----------
  snapshot(state_dict, change_summary, parent=None, edit_query="") -> int
  revert(version) -> dict
  history() -> list[dict]
  get_version_dir(version) -> Path
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import logging

logger = logging.getLogger(__name__)

# Files copied 1-to-1 from outputs/ root
_SNAPSHOT_FILES = [
    "scene_manifest.json",
    "character_db.json",
    "timing_manifest.json",
    "execution_log.json",
    "execution_log_phase2.json",
    "task_graph_log.json",
]

# Sub-directories copied recursively
_SNAPSHOT_DIRS = [
    "audio",
    "images",
    "raw_scenes",
]


class StateManager:
    """Append-only versioned snapshot store."""

    def __init__(self, outputs_dir: str = "outputs"):
        self.outputs_dir = Path(outputs_dir)
        self.versions_dir = self.outputs_dir / "versions"
        self.versions_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Core API                                                             #
    # ------------------------------------------------------------------ #

    def snapshot(
        self,
        state_dict: Dict[str, Any],
        change_summary: str,
        parent: Optional[int] = None,
        edit_query: str = "",
    ) -> int:
        """
        Create a new version snapshot.

        Parameters
        ----------
        state_dict      : Arbitrary pipeline state to preserve (JSON-serialisable).
        change_summary  : Human-readable description, e.g. "Initial run".
        parent          : Version number of the snapshot this was derived from.
        edit_query      : The raw user edit string that triggered this snapshot.

        Returns
        -------
        int : The new version number.
        """
        version = self._next_version()
        ver_dir = self.versions_dir / f"v{version:04d}"
        ver_dir.mkdir(parents=True, exist_ok=True)

        # Copy individual files
        for fname in _SNAPSHOT_FILES:
            src = self.outputs_dir / fname
            if src.exists():
                try:
                    shutil.copy2(src, ver_dir / fname)
                except Exception as exc:
                    logger.warning(f"Snapshot: could not copy {fname}: {exc}")

        # Copy directories
        for dname in _SNAPSHOT_DIRS:
            src = self.outputs_dir / dname
            dst = ver_dir / dname
            if src.exists():
                try:
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                except Exception as exc:
                    logger.warning(f"Snapshot: could not copy dir {dname}: {exc}")

        # Write pipeline_state.json
        try:
            with open(ver_dir / "pipeline_state.json", "w", encoding="utf-8") as f:
                json.dump(state_dict, f, indent=2, default=str)
        except Exception as exc:
            logger.warning(f"Snapshot: could not write pipeline_state.json: {exc}")

        # Write meta.json
        meta: Dict[str, Any] = {
            "version": version,
            "parent_version": parent,
            "change_summary": change_summary,
            "edit_query": edit_query,
            "timestamp": datetime.now().isoformat(),
        }
        with open(ver_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"StateManager: snapshot v{version:04d} saved to {ver_dir}")
        return version

    def revert(self, version: int) -> Dict[str, Any]:
        """
        Restore outputs/ to the state captured in version N.

        Returns the pipeline_state.json dict from that version.
        """
        ver_dir = self.versions_dir / f"v{version:04d}"
        if not ver_dir.exists():
            raise FileNotFoundError(f"Version v{version:04d} not found at {ver_dir}")

        # Restore files
        for fname in _SNAPSHOT_FILES:
            src = ver_dir / fname
            dst = self.outputs_dir / fname
            if src.exists():
                try:
                    shutil.copy2(src, dst)
                except Exception as exc:
                    logger.warning(f"Revert: could not restore {fname}: {exc}")

        # Restore directories
        for dname in _SNAPSHOT_DIRS:
            src = ver_dir / dname
            dst = self.outputs_dir / dname
            if src.exists():
                try:
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                except Exception as exc:
                    logger.warning(f"Revert: could not restore dir {dname}: {exc}")

        logger.info(f"StateManager: reverted outputs/ to v{version:04d}")

        state_path = ver_dir / "pipeline_state.json"
        if state_path.exists():
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def history(self) -> List[Dict[str, Any]]:
        """
        Return all snapshots newest-first.
        Each entry is the meta.json dict augmented with mp4_count and mp4_paths.
        """
        result: List[Dict[str, Any]] = []
        for ver_dir in sorted(self.versions_dir.iterdir()):
            if not ver_dir.is_dir() or not ver_dir.name.startswith("v"):
                continue
            num_part = ver_dir.name[1:]
            if not num_part.isdigit():
                continue
            meta_path = ver_dir / "meta.json"
            if not meta_path.exists():
                continue
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                continue
            raw_scenes = ver_dir / "raw_scenes"
            mp4s = (
                sorted(raw_scenes.glob("scene_*_final.mp4"))
                if raw_scenes.exists()
                else []
            )
            meta["mp4_count"] = len(mp4s)
            meta["mp4_paths"] = [str(p) for p in mp4s]
            result.append(meta)
        return sorted(result, key=lambda x: x.get("version", 0), reverse=True)

    def get_version_dir(self, version: int) -> Path:
        return self.versions_dir / f"v{version:04d}"

    def latest_version(self) -> Optional[int]:
        """Return the highest version number, or None if no snapshots exist."""
        existing = self._all_version_numbers()
        return existing[-1] if existing else None

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    def _next_version(self) -> int:
        existing = self._all_version_numbers()
        return (existing[-1] + 1) if existing else 1

    def _all_version_numbers(self) -> List[int]:
        nums = []
        for p in self.versions_dir.iterdir():
            if p.is_dir() and p.name.startswith("v") and p.name[1:].isdigit():
                nums.append(int(p.name[1:]))
        return sorted(nums)
