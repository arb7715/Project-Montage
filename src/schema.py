from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass, field as dc_field

class SceneMode(str, Enum):
    MANUAL = "manual"
    AUTONOMOUS = "autonomous"

class DialogueEntry(BaseModel):
    character: str
    dialogue: str

class SceneElement(BaseModel):
    scene_id: int
    heading: str  # e.g., "INT. COFFEE SHOP - MORNING"
    actions: List[str]
    dialogues: List[DialogueEntry]
    visual_cues: List[str] = Field(default_factory=list)

class ScriptManifest(BaseModel):
    mode: SceneMode
    scenes: List[SceneElement]
    character_list: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class CharacterProfile(BaseModel):
    name: str
    personality_traits: List[str]
    appearance_description: str
    reference_style: str
    gender: Optional[str] = None          # "male" / "female" / "unknown"
    image_reference: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class CharacterDatabase(BaseModel):
    characters: List[CharacterProfile]
    created_at: str
    updated_at: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ValidationResult(BaseModel):
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    corrected_script: Optional[ScriptManifest] = None

class MemoryEntry(BaseModel):
    entry_type: str  # "script_history", "character_metadata", "image_reference"
    content: Dict[str, Any]
    timestamp: str
    embedding: Optional[List[float]] = None


# ─── Phase 2 Models ───────────────────────────────────────────────────────────

class SceneTask(BaseModel):
    """A single parallelisable unit of work emitted by the Scene Parser."""
    scene_id: int
    heading: str
    actions: List[str] = Field(default_factory=list)
    dialogues: List[DialogueEntry] = Field(default_factory=list)
    visual_cues: List[str] = Field(default_factory=list)
    character_names: List[str] = Field(default_factory=list)


class AudioTrack(BaseModel):
    """Result produced by the Voice Synthesizer for one scene."""
    scene_id: int
    audio_path: str          # path to the WAV file
    duration_seconds: float = 0.0
    character_tracks: List[Dict[str, Any]] = Field(default_factory=list)  # [{character, path, start_ms, end_ms}]


class VideoFrame(BaseModel):
    """Metadata for a generated frame sequence for one scene."""
    scene_id: int
    frame_dir: str           # directory containing PNG frames
    frame_count: int = 0
    raw_video_path: str = ""  # assembled silent MP4


class SyncedScene(BaseModel):
    """Final lip-synced output for one scene."""
    scene_id: int
    video_path: str          # final MP4
    audio_path: str          # merged WAV
    lip_synced: bool = False
    duration_seconds: float = 0.0


@dataclass
class Phase2State:
    """Shared state across all Phase 2 agents in the workflow."""
    # Input
    scene_manifest_path: str = "outputs/scene_manifest.json"
    character_db_path: str = "outputs/character_db.json"
    output_dir: str = "outputs"

    # Scene Parser outputs
    scene_tasks: List = dc_field(default_factory=list)   # List[SceneTask]
    task_graph_logged: bool = False

    # Voice Synthesizer outputs (one per scene)
    audio_tracks: List = dc_field(default_factory=list)  # List[AudioTrack]
    audio_success: bool = False

    # Video Generator outputs (one per scene)
    video_frames: List = dc_field(default_factory=list)  # List[VideoFrame]
    video_success: bool = False

    # Face Swap outputs
    face_swapped_videos: List = dc_field(default_factory=list)  # List[str] paths
    face_swap_success: bool = False

    # Lip Sync outputs
    synced_scenes: List = dc_field(default_factory=list)  # List[SyncedScene]
    lip_sync_success: bool = False

    # Metadata
    execution_log: List = dc_field(default_factory=list)
    errors: List = dc_field(default_factory=list)
