"""
Voice Synthesizer Agent - Generate speech audio from scene dialogue using edge-tts.
Phase 2 - audio branch.
"""
import json
import logging
import io
import wave
import subprocess
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

from src.agents.base_agent import BaseAgent
from src.schema import SceneTask, AudioTrack
from src.utils.prompts import VOICE_SYNTHESIZER_SYSTEM_PROMPT
from src.utils.memory import MemoryStore

logger = logging.getLogger(__name__)

try:
    from pydub import AudioSegment
    _PYDUB_AVAILABLE = True
except ImportError:
    _PYDUB_AVAILABLE = False


class VoiceSynthesizerAgent(BaseAgent):
    """
    Voice Synthesizer Agent
    -----------------------
    Uses edge-tts to generate ultra-realistic distinct male and female voices.
    """

    def __init__(self, memory_store: Optional[MemoryStore] = None):
        super().__init__(
            name="Voice Synthesizer",
            role="voice_synthesizer",
            memory_store=memory_store,
        )
        self.system_prompt = VOICE_SYNTHESIZER_SYSTEM_PROMPT
        self.output_dir = Path("outputs/audio")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._char_db: Dict[str, Dict] = {}

    def _load_char_db(self, path: str):
        p = Path(path)
        if not p.exists():
            return
        try:
            with open(p, "r", encoding="utf-8") as f:
                db = json.load(f)
            for char in db.get("characters", []):
                name = char.get("name", "").lower()
                self._char_db[name] = char
        except Exception as exc:
            logger.warning(f"Could not load character_db: {exc}")

    def _get_gender(self, character_name: str) -> str:
        char_data = self._char_db.get(character_name.lower(), {})
        return char_data.get("gender", "unknown") or "unknown"

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        scene_task: Optional[SceneTask] = input_data.get("scene_task")
        if scene_task is None:
            raw = input_data.get("scene_task_dict")
            if raw:
                scene_task = SceneTask(**raw)

        if scene_task is None:
            return {"success": False, "error": "No scene_task provided", "audio_track": None}

        char_db_path = input_data.get("character_db_path", "outputs/character_db.json")
        self._load_char_db(char_db_path)

        output_dir = Path(input_data.get("output_dir", "outputs")) / "audio"
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Voice Synthesizer: processing scene {scene_task.scene_id}")

        char_tracks: List[Dict[str, str]] = []
        clip_paths: List[Path] = []

        for idx, dialogue in enumerate(scene_task.dialogues):
            char_name = dialogue.character or "Narrator"
            text = dialogue.dialogue.strip()
            if not text:
                continue

            gender = self._get_gender(char_name)
            clip_path = output_dir / f"scene_{scene_task.scene_id}_line_{idx:02d}.wav"

            logger.info(f"  Synthesizing (edge-tts): {char_name} ({gender}) -> {clip_path.name}")
            success = self._synthesize_line(char_name, gender, text, clip_path)

            if success:
                clip_paths.append(clip_path)
                char_tracks.append({"character": char_name, "path": str(clip_path)})
            else:
                silent_path = self._write_silence(clip_path, duration_seconds=1.5)
                clip_paths.append(silent_path)
                char_tracks.append({"character": char_name, "path": str(silent_path)})

        current_ms = 0
        for track_dict in char_tracks:
            p = Path(track_dict["path"])
            dur_ms = int(self._wav_duration(p) * 1000.0)
            track_dict["start_ms"] = current_ms
            track_dict["end_ms"] = current_ms + dur_ms
            current_ms += dur_ms + 300  # 0.3s silence between clips in _merge_clips

        scene_audio_path = output_dir / f"scene_{scene_task.scene_id}.wav"
        duration = self._merge_clips(clip_paths, scene_audio_path)

        audio_track = AudioTrack(
            scene_id=scene_task.scene_id,
            audio_path=str(scene_audio_path),
            duration_seconds=duration,
            character_tracks=char_tracks,
        )

        self.commit_to_memory("audio_reference", {
            "scene_id": scene_task.scene_id,
            "audio_path": str(scene_audio_path),
            "duration_seconds": duration,
            "lines_processed": len(char_tracks),
            "generated_at": datetime.now().isoformat(),
        })

        return {
            "success": True,
            "audio_track": audio_track,
            "scene_id": scene_task.scene_id,
        }

    def _synthesize_line(self, character_name: str, gender: str, text: str, out_path: Path) -> bool:
        if gender == "female":
            voice = "en-US-AriaNeural"
        else:
            voice = "en-US-ChristopherNeural"
            
        mp3_path = out_path.with_suffix(".mp3")
        try:
            cmd = ["edge-tts", "--voice", voice, "--text", text, "--write-media", str(mp3_path)]
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            
            if result.returncode == 0 and mp3_path.exists():
                import imageio_ffmpeg
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                cmd_ffmpeg = [ffmpeg_exe, "-y", "-i", str(mp3_path), str(out_path)]
                subprocess.run(cmd_ffmpeg, capture_output=True)
                
                if mp3_path.exists():
                    mp3_path.unlink()
                return out_path.exists()
            else:
                logger.warning(f"edge-tts failed: {result.stderr.decode()}")
                return False
        except Exception as exc:
            logger.warning(f"edge-tts exception: {exc}")
            return False

    def _write_silence(self, path: Path, duration_seconds: float = 1.0) -> Path:
        sample_rate = 22050
        n_samples = int(sample_rate * duration_seconds)
        with wave.open(str(path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(b"\x00\x00" * n_samples)
        return path

    def _wav_duration(self, path: Path) -> float:
        try:
            with wave.open(str(path), "r") as wf:
                return wf.getnframes() / float(wf.getframerate())
        except Exception:
            return 0.0

    def _merge_clips(self, clip_paths: List[Path], out_path: Path) -> float:
        valid_paths = [p for p in clip_paths if p.exists()]
        if not valid_paths:
            self._write_silence(out_path, 1.0)
            return 1.0

        # Use native wave module to concatenate
        data = []
        params = None
        for p in valid_paths:
            try:
                with wave.open(str(p), "r") as wf:
                    if params is None:
                        params = wf.getparams()
                    data.append(wf.readframes(wf.getnframes()))
            except Exception:
                pass
                
        if not data:
            self._write_silence(out_path, 1.0)
            return 1.0
            
        with wave.open(str(out_path), "w") as out:
            out.setparams(params)
            for d in data:
                out.writeframes(d)
                # write 0.3s silence between clips
                sample_rate = params.framerate
                silence_frames = int(sample_rate * 0.3)
                out.writeframes(b"\x00" * params.sampwidth * params.nchannels * silence_frames)
                
        return self._wav_duration(out_path)
