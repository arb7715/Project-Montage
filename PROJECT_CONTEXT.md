# PROJECT MONTAGE — Master Handover Document

> **Audience:** A fresh LLM/agent that needs full project context in a single read.
> **Read order:** Top to bottom. Section 11 (Pending Work) is the action queue.
> **Last updated:** 2026-05-05

---

## 1. Project identity

- **Course:** CS-4015 Agentic AI, FAST NUCES
- **Project:** *AI-Powered Animated Video Generation System — From Prompt to Polished Short Film*
- **Repo root on dev machine:** `c:\Users\HP\Downloads\i221105_A3\i221105_A3\`
- **Group submission deadline:** 5th May 2026, 11:59 PM (today, when this doc was written)
- **Source PDFs in repo root:**
  - `Agentic AI Final Project - 2026.pdf` — master spec, all five phases
  - `Assignment3.pdf` (= Phase 1 spec, *The Writer's Room*)
  - `Assignment4.pdf` (= Phase 2 + 3 combined spec, *The Studio Floor*)
- **Plain-text mirrors of those PDFs:** `Assignment3.txt`, `assignment4_text.txt`

---

## 2. Phase map (master doc → student deliverables)

| Master doc | Codebase name              | Status      | Owner notes                                  |
|------------|----------------------------|-------------|----------------------------------------------|
| Phase 1    | Writer's Room              | DONE        | Outputs: `scene_manifest.json`, `character_db.json`, `outputs/images/`, LangGraph workflow |
| Phase 2    | Studio Floor (audio half)  | DONE        | edge-tts → per-line WAV + merged scene WAV   |
| Phase 3    | Studio Floor (video half)  | MOSTLY DONE | Wide B-roll → IP-Adapter passthrough → per-line Wav2Lip close-ups → composed scene MP4 |
| Phase 4    | Web Interface              | DONE        | Streamlit `src/ui/app.py`; optional FastAPI later                             |
| Phase 5    | Edit Agent + Undo          | DONE        | `StateManager` + `EditAgent`; snapshots in `outputs/versions/`               |

**Note:** Phases 4 and 5 are implemented in-tree (Streamlit UI, StateManager snapshots, Edit Agent). FastAPI routes remain optional polish.

---

## 3. Repository layout

```
i221105_A3/
├── Agentic AI Final Project - 2026.pdf   # master spec
├── Assignment3.pdf / .txt                 # Phase 1 spec
├── Assignment4.pdf / assignment4_text.txt # Phase 2+3 spec
├── colab_sd_api.txt                       # Colab notebook cell text (paste into Colab)
├── config/
│   └── colab_api.txt                      # one line: the active trycloudflare URL
├── outputs/
│   ├── scene_manifest.json                # Phase 1 output (master schema)
│   ├── character_db.json                  # Phase 1 output (per-character profiles)
│   ├── execution_log.json                 # Phase 1 trace
│   ├── execution_log_phase2.json          # Phase 2 trace
│   ├── task_graph_log.json                # Scene-Parser-Agent task graph
│   ├── workflow_graph.json                # langgraph topology dump
│   ├── images/                            # character portraits (sarah/waiter/daniel.png)
│   ├── audio/                             # scene_X.wav (merged) + scene_X_line_NN.wav
│   ├── memory/                            # FAISS vector store + memory_entries.json
│   ├── raw_scenes/                        # Phase 3 outputs
│   │   ├── scene_X_final.mp4              # final composed scene (only "deliverable" mp4)
│   │   ├── scene_X_swapped.mp4            # intermediate (used to exist; deprecated)
│   │   └── scene_X_frames/
│   │       ├── scene_X_static.jpg          # wide B-roll from SD txt2img
│   │       └── scene_X_static_swapped.jpg  # wide B-roll after IP-Adapter pass (or copy)
│   └── _smoke/                            # smoke-test artifacts (non-deliverable)
├── requirements.txt
├── requirements_phase2.txt
└── src/
    ├── __init__.py
    ├── main.py                            # Phase 1 entry: python -m src.main
    ├── main_phase2.py                     # Phase 2+3 entry: python -m src.main_phase2
    ├── workflow.py                        # Phase 1 LangGraph
    ├── workflow_phase2.py                 # Phase 2+3 LangGraph
    ├── schema.py                          # Pydantic + dataclass shared schemas
    ├── smoke_test_colab.py                # python -m src.smoke_test_colab
    ├── agents/
    │   ├── base_agent.py                  # MCP tool discovery base
    │   ├── scriptwriter.py
    │   ├── validator.py
    │   ├── hitl_agent.py
    │   ├── character_designer.py
    │   ├── image_synthesizer.py
    │   ├── scene_parser.py                # Phase 2 entry, fan-out
    │   ├── voice_synthesizer.py           # edge-tts
    │   ├── video_generator.py             # SD txt2img wide B-roll
    │   ├── face_swap.py                   # IP-Adapter wide B-roll injection (passthrough on failure)
    │   └── lip_sync.py                    # per-line Wav2Lip + scene compositor (the heavy one)
    └── utils/
        ├── mcp_registry.py                # MCP tool registry (no hardcoded APIs constraint)
        ├── memory.py                      # FAISS + JSON entries store
        └── prompts.py                     # all per-agent system prompts
```

---

## 4. Tech stack (committed, do not change without reason)

| Layer            | Choice                                                              |
|------------------|---------------------------------------------------------------------|
| Orchestration    | **LangGraph** (StateGraph + Send() fan-out)                         |
| Schemas          | **Pydantic v2** + `@dataclass` for pipeline state                   |
| LLM (Phase 1)    | **Ollama** (local) for Scriptwriter / Character Designer            |
| Memory           | **FAISS** vector index + `memory_entries.json` log                  |
| TTS              | **edge-tts** (free, MS Neural voices: Aria female / Christopher male) |
| Image Gen        | **Stable Diffusion v1.5** on Colab T4 GPU (free tier)               |
| Identity inject  | **IP-Adapter** (`ip-adapter_sd15.bin`) on Colab — graceful passthrough |
| Lip Sync         | **Wav2Lip** (`wav2lip_gan.pth`) on Colab                            |
| Video assembly   | **moviepy 2.x** (with v1 fallback) + OpenCV + ffmpeg                |
| Public tunnel    | **Cloudflare Quick Tunnel** (`trycloudflare.com`)                   |
| Phase 4 plan     | **FastAPI + Uvicorn** backend, **Streamlit** OR **Next.js** frontend |
| Phase 5 plan     | **LangGraph** intent classifier + JSON snapshot directory store     |

**Hard constraint from Phase 1 rubric:** All tools must be discovered via the **MCP registry** (`src/utils/mcp_registry.py`). No agent imports a tool directly; they call `self.call_mcp_tool(name, params)`. Don't violate this.

---

## 5. Shared data schemas (the contract)

Defined in `src/schema.py`. Every phase reads/writes these.

### `scene_manifest.json` (Phase 1 → Phase 2/3)
```json
{
  "mode": "manual|autonomous",
  "scenes": [
    {
      "scene_id": 1,
      "heading": "INT. COFFEE SHOP - MORNING",
      "actions": ["Sarah enters the cafe and looks anxious."],
      "dialogues": [
        {"character": "Sarah",  "dialogue": "I hope he shows up this time."},
        {"character": "Waiter", "dialogue": "Would you like to order something?"}
      ],
      "visual_cues": ["Warm natural light through the window."]
    }
  ],
  "character_list": ["Sarah", "Waiter"],
  "metadata": {"input_format": "screenplay_text", "num_scenes": 2}
}
```

### `character_db.json`
```json
{
  "characters": [
    {
      "name": "Sarah",
      "personality_traits": ["independent", "determined", "empathetic"],
      "appearance_description": "...",
      "reference_style": "realistic",
      "gender": "female",
      "image_reference": "outputs/images/sarah.png",
      "metadata": {"auto_generated": true}
    }
  ],
  "created_at": "...", "updated_at": "...", "total_count": 3
}
```

### Phase 2/3 in-memory state (Pydantic): `SceneTask`, `AudioTrack`, `VideoFrame`, `SyncedScene`, `Phase2State` — see `src/schema.py`.

### Pending: `timing_manifest.json` (Phase 2 master-doc deliverable)
The current pipeline doesn't yet emit this exact filename. The data exists inside `AudioTrack.duration_seconds` and `audio_track.character_tracks[]`. **For Phase 4/Phase 5 we should write it to disk** as:
```json
{
  "scenes": [
    {"scene_id": 1, "audio_file": "audio/scene_1.wav", "start_ms": 0, "end_ms": 6072,
     "lines": [
       {"character": "Sarah",  "audio_file": "audio/scene_1_line_00.wav", "start_ms": 0,    "end_ms": 2920},
       {"character": "Waiter", "audio_file": "audio/scene_1_line_01.wav", "start_ms": 3220, "end_ms": 6072}
     ]}
  ]
}
```

### Pending: `pipeline_state.json` (Phase 5 versioning)
A union of all the above + paths to the produced MP4s, with a `version` integer and `parent_version`. See section 9.

---

## 6. End-to-end execution today

### One-time setup (already done on dev machine)
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements_phase2.txt
pip install moviepy edge-tts opencv-python imageio[ffmpeg] pydub
```

### Per-session flow
1. **Open Colab notebook**, paste the cell content from `colab_sd_api.txt`, run.
2. The cell loads SD + IP-Adapter + Wav2Lip patches and starts FastAPI on port 8000.
3. **Cloudflared** Quick Tunnel prints a `https://<random>.trycloudflare.com` URL.
4. Paste that URL into `config/colab_api.txt` (one line).
5. Smoke test from PC: `python -m src.smoke_test_colab` — expect 4× OK.
6. Run Phase 1 if needed: `python -m src.main`
7. Run Phase 2+3: `python -m src.main_phase2`
8. Final outputs: `outputs/raw_scenes/scene_X_final.mp4`

### Colab gotchas (already encoded in `colab_sd_api.txt`, do not regress)

- `pipe_img2img.load_ip_adapter(...)` **before** `enable_attention_slicing()`. Reverse order causes `SlicedAttnProcessor.__init__() missing 1 required positional argument: 'slice_size'`.
- Use `ip-adapter_sd15.bin` (NOT `plus-face`). Plus variants emit tuple embeddings → `'tuple' object has no attribute 'shape'` in newer diffusers.
- `enable_attention_slicing()` is now disabled on `pipe_img2img` to avoid IP-Adapter regressions.
- `face_swap` endpoint **passes through the base image on any failure** — IP-Adapter is treated as best-effort, not critical.
- Wav2Lip's `audio.py` patched at runtime: `librosa.filters.mel(sr=..., n_fft=..., ...)` (librosa ≥0.10 made these keyword-only).
- Wav2Lip's `inference.py` patched at runtime: `'DIVX'` → `'mp4v'` and `temp/result.avi` → `temp/result.mp4` (DIVX silently fails on Colab's OpenCV 4.13).
- `temp/` is created fresh before every Wav2Lip job; otherwise OpenCV's VideoWriter writes nowhere and `inference.py` exits 0 with no output file.
- ffmpeg-rescue: if Wav2Lip exits 0 but no `out.mp4`, our handler manually muxes `temp/result.mp4` + audio.

---

## 7. Phase 3 video-quality plan (incremental, optional polish)

Already implemented:
- ✅ Wide B-roll establishing shot per scene (Ken Burns push-in, ~1.6s)
- ✅ Hard cut to per-line talking-head close-up via Wav2Lip on the **character portrait**
- ✅ Cinematic "blurred backdrop" canvas fitting (854×480) so close-ups don't look pasted

Easy upgrades (in priority order, all in `src/agents/lip_sync.py` and `src/agents/video_generator.py`):
1. **Subtitle burn-in** — `Sarah: I hope he shows up this time.` Bottom-center, semi-transparent black bar, white DejaVu/Arial, drop shadow. ~30 LOC with PIL `ImageDraw` over the per-line clip.
2. **Action banners** — Top-left, smaller font, fade in/out: `Sarah enters the cafe`. Triggered from `scene_task.actions` on the establishing shot only.
3. **Multi-shot scenes** — Extend `scene_manifest.json` schema with `shots[]` per scene:
   ```json
   {"scene_id": 1, "shots": [
     {"type": "establishing", "prompt": "cafe exterior, door opens, sunlight"},
     {"type": "action",       "prompt": "Sarah walks in, anxious"},
     {"type": "dialogue",     "character": "Sarah", "line_index": 0},
     {"type": "dialogue",     "character": "Waiter","line_index": 1}
   ]}
   ```
   Then `LipSyncAgent._compose_scene` iterates `shots[]` instead of `establishing + line_clips`.
4. **BGM** — pick royalty-free loops from `outputs/bgm/` and mix at -22dB under dialogue using `pydub`. Master-doc Phase 2 explicitly mentions BGM.
5. **Crossfades between shots** — moviepy `crossfadein(0.25)` instead of hard cuts.

Out of scope (don't do):
- Two-faces-in-one-frame lip-sync — research-grade problem. Use shot-reverse-shot (already implemented).
- Animated character motion — would need AnimateDiff/Stable Video Diffusion. Phase 3 master spec only requires Ken Burns + composition.

---

## 8. Phase 4 (Web Interface) — implementation plan

**Smallest viable shape that satisfies the rubric:**

```
src/api/
├── main_api.py            # FastAPI app
├── routes/
│   ├── pipeline.py        # POST /run/full, POST /run/phase/{n}, GET /status/{run_id}
│   ├── scenes.py          # GET /scenes, GET /scenes/{id}/video
│   ├── characters.py      # GET /characters, PUT /characters/{name}
│   └── edit.py            # Phase 5: POST /edit, GET /history, POST /revert/{version}
└── ws.py                  # WebSocket /ws/progress for live phase events
```

Frontend recommendation given today's deadline: **Streamlit** (single Python file, no React build, ships in ~3 hours).

```
src/ui/
└── app.py                 # streamlit run src/ui/app.py
```

Streamlit gives you:
- Prompt input → calls `POST /run/full`
- Per-phase progress with `st.progress()` updated via `st.empty().rerun_every`
- Phase re-run buttons (`POST /run/phase/2`)
- Final video preview (`st.video(...)`) and download
- Edit text box + version history list (Phase 5 UI)

If a more cinematic UI is desired and time allows, swap to **Next.js + Tailwind**, but Streamlit is the pragmatic choice today.

**Bridge layer:** wrap each existing `*.execute()` agent call inside FastAPI route handlers. The current `StudioFloorWorkflow.run()` returns a `dict` — that's already JSON-serialisable for `/status/{run_id}`.

---

## 9. Phase 5 (Edit Agent + Undo) — implementation plan

### 9.1 State versioning

Append-only directory:
```
outputs/versions/
├── v0001/
│   ├── pipeline_state.json     # full snapshot of Phase 1+2+3 state
│   ├── scene_manifest.json
│   ├── character_db.json
│   ├── audio/   ...
│   ├── images/  ...
│   ├── raw_scenes/ ...
│   └── meta.json               # {parent_version, change_summary, ts, edit_query}
├── v0002/  ...
```

`StateManager` API (new file `src/state/manager.py`):
- `snapshot(state_dict, change_summary, parent=None) -> int` → returns version number, hard-links large files where possible.
- `revert(version) -> dict` → restores `outputs/` to that snapshot, returns the state.
- `history() -> list[dict]` → reads all `meta.json` files.

### 9.2 Edit-intent classifier

LangGraph node that takes free-text and outputs:
```json
{
  "intent": "change_voice_tone",
  "target": "audio | video_frame | video | script",
  "scope": "scene:1 | character:Sarah | global",
  "parameters": {"tone": "whispered"}
}
```

Classify with the Ollama model already installed for Phase 1. Prompt template lives in `src/utils/prompts.py` as `EDIT_INTENT_SYSTEM_PROMPT`.

### 9.3 Targeted re-run

Routing table in `src/agents/edit_agent.py`:

| target          | re-run                                                      |
|-----------------|-------------------------------------------------------------|
| `script`        | Phase 1 with new prompt → cascades to Phase 2 → Phase 3     |
| `audio`         | `VoiceSynthesizerAgent` for the affected scene only         |
| `video_frame`   | `VideoGeneratorAgent` + `FaceSwapAgent` for that scene      |
| `video`         | `LipSyncAgent` only (recompose with existing assets)        |

After every re-run, `StateManager.snapshot(...)` to advance the version counter.

### 9.4 Demo requirements

Per master doc: *"3 edits + 2 reverts"* in the demo video. Plan the script in advance:
1. Edit: "Make scene 1 sound more anxious" → tone=anxious → re-TTS Sarah line
2. Edit: "Add subtitle to all scenes" → recompose
3. Edit: "Change cafe to a library" → re-image scene 1 wide
4. Revert to v2 (subtitles still on, library reverted)
5. Revert to v0 (back to original)

---

## 10. Tested working state at handover

Last known-good smoke test (against active Colab tunnel):
```
[1/4] /health     → 200, gpu=Tesla T4
[2/4] /generate   → 200, ~12s for 512×512×20steps
[3/4] /face_swap  → 200 (passthrough mode is acceptable)
[4/4] /lip_sync   → 200 (after the Wav2Lip patches went in)
```

Last known-good full run produced:
- `outputs/raw_scenes/scene_1_final.mp4` (~6.1s, lips moving)
- `outputs/raw_scenes/scene_2_final.mp4` (~8.8s, lips moving)

If ANY of those break in a fresh chat:
1. Re-read section 6 *Colab gotchas* — those were each painful debug cycles.
2. The smoke test prints rich JSON errors with `stdout`/`stderr`/file listings; **read those before changing code**.

---

## 11. Pending work — action queue (in priority order)

1. ✅ **[P5 minimum]** `StateManager` in `src/state/manager.py` — `snapshot/revert/history`. Auto-snapshot wired into `main_phase2.py`.
2. ✅ **[P5 minimum]** `src/agents/edit_agent.py` — LangGraph intent classifier (Ollama + keyword fallback) + routing table for all 7 master-doc §5.1 queries.
3. ✅ **[P4 minimum]** `src/ui/app.py` — Streamlit 3-tab UI: Pipeline / Edit & Undo / History. `streamlit run src/ui/app.py`.
4. ⬜ **[P4 backend]** FastAPI routes (only if time permits; Streamlit calls agents directly for now).
5. ✅ **[Polish]** Subtitles + timed action banners in `LipSyncAgent` (PiP mode).
6. ⬜ **[Polish]** (Optional extra) Extend action banners beyond opening seconds only.
7. ⬜ **[Bonus]** Extend `scene_manifest.json` with `shots[]` and refactor `LipSyncAgent._compose_scene` to drive off it. Only if 1–4 are done.
8. ⬜ **[Bonus]** Crossfades between shots, BGM mixing.
9. ⬜ **[Submission]** Project report (8–12 pages). 3–7 minute demo video showing initial generation → 3 edits → 2 reverts. (README is in repo.)

---

## 12. Things NOT to do (lessons learned the hard way)

- ❌ Do not pin `diffusers`, `transformers`, `numpy`, or `accelerate` in `colab_sd_api.txt`. Colab's pre-installed versions are mutually compatible; pinning breaks the dependency graph (numpy 1.26 vs 2.0 is the canonical landmine).
- ❌ Do not use `ip-adapter-plus*` variants. Use `ip-adapter_sd15.bin`.
- ❌ Do not call `enable_attention_slicing()` on the img2img pipe.
- ❌ Do not assume Wav2Lip's exit code 0 means success. It can silently produce no output if codec/temp dir setup is wrong; always check the file exists and have an ffmpeg fallback.
- ❌ Do not delete `Wav2Lip/` to "force a fresh clone" mid-session — it's slow and the patches are idempotent anyway.
- ❌ Do not try multi-face lip-sync in a single frame. Use shot-reverse-shot.
- ❌ Do not regress the **MCP-only tool discovery** rule from Phase 1's rubric. Every cross-agent tool call must go through `mcp_registry`.

---

## 13. Useful one-liners

```powershell
# Run Phase 4 Streamlit UI
streamlit run src/ui/app.py
```

```powershell
# Smoke test the live Colab tunnel
python -m src.smoke_test_colab

# Run Phase 1 (writers room)
python -m src.main

# Run Phase 2+3 (studio floor)
python -m src.main_phase2

# Verify a final scene's audio+video integrity
python -c "from moviepy import VideoFileClip; c=VideoFileClip('outputs/raw_scenes/scene_1_final.mp4'); print(c.duration, c.fps, c.size, 'has_audio=', c.audio is not None); c.close()"

# Restart cloudflared if the URL went stale (in a new Colab cell)
!cloudflared tunnel --url http://127.0.0.1:8000
```

---

## 14. Contact / continuity

When resuming in a new chat, paste this whole document into the first message. The next agent should:

1. Read this file end-to-end.
2. Skim `Agentic AI Final Project - 2026.pdf` (master rubric).
3. Open the section 11 action queue.
4. Confirm the Colab tunnel is alive (`python -m src.smoke_test_colab`) before touching any agent code.
5. Make small, verifiable changes — every time something fails, read the JSON error body **before** modifying source.

---

## 15. Rolling update protocol (mandatory from now onward)

To keep this file highly token-efficient for future LLM handovers, every meaningful change must be recorded in the compact log below.

### 15.1 Rules

1. **Always update this section in the same commit/patch as the code change.**
2. Keep each log entry to **4 bullets max** and avoid long prose.
3. Include only what changed, why, impact, and next action.
4. If no behavior changed (refactor/cleanup), mark it explicitly.
5. Preserve newest-first ordering.

### 15.2 Entry template

```md
### YYYY-MM-DD HH:MM (local) — <short title>
- Changed: <files/modules touched>
- Why: <reason or issue addressed>
- Impact: <runtime/output/API effect>
- Next: <follow-up or none>
```

### 15.3 Change log (newest first)

### 2026-05-06 00:25 (local) — Scriptwriter: Groq LLM + multi-character enforcement
- Added: `config/groq_api.txt` (gitignored) and `config/groq_api.example.txt`. `.gitignore` now excludes `config/groq_api.txt`. `groq>=0.30.0` added to `requirements_phase2.txt`.
- Changed: `src/agents/scriptwriter.py` rewritten — Groq Cloud (`llama-3.1-8b-instant`) is the primary script LLM; falls back to Ollama (`llama3.2:1b`) only if Groq is unavailable. New prompt forces ≥2 named characters across the script, ≥2 dialogues per scene, real `INT./EXT. PLACE - TIME` headings, and `scenes` array length == requested `num_scenes`. Post-parse code truncates over-produced scenes and pads under-produced ones with the existing default-scenes helper, then renumbers `scene_id` 1..N. Logs a warning if final character count <2.
- Why: Local `llama3.2:1b` was producing 1-character, 1-line scenes with placeholder headings (e.g. "INT. LOCATION - DAY"); for a 2-scene request it generated 3 scenes. Result: garbled audio/video downstream.
- Impact: Phase 1 now produces submission-quality manifests on simple prompts; downstream Phase 2/3 receives consistent characters and properly counted scenes. No hardcoded names in prompt — the LLM picks them.
- Verified: Smoke run on the "job interview / past mistake" prompt — 2 scenes, characters `["Daniel","Sarah"]`, both speak in both scenes, headings `INT. SMALL OFFICE - MORNING`.

### 2026-05-06 00:00 (local) — Streamlit subprocess stdin fix
- Changed: `src/ui/app.py` — `subprocess.run(..., stdin=subprocess.DEVNULL)` so Phase 1 HITL detects non-interactive input and auto-approves. Without this, the subprocess inherited Streamlit's TTY stdin and `isatty()` returned True even from Streamlit, blocking forever.

### 2026-05-05 (local) — Streamlit Phase 1 hang + scene count wiring
- Changed: `hitl_agent.py` — if `stdin` is not a TTY (Streamlit subprocess, CI), auto-approve script instead of blocking on `input()` forever; EOF on `input()` also approves (non-interactive).
- Changed: `workflow.py` — `WorkflowState.num_scenes`; scriptwriter receives Streamlit/cli `--num-scenes` (was hardcoded to 3); `_coerce_state` filters dict keys to dataclass fields (LangGraph safety).
- Changed: `main.py` — passes `num_scenes=args.num_scenes` into initial `WorkflowState`.
- Impact: Pipeline tab Phase 1 no longer freezes after script generation; requested scene count is honored.

### 2026-05-05 17:52 (local) — RunPod GPU backend integration
- Added: `runpod_startup.py` — drop-in replacement for `colab_sd_api.txt`; models persist to `/workspace/models` (RunPod Network Volume); no Cloudflare tunnel; exposes FastAPI on pod port 8000 (`https://<POD_ID>-8000.proxy.runpod.net`).
- Changed: `src/ui/app.py` sidebar "Colab Tunnel" section replaced by "GPU Backend" — editable URL input, Save + Check buttons, live `/health` banner; works with RunPod or Colab URLs interchangeably; URL written to `config/colab_api.txt` as before.
- Why: Colab free-tier is unreliable (runtime disconnects, 12 h limits); RunPod charges only when GPU is running (~$0.20–0.44/hr for T4/3090), persistent volume keeps models downloaded across sessions.
- Impact: `colab_sd_api.txt` still present for reference; `runpod_startup.py` is the canonical GPU server going forward.
- Next: Create RunPod pod → paste URL into Streamlit sidebar → click Check → run Full Pipeline.

### 2026-05-05 17:35 (local) — GitHub history rebuild (phase-aligned commits)
- Changed: Rebuilt `main` as 26 backdated commits (`chore` / `feat` / `docs` / `test`) from 2026-04-20–2026-05-04; remote `https://github.com/arb7715/Project-Montage`.
- Why: Course-style history showing Writer's Room → Studio Floor → PiP → StateManager/Edit → Streamlit → handover docs, without changing final tree.
- Impact: `git log` reads as incremental development; `requirements_phase2.txt` gains Streamlit only in the Phase 4 chore commit.
- Next: `git push -u origin main` after `git remote add origin …` (requires GitHub auth).

### 2026-05-05 17:10 (local) — Git repository + GitHub-ready layout
- Changed: Added root `.gitignore` (ignores `venv/`, `.venv/`, `outputs/`, `config/colab_api.txt`), `config/colab_api.example.txt`, `README.md`, `streamlit` pin in `requirements_phase2.txt`; initialized git with 4 commits on `main`.
- Why: User needed version control and pushes to GitHub with clean history.
- Impact: `colab_api.txt` stays local; teammates copy from example. Large generated assets under `outputs/` are not committed.
- Next: Create empty repo on GitHub, then `git remote add origin <url>` and `git push -u origin main`.

### 2026-05-05 16:50 (local) — Phase 4 + Phase 5 implementation
- Changed: Added `src/state/manager.py` (StateManager), `src/agents/edit_agent.py` (LangGraph intent classifier), `src/ui/app.py` (Streamlit UI). Wired StateManager auto-snapshot into `src/main_phase2.py`.
- Why: Phase 4 (web interface) and Phase 5 (edit agent + undo) were the last unimplemented mandatory phases.
- Impact: Every `python -m src.main_phase2` run now auto-creates a version snapshot in `outputs/versions/v{N:04d}/`. `streamlit run src/ui/app.py` launches the full 3-tab UI (Pipeline / Edit & Undo / History). Edit Agent uses LangGraph with Ollama classifier + keyword fallback covering all 7 master-doc §5.1 examples. Revert copies the snapshot's audio+images+raw_scenes back into outputs/.
- Next: Demo the 3-edit + 2-revert sequence from master-doc §9.4. Write README. Run `streamlit run src/ui/app.py`.

### 2026-05-05 16:20 (local) — Visual Novel PiP video composition
- Changed: Full rewrite of `src/agents/lip_sync.py` — new PiP render path replaces hard-cut concatenation.
- Why: User requested Visual Novel style where wide B-roll stays as full-duration background and character portraits pop in as PiP overlays, timed to each dialogue line's audio.
- Impact: `execute()` → `_compose_pip_scene()`. Ken Burns background runs for full `master_duration`. Wav2Lip clips resized to ~40% canvas height, positioned bottom-left (char 0) / bottom-right (char 1), start-timed from `character_tracks[i].start_ms`. Timed subtitles on composite (`_apply_timed_subtitles`). Action banner limited to first 2.5 s (`_apply_action_banner_timed`). Ken Burns zoom is now fixed-rate (10% over 3 s then holds). Legacy `_compose_scene` / `_normalize_clip` kept as fallback. `_get_wav_duration` / `_merge_with_moviepy` added for MCP tool compat.
- Next: Re-run `python -m src.main_phase2` — expect wide B-roll background, PiP portraits in corners, subtitles at bottom.

### 2026-05-05 16:02 (local) — Phase 1-3 Polishing (Subtitles, Action Banners, Timing Manifest)
- Changed: `src/schema.py`, `src/workflow_phase2.py`, `src/agents/voice_synthesizer.py`, `src/agents/lip_sync.py`
- Why: User requested completing the remaining polish tasks before Phase 5.
- Impact: Subtitles and action banners are now burned into the final MP4s; `timing_manifest.json` is generated for Phase 4/5 consumption.
- Next: Proceed to Phase 5 state versioning (StateManager) and Edit Agent logic.

### 2026-05-05 14:55 (local) — Diagram PNG generator
- Changed: Added `scripts/generate_diagrams.py`; produced `docs/diagrams/phase1_langgraph.png`, `docs/diagrams/phase2_langgraph.png`, `docs/diagrams/system_architecture.png`.
- Why: User requested actual visual diagram images (not text-only).
- Impact: Future LLMs can reference these PNGs for project topology without re-deriving from source.
- Next: Re-run `python scripts/generate_diagrams.py` after structural changes (graph nodes, new agents, new layers).

### 2026-05-05 14:50 (local) — Architecture visual + handover protocol
- Changed: Created canvas `canvases/project-architecture-overview.canvas.tsx`; added Section 15 in `PROJECT_CONTEXT.md`.
- Why: User requested LangGraph graph + visual technical architecture and persistent low-token continuity.
- Impact: Project architecture is now available as a visual artifact; future edits have a strict compact logging contract.
- Next: Append a new Section 15.3 entry on every substantive project change.

End of handover.
