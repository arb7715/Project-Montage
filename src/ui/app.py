"""
Phase 4 — Streamlit Web Interface
Run with:  streamlit run src/ui/app.py

Tabs
----
  1. Pipeline  — prompt input, run Phase 1 / Phase 2/3 / Full pipeline,
                 live log display, final video player.
  2. Edit      — free-text edit requests, intent display, updated video preview.
  3. History   — version list with revert buttons, per-version video previews.

Architecture note
-----------------
Streamlit calls the pipeline as a subprocess so the UI stays responsive
while the (slow) SD / Wav2Lip steps run.  Progress is tracked by polling
output files that the pipeline writes incrementally.

GPU backend URL is configured in the sidebar (saved to config/colab_api.txt).
Works with RunPod, Colab tunnels, or any FastAPI endpoint that implements
the /health, /generate, /face_swap, and /lip_sync routes.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import List, Optional

import streamlit as st

# ── Ensure repo root is importable ──────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.state.manager import StateManager
from src.agents.edit_agent import EditAgent

# ── Paths ────────────────────────────────────────────────────────────────────
OUTPUTS      = _REPO_ROOT / "outputs"
SCENES_DIR   = OUTPUTS / "raw_scenes"
AUDIO_DIR    = OUTPUTS / "audio"
IMAGES_DIR   = OUTPUTS / "images"
CONFIG_DIR   = _REPO_ROOT / "config"

_mgr = StateManager(str(OUTPUTS))


# ── Subprocess helpers ────────────────────────────────────────────────────────

def _run_phase(phase: int, extra_args: Optional[List[str]] = None
               ) -> tuple[bool, str]:
    """Run src.main (phase=1) or src.main_phase2 (phase=2) as subprocess."""
    module = "src.main" if phase == 1 else "src.main_phase2"
    cmd = [sys.executable, "-m", module] + (extra_args or [])
    env = {**os.environ, "PYTHONPATH": str(_REPO_ROOT)}
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True,
            stdin=subprocess.DEVNULL,   # detach stdin so HITL auto-approves
            timeout=900, cwd=str(_REPO_ROOT), env=env,
        )
        return r.returncode == 0, (r.stdout + "\n" + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return False, "Timed out after 15 minutes."
    except Exception as exc:
        return False, str(exc)


def _get_videos() -> List[Path]:
    if not SCENES_DIR.exists():
        return []
    return sorted(SCENES_DIR.glob("scene_*_final.mp4"))


def _gpu_url() -> str:
    p = CONFIG_DIR / "colab_api.txt"
    return p.read_text().strip() if p.exists() else ""


def _save_gpu_url(url: str) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    (CONFIG_DIR / "colab_api.txt").write_text(url.strip(), encoding="utf-8")


def _check_health(url: str) -> tuple[bool, str]:
    """Return (ok, message) from the /health endpoint."""
    base = (url or "").strip()
    if base.endswith("/health"):
        base = base[:-7]
    health_url = base.rstrip("/") + "/health"
    try:
        req_obj = urllib.request.Request(
            health_url,
            headers={
                # RunPod/Cloudflare can reject header-less urllib requests.
                "User-Agent": "Mozilla/5.0 (ProjectMontage/1.0)",
                "Accept": "application/json",
            },
        )
        req = urllib.request.urlopen(req_obj, timeout=8)
        import json
        data = json.loads(req.read().decode())
        gpu  = data.get("gpu", "unknown GPU")
        return True, f"Online · {gpu}"
    except Exception as exc:
        return False, f"{exc} ({health_url})"[:180]


# Alias kept for backward compatibility with pipeline agents that read this file
_colab_url = _gpu_url


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Project Montage",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Session state defaults ────────────────────────────────────────────────────
_DEFAULTS = {
    "current_version":   None,
    "pipeline_log":      "",
    "last_edit_result":  None,
    "edit_prefill":      "",
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Project Montage")
    st.caption("AI-Powered Animated Video Generation")
    st.divider()

    videos = _get_videos()
    history = _mgr.history()
    cur_ver = st.session_state.current_version

    c1, c2 = st.columns(2)
    c1.metric("Scenes", len(videos))
    c2.metric("Versions", len(history))

    if cur_ver:
        st.success(f"Active: v{cur_ver:04d}")
    else:
        st.info("No active version yet")

    st.divider()
    st.subheader("GPU Backend")

    saved_url = _gpu_url()
    new_url = st.text_input(
        "API URL",
        value=saved_url,
        placeholder="https://<pod-id>-8000.proxy.runpod.net",
        help=(
            "RunPod: https://<POD_ID>-8000.proxy.runpod.net\n"
            "Colab:  https://<random>.trycloudflare.com"
        ),
    )

    col_save, col_check = st.columns(2)
    save_clicked  = col_save.button("Save", use_container_width=True)
    check_clicked = col_check.button("Check", use_container_width=True)

    if save_clicked and new_url.strip():
        _save_gpu_url(new_url.strip())
        st.success("Saved.")

    if check_clicked:
        url_to_check = new_url.strip() or saved_url
        if url_to_check:
            ok, msg = _check_health(url_to_check)
            if ok:
                st.success(f"✅ {msg}")
            else:
                st.error(f"❌ {msg}")
        else:
            st.warning("Enter a URL first.")

    # Persistent status indicator (shown every render)
    current_url = new_url.strip() or saved_url
    if current_url:
        if current_url.startswith("https://") or current_url.startswith("http://"):
            st.caption(f"Configured: `{current_url[:55]}…`" if len(current_url) > 55
                       else f"Configured: `{current_url}`")
        else:
            st.warning("URL looks invalid — should start with https://")
    else:
        st.warning("Not configured — paste your RunPod URL above.")

    st.divider()
    st.caption("CS-4015 Agentic AI · FAST NUCES")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_pipeline, tab_edit, tab_history = st.tabs(
    ["🎬  Pipeline", "✏️  Edit & Undo", "🕐  History"]
)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_pipeline:
    st.header("Generate Your Video")

    # ── Input controls ─────────────────────────────────────────────────────
    col_mode, col_scenes = st.columns([2, 1])
    mode = col_mode.selectbox(
        "Generation mode", ["autonomous", "manual"],
        help="Autonomous: LLM writes the script. Manual: paste your screenplay.",
    )
    num_scenes = col_scenes.number_input("Scenes", min_value=1, max_value=10,
                                         value=2, step=1)

    if mode == "autonomous":
        prompt = st.text_area(
            "Creative prompt",
            value="A mysterious stranger arrives in a quiet town, bringing secrets that will change everything.",
            height=90,
        )
        script_text = None
    else:
        prompt = None
        script_text = st.text_area(
            "Paste screenplay  (INT./EXT. headings, CHARACTER NAME, dialogue)",
            height=200,
            placeholder="INT. COFFEE SHOP - MORNING\n\nSARAH\nI hope he shows up this time.\n\nWAITER\nWould you like to order something?",
        )

    # ── Action buttons ──────────────────────────────────────────────────────
    bc1, bc2, bc3 = st.columns(3)
    run_p1  = bc1.button("Phase 1 — Script & Characters", use_container_width=True)
    run_p2  = bc2.button("Phase 2/3 — Audio & Video",     use_container_width=True)
    run_all = bc3.button("Full Pipeline",  type="primary", use_container_width=True)

    log_box = st.empty()
    pbar    = st.empty()

    # ── Phase 1 ─────────────────────────────────────────────────────────────
    if run_p1 or run_all:
        pbar.progress(0.05, "Phase 1: generating script…")
        extra: List[str] = []
        if mode == "autonomous":
            extra = ["--prompt", prompt or "", "--num-scenes", str(num_scenes)]
        else:
            if script_text:
                tmp = CONFIG_DIR / "_ui_script.txt"
                tmp.write_text(script_text, encoding="utf-8")
                extra = ["--mode", "manual", "--script-file", str(tmp)]
            else:
                st.warning("Please paste a screenplay first.")
                st.stop()

        with st.spinner("Running Phase 1 — Writer's Room…"):
            ok, log = _run_phase(1, extra)

        st.session_state.pipeline_log = log
        log_box.text_area("Phase 1 log", log, height=220)
        pbar.progress(0.45 if run_all else 1.0, "Phase 1 done" if ok else "Phase 1 failed")

        if ok:
            st.success("Phase 1 complete — script and character portraits generated.")
            try:
                ver = _mgr.snapshot(
                    {"phase": 1, "mode": mode, "prompt": prompt},
                    change_summary="Phase 1 initial run",
                )
                st.session_state.current_version = ver
                st.info(f"Snapshot created: v{ver:04d}")
            except Exception:
                pass
        else:
            st.error("Phase 1 failed. Check the log.")
            if not run_all:
                st.stop()

    # ── Phase 2/3 ───────────────────────────────────────────────────────────
    if run_p2 or run_all:
        pbar.progress(0.50, "Phase 2/3: audio → SD → face-swap → Wav2Lip…")
        with st.spinner("Running Phase 2/3 — Studio Floor (this takes a few minutes)…"):
            ok2, log2 = _run_phase(2)

        st.session_state.pipeline_log += "\n\n" + log2
        log_box.text_area("Phase 2/3 log", log2, height=220)
        pbar.progress(1.0, "Done!" if ok2 else "Phase 2/3 failed")

        if ok2:
            st.success("Phase 2/3 complete — final MP4s rendered.")
            try:
                ver = _mgr.snapshot(
                    {"phase": 2},
                    change_summary="Phase 2/3 pipeline run",
                    parent=st.session_state.current_version,
                )
                st.session_state.current_version = ver
                st.info(f"Snapshot created: v{ver:04d}")
            except Exception:
                pass
        else:
            st.error("Phase 2/3 failed. Check the log.")

    # ── Video preview ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Final Scenes")
    videos = _get_videos()
    if videos:
        vcols = st.columns(min(len(videos), 2))
        for i, vp in enumerate(videos):
            with vcols[i % 2]:
                st.caption(f"Scene {i + 1} — {vp.name}")
                st.video(str(vp))
    else:
        st.info("No final videos yet — run the pipeline above.")

    # ── Character portraits ──────────────────────────────────────────────────
    if IMAGES_DIR.exists():
        imgs = sorted(IMAGES_DIR.glob("*.png")) + sorted(IMAGES_DIR.glob("*.jpg"))
        if imgs:
            st.divider()
            st.subheader("Character Portraits")
            icols = st.columns(min(len(imgs), 4))
            for i, ip in enumerate(imgs):
                with icols[i % 4]:
                    st.image(str(ip), caption=ip.stem.capitalize(), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDIT & UNDO
# ═══════════════════════════════════════════════════════════════════════════════
with tab_edit:
    st.header("Edit Your Video")
    st.caption(
        "Describe the change in plain English. "
        "The Edit Agent classifies your intent and re-runs only the necessary pipeline stage."
    )

    col_form, col_examples = st.columns([3, 2])

    with col_examples:
        st.subheader("Example edits")
        _examples = [
            "Make scene 1 sound more anxious",
            "Change the cafe to a library",
            "Add subtitles to all scenes",
            "Make Daniel sound older",
            "Change the weather to rainy in scene 2",
            "Make Sarah's dialogue more formal",
            "Speed up scene 2",
        ]
        for ex in _examples:
            if st.button(ex, key=f"ex_{hash(ex)}", use_container_width=True):
                st.session_state.edit_prefill = ex
                st.rerun()

    with col_form:
        edit_query = st.text_area(
            "What would you like to change?",
            value=st.session_state.edit_prefill,
            height=110,
            placeholder="e.g. Make scene 1 sound more anxious",
            key="edit_query_input",
        )

        apply_btn = st.button(
            "Apply Edit", type="primary", use_container_width=True,
            disabled=not edit_query.strip(),
        )

        if apply_btn and edit_query.strip():
            with st.spinner("Classifying intent and applying edit…"):
                agent  = EditAgent(outputs_dir=str(OUTPUTS))
                result = agent.execute({
                    "query":          edit_query,
                    "parent_version": st.session_state.current_version,
                })
            st.session_state.last_edit_result = result
            st.session_state.edit_prefill = ""
            if result.get("new_version"):
                st.session_state.current_version = result["new_version"]
            st.rerun()

    # ── Edit result display ─────────────────────────────────────────────────
    if st.session_state.last_edit_result:
        r      = st.session_state.last_edit_result
        intent = r.get("intent_result", {})

        st.divider()
        st.subheader("Edit Result")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Intent",      intent.get("intent",     "—"))
        m2.metric("Target",      intent.get("target",     "—"))
        m3.metric("Scope",       intent.get("scope",      "—"))
        m4.metric("New version", f"v{r.get('new_version', 0):04d}")

        conf = intent.get("confidence", 0)
        st.progress(conf, f"Confidence: {conf:.0%}")
        st.caption(f"Reasoning: {intent.get('reasoning', '—')}")

        if r.get("success"):
            st.success("Edit applied successfully.")
        else:
            st.error(f"Edit failed: {r.get('error', 'unknown error')}")

        with st.expander("Execution log"):
            for entry in r.get("execution_log", []):
                st.text(f"  • {entry}")

    # ── Updated video preview ────────────────────────────────────────────────
    st.divider()
    st.subheader("Current Scenes")
    videos = _get_videos()
    if videos:
        vcols = st.columns(min(len(videos), 2))
        for i, vp in enumerate(videos):
            with vcols[i % 2]:
                st.caption(vp.name)
                st.video(str(vp))
    else:
        st.info("No final videos available.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — HISTORY
# ═══════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.header("Version History")
    st.caption(
        "Every successful pipeline run and edit creates a snapshot. "
        "Click **Revert** to restore any previous state."
    )

    history = _mgr.history()
    cur_ver = st.session_state.current_version

    if not history:
        st.info("No snapshots yet — run the pipeline to create the first snapshot.")
    else:
        for snap in history:
            ver      = snap.get("version", 0)
            is_cur   = (ver == cur_ver)
            label    = f"v{ver:04d}"
            ts       = snap.get("timestamp", "")[:19].replace("T", " ")
            summary  = snap.get("change_summary", "—")
            eq       = snap.get("edit_query", "")
            parent   = snap.get("parent_version")
            mp4_cnt  = snap.get("mp4_count", 0)

            with st.container(border=True):
                h1, h2, h3 = st.columns([4, 3, 1])

                with h1:
                    badge = " ← **active**" if is_cur else ""
                    st.markdown(f"### {label}{badge}")
                    st.caption(summary)
                    if eq:
                        st.caption(f'Edit: *"{eq}"*')

                with h2:
                    st.caption(f"🕐  {ts}")
                    if parent:
                        st.caption(f"Parent: v{parent:04d}")
                    st.caption(f"🎬  {mp4_cnt} scene(s)")

                with h3:
                    if is_cur:
                        st.markdown("*(active)*")
                    else:
                        if st.button(
                            "Revert", key=f"rev_{ver}",
                            use_container_width=True, type="secondary"
                        ):
                            with st.spinner(f"Reverting to v{ver:04d}…"):
                                try:
                                    _mgr.revert(ver)
                                    st.session_state.current_version = ver
                                    st.success(f"Reverted to v{ver:04d}")
                                    time.sleep(0.5)
                                    st.rerun()
                                except Exception as exc:
                                    st.error(f"Revert failed: {exc}")

                # Show per-version video previews (collapsed by default)
                mp4s = snap.get("mp4_paths", [])
                if mp4s:
                    with st.expander(f"Preview {mp4_cnt} scene(s) from this version"):
                        pc = st.columns(min(len(mp4s), 2))
                        for i, p in enumerate(mp4s):
                            if Path(p).exists():
                                with pc[i % 2]:
                                    st.video(p)
