"""
Generate technical diagram PNGs for Project Montage:
  - docs/diagrams/phase1_langgraph.png
  - docs/diagrams/phase2_langgraph.png
  - docs/diagrams/system_architecture.png

Run: python scripts/generate_diagrams.py
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = Path("docs/diagrams")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----- Theme -----
BG = "#0f1115"
FG = "#e6e6e6"
MUTED = "#9aa3af"
ACCENT = "#7aa2f7"
OK = "#9ece6a"
WARN = "#e0af68"
ERR = "#f7768e"
NODE = "#1a1d24"
EDGE = "#5c6370"
STROKE = "#2b2f38"


def style_axes(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def node(ax, x, y, w, h, text, color=NODE, text_color=FG, fontsize=10, bold=False):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        linewidth=1.4,
        edgecolor=STROKE,
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        color=text_color,
        fontsize=fontsize,
        fontweight="bold" if bold else "normal",
        family="DejaVu Sans",
    )


def arrow(ax, x1, y1, x2, y2, label=None, color=EDGE, style="-|>", curve=0.0, label_color=MUTED):
    a = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle=style,
        mutation_scale=14,
        linewidth=1.3,
        color=color,
        connectionstyle=f"arc3,rad={curve}",
    )
    ax.add_patch(a)
    if label:
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        ax.text(
            mx,
            my,
            label,
            color=label_color,
            fontsize=8,
            ha="center",
            va="center",
            bbox=dict(facecolor=BG, edgecolor="none", pad=1.5),
        )


def title(ax, text, subtitle=None):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.text(0.2, 9.6, text, color=FG, fontsize=15, fontweight="bold")
    if subtitle:
        ax.text(0.2, 9.25, subtitle, color=MUTED, fontsize=10)


# ---------------------------------------------------------------------------
# Phase 1 LangGraph
# ---------------------------------------------------------------------------
def draw_phase1():
    fig, ax = plt.subplots(figsize=(12, 8), dpi=160)
    fig.patch.set_facecolor(BG)
    style_axes(ax)
    title(
        ax,
        "Phase 1 LangGraph - The Writer's Room",
        "Mode-aware multi-agent workflow with HITL approval loop",
    )

    W, H = 1.9, 0.7

    node(ax, 5.0, 8.4, W, H, "mode_selector", color=ACCENT, text_color="#0b1020", bold=True)

    node(ax, 2.2, 7.2, W, H, "validator", color=NODE)
    node(ax, 7.8, 7.2, W, H, "scriptwriter", color=NODE)

    node(ax, 5.0, 5.9, W, H, "hitl", color=WARN, text_color="#0b1020", bold=True)

    node(ax, 5.0, 4.5, W + 0.3, H, "character_designer", color=NODE)
    node(ax, 5.0, 3.3, W + 0.3, H, "image_synthesizer", color=NODE)
    node(ax, 5.0, 2.1, W + 0.3, H, "memory_commit", color=NODE)
    node(ax, 5.0, 0.95, W, H, "finalize", color=OK, text_color="#0b1020", bold=True)
    node(ax, 8.6, 0.95, 1.1, 0.55, "END", color="#2b2f38", text_color=FG, fontsize=9)

    arrow(ax, 4.2, 8.2, 2.6, 7.55, label="manual", curve=-0.15)
    arrow(ax, 5.8, 8.2, 7.4, 7.55, label="autonomous", curve=0.15)

    arrow(ax, 2.2, 6.85, 4.2, 6.1, label="pass", curve=0.1)
    arrow(ax, 2.0, 6.85, 1.4, 1.2, label="fail", color=ERR, curve=-0.4)
    arrow(ax, 1.4, 1.2, 4.2, 0.95, color=ERR, curve=0.0)

    arrow(ax, 7.8, 6.85, 5.8, 6.1, curve=-0.1)

    arrow(ax, 5.0, 5.55, 5.0, 4.85, label="approve / max revisions")
    arrow(ax, 4.2, 5.9, 7.0, 7.05, label="revise (loop)", color=WARN, curve=0.35)
    arrow(ax, 5.8, 5.9, 6.4, 1.2, label="reject", color=ERR, curve=0.4)
    arrow(ax, 6.4, 1.2, 5.85, 0.95, color=ERR)

    arrow(ax, 5.0, 4.15, 5.0, 3.65)
    arrow(ax, 5.0, 2.95, 5.0, 2.45)
    arrow(ax, 5.0, 1.75, 5.0, 1.3)
    arrow(ax, 5.95, 0.95, 8.05, 0.95)

    legend = [
        ("Entry / decision", ACCENT),
        ("Human-in-the-loop", WARN),
        ("Terminal", OK),
        ("Reject path", ERR),
    ]
    for i, (lbl, col) in enumerate(legend):
        ax.add_patch(
            mp.Rectangle((0.3, 0.6 - i * 0.35), 0.25, 0.18, color=col, ec=STROKE)
        )
        ax.text(0.62, 0.69 - i * 0.35, lbl, color=MUTED, fontsize=8, va="center")

    ax.text(
        9.8,
        0.2,
        "Source: src/workflow.py",
        color=MUTED,
        fontsize=8,
        ha="right",
    )

    out = OUT_DIR / "phase1_langgraph.png"
    fig.tight_layout()
    fig.savefig(out, facecolor=BG, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


# ---------------------------------------------------------------------------
# Phase 2/3 LangGraph
# ---------------------------------------------------------------------------
def draw_phase2():
    fig, ax = plt.subplots(figsize=(13, 6), dpi=160)
    fig.patch.set_facecolor(BG)
    style_axes(ax)
    title(
        ax,
        "Phase 2 / 3 LangGraph - The Studio Floor",
        "Sequential pipeline (compiled) - audio duration -> video composition",
    )

    W, H = 1.65, 0.7
    y = 5.5
    nodes = [
        (1.2, "scene_parser", ACCENT, "#0b1020", True),
        (3.1, "voice_synth", NODE, FG, False),
        (5.0, "video_gen", NODE, FG, False),
        (6.9, "face_swap", NODE, FG, False),
        (8.8, "lip_sync", NODE, FG, False),
        (10.7, "finalize", OK, "#0b1020", True),
    ]
    for x, t, c, tc, b in nodes:
        node(ax, x, y, W, H, t, color=c, text_color=tc, bold=b)

    for i in range(len(nodes) - 1):
        arrow(ax, nodes[i][0] + W / 2 + 0.02, y, nodes[i + 1][0] - W / 2 - 0.02, y)

    arrow(ax, 10.7 + W / 2 + 0.02, y, 11.7, y)
    node(ax, 12.0, y, 0.7, 0.45, "END", color="#2b2f38", text_color=FG, fontsize=9)

    captions = [
        (1.2, "Manifest -> tasks", "scene_tasks[]"),
        (3.1, "edge-tts per line", "audio_tracks[]"),
        (5.0, "SD wide B-roll", "video_frames[]"),
        (6.9, "IP-Adapter blend", "face_swapped_videos[]"),
        (8.8, "Wav2Lip closeups", "synced_scenes[]"),
        (10.7, "Logs + summary", "execution_log"),
    ]
    for x, top, bot in captions:
        ax.text(x, y - 0.85, top, color=FG, fontsize=8.5, ha="center")
        ax.text(x, y - 1.2, bot, color=MUTED, fontsize=7.5, ha="center", style="italic")

    note = (
        "Note: The graph design supports Send() fan-out per scene (audio + video branches),\n"
        "but the *compiled* pipeline runs sequentially so video_gen knows audio durations."
    )
    ax.text(0.4, 2.4, note, color=MUTED, fontsize=9, va="top")

    ax.text(
        12.7,
        0.4,
        "Source: src/workflow_phase2.py",
        color=MUTED,
        fontsize=8,
        ha="right",
    )

    ax.set_xlim(0, 13)
    ax.set_ylim(0, 10)
    out = OUT_DIR / "phase2_langgraph.png"
    fig.tight_layout()
    fig.savefig(out, facecolor=BG, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


# ---------------------------------------------------------------------------
# System architecture
# ---------------------------------------------------------------------------
def draw_architecture():
    fig, ax = plt.subplots(figsize=(14, 11), dpi=160)
    fig.patch.set_facecolor(BG)
    style_axes(ax)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 14)

    ax.text(
        0.3,
        13.4,
        "Project Montage - System Architecture",
        color=FG,
        fontsize=16,
        fontweight="bold",
    )
    ax.text(
        0.3,
        13.0,
        "Local LangGraph orchestration + remote Colab GPU inference + filesystem state (Phase 4/5 future bridge)",
        color=MUTED,
        fontsize=10,
    )

    def band(x, y, w, h, label, label_color=MUTED):
        rect = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.0,rounding_size=0.12",
            linewidth=1.0,
            edgecolor=STROKE,
            facecolor="#13161c",
        )
        ax.add_patch(rect)
        ax.text(x + 0.18, y + h - 0.22, label, color=label_color, fontsize=9.5, fontweight="bold")

    def box(x, y, w, h, text, sub=None, color=NODE, text_color=FG, bold=True):
        b = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=1.2,
            edgecolor=STROKE,
            facecolor=color,
        )
        ax.add_patch(b)
        ax.text(
            x + w / 2,
            y + (h - 0.18) / 2 + (0.18 if sub else 0.0),
            text,
            color=text_color,
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold" if bold else "normal",
        )
        if sub:
            sub_color = "#1a1d24" if text_color == "#0b1020" else MUTED
            ax.text(
                x + w / 2,
                y + 0.22,
                sub,
                color=sub_color,
                ha="center",
                va="center",
                fontsize=7.8,
            )

    BAND_HEAD = 0.55

    y_top = 12.6
    h1 = 1.85
    band(0.3, y_top - h1, 13.4, h1, "1) Client / UI layer  (Phase 4 - planned)")
    box(0.7, y_top - h1 + 0.2, 3.0, 1.05, "Streamlit UI", "src/ui/app.py", color=ACCENT, text_color="#0b1020")
    box(4.0, y_top - h1 + 0.2, 3.2, 1.05, "FastAPI orchestrator", "src/api/* (planned)", color=ACCENT, text_color="#0b1020")
    box(7.5, y_top - h1 + 0.2, 3.0, 1.05, "WebSocket /ws", "live phase events", color="#1f2530")
    box(10.7, y_top - h1 + 0.2, 2.7, 1.05, "Edit + Undo (P5)", "intent + history", color=WARN, text_color="#0b1020")

    y_top = 10.55
    h2 = 1.95
    band(0.3, y_top - h2, 13.4, h2, "2) Orchestration  (Local Python)")
    box(0.7, y_top - h2 + 0.2, 4.0, 1.05, "LangGraph: Writers Room", "workflow.py (Phase 1)", color="#1c2230")
    box(5.0, y_top - h2 + 0.2, 4.0, 1.05, "LangGraph: Studio Floor", "workflow_phase2.py (P2/3)", color="#1c2230")
    box(9.3, y_top - h2 + 0.2, 4.1, 1.05, "MCP Tool Registry", "src/utils/mcp_registry.py", color="#1c2230")
    ax.text(
        0.5,
        y_top - h2 + 0.05,
        "All cross-agent calls go through MCP tool discovery (rubric constraint).",
        color=MUTED,
        fontsize=8.5,
    )

    y_top = 8.45
    h3 = 2.0
    band(0.3, y_top - h3, 13.4, h3, "3) Agents  (Local Python)")
    agents = [
        "scriptwriter",
        "validator",
        "hitl",
        "character_designer",
        "image_synthesizer",
        "scene_parser",
        "voice_synth",
        "video_gen",
        "face_swap",
        "lip_sync",
    ]
    cols = 5
    cw, ch = 2.55, 0.55
    x0 = 0.5
    y_first_row = y_top - h3 + 1.05
    for i, a in enumerate(agents):
        r, c = divmod(i, cols)
        bx = x0 + c * (cw + 0.05)
        by = y_first_row - r * (ch + 0.1)
        box(bx, by, cw, ch, a, color="#1a1d24", bold=False)

    y_top = 6.3
    h4 = 2.1
    band(0.3, y_top - h4, 6.5, h4, "4) GPU inference  (Colab + Cloudflare tunnel)")
    box(0.7, y_top - h4 + 1.15, 6.0, 0.7, "FastAPI on Colab T4", "endpoints below", color="#1f2530")
    box(0.7, y_top - h4 + 0.4, 1.85, 0.55, "/health", color="#1a1d24", bold=False)
    box(2.7, y_top - h4 + 0.4, 1.85, 0.55, "/generate (SD1.5)", color="#1a1d24", bold=False)
    box(4.7, y_top - h4 + 0.4, 2.0, 0.55, "/face_swap (IP-Adapter)", color="#1a1d24", bold=False)
    ax.text(
        0.5,
        y_top - h4 + 0.05,
        "URL stored in config/colab_api.txt; /lip_sync via Wav2Lip patches",
        color=MUTED,
        fontsize=8,
    )

    band(7.0, y_top - h4, 6.7, h4, "5) Persistence  (Local filesystem)")
    box(7.4, y_top - h4 + 1.15, 6.0, 0.7, "outputs/", "JSON manifests + audio + video + memory", color="#1f2530")
    artifacts = [
        ("scene_manifest.json", 7.4, y_top - h4 + 0.4),
        ("character_db.json", 9.3, y_top - h4 + 0.4),
        ("raw_scenes/*.mp4", 11.2, y_top - h4 + 0.4),
    ]
    for name, x, y in artifacts:
        box(x, y, 1.85, 0.55, name, color="#1a1d24", bold=False)

    y_top = 4.0
    h5 = 1.55
    band(0.3, y_top - h5, 13.4, h5, "6) Phase 5 versioning  (planned)")
    box(0.7, y_top - h5 + 0.2, 4.0, 0.85, "StateManager", "snapshot / revert / history", color=WARN, text_color="#0b1020")
    box(5.0, y_top - h5 + 0.2, 4.0, 0.85, "Edit Agent (LLM intent)", "src/agents/edit_agent.py", color=WARN, text_color="#0b1020")
    box(9.3, y_top - h5 + 0.2, 4.1, 0.85, "outputs/versions/v0001 ...", "snapshot dirs + meta.json", color="#1f2530")

    def link(x1, y1, x2, y2, color=EDGE, curve=0.0, style="->"):
        a = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle=style,
            mutation_scale=12,
            linewidth=1.1,
            color=color,
            connectionstyle=f"arc3,rad={curve}",
        )
        ax.add_patch(a)

    link(2.2, 10.95, 2.7, 9.85)
    link(5.6, 10.95, 5.6, 9.85)
    link(2.7, 8.8, 2.7, 8.05)
    link(7.0, 8.8, 7.0, 8.05)
    link(3.7, 6.85, 3.7, 6.05, color=ACCENT, curve=0.0)
    link(10.0, 6.85, 10.4, 6.05, color=OK, curve=0.0)
    link(12.05, 10.95, 12.0, 3.5, color=WARN, curve=0.35, style="-|>")
    ax.text(
        13.7,
        2.2,
        "PROJECT_CONTEXT.md is the source of truth for state and conventions",
        color=MUTED,
        fontsize=8,
        ha="right",
    )

    out = OUT_DIR / "system_architecture.png"
    fig.tight_layout()
    fig.savefig(out, facecolor=BG, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


if __name__ == "__main__":
    draw_phase1()
    draw_phase2()
    draw_architecture()
    print(f"\nAll diagrams written to: {OUT_DIR.resolve()}")
