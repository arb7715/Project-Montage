"""
Quick smoke test for the Colab backend.

Run this after restarting Colab with the updated colab_sd_api.txt and
pasting the new trycloudflare URL into config/colab_api.txt.

  python -m src.smoke_test_colab

Tests, in order:
  1.  /health           - server reachable?
  2.  /generate          - stable-diffusion working?
  3.  /face_swap         - IP-Adapter working?
  4.  /lip_sync (image)  - Wav2Lip working on a still portrait?

Each step prints status + writes a small artefact under outputs/_smoke/.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

OUT = Path("outputs/_smoke")
OUT.mkdir(parents=True, exist_ok=True)
HEADERS = {"ngrok-skip-browser-warning": "true"}


def colab_url() -> str:
    p = Path("config/colab_api.txt")
    if not p.exists():
        print("[X] config/colab_api.txt not found"); sys.exit(2)
    url = p.read_text(encoding="utf-8").strip()
    if not url:
        print("[X] config/colab_api.txt is empty"); sys.exit(2)
    return url


def test_health(url: str):
    print("\n[1/4] /health ...", end=" ")
    t = time.time()
    try:
        r = requests.get(f"{url}/health", headers=HEADERS, timeout=20)
        print(f"status={r.status_code} elapsed={time.time()-t:.1f}s body={r.text[:120]}")
        return r.status_code == 200
    except Exception as e:
        print(f"FAIL: {e}")
        return False


def test_generate(url: str):
    print("\n[2/4] /generate ...", end=" ")
    t = time.time()
    try:
        r = requests.post(
            f"{url}/generate",
            data={"prompt": "a small red apple on a wooden table, photo",
                  "width": 512, "height": 512, "steps": 20},
            headers=HEADERS, timeout=180,
        )
        if r.status_code == 200:
            (OUT / "test_generate.png").write_bytes(r.content)
            print(f"OK ({len(r.content)/1024:.0f}KB) -> {OUT/'test_generate.png'} ({time.time()-t:.1f}s)")
            return True
        print(f"status={r.status_code} body={r.text[:300]}")
        return False
    except Exception as e:
        print(f"FAIL: {e}"); return False


def test_face_swap(url: str):
    print("\n[3/4] /face_swap ...", end=" ")
    portrait = Path("outputs/images/sarah.png")
    base = Path("outputs/raw_scenes/scene_1_frames/scene_1_static.jpg")
    if not portrait.exists() or not base.exists():
        print(f"SKIP (need {portrait} and {base})"); return True
    t = time.time()
    try:
        with open(portrait, "rb") as p, open(base, "rb") as b:
            r = requests.post(
                f"{url}/face_swap",
                files={"swap_image": ("p.png", p, "image/png"),
                       "base_image": ("b.jpg", b, "image/jpeg")},
                data={"prompt": "INT. COFFEE SHOP - MORNING, cinematic, a person facing the camera",
                      "ip_scale": "0.85", "strength": "0.45", "steps": "25"},
                headers=HEADERS, timeout=300,
            )
        if r.status_code == 200:
            (OUT / "test_faceswap.png").write_bytes(r.content)
            print(f"OK ({len(r.content)/1024:.0f}KB) -> {OUT/'test_faceswap.png'} ({time.time()-t:.1f}s)")
            return True
        print(f"status={r.status_code}")
        try:
            ct = r.headers.get("content-type", "")
            body_preview = r.text[:2500] if "json" in ct else r.text[:800]
            print(body_preview)
        except Exception:
            print("(could not decode body)")
        return False
    except Exception as e:
        print(f"FAIL: {e}"); return False


def test_lip_sync(url: str):
    print("\n[4/4] /lip_sync ...", end=" ")
    portrait = Path("outputs/images/sarah.png")
    audio = Path("outputs/audio/scene_1_line_00.wav")
    if not portrait.exists() or not audio.exists():
        print(f"SKIP (need {portrait} and {audio})"); return True
    t = time.time()
    try:
        with open(portrait, "rb") as f, open(audio, "rb") as a:
            r = requests.post(
                f"{url}/lip_sync",
                files={"face": ("face.png", f, "image/png"),
                       "audio": ("audio.wav", a, "audio/wav")},
                headers=HEADERS, timeout=600,
            )
        if r.status_code == 200:
            (OUT / "test_lipsync.mp4").write_bytes(r.content)
            print(f"OK ({len(r.content)/1024:.0f}KB) -> {OUT/'test_lipsync.mp4'} ({time.time()-t:.1f}s)")
            return True
        print(f"status={r.status_code}")
        print(r.text[:3500])
        return False
    except Exception as e:
        print(f"FAIL: {e}"); return False


def main() -> int:
    url = colab_url()
    print(f"Testing Colab API at: {url}")
    ok = True
    ok &= test_health(url)
    ok &= test_generate(url)
    ok &= test_face_swap(url)
    ok &= test_lip_sync(url)
    print("\n" + ("All checks passed" if ok else "Some checks failed - see above"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
