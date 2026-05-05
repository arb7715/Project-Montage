"""
RunPod Startup Script — Project Montage GPU Backend
====================================================
Paste this entire file into a RunPod Jupyter terminal or run it as:
    python runpod_startup.py

What it does
------------
1. Installs missing pip packages (not re-downloaded if pod is restarted).
2. Downloads SD-1.5, IP-Adapter, and Wav2Lip ONCE into /workspace/models/.
   On every subsequent start the files are already there → ~2 min warmup.
3. Loads all models into GPU VRAM.
4. Starts a FastAPI server on port 8000.
5. Prints the public URL you paste into the Streamlit sidebar (or config/colab_api.txt).

No Cloudflare tunnel needed — RunPod exposes port 8000 directly via:
    https://<POD_ID>-8000.proxy.runpod.net

RunPod setup required (one-time in the web UI)
-----------------------------------------------
- Template : "RunPod Pytorch 2.x" (comes with torch + CUDA pre-installed)
- GPU      : RTX 3090 / A4000 / T4 (all work; 3090/A4000 faster)
- Volume   : Attach a Network Volume mounted at /workspace  (10 GB is enough)
- Expose   : TCP port 8000  (tick the checkbox in "Edit Pod" → Expose Ports)
"""

import io, os, re, sys, shutil, subprocess, threading, time, traceback, uuid, wave
import urllib.request
from pathlib import Path

# ── pip installs (idempotent) ────────────────────────────────────────────────
_PKGS = [
    "fastapi", "uvicorn[standard]", "python-multipart",
    # Keep these pinned; newer transformers builds can break with some
    # preinstalled torch versions on hosted GPU images.
    "diffusers==0.27.2", "transformers==4.41.2",
    "accelerate==0.30.1", "safetensors==0.4.3",
    "huggingface_hub==0.25.2",
    "invisible_watermark",          # required by SD safety checker bypass
    "imageio[ffmpeg]", "librosa", "soundfile", "tqdm", "numba",
    "nest-asyncio",
]
subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + _PKGS, check=False)

import nest_asyncio
nest_asyncio.apply()

import numpy as np
from PIL import Image
import torch
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import Response, JSONResponse
import uvicorn

# ── Paths ─────────────────────────────────────────────────────────────────────
WS          = Path("/workspace")          # RunPod persistent volume mount
MODELS      = WS / "models"
WAV2LIP_DIR = WS / "Wav2Lip"
WAV2LIP_CKP = WAV2LIP_DIR / "checkpoints" / "wav2lip_gan.pth"
SFD_CKP     = WAV2LIP_DIR / "face_detection" / "detection" / "sfd" / "s3fd.pth"
WORK_DIR    = WS / "work"

MODELS.mkdir(parents=True, exist_ok=True)
WORK_DIR.mkdir(parents=True, exist_ok=True)

# ── Wav2Lip clone + patches ───────────────────────────────────────────────────
if not WAV2LIP_DIR.exists():
    print("Cloning Wav2Lip …")
    subprocess.run(["git", "clone", "-q",
                    "https://github.com/Rudrabha/Wav2Lip.git",
                    str(WAV2LIP_DIR)], check=True)

def _patch_file(path: Path, old: str, new: str, label: str):
    if not path.exists():
        return
    t = path.read_text(encoding="utf-8")
    t2 = t.replace(old, new)
    if t2 != t:
        path.write_text(t2, encoding="utf-8")
        print(f"Patched {path.name}: {label}")

# librosa >= 0.10 keyword-arg change
_audio_py = WAV2LIP_DIR / "audio.py"
if _audio_py.exists():
    import re as _re
    t = _audio_py.read_text(encoding="utf-8")
    t2 = _re.sub(
        r"librosa\.filters\.mel\s*\(\s*hp\.sample_rate\s*,\s*hp\.n_fft\s*,",
        "librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft,",
        t, count=1,
    )
    if t2 != t:
        _audio_py.write_text(t2, encoding="utf-8")
        print("Patched Wav2Lip/audio.py: librosa keyword args")

_inf_py = WAV2LIP_DIR / "inference.py"
_patch_file(_inf_py, "cv2.VideoWriter_fourcc(*'DIVX')",
            "cv2.VideoWriter_fourcc(*'mp4v')", "DIVX→mp4v")
_patch_file(_inf_py, "'temp/result.avi'", "'temp/result.mp4'",
            "result.avi→result.mp4")

# ── Wav2Lip checkpoints (skip if already on /workspace) ──────────────────────
WAV2LIP_DIR.joinpath("checkpoints").mkdir(parents=True, exist_ok=True)
WAV2LIP_DIR.joinpath("face_detection/detection/sfd").mkdir(parents=True, exist_ok=True)

def _wget(url: str, dest: Path):
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"  Already cached: {dest.name}")
        return
    print(f"  Downloading {dest.name} …")
    subprocess.run(["wget", "-q", url, "-O", str(dest)], check=True)

_wget("https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip_gan.pth",
      WAV2LIP_CKP)
_wget("https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth",
      SFD_CKP)

# ── Load Stable Diffusion (cached to /workspace/models) ──────────────────────
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

SD_CACHE   = str(MODELS)
SD_MODEL   = "runwayml/stable-diffusion-v1-5"

print("\nLoading SD txt2img …")
pipe_txt2img = StableDiffusionPipeline.from_pretrained(
    SD_MODEL, torch_dtype=torch.float16,
    safety_checker=None, cache_dir=SD_CACHE,
).to("cuda")
pipe_txt2img.enable_attention_slicing()

print("Loading SD img2img …")
pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
    SD_MODEL, torch_dtype=torch.float16,
    safety_checker=None, cache_dir=SD_CACHE,
).to("cuda")

# ── IP-Adapter (optional, graceful passthrough) ───────────────────────────────
IP_ADAPTER_OK = False
try:
    pipe_img2img.load_ip_adapter(
        "h94/IP-Adapter", subfolder="models",
        weight_name="ip-adapter_sd15.bin",
        cache_dir=SD_CACHE,
    )
    IP_ADAPTER_OK = True
    print("IP-Adapter loaded.")
except Exception as e:
    print(f"[!] IP-Adapter failed ({e}). face_swap will passthrough.")

print("All models ready.\n")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Project Montage — RunPod GPU Backend")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "ip_adapter": IP_ADAPTER_OK,
    }


@app.post("/generate")
async def generate(
    prompt: str = Form(...),
    negative_prompt: str = Form(
        "text, watermark, logo, blurry, distorted, low quality, "
        "deformed, mutated, ugly, extra limbs, bad anatomy"
    ),
    width: int  = Form(896),
    height: int = Form(512),
    steps: int  = Form(28),
):
    print(f"[generate] {prompt[:120]}")
    image = pipe_txt2img(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width, height=height,
        num_inference_steps=steps,
        guidance_scale=7.5,
    ).images[0]
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.post("/face_swap")
async def face_swap(
    prompt: str = Form(...),
    swap_image: UploadFile  = File(...),
    base_image: UploadFile  = File(...),
    ip_scale: float = Form(0.85),
    strength: float = Form(0.50),
    steps: int      = Form(30),
):
    base_bytes = await base_image.read()
    swap_bytes = await swap_image.read()
    if not IP_ADAPTER_OK:
        return Response(content=base_bytes, media_type="image/png")
    try:
        def _lim(im: Image.Image, mx: int) -> Image.Image:
            w, h = im.size
            s = mx / max(w, h)
            return im.resize((int(w*s), int(h*s)), Image.LANCZOS) if max(w,h)>mx else im
        ref  = _lim(Image.open(io.BytesIO(swap_bytes)).convert("RGB"), 768)
        init = _lim(Image.open(io.BytesIO(base_bytes)).convert("RGB"), 896)
        torch.cuda.empty_cache()
        pipe_img2img.set_ip_adapter_scale(float(ip_scale))
        out = pipe_img2img(
            prompt=prompt, image=init, ip_adapter_image=ref,
            negative_prompt="bad quality, blurry, distorted",
            strength=float(strength), guidance_scale=7.0,
            num_inference_steps=int(steps),
        ).images[0]
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception:
        print("[face_swap] FAILED, passthrough.\n", traceback.format_exc()[-1500:])
        return Response(content=base_bytes, media_type="image/png")


def _ext(name: str, fallback: str = ".bin") -> str:
    e = os.path.splitext(name)[1].lower()
    return e if e else fallback


def _crop_face_for_wav2lip(img_path: Path) -> Path:
    """
    Detect the largest face in a portrait and return a tight crop
    (head-and-shoulders, 40% margin, resized to 256×256) for Wav2Lip.
    Falls back to a centered 60% crop when no face is detected by Haar.
    The original file is untouched; cropped version saved alongside it.
    """
    try:
        import cv2 as _cv2
        img = _cv2.imread(str(img_path))
        if img is None:
            return img_path
        h, w = img.shape[:2]
        gray = _cv2.cvtColor(img, _cv2.COLOR_BGR2GRAY)
        cascade_xml = _cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector = _cv2.CascadeClassifier(cascade_xml)

        faces = detector.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(24, 24)
        )
        if len(faces) > 0:
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            margin = int(max(fw, fh) * 0.40)
            x1, y1 = max(0, x - margin), max(0, y - margin)
            x2, y2 = min(w, x + fw + margin), min(h, y + fh + margin)
            crop = img[y1:y2, x1:x2]
            print(f"[lip_sync] face crop: ({x},{y},{fw},{fh}) → ({x1},{y1})-({x2},{y2})")
        else:
            # No face found — take the upper-center 60% (head is usually there)
            cw, ch = int(w * 0.65), int(h * 0.65)
            x1 = (w - cw) // 2
            y1 = max(0, int(h * 0.05))
            crop = img[y1: y1 + ch, x1: x1 + cw]
            print("[lip_sync] no face detected; using center-top crop fallback")

        crop_resized = _cv2.resize(crop, (256, 256), interpolation=_cv2.INTER_LANCZOS4)
        out_path = img_path.parent / ("face_cropped" + img_path.suffix)
        _cv2.imwrite(str(out_path), crop_resized)
        return out_path
    except Exception as exc:
        print(f"[lip_sync] face crop failed ({exc}); using original portrait")
        return img_path


def _wav_seconds(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                capture_output=True, text=True, timeout=15,
            )
            return float(r.stdout.strip() or 3.0)
        except Exception:
            return 3.0

def _image_to_video(img_path: Path, seconds: float, out: Path, fps: int = 25):
    import imageio
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    s = 480 / max(w, h)
    img = img.resize((int(w*s), int(h*s)), Image.LANCZOS)
    arr = np.array(img)
    n = max(int(round(seconds * fps)) + fps, fps * 2)
    writer = imageio.get_writer(str(out), fps=fps, codec="libx264", quality=8)
    for _ in range(n):
        writer.append_data(arr)
    writer.close()


@app.post("/lip_sync")
async def lip_sync(audio: UploadFile = File(...), face: UploadFile = File(...)):
    job = WORK_DIR / f"job_{uuid.uuid4().hex[:8]}"
    job.mkdir(parents=True, exist_ok=True)

    face_ext   = _ext(face.filename, ".png")
    audio_ext  = _ext(audio.filename, ".wav")
    face_in    = job / f"face_in{face_ext}"
    audio_path = job / f"audio{audio_ext}"

    face_in.write_bytes(await face.read())
    audio_path.write_bytes(await audio.read())

    is_image = face_ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    if is_image:
        # Optional pre-crop helps when SD portraits frame the face very small.
        # Disabled by default: the colab/runpod working baseline simply resized
        # the raw portrait to 480 px and SFD detected it fine. Set
        # WAV2LIP_FACE_CROP=1 in the pod env to opt in.
        face_for_video = face_in
        if os.environ.get("WAV2LIP_FACE_CROP", "0") == "1":
            face_for_video = _crop_face_for_wav2lip(face_in)
        face_video = job / "face_video.mp4"
        _image_to_video(face_for_video, _wav_seconds(audio_path), face_video)
    else:
        face_video = job / f"face_video{face_ext}"
        shutil.move(str(face_in), str(face_video))

    # Wav2Lip writes intermediate frames to `temp/result.mp4` *relative to the
    # subprocess cwd* (which is /workspace). If that directory doesn't exist
    # OpenCV's VideoWriter silently writes nowhere and inference.py exits 0
    # with no output. Anchor the path to /workspace so we stay consistent
    # regardless of where uvicorn was launched from.
    temp_dir = WS / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    for old in temp_dir.glob("result.*"):
        old.unlink(missing_ok=True)

    out_video = job / "out.mp4"
    cmd = [
        "python", str(WAV2LIP_DIR / "inference.py"),
        "--checkpoint_path", str(WAV2LIP_CKP),
        "--face", str(face_video),
        "--audio", str(audio_path),
        "--outfile", str(out_video),
        "--pads", "0", "10", "0", "0",
        "--resize_factor", "1", "--nosmooth",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=str(WS))
    print("[lip_sync] rc:", proc.returncode)

    # Recovery path: if Wav2Lip's internal ffmpeg muxing failed but the raw
    # frames file exists, mux it ourselves. Use absolute paths.
    if not out_video.exists():
        for candidate in (temp_dir / "result.mp4", temp_dir / "result.avi"):
            if candidate.exists() and candidate.stat().st_size > 0:
                rc = subprocess.run(
                    ["ffmpeg", "-y", "-i", str(candidate), "-i", str(audio_path),
                     "-c:v", "libx264", "-pix_fmt", "yuv420p",
                     "-c:a", "aac", "-b:a", "128k", "-shortest", str(out_video)],
                    capture_output=True, text=True, timeout=120,
                )
                print(f"[lip_sync] manual mux rc={rc.returncode}")
                break

    if not out_video.exists():
        # Surface enough state for the next debug pass.
        listing = []
        for p in job.glob("**/*"):
            try:
                listing.append(f"{p.relative_to(job)} ({p.stat().st_size if p.is_file() else 'dir'})")
            except Exception:
                pass
        for p in temp_dir.glob("*"):
            try:
                listing.append(f"temp/{p.name} ({p.stat().st_size})")
            except Exception:
                pass
        return JSONResponse(status_code=500, content={
            "error": "no output produced",
            "stderr": (proc.stderr or "")[-8000:],
            "stdout": (proc.stdout or "")[-3000:],
            "files": listing,
        })

    data = out_video.read_bytes()
    shutil.rmtree(job, ignore_errors=True)
    return Response(content=data, media_type="video/mp4")


# ── Detect and print public URL ───────────────────────────────────────────────
def _print_public_url():
    """
    RunPod exposes port 8000 at:
        https://<POD_ID>-8000.proxy.runpod.net
    The POD_ID is available as an environment variable.
    """
    pod_id = os.environ.get("RUNPOD_POD_ID", "")
    if pod_id:
        url = f"https://{pod_id}-8000.proxy.runpod.net"
    else:
        # Fallback: try to read from RunPod metadata endpoint
        try:
            r = urllib.request.urlopen(
                "http://169.254.169.254/latest/meta-data/public-hostname", timeout=3
            )
            url = f"http://{r.read().decode().strip()}:8000"
        except Exception:
            url = "http://localhost:8000  (set RUNPOD_POD_ID if this is wrong)"

    banner = "\n" + "=" * 72
    banner += f"\n  PROJECT MONTAGE — GPU BACKEND READY"
    banner += f"\n  Public API URL:  {url}"
    banner += f"\n  Paste this into the Streamlit sidebar (or config/colab_api.txt)"
    banner += "\n" + "=" * 72
    print(banner, flush=True)


# ── Start server ──────────────────────────────────────────────────────────────
def _run_server():
    uvicorn.Server(
        uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="warning")
    ).run()


threading.Thread(target=_run_server, daemon=True).start()

# Wait until local health check passes, then print public URL
print("Waiting for local API …", flush=True)
for _ in range(120):
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=1)
        break
    except Exception:
        time.sleep(0.5)

_print_public_url()

# Keep the main thread alive (Jupyter / terminal)
try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    print("Shutting down.")
