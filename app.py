"""
ASR (Automatic Speech Recognition) Tool - Flask Backend
Uses OpenAI Whisper for local, offline speech recognition.

Features:
- Auto-detects ffmpeg across standard PATH and Windows winget/chocolatey locations
- Injects ffmpeg directory into PATH and monkey-patches whisper.audio.load_audio
- Supports WAV, MP3, WebM, FLAC, OGG, M4A audio inputs
- Supports model selection (tiny, base, small, medium)
- Cache-busting for assets so client browser updates immediately
"""

import os
import sys
import shutil
import subprocess
import tempfile
import time
import logging
import wave

import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# ─── App Setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Locate ffmpeg ────────────────────────────────────────────────────────────
FFMPEG_SEARCH_PATHS = [
    shutil.which("ffmpeg"),
    r"C:\Users\sharb\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe",
    r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
    r"C:\ffmpeg\bin\ffmpeg.exe",
    r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
]

FFMPEG_PATH = None
for _p in FFMPEG_SEARCH_PATHS:
    if _p and os.path.isfile(_p):
        FFMPEG_PATH = _p
        break

if FFMPEG_PATH:
    ffmpeg_dir = os.path.dirname(FFMPEG_PATH)
    if ffmpeg_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
        logger.info(f"Injected ffmpeg directory into PATH: {ffmpeg_dir}")

    logger.info(f"ffmpeg located: {FFMPEG_PATH}")

    # ── MONKEY-PATCH whisper.audio ────────────────────────────────────────────
    import whisper.audio as _whisper_audio

    _WHISPER_SR = _whisper_audio.SAMPLE_RATE

    def _patched_load_audio(file: str, sr: int = _WHISPER_SR) -> np.ndarray:
        cmd = [
            FFMPEG_PATH,
            "-nostdin",
            "-threads", "0",
            "-i", file,
            "-f", "s16le",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-ar", str(sr),
            "-",
        ]
        try:
            out = subprocess.run(cmd, capture_output=True, check=True).stdout
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"ffmpeg failed to decode audio: {exc.stderr.decode(errors='replace')}"
            ) from exc
        return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

    _whisper_audio.load_audio = _patched_load_audio
    logger.info("whisper.audio.load_audio patched to use full ffmpeg path ✓")

else:
    logger.warning("ffmpeg not found — only 16kHz WAV files will work (wave fallback).")

# ─── Load Whisper Models ──────────────────────────────────────────────────────
import whisper

DEFAULT_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "base")
LOADED_MODELS = {}

def get_model(size: str = DEFAULT_MODEL_SIZE):
    size = size.lower().strip()
    if size not in LOADED_MODELS:
        logger.info(f"Loading Whisper model: '{size}' …")
        LOADED_MODELS[size] = whisper.load_model(size)
        logger.info(f"Whisper model '{size}' loaded successfully ✓")
    return LOADED_MODELS[size]

# Pre-warm default model
logger.info(f"Pre-warming default model: '{DEFAULT_MODEL_SIZE}'")
get_model(DEFAULT_MODEL_SIZE)


# ─── Audio Helpers ────────────────────────────────────────────────────────────
def _read_wav_numpy(wav_path: str) -> np.ndarray:
    """Read a WAV file with stdlib wave when ffmpeg is unavailable."""
    with wave.open(wav_path, "rb") as wf:
        n_ch = wf.getnchannels()
        sw = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sw, np.int16)
    pcm = np.frombuffer(raw, dtype=dtype).astype(np.float32)

    if n_ch > 1:
        pcm = pcm.reshape(-1, n_ch).mean(axis=1)

    pcm /= float(np.iinfo(dtype).max)
    return pcm


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", cache_bust=int(time.time()))


@app.route("/transcribe", methods=["POST"])
def transcribe():
    """
    Accepts WAV, WebM, MP3, FLAC, OGG, M4A audio blob or file.
    Transcribes with Whisper, returns transcript + timeline segments.
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    model_size = request.form.get("model", DEFAULT_MODEL_SIZE).strip()
    try:
        active_model = get_model(model_size)
    except Exception as e:
        logger.warning(f"Could not load requested model '{model_size}': {e}. Falling back to default.")
        active_model = get_model(DEFAULT_MODEL_SIZE)
        model_size = DEFAULT_MODEL_SIZE

    suffix = os.path.splitext(audio_file.filename)[-1].lower()
    if not suffix:
        suffix = ".wav"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    file_size = os.path.getsize(tmp_path)
    logger.info(f"Received: '{audio_file.filename}' (suffix={suffix}, size={file_size} bytes, model={model_size})")

    try:
        start = time.perf_counter()
        is_wav = suffix in (".wav", ".wave")

        if FFMPEG_PATH:
            # ffmpeg decodes any format and resamples to 16kHz mono float32
            result = active_model.transcribe(tmp_path, fp16=False, verbose=False)
        elif is_wav:
            # fallback for WAV if ffmpeg is somehow absent
            audio_np = _read_wav_numpy(tmp_path)
            result = active_model.transcribe(audio_np, fp16=False, verbose=False)
        else:
            return jsonify({
                "error": "ffmpeg is not found on this machine. Only WAV files are supported without ffmpeg."
            }), 400

        elapsed = round(time.perf_counter() - start, 2)
        text = result.get("text", "").strip()
        language = result.get("language", "unknown")
        segments = result.get("segments", [])

        logger.info(f"Transcription done in {elapsed}s | lang={language} | chars={len(text)}")

        timeline = [
            {
                "start": round(s["start"], 2),
                "end": round(s["end"], 2),
                "text": s["text"].strip(),
            }
            for s in segments
        ]

        return jsonify({
            "success": True,
            "transcript": text,
            "language": language,
            "duration_s": elapsed,
            "segments": timeline,
            "model": model_size,
        })

    except Exception as exc:
        logger.error(f"Transcription failed: {exc}", exc_info=True)
        return jsonify({"error": f"Transcription failed: {exc}"}), 500

    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "default_model": DEFAULT_MODEL_SIZE,
        "loaded_models": list(LOADED_MODELS.keys()),
        "ffmpeg": bool(FFMPEG_PATH),
        "ffmpeg_path": FFMPEG_PATH or "not found",
        "timestamp": time.time(),
    })


@app.route("/models")
def list_models():
    sizes = ["tiny", "base", "small", "medium", "large"]
    return jsonify({"available": sizes, "current": DEFAULT_MODEL_SIZE})


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
