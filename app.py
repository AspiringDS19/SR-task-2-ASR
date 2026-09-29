"""
ASR (Automatic Speech Recognition) Tool - Flask Backend
Uses OpenAI Whisper for local, offline speech recognition.

Audio pipeline:
  1. If ffmpeg is on PATH → Whisper handles all formats natively.
  2. If ffmpeg is NOT on PATH → we convert the temp file to a 16-kHz
     mono WAV using soundfile/pydub before passing to Whisper's
     load_audio(), so the app still works without ffmpeg.
"""

import io
import os
import shutil
import subprocess
import tempfile
import time
import logging
import struct
import wave

import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import whisper

# ─── App Setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Load Whisper Model ───────────────────────────────────────────────────────
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "base")   # tiny | base | small | medium | large
logger.info(f"Loading Whisper model: '{MODEL_SIZE}' …")
model = whisper.load_model(MODEL_SIZE)
logger.info("Whisper model loaded successfully ✓")

# ─── ffmpeg availability check ────────────────────────────────────────────────
# Check PATH first, then well-known WinGet install location as a fallback
FFMPEG_PATH = shutil.which("ffmpeg")

if not FFMPEG_PATH:
    # WinGet installs ffmpeg here when the shell PATH hasn't been refreshed yet
    _WINGET_FFMPEG = r"C:\Users\sharb\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe"
    if os.path.isfile(_WINGET_FFMPEG):
        FFMPEG_PATH = _WINGET_FFMPEG
        logger.info(f"ffmpeg found via WinGet fallback: {FFMPEG_PATH}")

if FFMPEG_PATH:
    logger.info(f"ffmpeg available: {FFMPEG_PATH}")
else:
    logger.warning(
        "ffmpeg NOT found on PATH. Will use numpy/soundfile fallback for WAV files. "
        "WebM/MP3 uploads require ffmpeg — install it for full format support."
    )


# ─── Audio helpers ────────────────────────────────────────────────────────────

def _convert_with_ffmpeg(src_path: str, dst_path: str) -> bool:
    """Convert any audio file to 16-kHz mono WAV using ffmpeg."""
    try:
        result = subprocess.run(
            [FFMPEG_PATH, "-y", "-i", src_path,
             "-ar", "16000", "-ac", "1", "-f", "wav", dst_path],
            capture_output=True, timeout=60,
        )
        return result.returncode == 0
    except Exception as exc:
        logger.error(f"ffmpeg conversion failed: {exc}")
        return False


def _load_audio_numpy(wav_path: str) -> np.ndarray:
    """
    Read a WAV file via stdlib `wave` module (no ffmpeg needed).
    Returns float32 numpy array normalised to [-1, 1] at the file's
    native sample rate. Whisper will resample internally if needed.
    """
    with wave.open(wav_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth  = wf.getsampwidth()
        n_frames   = wf.getnframes()
        raw        = wf.readframes(n_frames)

    fmt = {1: np.int8, 2: np.int16, 4: np.int32}.get(sampwidth, np.int16)
    audio = np.frombuffer(raw, dtype=fmt).astype(np.float32)

    # Mix down to mono
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    # Normalise
    audio /= float(np.iinfo(fmt).max)
    return audio


def prepare_audio(src_path: str, suffix: str):
    """
    Returns a path to a file Whisper can read (16-kHz mono WAV).
    Strategy:
      • ffmpeg available  → convert everything (any format).
      • ffmpeg missing    → if already a .wav, use numpy fallback;
                            otherwise raise a helpful error.
    Returns (wav_path, array_or_None, cleanup_needed).
    """
    wav_tmp = None

    if FFMPEG_PATH:
        # Convert via ffmpeg → guaranteed WAV
        wav_tmp = src_path.replace(suffix, "_converted.wav")
        if not _convert_with_ffmpeg(src_path, wav_tmp):
            raise RuntimeError("ffmpeg failed to convert the audio file.")
        return wav_tmp, None, True          # pass the wav path to Whisper

    # ── No ffmpeg ─────────────────────────────────────────────────────────────
    if suffix.lower() in (".wav", ".wave"):
        # Use numpy-based loader → pass ndarray directly to Whisper
        audio_array = _load_audio_numpy(src_path)
        return src_path, audio_array, False

    # Non-WAV without ffmpeg — try pydub if installed
    try:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(src_path)
        seg = seg.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        wav_tmp = src_path + "_pydub.wav"
        seg.export(wav_tmp, format="wav")
        return wav_tmp, None, True
    except Exception:
        pass

    raise RuntimeError(
        "ffmpeg is not installed. Please install it to transcribe MP3/WebM files. "
        "WAV files work without ffmpeg. "
        "Install: https://ffmpeg.org/download.html"
    )


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/transcribe", methods=["POST"])
def transcribe():
    """
    Accepts a WAV/WebM/MP3/MP4 audio blob, transcribes with Whisper,
    returns transcript + metadata.
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # ── Save upload to temp file ──────────────────────────────────────────────
    suffix = os.path.splitext(audio_file.filename)[-1].lower() or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    wav_path   = None
    needs_del  = False

    try:
        logger.info(f"Transcribing: {audio_file.filename}  ({os.path.getsize(tmp_path)} bytes)")
        start = time.perf_counter()

        # ── Prepare audio (convert if needed) ────────────────────────────────
        wav_path, audio_array, needs_del = prepare_audio(tmp_path, suffix)

        # ── Run Whisper ───────────────────────────────────────────────────────
        if audio_array is not None:
            # numpy path (no ffmpeg, WAV only)
            result = model.transcribe(audio_array, fp16=False, verbose=False)
        else:
            result = model.transcribe(wav_path,    fp16=False, verbose=False)

        elapsed  = round(time.perf_counter() - start, 2)
        text     = result.get("text", "").strip()
        language = result.get("language", "unknown")
        segments = result.get("segments", [])

        logger.info(f"Done in {elapsed}s | lang={language} | chars={len(text)}")

        timeline = [
            {
                "start": round(s["start"], 2),
                "end":   round(s["end"], 2),
                "text":  s["text"].strip(),
            }
            for s in segments
        ]

        return jsonify({
            "success":    True,
            "transcript": text,
            "language":   language,
            "duration_s": elapsed,
            "segments":   timeline,
            "model":      MODEL_SIZE,
        })

    except RuntimeError as exc:
        # User-facing error (ffmpeg missing, bad format, etc.)
        logger.warning(f"Transcription error: {exc}")
        return jsonify({"error": str(exc)}), 400

    except Exception as exc:
        logger.error(f"Transcription failed: {exc}", exc_info=True)
        return jsonify({"error": f"Transcription failed: {exc}"}), 500

    finally:
        # Clean up temp files
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        if needs_del and wav_path and os.path.exists(wav_path):
            try:
                os.unlink(wav_path)
            except OSError:
                pass


@app.route("/health")
def health():
    return jsonify({
        "status":   "ok",
        "model":    MODEL_SIZE,
        "ffmpeg":   bool(FFMPEG_PATH),
        "timestamp": time.time(),
    })


@app.route("/models")
def list_models():
    sizes = ["tiny", "base", "small", "medium", "large"]
    return jsonify({"available": sizes, "current": MODEL_SIZE})


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
