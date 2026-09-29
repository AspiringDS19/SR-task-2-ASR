"""
ASR (Automatic Speech Recognition) Tool - Flask Backend
Uses OpenAI Whisper for local, offline speech recognition.

Key fix: Whisper's load_audio() hardcodes "ffmpeg" as a command string.
We monkey-patch whisper.audio so it uses our full ffmpeg path, bypassing
the PATH lookup entirely.
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
# Search PATH first, then common Windows install locations
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
    logger.info(f"ffmpeg located: {FFMPEG_PATH}")
    # ── MONKEY-PATCH whisper.audio ────────────────────────────────────────────
    # Whisper hardcodes the string "ffmpeg" in its cmd list.
    # We replace load_audio with our own version that uses the full path.
    import whisper.audio as _whisper_audio

    _WHISPER_SR = _whisper_audio.SAMPLE_RATE

    def _patched_load_audio(file: str, sr: int = _WHISPER_SR) -> np.ndarray:
        cmd = [
            FFMPEG_PATH,           # <── full path instead of bare "ffmpeg"
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
    logger.warning("ffmpeg not found — only WAV files will work (numpy fallback).")

# ─── Load Whisper Model ───────────────────────────────────────────────────────
import whisper  # import AFTER patching audio module

MODEL_SIZE = os.environ.get("WHISPER_MODEL", "base")
logger.info(f"Loading Whisper model: '{MODEL_SIZE}' …")
model = whisper.load_model(MODEL_SIZE)
logger.info("Whisper model loaded successfully ✓")


# ─── Audio Helpers ────────────────────────────────────────────────────────────

def _read_wav_numpy(wav_path: str) -> np.ndarray:
    """
    Read a WAV file with stdlib `wave` — zero external dependencies.
    Returns float32 array at 16 kHz mono (matches Whisper's expected input).
    """
    with wave.open(wav_path, "rb") as wf:
        n_ch     = wf.getnchannels()
        sw       = wf.getsampwidth()      # bytes per sample
        n_frames = wf.getnframes()
        raw      = wf.readframes(n_frames)

    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sw, np.int16)
    pcm   = np.frombuffer(raw, dtype=dtype).astype(np.float32)

    # Stereo → mono
    if n_ch > 1:
        pcm = pcm.reshape(-1, n_ch).mean(axis=1)

    # Normalise to [-1, 1]
    pcm /= float(np.iinfo(dtype).max)
    return pcm


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

    # Determine file extension
    suffix = os.path.splitext(audio_file.filename)[-1].lower()
    if not suffix:
        suffix = ".wav"   # default for browser recordings

    # Save uploaded bytes to a temp file
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    logger.info(
        f"Received: '{audio_file.filename}'  "
        f"suffix={suffix}  size={os.path.getsize(tmp_path)} bytes"
    )

    extra_cleanup = []   # additional temp files to delete

    try:
        start = time.perf_counter()

        is_wav = suffix in (".wav", ".wave")

        if is_wav:
            # ── WAV path: read with stdlib wave, pass numpy array to Whisper ──
            # This completely bypasses Whisper's load_audio() / ffmpeg.
            logger.info("WAV detected → using numpy reader (no ffmpeg needed)")
            audio_np = _read_wav_numpy(tmp_path)
            result   = model.transcribe(audio_np, fp16=False, verbose=False)

        elif FFMPEG_PATH:
            # ── Non-WAV path: pass file path to Whisper (uses patched load_audio)
            logger.info(f"Non-WAV detected → using patched ffmpeg: {FFMPEG_PATH}")
            result = model.transcribe(tmp_path, fp16=False, verbose=False)

        else:
            return jsonify({
                "error": (
                    "ffmpeg is not installed on this machine. "
                    "Only WAV files are supported without ffmpeg. "
                    "Please install ffmpeg from https://ffmpeg.org/download.html "
                    "to transcribe MP3 / WebM / OGG / FLAC files."
                )
            }), 400

        elapsed  = round(time.perf_counter() - start, 2)
        text     = result.get("text", "").strip()
        language = result.get("language", "unknown")
        segments = result.get("segments", [])

        logger.info(
            f"Transcription done in {elapsed}s | lang={language} | chars={len(text)}"
        )

        timeline = [
            {
                "start": round(s["start"], 2),
                "end":   round(s["end"],   2),
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

    except Exception as exc:
        logger.error(f"Transcription failed: {exc}", exc_info=True)
        return jsonify({"error": f"Transcription failed: {exc}"}), 500

    finally:
        for path in [tmp_path] + extra_cleanup:
            try:
                if path and os.path.exists(path):
                    os.unlink(path)
            except OSError:
                pass


@app.route("/health")
def health():
    return jsonify({
        "status":    "ok",
        "model":     MODEL_SIZE,
        "ffmpeg":    bool(FFMPEG_PATH),
        "ffmpeg_path": FFMPEG_PATH or "not found",
        "timestamp": time.time(),
    })


@app.route("/models")
def list_models():
    sizes = ["tiny", "base", "small", "medium", "large"]
    return jsonify({"available": sizes, "current": MODEL_SIZE})


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
