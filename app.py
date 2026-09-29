"""
ASR (Automatic Speech Recognition) Tool - Flask Backend
Uses OpenAI Whisper for local, offline speech recognition.
"""

import os
import tempfile
import json
import time
import logging
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


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/transcribe", methods=["POST"])
def transcribe():
    """
    Accepts a WAV/WebM/MP4 audio blob, transcribes it with Whisper,
    and returns the transcript + metadata.
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save to a temp file so Whisper can read it
    suffix = os.path.splitext(audio_file.filename)[-1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        logger.info(f"Transcribing file: {tmp_path}")
        start = time.perf_counter()

        result = model.transcribe(
            tmp_path,
            fp16=False,          # works on CPU too
            verbose=False,
        )

        elapsed = round(time.perf_counter() - start, 2)
        text = result.get("text", "").strip()
        language = result.get("language", "unknown")
        segments = result.get("segments", [])

        logger.info(f"Transcription done in {elapsed}s | lang={language} | chars={len(text)}")

        # Build word-level timeline from segments
        timeline = [
            {
                "start": round(s["start"], 2),
                "end":   round(s["end"], 2),
                "text":  s["text"].strip(),
            }
            for s in segments
        ]

        return jsonify({
            "success": True,
            "transcript": text,
            "language":   language,
            "duration_s": elapsed,
            "segments":   timeline,
            "model":      MODEL_SIZE,
        })

    except Exception as exc:
        logger.error(f"Transcription failed: {exc}", exc_info=True)
        return jsonify({"error": str(exc)}), 500

    finally:
        os.unlink(tmp_path)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_SIZE, "timestamp": time.time()})


@app.route("/models")
def list_models():
    sizes = ["tiny", "base", "small", "medium", "large"]
    return jsonify({"available": sizes, "current": MODEL_SIZE})


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
