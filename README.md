# VoiceScribe – Automatic Speech Recognition (ASR) Tool

<div align="center">

![VoiceScribe Banner](https://img.shields.io/badge/VoiceScribe-ASR%20Tool-00e5ff?style=for-the-badge&logo=soundcloud&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=for-the-badge&logo=flask&logoColor=white)
![Whisper](https://img.shields.io/badge/OpenAI%20Whisper-ASR-412991?style=for-the-badge&logo=openai&logoColor=white)

**Real-time, browser-based Automatic Speech Recognition powered by OpenAI Whisper.**  
Record your voice or upload any audio file and receive an accurate transcript instantly — all processed locally on your machine.

</div>

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎙 **Live Recording** | Record directly from your browser microphone with real-time waveform visualization |
| 📂 **File Upload** | Drag-and-drop or browse to upload WAV, MP3, WebM, OGG, FLAC, M4A files |
| ⏱ **Timestamped Segments** | Each sentence is displayed with precise start/end timestamps |
| 🌐 **Language Detection** | Whisper automatically detects the spoken language |
| 📋 **Copy / Export** | Copy transcript, download as `.txt`, or export full JSON (transcript + segments) |
| 📊 **Session Stats** | Live counters for recordings, total words, characters, and average transcription speed |
| 🤖 **Model Selection** | Choose from `tiny`, `base`, `small`, or `medium` Whisper models |
| 🔒 **Privacy First** | 100% local — no audio leaves your device, no cloud APIs |

---

## 🛠 Technologies Used

### Backend
| Technology | Version | Purpose |
|---|---|---|
| **Python** | 3.9+ | Core language |
| **Flask** | 3.x | Web framework & REST API |
| **flask-cors** | 4.x | Cross-Origin Resource Sharing |
| **OpenAI Whisper** | latest | Core ASR model (runs locally) |
| **PyTorch** | 2.x | Whisper's ML runtime |
| **PyAudio** | 0.2.x | Audio I/O (optional, for future microphone server-side) |
| **SpeechRecognition** | 3.x | Supporting library |

### Frontend
| Technology | Purpose |
|---|---|
| **HTML5** | Semantic structure, SEO-ready |
| **Vanilla CSS** | Glassmorphism design, custom animations |
| **Vanilla JavaScript** | MediaRecorder API, Web Audio API, fetch |
| **Web Audio API** | Real-time frequency bar visualizer |
| **MediaRecorder API** | Browser-native audio capture |

---

## 📁 Project Structure

```
ASR proj/
├── app.py                  # Flask backend — Whisper transcription API
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html          # Jinja2 HTML template
├── static/
│   ├── css/
│   │   ├── style.css       # Main stylesheet (glassmorphism design)
│   │   └── visualizer.css  # Canvas visualizer styles
│   └── js/
│       └── main.js         # Frontend logic (recording, upload, UI)
└── README.md
```

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- A modern browser (Chrome / Edge / Firefox)
- **ffmpeg** (required by Whisper for audio decoding)

### 1. Install ffmpeg

**Windows (via Chocolatey):**
```bash
choco install ffmpeg
```
**Windows (manual):** Download from https://ffmpeg.org/download.html and add to PATH.

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt install ffmpeg
```

### 2. Clone the Repository

```bash
git clone https://github.com/<your-username>/voicescribe-asr.git
cd voicescribe-asr
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** On first run, Whisper will automatically download the model weights (~74MB for `base`).

### 4. Run the Application

```bash
python app.py
```

Open your browser and navigate to:

```
http://localhost:5000
```

---

## 🚀 Usage

### Live Recording
1. Click the **🎙 microphone button** — your browser will ask for microphone permission.
2. Speak clearly. The waveform visualizer shows live audio.
3. Click the button again (or **✨ Transcribe**) to stop and transcribe.
4. The transcript appears instantly with language, word count, and timing info.

### File Upload
1. Drag and drop an audio file onto the **Upload** zone, or click to browse.
2. Supported formats: `WAV`, `MP3`, `WebM`, `OGG`, `FLAC`, `M4A`
3. Transcription starts automatically after upload.

### Exporting Results
| Button | Action |
|---|---|
| 📋 **Copy** | Copies transcript text to clipboard |
| ⬇️ **TXT** | Downloads `transcript.txt` |
| 🗃 **JSON** | Downloads `transcript.json` (with segments) |
| 🗑 **Clear** | Clears the current transcript |

---

## 🔧 Configuration

You can control the default Whisper model via an environment variable:

```bash
# Windows PowerShell
$env:WHISPER_MODEL = "small"
python app.py

# Linux/macOS
WHISPER_MODEL=small python app.py
```

| Model | Size | Speed | Best For |
|---|---|---|---|
| `tiny` | ~39 MB | Very fast | Quick demos |
| `base` | ~74 MB | Fast | **Default, general use** |
| `small` | ~244 MB | Moderate | Better accuracy |
| `medium` | ~769 MB | Slow | High accuracy |

---

## 📡 API Reference

### `POST /transcribe`
Transcribes an uploaded audio file.

**Request:**
```
Content-Type: multipart/form-data
Body: audio=<file>
```

**Response:**
```json
{
  "success": true,
  "transcript": "Hello, this is a test transcription.",
  "language": "en",
  "duration_s": 2.45,
  "model": "base",
  "segments": [
    { "start": 0.0, "end": 2.5, "text": "Hello, this is a test transcription." }
  ]
}
```

### `GET /health`
Health check endpoint.

```json
{ "status": "ok", "model": "base", "timestamp": 1727648400.0 }
```

### `GET /models`
Lists available Whisper models.

```json
{ "available": ["tiny", "base", "small", "medium", "large"], "current": "base" }
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -m 'Add your feature'`)
4. Push to branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the **MIT License**.  
OpenAI Whisper is licensed under the [MIT License](https://github.com/openai/whisper/blob/main/LICENSE).

---

<div align="center">
Made with ❤️ using <strong>OpenAI Whisper</strong> + <strong>Flask</strong>
</div>
