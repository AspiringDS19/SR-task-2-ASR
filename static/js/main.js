/**
 * VoiceScribe – Main JavaScript
 *
 * KEY FIX: The browser records raw PCM via ScriptProcessorNode, then we
 * encode it as a proper 16-bit mono WAV blob before sending to /transcribe.
 * This means NO ffmpeg is needed on the server for microphone recordings.
 *
 * File uploads: WAV files work natively. MP3/WebM require ffmpeg on the server.
 */

"use strict";

// ── DOM Refs ─────────────────────────────────────────────────────────────────
const micBtn          = document.getElementById("micBtn");
const micLabel        = document.getElementById("micLabel");
const timerEl         = document.getElementById("timer");
const transcriptBox   = document.getElementById("transcriptBox");
const transcriptPH    = document.getElementById("transcriptPH");
const segmentsList    = document.getElementById("segmentsList");
const spinnerOverlay  = document.getElementById("spinnerOverlay");
const visualizerCanvas= document.getElementById("visualizer");
const dropZone        = document.getElementById("dropZone");
const fileInput       = document.getElementById("fileInput");
const modelSelect     = document.getElementById("modelSelect");
const statusDot       = document.getElementById("statusDot");
const statusLabel     = document.getElementById("statusLabel");
const progressBarWrap = document.getElementById("progressBarWrap");
const progressBar     = document.getElementById("progressBar");
const micRing         = document.getElementById("micRing");
const visIdleText     = document.getElementById("visIdleText");

const statRecordings = document.getElementById("statRecordings");
const statWords      = document.getElementById("statWords");
const statChars      = document.getElementById("statChars");
const statAvgTime    = document.getElementById("statAvgTime");

// ── State ─────────────────────────────────────────────────────────────────────
let isRecording     = false;
let timerInterval   = null;
let elapsedSecs     = 0;
let audioCtx        = null;
let analyser        = null;
let animFrame       = null;
let micStream       = null;
let scriptProcessor = null;
let pcmChunks       = [];       // raw Float32 PCM frames
const SAMPLE_RATE   = 16000;    // Whisper expects 16 kHz

let stats = { recordings:0, totalWords:0, totalChars:0, transcribeTimes:[] };

// Canvas
const ctx2d = visualizerCanvas.getContext("2d");
let canvasW, canvasH;

// ── Canvas Setup ──────────────────────────────────────────────────────────────
function resizeCanvas() {
  canvasW = visualizerCanvas.width  = visualizerCanvas.offsetWidth;
  canvasH = visualizerCanvas.height = visualizerCanvas.offsetHeight;
}
// ── Visualizer Palette (Crimson Velvet & Obsidian Red) ─────────────────────
const VIS_COLORS = {
  idle: "rgba(250, 45, 72, 0.35)",
  c1:   "250, 45, 72",
  c2:   "255, 45, 85",
  c3:   "255, 94, 98"
};

window.addEventListener("resize", resizeCanvas);
resizeCanvas();

// ── Idle sine wave ────────────────────────────────────────────────────────────
let idlePhase = 0;
function drawIdle() {
  ctx2d.clearRect(0, 0, canvasW, canvasH);
  ctx2d.beginPath();
  for (let x = 0; x <= canvasW; x++) {
    const y = canvasH/2
            + Math.sin((x + idlePhase) * 0.02) * 6
            + Math.sin((x + idlePhase*0.5) * 0.034) * 2.5;
    x === 0 ? ctx2d.moveTo(x, y) : ctx2d.lineTo(x, y);
  }
  ctx2d.strokeStyle = VIS_COLORS.idle;
  ctx2d.lineWidth = 1.8;
  ctx2d.stroke();
  idlePhase += 1.2;
  animFrame = requestAnimationFrame(drawIdle);
}

// ── Live bar visualizer ───────────────────────────────────────────────────────
function drawBars() {
  if (!analyser) return;
  const bufLen  = analyser.frequencyBinCount;
  const dataArr = new Uint8Array(bufLen);
  analyser.getByteFrequencyData(dataArr);
  ctx2d.clearRect(0, 0, canvasW, canvasH);

  const barCount = 80;
  const barWidth = (canvasW / barCount) - 2;
  for (let i = 0; i < barCount; i++) {
    const idx   = Math.floor((i / barCount) * bufLen * 0.6);
    const value = dataArr[idx] / 255;
    const barH  = value * canvasH * 0.85;
    const x     = i * (barWidth + 2);
    const y     = (canvasH - barH) / 2;
    const grad  = ctx2d.createLinearGradient(0, y, 0, y + barH);
    grad.addColorStop(0,   `rgba(${VIS_COLORS.c1}, ${0.35 + value * 0.65})`);
    grad.addColorStop(0.5, `rgba(${VIS_COLORS.c2}, ${0.25 + value * 0.65})`);
    grad.addColorStop(1,   `rgba(${VIS_COLORS.c3}, ${0.35 + value * 0.65})`);
    ctx2d.fillStyle = grad;
    ctx2d.beginPath();
    ctx2d.roundRect(x, y, barWidth, barH, 3);
    ctx2d.fill();
  }
  animFrame = requestAnimationFrame(drawBars);
}

drawIdle();   // start idle animation immediately

// ── Timer ─────────────────────────────────────────────────────────────────────
function formatTime(s) {
  const m = Math.floor(s/60).toString().padStart(2,"0");
  return `${m}:${(s%60).toString().padStart(2,"0")}`;
}
function startTimer() {
  elapsedSecs = 0; timerEl.textContent = "00:00";
  timerInterval = setInterval(() => { elapsedSecs++; timerEl.textContent = formatTime(elapsedSecs); }, 1000);
}
function stopTimer() { clearInterval(timerInterval); timerInterval = null; }

// ── Toast ─────────────────────────────────────────────────────────────────────
const toastContainer = document.getElementById("toastContainer");
function toast(message, type="info", duration=3500) {
  const icons = { success:"✅", error:"❌", info:"💡" };
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.innerHTML = `<span>${icons[type]}</span><span>${message}</span>`;
  toastContainer.appendChild(el);
  setTimeout(() => {
    el.style.cssText += ";opacity:0;transform:translateX(40px);transition:all .3s ease";
    setTimeout(() => el.remove(), 300);
  }, duration);
}
window.toast = toast;   // expose for inline HTML handler

// ── WAV Encoder ───────────────────────────────────────────────────────────────
/**
 * Encode raw Float32 PCM frames into a proper 16-bit mono WAV Blob.
 * This is sent to Flask and does NOT require ffmpeg on the server.
 */
function encodeWav(samples, sampleRate) {
  const numSamples = samples.length;
  const buffer     = new ArrayBuffer(44 + numSamples * 2);
  const view       = new DataView(buffer);

  function writeStr(off, str) {
    for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i));
  }
  function clamp(n) { return Math.max(-1, Math.min(1, n)); }

  writeStr(0,  "RIFF");
  view.setUint32(4,  36 + numSamples*2, true);
  writeStr(8,  "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);           // chunk size
  view.setUint16(20,  1, true);           // PCM
  view.setUint16(22,  1, true);           // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32,  2, true);           // block align
  view.setUint16(34, 16, true);           // bits per sample
  writeStr(36, "data");
  view.setUint32(40, numSamples*2, true);

  let offset = 44;
  for (let i = 0; i < numSamples; i++) {
    const s = clamp(samples[i]);
    view.setInt16(offset, s < 0 ? s*0x8000 : s*0x7FFF, true);
    offset += 2;
  }
  return new Blob([buffer], { type: "audio/wav" });
}

// Merge multiple Float32Arrays into one
function mergeChunks(chunks) {
  const total = chunks.reduce((n, c) => n + c.length, 0);
  const merged = new Float32Array(total);
  let off = 0;
  for (const c of chunks) { merged.set(c, off); off += c.length; }
  return merged;
}

let lastRecordedBlob     = null;
let lastRecordedFilename = "recording.wav";
let isTranscribing       = false;

// ── Microphone Recording (PCM → WAV) ─────────────────────────────────────────
async function startRecording() {
  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { sampleRate: SAMPLE_RATE, channelCount: 1, echoCancellation: true }
    });

    // AudioContext at desired sample rate
    audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SAMPLE_RATE });
    if (audioCtx.state === "suspended") {
      await audioCtx.resume();
    }
    const source = audioCtx.createMediaStreamSource(micStream);

    // Analyser for visualizer
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 256;
    source.connect(analyser);

    // ScriptProcessor to capture raw PCM
    const bufSize = 4096;
    scriptProcessor = audioCtx.createScriptProcessor(bufSize, 1, 1);
    pcmChunks = [];
    scriptProcessor.onaudioprocess = (e) => {
      const data = e.inputBuffer.getChannelData(0);
      pcmChunks.push(new Float32Array(data));   // copy
    };
    source.connect(scriptProcessor);
    scriptProcessor.connect(audioCtx.destination);

    // UI updates
    isRecording = true;
    lastRecordedBlob = null;
    micBtn.classList.add("recording");
    micRing.classList.add("active");
    micLabel.classList.add("recording");
    timerEl.classList.add("recording");
    micBtn.innerHTML = "⏹";
    micLabel.textContent = "Recording… click to stop";
    visIdleText.style.opacity = "0";
    cancelAnimationFrame(animFrame);
    drawBars();
    startTimer();
    setServerStatus("recording");

  } catch (err) {
    toast("Microphone access denied. Please allow microphone in browser settings.", "error", 5000);
    console.error(err);
  }
}

function stopRecording(autoTranscribe = false) {
  if (!scriptProcessor && !isRecording) return;

  const actualSr = audioCtx ? audioCtx.sampleRate : SAMPLE_RATE;

  // Disconnect audio nodes
  if (scriptProcessor) {
    scriptProcessor.disconnect();
    scriptProcessor.onaudioprocess = null;
  }
  if (micStream) micStream.getTracks().forEach(t => t.stop());
  if (audioCtx) { audioCtx.close(); audioCtx = null; }
  analyser = null;
  scriptProcessor = null;

  // Restore UI
  isRecording = false;
  micBtn.classList.remove("recording");
  micRing.classList.remove("active");
  micLabel.classList.remove("recording");
  timerEl.classList.remove("recording");
  micBtn.innerHTML = "🎙";
  stopTimer();
  cancelAnimationFrame(animFrame);
  visIdleText.style.opacity = "1";
  drawIdle();

  // Encode
  if (pcmChunks.length === 0) {
    toast("No audio captured.", "error");
    micLabel.textContent = "Click to record";
    return;
  }
  const merged = mergeChunks(pcmChunks);
  lastRecordedBlob = encodeWav(merged, actualSr);
  lastRecordedFilename = "recording.wav";
  pcmChunks = [];

  const recordedDuration = timerEl.textContent;
  micLabel.textContent = `Recorded (${recordedDuration}) · Click ✨ Transcribe`;

  if (autoTranscribe) {
    transcribeBlob(lastRecordedBlob, lastRecordedFilename);
  } else {
    toast(`Recording saved (${recordedDuration}). Click "✨ Transcribe" to proceed.`, "info", 4000);
  }
}

// ── Mic Button ────────────────────────────────────────────────────────────────
micBtn.addEventListener("click", () => {
  if (isRecording) {
    stopRecording(false);
  } else {
    startRecording();
  }
});

// ── "✨ Transcribe" button ────────────────────────────────────────────────────
document.getElementById("transcribeNowBtn").addEventListener("click", () => {
  if (isRecording) {
    stopRecording(true);
  } else if (lastRecordedBlob) {
    transcribeBlob(lastRecordedBlob, lastRecordedFilename);
  } else {
    toast("Record your voice or upload a file first.", "info");
  }
});

// ── "🗑 Discard" button ────────────────────────────────────────────────────────
document.getElementById("discardBtn").addEventListener("click", () => {
  if (isRecording) {
    if (scriptProcessor) {
      scriptProcessor.disconnect();
      scriptProcessor.onaudioprocess = null;
    }
    if (micStream) micStream.getTracks().forEach(t => t.stop());
    if (audioCtx) { audioCtx.close(); audioCtx = null; }
    analyser = null;
    scriptProcessor = null;
    isRecording = false;
  }
  pcmChunks = [];
  lastRecordedBlob = null;
  micBtn.classList.remove("recording");
  micRing.classList.remove("active");
  micLabel.classList.remove("recording");
  timerEl.classList.remove("recording");
  timerEl.textContent = "00:00";
  micBtn.innerHTML = "🎙";
  micLabel.textContent = "Click to record";
  stopTimer();
  cancelAnimationFrame(animFrame);
  visIdleText.style.opacity = "1";
  drawIdle();
  setServerStatus("ready");
  toast("Recording discarded.", "info");
});

// ── File Upload / Drag-Drop ───────────────────────────────────────────────────
dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("dragover",  (e) => { e.preventDefault(); dropZone.classList.add("drag-over"); });
dropZone.addEventListener("dragleave", ()  => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault(); dropZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) handleFileUpload(file);
});
fileInput.addEventListener("change", () => { if (fileInput.files[0]) handleFileUpload(fileInput.files[0]); });

async function handleFileUpload(file) {
  const ok = /\.(wav|mp3|webm|ogg|flac|m4a)$/i.test(file.name) || file.type.startsWith("audio/");
  if (!ok) { toast("Unsupported file type. Please upload an audio file.", "error"); return; }
  lastRecordedBlob = file;
  lastRecordedFilename = file.name;
  micLabel.textContent = `File ready: ${file.name}`;
  toast(`Uploading "${file.name}"…`, "info");
  await transcribeBlob(file, file.name);
}

// ── Core Transcription ────────────────────────────────────────────────────────
async function transcribeBlob(blob, filename) {
  if (isTranscribing) {
    toast("Transcription is already running…", "info");
    return;
  }
  isTranscribing = true;
  showSpinner(true);
  showProgress(true);
  animateProgress();
  setServerStatus("transcribing");

  const formData = new FormData();
  formData.append("audio", blob, filename);
  const selectedModel = modelSelect ? modelSelect.value : "base";
  formData.append("model", selectedModel);

  try {
    const res  = await fetch("/transcribe", { method:"POST", body: formData });
    const data = await res.json();

    if (!res.ok || data.error) throw new Error(data.error || "Transcription failed");

    renderTranscript(data);
    updateStats(data);
    toast("Transcription complete!", "success");

  } catch (err) {
    toast(`Error: ${err.message}`, "error", 6000);
    console.error(err);
  } finally {
    isTranscribing = false;
    showSpinner(false);
    showProgress(false);
    setServerStatus("ready");
  }
}

// ── Render Results ────────────────────────────────────────────────────────────
function renderTranscript(data) {
  const text = data.transcript || "";
  transcriptBox.textContent = text;
  transcriptPH.style.display = text ? "none" : "flex";
  transcriptBox.classList.toggle("has-text", !!text);

  document.getElementById("badgeLang").textContent  = `🌐 ${(data.language || "?").toUpperCase()}`;
  document.getElementById("badgeTime").textContent  = `⏱ ${data.duration_s}s`;
  document.getElementById("badgeWords").textContent = `📝 ${countWords(text)} words`;
  document.getElementById("badgeModel").textContent = `🤖 whisper-${data.model || "base"}`;
  document.getElementById("transcriptMeta").style.display = "flex";

  segmentsList.innerHTML = "";
  if (data.segments && data.segments.length > 0) {
    data.segments.forEach((seg, i) => {
      const item = document.createElement("div");
      item.className = "segment-item";
      item.style.animationDelay = `${i * 0.04}s`;
      item.innerHTML = `<span class="seg-time">${formatTime2(seg.start)} → ${formatTime2(seg.end)}</span>
                        <span class="seg-text">${escapeHtml(seg.text)}</span>`;
      segmentsList.appendChild(item);
    });
  } else {
    segmentsList.innerHTML = `<div class="no-segments">No segment timeline available</div>`;
  }
}

function formatTime2(secs) {
  const m = Math.floor(secs/60).toString().padStart(2,"0");
  const s = Math.floor(secs%60).toString().padStart(2,"0");
  return `${m}:${s}`;
}

function countWords(text) { return text ? text.trim().split(/\s+/).filter(Boolean).length : 0; }
function escapeHtml(str)  { return str.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }

// ── Stats ─────────────────────────────────────────────────────────────────────
function updateStats(data) {
  stats.recordings++;
  const wc = countWords(data.transcript || "");
  stats.totalWords += wc;
  stats.totalChars += (data.transcript || "").length;
  stats.transcribeTimes.push(data.duration_s || 0);
  const avg = (stats.transcribeTimes.reduce((a,b)=>a+b,0)/stats.transcribeTimes.length).toFixed(1);
  statRecordings.textContent = stats.recordings;
  statWords.textContent      = stats.totalWords;
  statChars.textContent      = stats.totalChars;
  statAvgTime.textContent    = `${avg}s`;
}

// ── Spinner ───────────────────────────────────────────────────────────────────
function showSpinner(show) { spinnerOverlay.classList.toggle("active", show); }

// ── Progress Bar ──────────────────────────────────────────────────────────────
let progressInterval = null;
function showProgress(show) {
  progressBarWrap.style.display = show ? "block" : "none";
  if (!show) { progressBar.style.width = "0%"; clearInterval(progressInterval); }
}
function animateProgress() {
  let pct = 0; clearInterval(progressInterval);
  progressInterval = setInterval(() => {
    pct += Math.random() * 8;
    if (pct >= 90) { clearInterval(progressInterval); pct = 90; }
    progressBar.style.width = `${pct}%`;
  }, 300);
}

// ── Status ────────────────────────────────────────────────────────────────────
function setServerStatus(state) {
  const map = {
    ready:        { color:"var(--green)", label:"Ready",         dot:"var(--green)"  },
    recording:    { color:"var(--red)",   label:"Recording…",    dot:"var(--red)"    },
    transcribing: { color:"var(--amber)", label:"Transcribing…", dot:"var(--amber)"  },
  };
  const s = map[state] || map.ready;
  statusDot.style.background = s.dot;
  statusLabel.textContent    = s.label;
  const pill = document.querySelector(".status-pill");
  pill.style.borderColor = `${s.color}44`;
  pill.style.color       = s.color;
}

// ── Clipboard ─────────────────────────────────────────────────────────────────
document.getElementById("copyBtn").addEventListener("click", () => {
  const text = transcriptBox.textContent.trim();
  if (!text) { toast("Nothing to copy yet!", "info"); return; }
  navigator.clipboard.writeText(text).then(() => toast("Copied to clipboard!", "success"));
});

// ── Download TXT ──────────────────────────────────────────────────────────────
document.getElementById("downloadBtn").addEventListener("click", () => {
  const text = transcriptBox.textContent.trim();
  if (!text) { toast("Nothing to download yet!", "info"); return; }
  const a = Object.assign(document.createElement("a"), {
    href: URL.createObjectURL(new Blob([text], { type:"text/plain" })),
    download: "transcript.txt"
  });
  a.click(); URL.revokeObjectURL(a.href);
  toast("Transcript downloaded!", "success");
});

// ── Export JSON ───────────────────────────────────────────────────────────────
document.getElementById("jsonBtn").addEventListener("click", () => {
  const text = transcriptBox.textContent.trim();
  if (!text) { toast("No transcript to export!", "info"); return; }
  const segments = [...document.querySelectorAll(".segment-item")].map(el => ({
    time: el.querySelector(".seg-time").textContent,
    text: el.querySelector(".seg-text").textContent,
  }));
  const a = Object.assign(document.createElement("a"), {
    href: URL.createObjectURL(new Blob([JSON.stringify({transcript:text, segments}, null, 2)], {type:"application/json"})),
    download: "transcript.json"
  });
  a.click(); URL.revokeObjectURL(a.href);
  toast("JSON downloaded!", "success");
});

// ── Clear ─────────────────────────────────────────────────────────────────────
document.getElementById("clearBtn").addEventListener("click", () => {
  transcriptBox.textContent = "";
  transcriptPH.style.display = "flex";
  transcriptBox.classList.remove("has-text");
  segmentsList.innerHTML = `<div class="no-segments">No segments yet</div>`;
  ["badgeLang","badgeTime","badgeWords","badgeModel"].forEach(id => {
    document.getElementById(id).textContent = id === "badgeLang" ? "🌐 –"
      : id === "badgeTime" ? "⏱ –" : id === "badgeWords" ? "📝 –" : "🤖 –";
  });
  toast("Cleared.", "info");
});

// ── Discard ───────────────────────────────────────────────────────────────────
document.getElementById("discardBtn").addEventListener("click", () => {
  if (isRecording) stopRecording();
  pcmChunks = [];
  toast("Recording discarded.", "info");
});

// ── Health check on load ──────────────────────────────────────────────────────
(async () => {
  try {
    const res  = await fetch("/health");
    const data = await res.json();
    if (res.ok) {
      setServerStatus("ready");
      const ffmpegNote = data.ffmpeg ? "" : " (WAV only — install ffmpeg for MP3/WebM)";
      toast(`Whisper "${data.model}" loaded ✓${ffmpegNote}`, "success", 5000);
    }
  } catch { setServerStatus("ready"); }
})();
