/**
 * VoiceScribe – Main JavaScript
 * Handles: MediaRecorder, AudioContext visualizer, Fetch to /transcribe,
 * file drag-drop upload, clipboard copy, download, history stats.
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

// Stats
const statRecordings = document.getElementById("statRecordings");
const statWords      = document.getElementById("statWords");
const statChars      = document.getElementById("statChars");
const statAvgTime    = document.getElementById("statAvgTime");

// ── State ─────────────────────────────────────────────────────────────────────
let mediaRecorder = null;
let audioChunks   = [];
let isRecording   = false;
let timerInterval = null;
let elapsedSecs   = 0;
let audioCtx      = null;
let analyser      = null;
let animFrame     = null;

// Session stats
let stats = {
  recordings: 0,
  totalWords: 0,
  totalChars: 0,
  transcribeTimes: [],
};

// Canvas
const ctx2d = visualizerCanvas.getContext("2d");
let canvasW, canvasH;

// ── Canvas Setup ──────────────────────────────────────────────────────────────
function resizeCanvas() {
  canvasW = visualizerCanvas.width  = visualizerCanvas.offsetWidth;
  canvasH = visualizerCanvas.height = visualizerCanvas.offsetHeight;
}

window.addEventListener("resize", resizeCanvas);
resizeCanvas();

// ── Idle waveform (decorative sine) ──────────────────────────────────────────
let idlePhase = 0;

function drawIdle() {
  ctx2d.clearRect(0, 0, canvasW, canvasH);
  ctx2d.beginPath();
  const amp = 6;
  const freq = 0.02;
  for (let x = 0; x <= canvasW; x++) {
    const y = canvasH / 2 + Math.sin((x + idlePhase) * freq) * amp
                           + Math.sin((x + idlePhase * 0.5) * freq * 1.7) * (amp * 0.4);
    x === 0 ? ctx2d.moveTo(x, y) : ctx2d.lineTo(x, y);
  }
  ctx2d.strokeStyle = "rgba(0,229,255,0.18)";
  ctx2d.lineWidth = 1.5;
  ctx2d.stroke();
  idlePhase += 1.2;
  animFrame = requestAnimationFrame(drawIdle);
}

// ── Live bar visualizer ───────────────────────────────────────────────────────
function drawBars() {
  if (!analyser) return;
  const bufLen = analyser.frequencyBinCount;
  const dataArr = new Uint8Array(bufLen);
  analyser.getByteFrequencyData(dataArr);

  ctx2d.clearRect(0, 0, canvasW, canvasH);

  const barCount = 80;
  const barWidth = (canvasW / barCount) - 2;

  for (let i = 0; i < barCount; i++) {
    const idx = Math.floor((i / barCount) * bufLen * 0.6);
    const value = dataArr[idx] / 255;
    const barH = value * canvasH * 0.85;
    const x = i * (barWidth + 2);
    const y = (canvasH - barH) / 2;

    // Gradient bar
    const grad = ctx2d.createLinearGradient(0, y, 0, y + barH);
    grad.addColorStop(0, `rgba(0,229,255,${0.3 + value * 0.7})`);
    grad.addColorStop(0.5, `rgba(179,107,255,${0.2 + value * 0.6})`);
    grad.addColorStop(1, `rgba(0,229,255,${0.3 + value * 0.7})`);
    ctx2d.fillStyle = grad;
    ctx2d.beginPath();
    ctx2d.roundRect(x, y, barWidth, barH, 3);
    ctx2d.fill();
  }

  animFrame = requestAnimationFrame(drawBars);
}

// Start idle animation on load
drawIdle();

// ── Timer ─────────────────────────────────────────────────────────────────────
function formatTime(s) {
  const m = Math.floor(s / 60).toString().padStart(2, "0");
  const sec = (s % 60).toString().padStart(2, "0");
  return `${m}:${sec}`;
}

function startTimer() {
  elapsedSecs = 0;
  timerEl.textContent = "00:00";
  timerInterval = setInterval(() => {
    elapsedSecs++;
    timerEl.textContent = formatTime(elapsedSecs);
  }, 1000);
}

function stopTimer() {
  clearInterval(timerInterval);
  timerInterval = null;
}

// ── Toast Notifications ───────────────────────────────────────────────────────
const toastContainer = document.getElementById("toastContainer");

function toast(message, type = "info", duration = 3500) {
  const icons = { success: "✅", error: "❌", info: "💡" };
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.innerHTML = `<span>${icons[type]}</span><span>${message}</span>`;
  toastContainer.appendChild(el);
  setTimeout(() => {
    el.style.animation = "none";
    el.style.opacity = "0";
    el.style.transform = "translateX(40px)";
    el.style.transition = "all 0.3s ease";
    setTimeout(() => el.remove(), 300);
  }, duration);
}

// ── Microphone Recording ──────────────────────────────────────────────────────
async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    // Audio context + analyser
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioCtx.createMediaStreamSource(stream);
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 256;
    source.connect(analyser);

    // MediaRecorder
    mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
    audioChunks = [];
    mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) audioChunks.push(e.data); };
    mediaRecorder.onstop = handleRecordingStop;
    mediaRecorder.start(100);

    // UI
    isRecording = true;
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
    toast("Microphone access denied. Please allow microphone access.", "error");
    console.error(err);
  }
}

function stopRecording() {
  if (!mediaRecorder) return;
  mediaRecorder.stop();
  mediaRecorder.stream.getTracks().forEach((t) => t.stop());

  isRecording = false;
  micBtn.classList.remove("recording");
  micRing.classList.remove("active");
  micLabel.classList.remove("recording");
  timerEl.classList.remove("recording");
  micBtn.innerHTML = "🎙";
  micLabel.textContent = "Click to record";
  stopTimer();

  cancelAnimationFrame(animFrame);
  if (audioCtx) { audioCtx.close(); audioCtx = null; analyser = null; }
  visIdleText.style.opacity = "1";
  drawIdle();
}

async function handleRecordingStop() {
  const blob = new Blob(audioChunks, { type: "audio/webm" });
  setServerStatus("transcribing");
  await transcribeBlob(blob, "recording.webm");
  setServerStatus("ready");
}

// ── Mic Button Toggle ─────────────────────────────────────────────────────────
micBtn.addEventListener("click", () => {
  isRecording ? stopRecording() : startRecording();
});

// ── File Upload / Drag-Drop ───────────────────────────────────────────────────
dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) handleFileUpload(file);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handleFileUpload(fileInput.files[0]);
});

async function handleFileUpload(file) {
  const allowed = ["audio/wav", "audio/mp3", "audio/mpeg", "audio/webm",
                   "audio/ogg", "audio/flac", "audio/x-wav", "audio/mp4"];
  if (!allowed.includes(file.type) && !file.name.match(/\.(wav|mp3|webm|ogg|flac|m4a)$/i)) {
    toast("Unsupported file type. Please upload an audio file.", "error");
    return;
  }
  toast(`Uploading "${file.name}"…`, "info");
  await transcribeBlob(file, file.name);
}

// ── Core Transcription ────────────────────────────────────────────────────────
async function transcribeBlob(blob, filename) {
  showSpinner(true);
  showProgress(true);
  animateProgress();

  const formData = new FormData();
  formData.append("audio", blob, filename);

  try {
    const res = await fetch("/transcribe", { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok || data.error) {
      throw new Error(data.error || "Transcription failed");
    }

    renderTranscript(data);
    updateStats(data);
    toast("Transcription complete!", "success");

  } catch (err) {
    toast(`Error: ${err.message}`, "error");
    console.error(err);
  } finally {
    showSpinner(false);
    showProgress(false);
  }
}

// ── Render Results ────────────────────────────────────────────────────────────
function renderTranscript(data) {
  const text = data.transcript || "";

  // Transcript box
  transcriptBox.textContent = text;
  transcriptPH.style.display = text ? "none" : "flex";
  transcriptBox.classList.toggle("has-text", !!text);

  // Metadata badges
  document.getElementById("badgeLang").textContent  = `🌐 ${(data.language || "?").toUpperCase()}`;
  document.getElementById("badgeTime").textContent  = `⏱ ${data.duration_s}s`;
  document.getElementById("badgeWords").textContent = `📝 ${countWords(text)} words`;
  document.getElementById("badgeModel").textContent = `🤖 whisper-${data.model || "base"}`;
  document.getElementById("transcriptMeta").style.display = "flex";

  // Segments
  segmentsList.innerHTML = "";
  if (data.segments && data.segments.length > 0) {
    data.segments.forEach((seg, i) => {
      const item = document.createElement("div");
      item.className = "segment-item";
      item.style.animationDelay = `${i * 0.04}s`;
      item.innerHTML = `
        <span class="seg-time">${formatTime2(seg.start)} → ${formatTime2(seg.end)}</span>
        <span class="seg-text">${escapeHtml(seg.text)}</span>`;
      segmentsList.appendChild(item);
    });
  } else {
    segmentsList.innerHTML = `<div class="no-segments">No segment timeline available</div>`;
  }
}

function formatTime2(secs) {
  const m = Math.floor(secs / 60).toString().padStart(2, "0");
  const s = Math.floor(secs % 60).toString().padStart(2, "0");
  return `${m}:${s}`;
}

function countWords(text) {
  return text ? text.trim().split(/\s+/).filter(Boolean).length : 0;
}

function escapeHtml(str) {
  return str.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

// ── Stats ─────────────────────────────────────────────────────────────────────
function updateStats(data) {
  stats.recordings++;
  const wc = countWords(data.transcript || "");
  stats.totalWords += wc;
  stats.totalChars += (data.transcript || "").length;
  stats.transcribeTimes.push(data.duration_s || 0);

  const avgTime = (stats.transcribeTimes.reduce((a, b) => a + b, 0) / stats.transcribeTimes.length).toFixed(1);

  statRecordings.textContent = stats.recordings;
  statWords.textContent      = stats.totalWords;
  statChars.textContent      = stats.totalChars;
  statAvgTime.textContent    = `${avgTime}s`;
}

// ── Spinner ───────────────────────────────────────────────────────────────────
function showSpinner(show) {
  spinnerOverlay.classList.toggle("active", show);
}

// ── Progress Bar ──────────────────────────────────────────────────────────────
let progressInterval = null;

function showProgress(show) {
  progressBarWrap.style.display = show ? "block" : "none";
  if (!show) {
    progressBar.style.width = "0%";
    clearInterval(progressInterval);
  }
}

function animateProgress() {
  let pct = 0;
  clearInterval(progressInterval);
  progressInterval = setInterval(() => {
    pct += Math.random() * 8;
    if (pct >= 90) { clearInterval(progressInterval); pct = 90; }
    progressBar.style.width = `${pct}%`;
  }, 300);
}

// ── Status Indicator ──────────────────────────────────────────────────────────
function setServerStatus(state) {
  const states = {
    ready:        { color: "var(--green)",  label: "Ready",        dotBg: "var(--green)" },
    recording:    { color: "var(--red)",    label: "Recording…",   dotBg: "var(--red)"   },
    transcribing: { color: "var(--amber)",  label: "Transcribing…",dotBg: "var(--amber)" },
  };
  const s = states[state] || states.ready;
  statusDot.style.background = s.dotBg;
  statusLabel.textContent = s.label;
  document.querySelector(".status-pill").style.borderColor = `${s.color}44`;
  document.querySelector(".status-pill").style.color = s.color;
}

// ── Clipboard Copy ────────────────────────────────────────────────────────────
document.getElementById("copyBtn").addEventListener("click", () => {
  const text = transcriptBox.textContent.trim();
  if (!text) { toast("Nothing to copy yet!", "info"); return; }
  navigator.clipboard.writeText(text).then(() => toast("Copied to clipboard!", "success"));
});

// ── Download TXT ──────────────────────────────────────────────────────────────
document.getElementById("downloadBtn").addEventListener("click", () => {
  const text = transcriptBox.textContent.trim();
  if (!text) { toast("Nothing to download yet!", "info"); return; }
  const blob = new Blob([text], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = "transcript.txt"; a.click();
  URL.revokeObjectURL(url);
  toast("Transcript downloaded!", "success");
});

// ── Clear Transcript ──────────────────────────────────────────────────────────
document.getElementById("clearBtn").addEventListener("click", () => {
  transcriptBox.textContent = "";
  transcriptPH.style.display = "flex";
  transcriptBox.classList.remove("has-text");
  segmentsList.innerHTML = `<div class="no-segments">No segments yet</div>`;
  document.getElementById("badgeLang").textContent  = "🌐 –";
  document.getElementById("badgeTime").textContent  = "⏱ –";
  document.getElementById("badgeWords").textContent = "📝 –";
  document.getElementById("badgeModel").textContent = "🤖 –";
  toast("Cleared.", "info");
});

// ── Download JSON ─────────────────────────────────────────────────────────────
document.getElementById("jsonBtn").addEventListener("click", async () => {
  const text = transcriptBox.textContent.trim();
  if (!text) { toast("No transcript to export!", "info"); return; }
  const segments = [];
  document.querySelectorAll(".segment-item").forEach((el) => {
    segments.push({
      time: el.querySelector(".seg-time").textContent,
      text: el.querySelector(".seg-text").textContent,
    });
  });
  const payload = { transcript: text, segments };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = "transcript.json"; a.click();
  URL.revokeObjectURL(url);
  toast("JSON downloaded!", "success");
});

// ── Init health check ─────────────────────────────────────────────────────────
(async () => {
  try {
    const res = await fetch("/health");
    if (res.ok) {
      const data = await res.json();
      setServerStatus("ready");
      toast(`Whisper model "${data.model}" loaded ✓`, "success", 4000);
    }
  } catch {
    setServerStatus("ready");
  }
})();
