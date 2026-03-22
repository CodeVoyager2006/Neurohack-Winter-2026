/**
 * app.js  –  Main Application Controller
 * ========================================
 * Orchestrates the entire lifecycle of the facial mapping application:
 *
 *   1. Side selection  (setup panel)
 *   2. Webcam initialisation
 *   3. Calibration     (accumulate N stable landmark frames)
 *   4. Live loop       (send frames → server → render)
 *   5. Expression button state management
 *   6. EEG prediction polling
 *
 * ── Position-lock design (KEY FIX) ──────────────────────────────────
 *
 * The projected (affected) side must ONLY move in response to intentional
 * control signals — EEG prediction or manual expression buttons.
 * It must NOT mirror any uncontrolled movement of the healthy side.
 *
 * To enforce this, THREE landmark objects are maintained after calibration:
 *
 *   State.calibrated        — frozen average snapshot taken at calibration.
 *                             Never updated after calibration is complete.
 *                             Dual role:
 *                               1. BASE POSITION for the overlay (the
 *                                  projected side always starts here).
 *                               2. SCALE REFERENCE for expression offsets
 *                                  (e.g. "lift brow by 1.5× the eye-height
 *                                  measured at calibration").
 *
 *   State.liveLandmarks     — updated every frame from the healthy-side
 *                             camera. Used ONLY for head-tilt/translation
 *                             correction (global rigid movement), NOT for
 *                             expression state.
 *
 *   State.positionLocked    — boolean flag. When true (default after
 *                             calibration), the overlay base position is
 *                             State.calibrated, NOT State.liveLandmarks.
 *                             This is the "affected side boolean" described
 *                             in the design: the projected side is locked
 *                             to neutral calibration position and can only
 *                             move via EEG or manual button.
 *
 * Result:
 *   - Healthy side twitches / blinks → overlay does NOT move  ✓
 *   - Patient moves head (rotation/translation) → overlay follows  ✓ *
 *   - EEG fires "lookup" → overlay raises brow  ✓
 *   - Manual button pressed → overlay changes expression  ✓
 *
 * * Head-tracking correction uses a global offset computed from the
 *   iris centre shift between calibrated and live positions, so only
 *   rigid head movement is tracked, not facial muscle activity.
 */

"use strict";

// ── Configuration ────────────────────────────────────────────────────
const API_URL            = "/api/process-frame";
const EEG_API_URL        = "/api/eeg-predict";
const EEG_PORTS_URL      = "/api/eeg-ports";
const EEG_CONNECT_URL    = "/api/eeg-connect";
const FRAME_INTERVAL_MS  = 80;    // ~12 fps polling rate to server
const CALIBRATION_FRAMES = 10;    // frames averaged before going live
const JPEG_QUALITY       = 0.6;   // trade-off between speed and accuracy
const EEG_POLL_MS        = 300;   // new EEG sample every 300 ms (~3 Hz)

// ── Application State ────────────────────────────────────────────────
const State = {
  /** "left" or "right" — which side to black out and map */
  mappedSide: null,

  /** The SOURCE side (the opposite of mappedSide) */
  sourceSide: null,

  /** Currently active expression name */
  expression: "neutral",

  /**
   * Frozen calibration snapshot.
   * Set once when calibration completes, never updated afterwards.
   * Acts as BOTH the overlay base position AND the expression scale ref.
   */
  calibrated: null,

  /**
   * Live landmarks from the most recent healthy-side camera frame.
   * Used ONLY to extract global head-movement delta (iris centre shift).
   * Never used directly as the overlay base position.
   */
  liveLandmarks: null,

  /** Whether we are still in the calibration accumulation phase. */
  isCalibrating: true,

  /** Accumulation buffer for calibration averaging. */
  calibrationBuffer: [],

  /**
   * POSITION LOCK — the core boolean fix.
   *
   * true  (default after calibration):
   *   The overlay for the affected side is drawn from State.calibrated
   *   as its base position. Only expression offsets (from EEG or manual
   *   buttons) can move the projected landmarks. Healthy-side muscle
   *   movement has zero effect on the overlay.
   *
   * false:
   *   Legacy behaviour — overlay follows live healthy-side landmarks.
   *   Retained for debugging / comparison only; not exposed in the UI.
   */
  positionLocked: true,

  /** Whether the animation loop is running. */
  loopRunning: false,

  /** Interval handle for the camera polling loop. */
  loopHandle: null,

  /**
   * EEG mode:
   *   "manual" — expression buttons control the projected side.
   *   "auto"   — EEG model prediction drives setExpression().
   */
  eegMode: "manual",

  /** Interval handle for the EEG polling loop. */
  eegLoopHandle: null,

  /** True once the board has successfully connected at least once this session. */
  eegEverConnected: false,
};

// ── DOM References ───────────────────────────────────────────────────
const DOM = {
  setupPanel:       document.getElementById("setup-panel"),
  visualiserPanel:  document.getElementById("visualiser-panel"),
  webcam:           document.getElementById("webcam"),
  canvas:           document.getElementById("main-canvas"),
  noFaceBadge:      document.getElementById("no-face-badge"),
  calibBadge:       document.getElementById("calibrating-badge"),
  activeSideLabel:  document.getElementById("active-side-label"),
  recalibrateBtn:   document.getElementById("recalibrate-btn"),
  changeSideBtn:    document.getElementById("change-side-btn"),
  sideButtons:      document.querySelectorAll(".side-btn"),
  exprButtons:      document.querySelectorAll(".expr-btn"),
  // EEG panel
  eegModeBtn:       document.getElementById("eeg-mode-btn"),
  eegPortSelect:    document.getElementById("eeg-port-select"),
  eegConnectBtn:    document.getElementById("eeg-connect-btn"),
  eegConnectError:  document.getElementById("eeg-connect-error"),
  eegCsvIndex:      document.getElementById("eeg-csv-index"),
  eegSession:       document.getElementById("eeg-session"),
  eegWindow:        document.getElementById("eeg-window"),
  eegTrueLabel:     document.getElementById("eeg-true-label"),
  eegPrediction:    document.getElementById("eeg-prediction"),
  eegConfidence:    document.getElementById("eeg-confidence"),
  eegExpression:    document.getElementById("eeg-expression"),
  eegMatch:         document.getElementById("eeg-match"),
  eegStatusLabel:   document.getElementById("eeg-status-label"),
};

// ── Initialisation ───────────────────────────────────────────────────

function initEventListeners() {

  // Side selection
  DOM.sideButtons.forEach(btn => {
    btn.addEventListener("click", () => startSession(btn.dataset.side));
  });

  // Manual expression buttons
  DOM.exprButtons.forEach(btn => {
    btn.addEventListener("click", () => setExpression(btn.dataset.expr));
  });

  // EEG auto mode toggle
  DOM.eegModeBtn.addEventListener("click", toggleEEGMode);

  // EEG board connect button
  DOM.eegConnectBtn.addEventListener("click", connectBoard);

  // Recalibrate (same side)
  DOM.recalibrateBtn.addEventListener("click", resetCalibration);

  // Change side → back to setup
  DOM.changeSideBtn.addEventListener("click", () => {
    stopLoop();
    stopEEGLoop();
    showSetupPanel();
  });

  // Populate port dropdown on load
  loadPortList();
}

// ── Board connection ─────────────────────────────────────────────────

async function loadPortList() {
  try {
    const resp  = await fetch(EEG_PORTS_URL);
    const data  = await resp.json();
    const ports = data.ports || [];

    // Clear existing options except "Auto-detect"
    while (DOM.eegPortSelect.options.length > 1) {
      DOM.eegPortSelect.remove(1);
    }

    ports.forEach(p => {
      const opt   = document.createElement("option");
      opt.value   = p.device;
      // Highlight FTDI (OpenBCI dongle) ports
      const isFtdi = p.description.toLowerCase().includes("usb serial") ||
                     p.description.toLowerCase().includes("ftdi");
      opt.textContent = isFtdi ? `★ ${p.device}` : p.device;
      if (isFtdi && DOM.eegPortSelect.value === "") {
        opt.selected = true;   // pre-select likely Cyton port
      }
      DOM.eegPortSelect.appendChild(opt);
    });
  } catch (_) { /* silently ignore if server not up yet */ }
}

async function connectBoard() {
  DOM.eegConnectBtn.textContent = "Connecting…";
  DOM.eegConnectBtn.classList.add("connecting");
  DOM.eegConnectError.classList.add("hidden");

  const port = DOM.eegPortSelect.value || null;  // null → server auto-detects

  try {
    const resp = await fetch(EEG_CONNECT_URL, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ port }),
    });
    const data = await resp.json();

    if (data.success) {
      DOM.eegConnectBtn.textContent = "Connected";
      DOM.eegConnectError.classList.add("hidden");
      // Auto-enable EEG mode so projections start moving immediately
      if (State.eegMode !== "auto") toggleEEGMode();
    } else {
      DOM.eegConnectBtn.textContent = "Retry";
      DOM.eegConnectError.textContent = data.error || "Connection failed";
      DOM.eegConnectError.classList.remove("hidden");
    }
  } catch (err) {
    DOM.eegConnectBtn.textContent = "Retry";
    DOM.eegConnectError.textContent = "Server unreachable";
    DOM.eegConnectError.classList.remove("hidden");
  } finally {
    DOM.eegConnectBtn.classList.remove("connecting");
  }
}

// ── Panel switching ──────────────────────────────────────────────────

function showSetupPanel() {
  DOM.setupPanel.classList.remove("hidden");
  DOM.visualiserPanel.classList.add("hidden");
}

function showVisualiserPanel() {
  DOM.setupPanel.classList.add("hidden");
  DOM.visualiserPanel.classList.remove("hidden");
}

// ── Session management ───────────────────────────────────────────────

async function startSession(mappedSide) {
  State.mappedSide    = mappedSide;
  State.sourceSide    = mappedSide === "left" ? "right" : "left";
  State.positionLocked = true;   // enforce position lock from the start

  DOM.activeSideLabel.textContent =
    `Affected: ${mappedSide}  ·  Source: ${State.sourceSide}  ·  Position locked`;

  showVisualiserPanel();
  setExpression("neutral");

  const webcamOk = await startWebcam();
  if (!webcamOk) return;

  Renderer.init(DOM.canvas, DOM.webcam);
  resetCalibration();
  startLoop();
  startEEGLoop();
}

// ── Webcam ───────────────────────────────────────────────────────────

async function startWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      audio: false,
    });

    DOM.webcam.srcObject = stream;
    DOM.webcam.style.transform = "scaleX(-1)";

    await new Promise(resolve => { DOM.webcam.onloadedmetadata = resolve; });
    return true;
  } catch (err) {
    console.error("Webcam error:", err);
    alert("Could not access webcam.\n\nPlease allow camera permissions and reload the page.");
    showSetupPanel();
    return false;
  }
}

// ── Calibration ──────────────────────────────────────────────────────

function resetCalibration() {
  State.calibrated        = null;
  State.liveLandmarks     = null;
  State.calibrationBuffer = [];
  State.isCalibrating     = true;

  DOM.calibBadge.classList.remove("hidden");
  DOM.noFaceBadge.classList.add("hidden");
}

/**
 * Accumulate one healthy-side frame into the calibration buffer.
 * Once CALIBRATION_FRAMES have been collected, freeze State.calibrated.
 * This frozen snapshot becomes the permanent base position for the
 * affected-side overlay — it will not change again until Recalibrate.
 */
function accumulateCalibration(landmarks) {
  State.calibrationBuffer.push(landmarks);

  if (State.calibrationBuffer.length >= CALIBRATION_FRAMES) {
    State.calibrated    = averageLandmarks(State.calibrationBuffer);
    State.isCalibrating = false;
    DOM.calibBadge.classList.add("hidden");
  }
}

function averageLandmarks(buffer) {
  const n = buffer.length;

  function avgPoints(arrays) {
    return arrays[0].map((_, i) => ({
      x: arrays.reduce((s, arr) => s + arr[i].x, 0) / n,
      y: arrays.reduce((s, arr) => s + arr[i].y, 0) / n,
    }));
  }

  function avgPoint(points) {
    return {
      x: points.reduce((s, p) => s + p.x, 0) / n,
      y: points.reduce((s, p) => s + p.y, 0) / n,
    };
  }

  return {
    eyebrow: avgPoints(buffer.map(lm => lm.eyebrow)),
    eye:     avgPoints(buffer.map(lm => lm.eye)),
    iris:    avgPoint(buffer.map(lm => lm.iris)),
    mouth:   avgPoints(buffer.map(lm => lm.mouth)),
  };
}

// ── Polling loop ─────────────────────────────────────────────────────

function startLoop() {
  if (State.loopRunning) return;
  State.loopRunning = true;
  State.loopHandle  = setInterval(pollFrame, FRAME_INTERVAL_MS);
}

function stopLoop() {
  if (!State.loopRunning) return;
  clearInterval(State.loopHandle);
  State.loopRunning = false;
}

/**
 * Main render loop — runs every FRAME_INTERVAL_MS.
 *
 * Camera frames are still sent to MediaPipe every tick so that:
 *   1. Calibration can accumulate frames.
 *   2. Global head movement (iris centre shift) can be tracked for
 *      rigid-body compensation of the overlay position.
 *
 * However, the overlay BASE POSITION is determined by positionLocked:
 *
 *   positionLocked = true  →  base = State.calibrated  (affected side
 *                              stays at neutral pose; only expression
 *                              offsets from EEG/button move it)
 *
 *   positionLocked = false →  base = State.liveLandmarks  (legacy mode)
 */
async function pollFrame() {
  const frameDataUrl = captureFrame();
  if (!frameDataUrl) return;

  try {
    const resp = await fetch(API_URL, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ image: frameDataUrl, side: State.sourceSide }),
    });

    const data = await resp.json();

    if (data.success) {
      DOM.noFaceBadge.classList.add("hidden");

      if (State.isCalibrating) {
        accumulateCalibration(data.landmarks);
      } else {
        // Always update liveLandmarks so head-tracking delta stays fresh,
        // but this does NOT affect the overlay base position when locked.
        State.liveLandmarks = data.landmarks;
      }
    } else {
      DOM.noFaceBadge.classList.remove("hidden");
    }
  } catch (err) {
    console.warn("Frame poll error:", err);
  }

  // ── Render ───────────────────────────────────────────────────────
  if (State.isCalibrating || !State.calibrated) {
    Renderer.drawCalibrating(State.mappedSide);
    return;
  }

  /**
   * POSITION SOURCE SELECTION — the core fix.
   *
   * positionLocked = true:
   *   Use State.calibrated as the base. The affected side is always
   *   drawn at the neutral calibration pose. Only the expression
   *   offset (computed from State.calibrated scale) moves landmarks.
   *   Healthy-side muscle movement has ZERO effect here.
   *
   * positionLocked = false (legacy / debug):
   *   Use State.liveLandmarks — overlay follows healthy-side motion.
   */
  const positionSource = State.positionLocked
    ? State.calibrated
    : (State.liveLandmarks || State.calibrated);

  Renderer.drawFrame(
    positionSource,       // base position for the affected-side overlay
    State.mappedSide,
    State.expression,
    State.calibrated,     // scale reference for expression offsets
  );
}

// ── Frame capture ────────────────────────────────────────────────────

function captureFrame() {
  if (DOM.webcam.readyState < 2) return null;

  const w = DOM.webcam.videoWidth;
  const h = DOM.webcam.videoHeight;
  if (!w || !h) return null;

  const tmpCanvas    = document.createElement("canvas");
  tmpCanvas.width    = w;
  tmpCanvas.height   = h;
  tmpCanvas.getContext("2d").drawImage(DOM.webcam, 0, 0, w, h);

  return tmpCanvas.toDataURL("image/jpeg", JPEG_QUALITY);
}

// ── Expression management ────────────────────────────────────────────

/**
 * Set the active expression.
 * This is the ONLY mechanism that moves the affected-side overlay
 * (when positionLocked = true). Called by:
 *   - Manual expression buttons (manual mode)
 *   - pollEEG() (auto mode)
 *
 * @param {string} name  neutral | raise | knit | lookup | lookdown
 */
function setExpression(name) {
  State.expression = name;
  DOM.exprButtons.forEach(btn => {
    btn.classList.toggle("active", btn.dataset.expr === name);
  });
}

// ── EEG Prediction ───────────────────────────────────────────────────

/**
 * Toggle between manual and EEG-driven auto mode.
 *
 * In both modes State.positionLocked remains true — the position of
 * the overlay is ALWAYS frozen to calibration. The only difference is
 * whether the expression offset is set by a button or by the model.
 */
function toggleEEGMode() {
  State.eegMode = State.eegMode === "manual" ? "auto" : "manual";
  const isAuto  = State.eegMode === "auto";

  DOM.eegModeBtn.textContent = isAuto ? "Auto: ON" : "Auto: OFF";
  DOM.eegModeBtn.classList.toggle("active", isAuto);

  // Disable manual buttons in auto mode
  DOM.exprButtons.forEach(btn => { btn.disabled = isAuto; });
}

function startEEGLoop() {
  if (State.eegLoopHandle) return;
  pollEEG();
  State.eegLoopHandle = setInterval(pollEEG, EEG_POLL_MS);
}

function stopEEGLoop() {
  if (!State.eegLoopHandle) return;
  clearInterval(State.eegLoopHandle);
  State.eegLoopHandle = null;
}

/**
 * Fetch one EEG prediction from the server.
 * In auto mode, the returned expression key is passed directly to
 * setExpression() — this is the only path that moves the overlay in
 * auto mode.
 */
async function pollEEG() {
  try {
    const resp = await fetch(EEG_API_URL);
    const data = await resp.json();

    if (!data.success) {
      updateEEGPanelError(data);
      return;
    }

    updateEEGPanel(data);

    // In auto mode: EEG prediction is the ONLY thing that changes the
    // expression on the affected side. Position is still locked to
    // State.calibrated — only the expression offset changes.
    if (State.eegMode === "auto") {
      setExpression(data.expression);
    }
  } catch (err) {
    console.warn("EEG poll failed:", err);
  }
}

const _STATUS_LABEL = {
  neutral:      { text: "◉ Neutral",      cls: "eeg-status-label--neutral" },
  looking_up:   { text: "↑ Looking Up",   cls: "eeg-status-label--up"      },
  looking_down: { text: "↓ Looking Down", cls: "eeg-status-label--down"    },
};

function updateEEGPanel(data) {
  DOM.eegCsvIndex.textContent   = "CONNECTED";
  DOM.eegSession.textContent    = data.port || "—";
  DOM.eegWindow.textContent     = data.samples ? `${data.samples} samples` : "—";
  DOM.eegTrueLabel.textContent  = data.raw;
  DOM.eegPrediction.textContent = data.prediction;
  DOM.eegConfidence.textContent = `${(data.confidence * 100).toFixed(1)}%`;
  DOM.eegExpression.textContent = data.expression;

  DOM.eegMatch.textContent = "● LIVE";
  DOM.eegMatch.className   = "eeg-match eeg-match--ok";

  DOM.eegConnectBtn.textContent = "Connected";
  DOM.eegConnectError.classList.add("hidden");

  // Auto-enable EEG mode the first time a live signal arrives
  if (!State.eegEverConnected) {
    State.eegEverConnected = true;
    if (State.eegMode !== "auto") toggleEEGMode();
  }

  // Big status label overlaid on the canvas
  const info = _STATUS_LABEL[data.prediction] || _STATUS_LABEL.neutral;
  DOM.eegStatusLabel.textContent = `${info.text}  ${(data.confidence * 100).toFixed(0)}%`;
  DOM.eegStatusLabel.className   = `eeg-status-label ${info.cls}`;
}

function updateEEGPanelError(data) {
  const status = data.board_status || "disconnected";
  const errMsg = data.error || "";

  DOM.eegCsvIndex.textContent   = status.toUpperCase();
  DOM.eegSession.textContent    = "—";
  DOM.eegWindow.textContent     = "—";
  DOM.eegTrueLabel.textContent  = "—";
  DOM.eegPrediction.textContent = "—";
  DOM.eegConfidence.textContent = "—";
  DOM.eegExpression.textContent = "—";

  DOM.eegMatch.textContent = "● OFFLINE";
  DOM.eegMatch.className   = "eeg-match eeg-match--fail";

  // Show error hint (e.g. "Access denied — run as Administrator")
  if (errMsg) {
    DOM.eegConnectError.textContent = errMsg;
    DOM.eegConnectError.classList.remove("hidden");
  }

  if (DOM.eegConnectBtn.textContent === "Connected") {
    DOM.eegConnectBtn.textContent = "Reconnect";
  }

  // Reset label to no-signal state
  DOM.eegStatusLabel.textContent = "— NO SIGNAL";
  DOM.eegStatusLabel.className   = "eeg-status-label";
}

// ── Bootstrap ────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  initEventListeners();
  showSetupPanel();
});