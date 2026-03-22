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
 *
 * Module dependencies (loaded before this file via HTML <script> tags):
 *   utils.js → Utils
 *   expressions.js → Expressions
 *   renderer.js → Renderer
 *
 * ── Calibration strategy ────────────────────────────────────────────
 * We accumulate CALIBRATION_FRAMES consecutive successful detections
 * and average them to form a stable baseline. If the face moves away
 * the frame is simply skipped; the calibrated snapshot is not replaced
 * until the user explicitly clicks "Recalibrate".
 */

"use strict";

// ── Configuration ────────────────────────────────────────────────────
const API_URL           = "/api/process-frame";
const FRAME_INTERVAL_MS = 80;       // ~12 fps polling rate to server
const CALIBRATION_FRAMES = 10;      // frames averaged before going live
const JPEG_QUALITY       = 0.6;     // trade-off between speed and accuracy

// ── Application State ────────────────────────────────────────────────
const State = {
  /** "left" or "right" — which side to black out and map */
  mappedSide: null,

  /** The SOURCE side (the opposite of mappedSide) */
  sourceSide: null,

  /** Currently active expression name */
  expression: "neutral",

  /** Locked-in calibration snapshot (normalised landmarks) */
  calibrated: null,

  /** Whether we are still in the calibration accumulation phase */
  isCalibrating: true,

  /** Accumulation buffer for calibration averaging */
  calibrationBuffer: [],

  /** Whether the animation loop is running */
  loopRunning: false,

  /** Interval handle for the polling loop */
  loopHandle: null,
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
};

// ── Initialisation ───────────────────────────────────────────────────

/** Bind all static event listeners on page load. */
function initEventListeners() {

  // Side selection buttons (setup panel)
  DOM.sideButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      const side = btn.dataset.side;   // "left" or "right"
      startSession(side);
    });
  });

  // Expression toggle buttons
  DOM.exprButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      setExpression(btn.dataset.expr);
    });
  });

  // Recalibrate – keep the same side, reset calibration state
  DOM.recalibrateBtn.addEventListener("click", () => {
    resetCalibration();
  });

  // Change side – go back to setup panel
  DOM.changeSideBtn.addEventListener("click", () => {
    stopLoop();
    showSetupPanel();
  });
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

/**
 * Start a new mapping session for the given side.
 * Initialises the webcam, renderer, and polling loop.
 *
 * @param {"left"|"right"} mappedSide
 */
async function startSession(mappedSide) {
  State.mappedSide = mappedSide;
  State.sourceSide = mappedSide === "left" ? "right" : "left";

  // Update the UI label
  DOM.activeSideLabel.textContent =
    `Mapping ${mappedSide} side  ·  reading ${State.sourceSide} side`;

  showVisualiserPanel();

  // Set the default expression
  setExpression("neutral");

  // Attempt to start the webcam
  const webcamOk = await startWebcam();
  if (!webcamOk) return;

  // Initialise the canvas renderer
  Renderer.init(DOM.canvas, DOM.webcam);

  // Begin calibration + polling loop
  resetCalibration();
  startLoop();
}

// ── Webcam ───────────────────────────────────────────────────────────

/**
 * Request webcam access and attach the stream to the video element.
 * Uses rear camera on mobile if available (facingMode: "user" for selfie).
 *
 * @returns {Promise<boolean>}  true on success, false on error
 */
async function startWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width:      { ideal: 640 },
        height:     { ideal: 480 },
        facingMode: "user",
      },
      audio: false,
    });

    DOM.webcam.srcObject = stream;

    // Mirror the video so it acts as a natural selfie mirror
    DOM.webcam.style.transform = "scaleX(-1)";

    // Wait for metadata so we know the stream dimensions
    await new Promise(resolve => {
      DOM.webcam.onloadedmetadata = resolve;
    });

    return true;
  } catch (err) {
    console.error("Webcam error:", err);
    alert(
      "Could not access webcam.\n\n" +
      "Please allow camera permissions and reload the page."
    );
    showSetupPanel();
    return false;
  }
}

// ── Calibration ──────────────────────────────────────────────────────

/** Reset calibration state and show the calibrating badge. */
function resetCalibration() {
  State.calibrated         = null;
  State.calibrationBuffer  = [];
  State.isCalibrating      = true;

  DOM.calibBadge.classList.remove("hidden");
  DOM.noFaceBadge.classList.add("hidden");
}

/**
 * Accumulate a new landmark frame into the calibration buffer.
 * Once CALIBRATION_FRAMES samples are collected, average them and lock in.
 *
 * @param {object} landmarks  - { eyebrow[], eye[], iris, mouth[] }
 */
function accumulateCalibration(landmarks) {
  State.calibrationBuffer.push(landmarks);

  if (State.calibrationBuffer.length >= CALIBRATION_FRAMES) {
    State.calibrated    = averageLandmarks(State.calibrationBuffer);
    State.isCalibrating = false;
    DOM.calibBadge.classList.add("hidden");
  }
}

/**
 * Average an array of landmark snapshots into a single snapshot.
 * Each field is averaged point-by-point.
 *
 * @param {object[]} buffer  - array of landmark objects
 * @returns {object}         averaged landmark object
 */
function averageLandmarks(buffer) {
  const n = buffer.length;

  function avgPoints(arrays) {
    // arrays is an array of arrays of {x,y}
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

/** Start the frame polling interval. */
function startLoop() {
  if (State.loopRunning) return;
  State.loopRunning = true;
  State.loopHandle  = setInterval(pollFrame, FRAME_INTERVAL_MS);
}

/** Stop the frame polling interval. */
function stopLoop() {
  if (!State.loopRunning) return;
  clearInterval(State.loopHandle);
  State.loopRunning = false;
}

/**
 * Capture a frame from the webcam, send it to the Flask backend,
 * handle the landmark response, and render the canvas.
 *
 * Called by setInterval – errors are caught so the loop continues.
 *
 * ── Why landmarks are frozen after calibration ──────────────────────
 * Once calibration completes, State.calibrated holds a single averaged
 * landmark snapshot. That snapshot is the ONLY source used for drawing
 * from that point on.
 *
 * Live server responses are intentionally ignored after calibration so
 * that real face movements (twitches, blinks, head turns) on the SOURCE
 * side do NOT move the overlay on the mapped side. The overlay only
 * changes when the user clicks an expression button, which modifies the
 * mathematical offsets applied to the frozen snapshot.
 *
 * To update the frozen snapshot the user must click "Recalibrate".
 */
async function pollFrame() {
  // Capture the current video frame as a JPEG data-URL
  const frameDataUrl = captureFrame();
  if (!frameDataUrl) return;

  // ── During calibration: keep polling the server to build the buffer ──
  if (State.isCalibrating) {
    try {
      const resp = await fetch(API_URL, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({
          image: frameDataUrl,
          side:  State.sourceSide,
        }),
      });

      const data = await resp.json();

      if (data.success) {
        DOM.noFaceBadge.classList.add("hidden");
        accumulateCalibration(data.landmarks);
      } else {
        // No face detected yet – keep showing the calibrating badge
        DOM.noFaceBadge.classList.remove("hidden");
      }
    } catch (err) {
      console.warn("Calibration poll error:", err);
    }

    // Render video + blackout only; no landmarks until calibration is done
    Renderer.drawCalibrating(State.mappedSide);
    return;
  }

  // ── After calibration: stop polling the server entirely ─────────────
  // The overlay is driven solely by State.calibrated + the active
  // expression. Real-time face movements are deliberately ignored.
  Renderer.drawFrame(
    State.calibrated,   // frozen snapshot – never changes until recalibrate
    State.mappedSide,
    State.expression,
    State.calibrated,
  );
}

// ── Frame capture ────────────────────────────────────────────────────

/**
 * Draw one video frame onto a temporary off-screen canvas and
 * return it as a base-64 JPEG data-URL.
 *
 * @returns {string|null}  data-URL or null if the video isn't ready
 */
function captureFrame() {
  if (DOM.webcam.readyState < 2) return null;   // HAVE_CURRENT_DATA

  const w = DOM.webcam.videoWidth;
  const h = DOM.webcam.videoHeight;
  if (!w || !h) return null;

  const tmpCanvas = document.createElement("canvas");
  tmpCanvas.width  = w;
  tmpCanvas.height = h;

  const tmpCtx = tmpCanvas.getContext("2d");

  // We send the UN-mirrored image to MediaPipe (it works on the real image)
  tmpCtx.drawImage(DOM.webcam, 0, 0, w, h);

  return tmpCanvas.toDataURL("image/jpeg", JPEG_QUALITY);
}

// ── Expression management ────────────────────────────────────────────

/**
 * Set the active expression and update button states.
 * Only one expression can be active at a time.
 *
 * @param {string} name  - expression key (neutral | raise | knit | lookup | lookdown)
 */
function setExpression(name) {
  State.expression = name;

  // Toggle the "active" class – only the clicked button gets it
  DOM.exprButtons.forEach(btn => {
    btn.classList.toggle("active", btn.dataset.expr === name);
  });
}

// ── Bootstrap ────────────────────────────────────────────────────────

/** Entry point – runs when the DOM is ready. */
document.addEventListener("DOMContentLoaded", () => {
  initEventListeners();
  showSetupPanel();
});