/**
 * renderer.js  –  Canvas Compositor & Landmark Renderer
 * =======================================================
 * Responsible for EVERY pixel drawn on #main-canvas each frame.
 *
 * Rendering pipeline (in order)
 * ────────────────────────────────────────────────────────────────
 * 1. Draw the raw webcam frame as a background
 * 2. Black out the chosen half of the face (the "mapped" side)
 * 3. Mirror the SOURCE side's landmarks to the blacked-out side
 * 4. Apply expression offsets from expressions.js
 * 5. Draw landmark overlays: eyebrow, eye contour, iris, mouth
 *
 * Coordinate system
 * ────────────────────────────────────────────────────────────────
 * MediaPipe returns normalised [0-1] coordinates.
 * All arithmetic is done in normalised space; we convert to pixels
 * only at draw time via Utils.toPixel().
 *
 * Mirroring
 * ────────────────────────────────────────────────────────────────
 * If the user chose to map the LEFT side we read the RIGHT landmarks
 * and flip their X coordinate (Utils.mirrorX) before drawing on the left.
 * Vice-versa for mapping the right side.
 *
 * The video feed itself is NOT flipped by this module – the video
 * element uses CSS `transform: scaleX(-1)` (applied via JS in app.js)
 * to act as a natural mirror, which matches user expectation.
 */

"use strict";

const Renderer = (() => {

  // ── Colour palette ─────────────────────────────────────────────
  const COLOUR = {
    eyebrow:    "#ffcc44",   // warm yellow
    eye:        "#44aaff",   // cool blue
    iris:       "#ff6644",   // orange-red
    mouth:      "#44ff88",   // mint green
    irisRing:   "rgba(255,100,68,0.35)",
    nodeRadius: 3,
    irisRadius: 10,
    lineWidth:  2,
  };

  // ── Private state ───────────────────────────────────────────────
  let _canvas = null;   // HTMLCanvasElement
  let _ctx    = null;   // CanvasRenderingContext2D
  let _video  = null;   // HTMLVideoElement

  // ── Init ────────────────────────────────────────────────────────

  /**
   * Bind the renderer to the canvas and video elements.
   * Must be called once before any draw calls.
   *
   * @param {HTMLCanvasElement}  canvas
   * @param {HTMLVideoElement}   video
   */
  function init(canvas, video) {
    _canvas = canvas;
    _ctx    = canvas.getContext("2d");
    _video  = video;
  }

  // ── Frame drawing ───────────────────────────────────────────────

  /**
   * Sync canvas dimensions to the actual displayed video size.
   * Called each frame so it reacts to window resizes.
   */
  function _syncSize() {
    const rect = _canvas.getBoundingClientRect();
    if (_canvas.width  !== rect.width  ||
        _canvas.height !== rect.height) {
      _canvas.width  = rect.width;
      _canvas.height = rect.height;
    }
  }

  /**
   * Draw the webcam video onto the canvas (full frame).
   * The video is naturally mirror-flipped by app.js CSS, so we
   * must also flip the canvas draw to stay consistent.
   */
  function _drawVideo() {
    const cw = _canvas.width;
    const ch = _canvas.height;

    // Flip horizontally so the canvas matches the mirror video
    _ctx.save();
    _ctx.translate(cw, 0);
    _ctx.scale(-1, 1);
    _ctx.drawImage(_video, 0, 0, cw, ch);
    _ctx.restore();
  }

  /**
   * Black out one vertical half of the canvas.
   *
   * @param {"left"|"right"} mappedSide  - the side to black out
   */
  function _blackoutSide(mappedSide) {
    const cw = _canvas.width;
    const ch = _canvas.height;
    const x  = mappedSide === "left" ? 0 : cw / 2;

    _ctx.fillStyle = "#000000";
    _ctx.fillRect(x, 0, cw / 2, ch);
  }

  // ── Landmark mirroring ──────────────────────────────────────────

  /**
   * Take the SOURCE side's normalised landmarks and mirror their X
   * coordinates to produce landmarks for the MAPPED (blacked-out) side.
   *
   * @param {object} sourceLandmarks  - { eyebrow, eye, iris, mouth }
   * @returns {object} mirrored landmarks in the same shape
   */
  function _mirrorLandmarks(sourceLandmarks) {
    function flipPt(p) {
      return { x: Utils.mirrorX(p.x), y: p.y };
    }

    return {
      eyebrow: sourceLandmarks.eyebrow.map(flipPt),
      eye:     sourceLandmarks.eye.map(flipPt),
      iris:    flipPt(sourceLandmarks.iris),
      mouth:   sourceLandmarks.mouth.map(flipPt),
    };
  }

  // ── Offset application ──────────────────────────────────────────

  /**
   * Add normalised {dx, dy} offsets from an expression to the
   * mirrored landmarks, returning new landmark objects.
   *
   * @param {object} mirrored   - { eyebrow[], eye[], iris, mouth[] }
   * @param {object} offsets    - { eyebrow: {dx,dy}[], iris: {dx,dy} }
   * @returns {object} adjusted landmark set
   */
  function _applyOffsets(mirrored, offsets) {
    return {
      eyebrow: mirrored.eyebrow.map((pt, i) => ({
        x: pt.x + (offsets.eyebrow[i]?.dx ?? 0),
        y: pt.y + (offsets.eyebrow[i]?.dy ?? 0),
      })),
      eye:   mirrored.eye,       // eye contour is static in all expressions
      iris:  {
        x: mirrored.iris.x + offsets.iris.dx,
        y: mirrored.iris.y + offsets.iris.dy,
      },
      mouth: mirrored.mouth,     // mouth is static in these expressions
    };
  }

  // ── Landmark drawing ────────────────────────────────────────────

  /**
   * Convert all normalised points to pixel coordinates and draw the
   * full facial landmark overlay for the mapped side.
   *
   * @param {object} lm   - finalised { eyebrow[], eye[], iris, mouth[] } (normalised)
   */
  function _drawLandmarks(lm) {
    const cw = _canvas.width;
    const ch = _canvas.height;

    const toBPx = (p) => Utils.toPixel(p, cw, ch);

    const browPx  = lm.eyebrow.map(toBPx);
    const eyePx   = lm.eye.map(toBPx);
    const irisPx  = toBPx(lm.iris);
    const mouthPx = lm.mouth.map(toBPx);

    // ── Eyebrow ────────────────────────────────────────────────────
    Utils.drawPolyline(_ctx, browPx, COLOUR.eyebrow, COLOUR.lineWidth);
    browPx.forEach(p =>
      Utils.drawCircle(_ctx, p.x, p.y, COLOUR.nodeRadius, COLOUR.eyebrow)
    );

    // ── Eye contour ────────────────────────────────────────────────
    Utils.drawPolygon(_ctx, eyePx, COLOUR.eye, COLOUR.lineWidth);
    eyePx.forEach(p =>
      Utils.drawCircle(_ctx, p.x, p.y, COLOUR.nodeRadius, COLOUR.eye)
    );

    // ── Iris (outer ring + centre dot) ─────────────────────────────
    _ctx.beginPath();
    _ctx.arc(irisPx.x, irisPx.y, COLOUR.irisRadius + 4, 0, Math.PI * 2);
    _ctx.strokeStyle = COLOUR.irisRing;
    _ctx.lineWidth   = 3;
    _ctx.stroke();

    Utils.drawCircle(_ctx, irisPx.x, irisPx.y, COLOUR.irisRadius, COLOUR.iris);

    // ── Mouth corners ──────────────────────────────────────────────
    mouthPx.forEach(p =>
      Utils.drawCircle(_ctx, p.x, p.y, COLOUR.nodeRadius + 1, COLOUR.mouth)
    );
    // Connect with a subtle line between corner 0 (left) and corner 1 (right)
    if (mouthPx.length >= 2) {
      Utils.drawPolyline(_ctx, [mouthPx[0], mouthPx[1]], COLOUR.mouth, 1);
    }
  }

  // ── Public draw API ─────────────────────────────────────────────

  /**
   * Render one complete frame.
   *
   * @param {object}           sourceLandmarks  - raw { eyebrow,eye,iris,mouth } from server
   * @param {"left"|"right"}   mappedSide       - side to black out and draw onto
   * @param {string}           expression       - current expression name
   * @param {object}           calibrated       - locked-in calibration snapshot
   */
  function drawFrame(sourceLandmarks, mappedSide, expression, calibrated) {
    if (!_canvas || !_ctx || !_video) return;

    _syncSize();

    // 1. Draw the webcam feed
    _drawVideo();

    // 2. Black out the mapped half
    _blackoutSide(mappedSide);

    if (!sourceLandmarks || !calibrated) return;

    // 3. Mirror source landmarks to the mapped side
    const mirrored = _mirrorLandmarks(sourceLandmarks);

    // 4. Compute expression offsets relative to the CALIBRATED mirrored position
    const mirroredCalib = _mirrorLandmarks(calibrated);
    const offsets       = Expressions.getOffsets(expression, mirroredCalib);

    // 5. Apply offsets
    const final = _applyOffsets(mirrored, offsets);

    // 6. Draw the landmarks
    _drawLandmarks(final);
  }

  /**
   * Draw only the video + blackout (no landmarks).
   * Called while calibrating before any landmarks are confirmed.
   *
   * @param {"left"|"right"} mappedSide
   */
  function drawCalibrating(mappedSide) {
    if (!_canvas || !_ctx || !_video) return;
    _syncSize();    
    _drawVideo();
    _blackoutSide(mappedSide);
  }

  return { init, drawFrame, drawCalibrating };

})();