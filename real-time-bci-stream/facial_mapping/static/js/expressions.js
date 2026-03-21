/**
 * expressions.js  –  Expression Offset Calculator
 * =================================================
 * Maps each named expression to a function that returns delta {dx, dy}
 * values for the eyebrow points and the iris, given the calibrated
 * landmark data.
 *
 * All offsets are in NORMALISED coordinates (same space as MediaPipe
 * output, i.e. 0-1). The renderer converts to pixels before drawing.
 *
 * How the offsets are designed
 * ----------------------------
 * We derive scale-invariant offsets from the eye height:
 *   eyeHeight = bottom_of_eye.y  –  top_of_eye.y   (in norm. coords)
 *
 * Using eyeHeight as a reference means the animation magnitudes
 * automatically scale with face size / distance from camera.
 *
 * Eyebrow landmark order (inner → outer):
 *   index 0 = inner corner (closest to nose bridge)
 *   index 4 = outer corner
 *
 * Expression rules
 * ─────────────────────────────────────────────────────
 * neutral    → no offset on anything
 * raise      → all eyebrow points move UP uniformly; iris stays
 * knit       → all eyebrow points move toward face centre (inner = more X shift);
 *              iris stays
 * lookup     → inner eyebrow corner lifts slightly; iris moves UP
 * lookdown   → inner eyebrow corner drops slightly; iris moves DOWN
 */

"use strict";

const Expressions = (() => {

  // ── Tuning constants (tweak here to adjust animation feel) ────────

  // Fraction of eyeHeight used as the base animation unit
  const RAISE_FACTOR     = 1.5;   // how many "eye heights" the brow lifts
  const KNIT_X_FACTOR    = 0.8;   // horizontal inward shift for knit (inner point)
  const KNIT_FALLOFF     = 0.6;   // multiplier reduction per landmark from inner → outer
  const IRIS_SHIFT       = 1.0;   // iris vertical travel in eye-height units
  const LOOKUP_BROW_LIFT = 0.4;   // fraction of eyeHeight the inner brow lifts on lookup
  const LOOKDOWN_DROP    = 0.3;   // same for lookdown

  // ── Helpers ───────────────────────────────────────────────────────

  /**
   * Estimate the vertical height of the eye from its 6 contour points.
   * Points are: [outer-corner, top-outer, top-inner, inner-corner, bottom-inner, bottom-outer]
   * (MediaPipe ordering after the index mapping in landmarks.py)
   *
   * We simply take max_y – min_y across all 6 points.
   *
   * @param {{ x:number, y:number }[]} eyePts  - 6 normalised eye contour points
   * @returns {number} eye height in normalised coords
   */
  function eyeHeight(eyePts) {
    const ys  = eyePts.map(p => p.y);
    return Math.max(...ys) - Math.min(...ys);
  }

  /**
   * Determine the sign of the "inward" X direction so the brow
   * moves toward the nose bridge regardless of which side we are on.
   *
   * For a RIGHT-side face (drawn in the right half of screen) the
   * inner corner has a SMALLER x value → moving inward means subtracting x.
   * For a LEFT-side face the inner corner has a LARGER x value → adding x.
   *
   * We detect which side we're on by comparing the iris x to 0.5.
   *
   * @param {{ x:number, y:number }} iris  - iris centre in normalised coords
   * @returns {number} +1 or -1
   */
  function inwardXSign(iris) {
    // iris.x > 0.5 means face is on the right half → inner corner is to the LEFT → subtract
    return iris.x > 0.5 ? -1 : 1;
  }

  // ── Expression definitions ────────────────────────────────────────

  /**
   * Returns zero offsets for all points (neutral resting position).
   *
   * @param {object} calibrated  - the calibrated landmark snapshot
   * @returns {{ eyebrow: {dx,dy}[], iris: {dx,dy} }}
   */
  function neutral(calibrated) {
    const zero = { dx: 0, dy: 0 };
    return {
      eyebrow: calibrated.eyebrow.map(() => ({ ...zero })),
      iris:    { ...zero },
    };
  }

  /**
   * RAISE EYEBROW – whole brow lifts uniformly upward.
   * Iris stays at resting position.
   */
  function raise(calibrated) {
    const eh  = eyeHeight(calibrated.eye);
    const dy  = -(eh * RAISE_FACTOR);   // negative y = upward on canvas

    return {
      eyebrow: calibrated.eyebrow.map(() => ({ dx: 0, dy })),
      iris:    { dx: 0, dy: 0 },
    };
  }

  /**
   * KNIT EYEBROW – brow moves inward (toward nose bridge).
   * The inner corner moves more than the outer corner (falloff).
   * Iris stays at resting position.
   */
  function knit(calibrated) {
    const eh    = eyeHeight(calibrated.eye);
    const xSign = inwardXSign(calibrated.iris);
    const baseX = eh * KNIT_X_FACTOR;

    // Index 0 = inner, 4 = outer → apply decreasing X shift
    const eyebrowOffsets = calibrated.eyebrow.map((_, i) => {
      // Falloff: inner point gets full shift, outer gets less
      const factor = Math.pow(KNIT_FALLOFF, i);
      return { dx: xSign * baseX * factor, dy: 0 };
    });

    return {
      eyebrow: eyebrowOffsets,
      iris:    { dx: 0, dy: 0 },
    };
  }

  /**
   * LOOK UP – inner brow corner lifts slightly; iris moves upward.
   */
  function lookUp(calibrated) {
    const eh    = eyeHeight(calibrated.eye);
    const dy    = -(eh * IRIS_SHIFT);               // iris moves up
    const browLift = -(eh * LOOKUP_BROW_LIFT);      // inner brow lifts

    // Only the inner corner (index 0) lifts; outer stays flat
    const eyebrowOffsets = calibrated.eyebrow.map((_, i) => ({
      dx: 0,
      dy: i === 0 ? browLift : 0,
    }));

    return {
      eyebrow: eyebrowOffsets,
      iris:    { dx: 0, dy },
    };
  }

  /**
   * LOOK DOWN – inner brow corner drops slightly; iris moves downward.
   */
  function lookDown(calibrated) {
    const eh    = eyeHeight(calibrated.eye);
    const dy    = eh * IRIS_SHIFT;                  // iris moves down (+y)
    const browDrop = eh * LOOKDOWN_DROP;            // inner brow drops

    const eyebrowOffsets = calibrated.eyebrow.map((_, i) => ({
      dx: 0,
      dy: i === 0 ? browDrop : 0,
    }));

    return {
      eyebrow: eyebrowOffsets,
      iris:    { dx: 0, dy },
    };
  }

  // ── Map expression name → function ──────────────────────────────

  const MAP = {
    neutral:  neutral,
    raise:    raise,
    knit:     knit,
    lookup:   lookUp,
    lookdown: lookDown,
  };

  /**
   * Compute the offsets for the current expression.
   *
   * @param {string} expressionName  - one of: neutral | raise | knit | lookup | lookdown
   * @param {object} calibrated      - snapshot of normalised landmarks at calibration time
   * @returns {{ eyebrow: {dx,dy}[], iris: {dx,dy} }}
   */
  function getOffsets(expressionName, calibrated) {
    const fn = MAP[expressionName] || MAP.neutral;
    return fn(calibrated);
  }

  return { getOffsets };

})();