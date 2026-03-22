/**
 * utils.js  –  Shared Utility Functions
 * ======================================
 * Pure helper functions with no side-effects used by all other modules.
 */

"use strict";

const Utils = (() => {

  // ──────────────────────────────────────────────────────────────────
  // Coordinate helpers
  // ──────────────────────────────────────────────────────────────────

  /**
   * Convert a normalised landmark point {x,y} (range 0-1) to
   * canvas pixel coordinates.
   *
   * @param {{ x: number, y: number }} pt  - normalised point
   * @param {number} cw  - canvas width  in pixels
   * @param {number} ch  - canvas height in pixels
   * @returns {{ x: number, y: number }}
   */
  function toPixel(pt, cw, ch) {
    return { x: pt.x * cw, y: pt.y * ch };
  }

  /**
   * Mirror a normalised X coordinate around the vertical centre (0.5).
   * Used to flip source landmarks from one side to the other.
   *
   * @param {number} normX  - normalised x in [0, 1]
   * @returns {number}
   */
  function mirrorX(normX) {
    return 1.0 - normX;
  }

  /**
   * Compute the centroid (average point) of an array of {x,y} points.
   *
   * @param {{ x: number, y: number }[]} points
   * @returns {{ x: number, y: number }}
   */
  function centroid(points) {
    const sum = points.reduce(
      (acc, p) => ({ x: acc.x + p.x, y: acc.y + p.y }),
      { x: 0, y: 0 }
    );
    return { x: sum.x / points.length, y: sum.y / points.length };
  }

  /**
   * Linearly interpolate between two values.
   *
   * @param {number} a
   * @param {number} b
   * @param {number} t  - blend factor in [0, 1]
   * @returns {number}
   */
  function lerp(a, b, t) {
    return a + (b - a) * t;
  }

  /**
   * Clamp a value between min and max.
   */
  function clamp(val, min, max) {
    return Math.max(min, Math.min(max, val));
  }

  // ──────────────────────────────────────────────────────────────────
  // Canvas helpers
  // ──────────────────────────────────────────────────────────────────

  /**
   * Draw a filled circle on a canvas context.
   *
   * @param {CanvasRenderingContext2D} ctx
   * @param {number} x
   * @param {number} y
   * @param {number} radius
   * @param {string} colour  - CSS colour string
   */
  function drawCircle(ctx, x, y, radius, colour) {
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fillStyle = colour;
    ctx.fill();
  }

  /**
   * Draw an open polyline through an array of pixel points.
   *
   * @param {CanvasRenderingContext2D} ctx
   * @param {{ x: number, y: number }[]} pixelPoints
   * @param {string} colour
   * @param {number} lineWidth
   */
  function drawPolyline(ctx, pixelPoints, colour, lineWidth = 1.5) {
    if (pixelPoints.length < 2) return;
    ctx.beginPath();
    ctx.moveTo(pixelPoints[0].x, pixelPoints[0].y);
    for (let i = 1; i < pixelPoints.length; i++) {
      ctx.lineTo(pixelPoints[i].x, pixelPoints[i].y);
    }
    ctx.strokeStyle = colour;
    ctx.lineWidth   = lineWidth;
    ctx.lineJoin    = "round";
    ctx.lineCap     = "round";
    ctx.stroke();
  }

  /**
   * Draw a closed polygon (connects last point back to first).
   */
  function drawPolygon(ctx, pixelPoints, colour, lineWidth = 1.5) {
    if (pixelPoints.length < 2) return;
    ctx.beginPath();
    ctx.moveTo(pixelPoints[0].x, pixelPoints[0].y);
    for (let i = 1; i < pixelPoints.length; i++) {
      ctx.lineTo(pixelPoints[i].x, pixelPoints[i].y);
    }
    ctx.closePath();
    ctx.strokeStyle = colour;
    ctx.lineWidth   = lineWidth;
    ctx.lineJoin    = "round";
    ctx.stroke();
  }

  // ──────────────────────────────────────────────────────────────────
  // Public API
  // ──────────────────────────────────────────────────────────────────
  return { toPixel, mirrorX, centroid, lerp, clamp, drawCircle, drawPolyline, drawPolygon };

})();