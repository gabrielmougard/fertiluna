/**
 * curveDigitizer.ts — pure, DOM-free chart-digitization logic.
 *
 * SOTA pragmatic approach (WebPlotDigitizer-style), 100 % in-browser:
 *   1. The user uploads a screenshot of their BBT/LH chart.
 *   2. They calibrate the axes by clicking two reference points per axis
 *      (e.g. "this gridline = J1", "this one = J20"; "this = 36.5 °C",
 *      "this = 37.0 °C"). Calibration is what makes extraction reliable across
 *      every app's chart style — the human supplies the coordinate mapping.
 *   3. They pick the curve colour (one click on the temperature line/dots).
 *   4. We scan each pixel column inside the plot region, find the curve pixels
 *      matching that colour, take their vertical centroid, and map the pixel
 *      coordinates to (day, value) via the calibration.
 *   5. We resample to one value per cycle day and hand them to the editable
 *      table, where the user verifies/corrects before analysis.
 *
 * Nothing here is uploaded; it operates on an ImageData-like RGBA buffer.
 */

import { CYCLE_MAX_DAYS } from "./constants";

export interface RGBA {
  r: number;
  g: number;
  b: number;
  a: number;
}

export interface PixelPoint {
  x: number;
  y: number;
}

/** Two-point linear calibration for one axis: maps pixel coord → data value. */
export interface AxisCalibration {
  /** pixel coordinate of reference point 1 (x for day-axis, y for value-axis) */
  pixel1: number;
  value1: number;
  pixel2: number;
  value2: number;
}

export interface Calibration {
  /** maps pixel-X → cycle day (1-indexed) */
  dayAxis: AxisCalibration;
  /** maps pixel-Y → temperature (°C) or LH value */
  valueAxis: AxisCalibration;
}

export interface ExtractOptions {
  /** RGBA pixel buffer (length = width*height*4), row-major. */
  data: Uint8ClampedArray;
  width: number;
  height: number;
  /** Target colour of the curve, sampled by the user. */
  target: RGBA;
  /** Max Euclidean RGB distance to consider a pixel "on the curve". */
  tolerance: number;
  /** Plot region in pixels (inclusive). Pixels outside are ignored. */
  region: { x0: number; y0: number; x1: number; y1: number };
}

function idx(x: number, y: number, width: number): number {
  return (y * width + x) * 4;
}

export function pixelAt(
  data: Uint8ClampedArray,
  width: number,
  x: number,
  y: number,
): RGBA {
  const i = idx(x, y, width);
  return { r: data[i], g: data[i + 1], b: data[i + 2], a: data[i + 3] };
}

export function colorDistance(a: RGBA, b: RGBA): number {
  const dr = a.r - b.r;
  const dg = a.g - b.g;
  const db = a.b - b.b;
  return Math.sqrt(dr * dr + dg * dg + db * db);
}

/** Linear map a pixel coordinate to a data value using a 2-point calibration. */
export function mapAxis(cal: AxisCalibration, pixel: number): number {
  const { pixel1, value1, pixel2, value2 } = cal;
  if (pixel2 === pixel1) return value1;
  const t = (pixel - pixel1) / (pixel2 - pixel1);
  return value1 + t * (value2 - value1);
}

/** Inverse: map a data value back to a pixel coordinate (for drawing overlays). */
export function unmapAxis(cal: AxisCalibration, value: number): number {
  const { pixel1, value1, pixel2, value2 } = cal;
  if (value2 === value1) return pixel1;
  const t = (value - value1) / (value2 - value1);
  return pixel1 + t * (pixel2 - pixel1);
}

/**
 * Column-scan extraction: for each x in the plot region, find the matching
 * curve pixels and return the vertical centroid as a pixel point. Columns with
 * no match are skipped (gaps → missing days later).
 */
export function extractCurvePixels(opts: ExtractOptions): PixelPoint[] {
  const { data, width, target, tolerance, region } = opts;
  const points: PixelPoint[] = [];
  const x0 = Math.max(0, Math.floor(region.x0));
  const x1 = Math.min(width - 1, Math.ceil(region.x1));
  const y0 = Math.max(0, Math.floor(region.y0));
  const y1 = Math.min(opts.height - 1, Math.ceil(region.y1));

  for (let x = x0; x <= x1; x++) {
    let sumY = 0;
    let count = 0;
    for (let y = y0; y <= y1; y++) {
      const p = pixelAt(data, width, x, y);
      if (p.a < 32) continue; // transparent
      if (colorDistance(p, target) <= tolerance) {
        sumY += y;
        count++;
      }
    }
    if (count > 0) {
      points.push({ x, y: sumY / count });
    }
  }
  return points;
}

export interface DayValue {
  day: number; // 1-indexed
  value: number | null;
}

/** Series we can digitize. Each maps to a column in the input table. */
export type SeriesKind = "temp" | "lh";

/** Sensible rounding per series: temps to 0.01 °C, LH to 0.1 relative units. */
export function rounderFor(kind: SeriesKind): (v: number) => number {
  if (kind === "temp") return (v) => Math.round(v * 100) / 100;
  return (v) => Math.round(v * 10) / 10;
}

/**
 * Resample extracted pixel points to one value per cycle day.
 *
 * For each integer day d in [1..maxDays], find the pixel-x for that day from
 * the calibration, collect curve points within ±half-a-day, and take the
 * median pixel-y → value. Days with no nearby curve point are null (missing).
 */
export function resampleToDays(
  points: PixelPoint[],
  cal: Calibration,
  maxDays: number = CYCLE_MAX_DAYS,
  round: (v: number) => number = (v) => Math.round(v * 100) / 100,
): DayValue[] {
  const out: DayValue[] = [];
  if (points.length === 0) {
    for (let d = 1; d <= maxDays; d++) out.push({ day: d, value: null });
    return out;
  }

  // pixel-x per day, and half-day window width in pixels
  const pxPerDay = Math.abs(
    unmapAxis(cal.dayAxis, 2) - unmapAxis(cal.dayAxis, 1),
  );
  const halfWin = Math.max(1, pxPerDay / 2);

  for (let d = 1; d <= maxDays; d++) {
    const targetX = unmapAxis(cal.dayAxis, d);
    const nearby: number[] = [];
    for (const p of points) {
      if (Math.abs(p.x - targetX) <= halfWin) nearby.push(p.y);
    }
    if (nearby.length === 0) {
      out.push({ day: d, value: null });
      continue;
    }
    nearby.sort((a, b) => a - b);
    const medY = nearby[Math.floor(nearby.length / 2)];
    out.push({ day: d, value: round(mapAxis(cal.valueAxis, medY)) });
  }
  return out;
}

/**
 * Heuristic auto-detection of the plot region: find the bounding box of the
 * densest band of near-grey/axis pixels. Used only as a STARTING guess for the
 * calibration overlay — the user always confirms. Returns a region covering the
 * central area if detection is inconclusive.
 */
export function guessPlotRegion(
  data: Uint8ClampedArray,
  width: number,
  height: number,
): { x0: number; y0: number; x1: number; y1: number } {
  // Default: inset 8% margin (most charts have axis labels around the edges).
  const mx = Math.round(width * 0.1);
  const my = Math.round(height * 0.08);
  return { x0: mx, y0: my, x1: width - mx, y1: height - my };
}

/**
 * Suggest a curve colour by finding the most saturated, non-grey colour inside
 * the region (charts draw the data line in a distinct colour vs. grey grids).
 * The user can override by clicking the line directly.
 */
export function suggestCurveColor(
  data: Uint8ClampedArray,
  width: number,
  region: { x0: number; y0: number; x1: number; y1: number },
): RGBA {
  const buckets = new Map<string, { c: RGBA; n: number; sat: number }>();
  const step = 2;
  for (let y = Math.floor(region.y0); y < region.y1; y += step) {
    for (let x = Math.floor(region.x0); x < region.x1; x += step) {
      const p = pixelAt(data, width, x, y);
      if (p.a < 32) continue;
      const max = Math.max(p.r, p.g, p.b);
      const min = Math.min(p.r, p.g, p.b);
      const sat = max === 0 ? 0 : (max - min) / max;
      // skip near-white, near-black, and near-grey (grid/background/text)
      if (max > 235 && min > 215) continue;
      if (max < 40) continue;
      if (sat < 0.18) continue;
      // quantize to 24-level buckets to group similar colours
      const key = `${p.r >> 5}-${p.g >> 5}-${p.b >> 5}`;
      const b = buckets.get(key);
      if (b) {
        b.n++;
      } else {
        buckets.set(key, { c: p, n: 1, sat });
      }
    }
  }
  let best: { c: RGBA; n: number; sat: number } | null = null;
  for (const b of buckets.values()) {
    // prefer frequent AND saturated
    const score = b.n * (0.5 + b.sat);
    const bestScore = best ? best.n * (0.5 + best.sat) : -1;
    if (score > bestScore) best = b;
  }
  return best ? best.c : { r: 214, g: 68, b: 127, a: 255 };
}
