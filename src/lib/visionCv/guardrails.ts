/**
 * Post-OCR guardrails — port of guardrails.py.
 *
 *  - snapAxisColumn: rewrite each axis label onto the inferred arithmetic grid
 *    (the fit already gives a robust a,b; this repairs the garbled/blank
 *    labels to their on-grid value, e.g. [36,"5",37] → [36,36.5,37]).
 *  - interpolateSeries: fill short gaps so the curve draws continuously, in a
 *    SEPARATE mask (never merged into `present` — measured ≠ synthesised).
 */

import type { AxisColumn } from "./axisColumns";

const NICE_STEPS = [0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0];

function inferStep(b: number, ys: number[]): number {
  let raw = 0.5;
  if (ys.length >= 2) {
    const sorted = [...ys].sort((a, c) => a - c);
    const diffs: number[] = [];
    for (let i = 1; i < sorted.length; i++) diffs.push(sorted[i] - sorted[i - 1]);
    diffs.sort((a, c) => a - c);
    const med = diffs[diffs.length >> 1];
    raw = Math.abs(b) * med;
  }
  return NICE_STEPS.reduce((best, s) => (Math.abs(s - raw) < Math.abs(best - raw) ? s : best), NICE_STEPS[0]);
}

/** Snap every label value onto the column's inferred grid (in place). */
export function snapAxisColumn(col: AxisColumn): void {
  if (!col.boxes.length || Math.abs(col.b) < 1e-12) return;
  const ys = col.boxes.map((b) => b.cy);
  const step = inferStep(col.b, ys);
  for (const bx of col.boxes) {
    const pred = col.a + col.b * bx.cy;
    bx.value = Math.round((Math.round(pred / step) * step) * 1000) / 1000;
  }
}

/**
 * Linear-interpolate gaps ≤ maxGap between measured days. Returns the filled
 * value array + an `interpolated` mask (1 = synthesised). `present` is NOT
 * modified by the caller — interpolated days stay absent from measured.
 */
export function interpolateSeries(
  value: number[],
  present: number[],
  maxGap = 3,
): { value: number[]; interpolated: number[] } {
  const n = value.length;
  const out = value.slice();
  const interp = new Array<number>(n).fill(0);
  const idx: number[] = [];
  for (let i = 0; i < n; i++) if (present[i] > 0.5) idx.push(i);
  for (let j = 0; j < idx.length - 1; j++) {
    const a = idx[j], b = idx[j + 1];
    const gap = b - a - 1;
    if (gap >= 1 && gap <= maxGap) {
      const va = value[a], vb = value[b];
      for (let k = a + 1; k < b; k++) {
        const t = (k - a) / (b - a);
        out[k] = va * (1 - t) + vb * t;
        interp[k] = 1;
      }
    }
  }
  return { value: out, interpolated: interp };
}
