/**
 * Plot-region detection — port of plot_region.py (the adaptive ink-bbox path).
 *
 * Uses ONLY chart-series ink (blue ∪ orange; purple is excluded because the
 * Level line + alpha-blended period bands leak into the table area). Drops
 * long-thin horizontal cover-lines, then keeps only LARGE components (≥25 %
 * of the biggest) — the chart line+markers form one big blob; table icons are
 * small isolated blobs. The bbox of what survives is the plot region.
 *
 * The gridline-morphology reconciliation in the Python `detect()` is deferred
 * (it needs morphologyEx); the ink-bbox path is the dominant, robust one and
 * the pipeline later refines plot.y from the axis-label span anyway.
 */

import { connectedComponents } from "./ops";
import type { Mask, PlotRegion, SeriesMasks } from "./types";

const FALLBACK = [0.1, 0.1, 0.92, 0.72]; // left, top, right, bottom ratios

function unionBlueOrange(masks: SeriesMasks): Mask {
  const { width, height } = masks.blue;
  const data = new Uint8Array(width * height);
  const b = masks.blue.data, o = masks.orange.data;
  for (let i = 0; i < data.length; i++) data[i] = b[i] || o[i] ? 255 : 0;
  return { data, width, height };
}

export function detectPlotRegion(masks: SeriesMasks): PlotRegion {
  const W = masks.blue.width, H = masks.blue.height;
  const union = unionBlueOrange(masks);
  const { stats } = connectedComponents(union, 8);

  // 1) drop long-thin horizontal cover-lines / axis frames
  const kept = stats.filter((s) => {
    const w = s.x1 - s.x0 + 1, h = s.y1 - s.y0 + 1;
    return !(w > 200 && h < Math.max(6, w / 18));
  });
  if (kept.length === 0) {
    return {
      x0: Math.round(W * FALLBACK[0]), y0: Math.round(H * FALLBACK[1]),
      x1: Math.round(W * FALLBACK[2]), y1: Math.round(H * FALLBACK[3]),
      method: "fallback",
    };
  }

  // 2) keep only large components (≥25 % of the biggest area)
  const maxArea = kept.reduce((m, s) => Math.max(m, s.area), 0);
  const areaThresh = Math.max(80, 0.25 * maxArea);
  const big = kept.filter((s) => s.area >= areaThresh);
  const use = big.length ? big : kept;

  let x0 = W, y0 = H, x1 = 0, y1 = 0;
  for (const s of use) {
    if (s.x0 < x0) x0 = s.x0;
    if (s.y0 < y0) y0 = s.y0;
    if (s.x1 > x1) x1 = s.x1;
    if (s.y1 > y1) y1 = s.y1;
  }
  return { x0, y0, x1, y1, method: "ink" };
}
