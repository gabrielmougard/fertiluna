/**
 * Marker detection — port of marker_detection.py.
 *
 * Signature: a data marker is a SMALL WHITE BLOB whose neighbourhood is
 * dominated by one chart colour. We find white blobs (CC on a brightness
 * mask), then sample a ring of pixels around each in the blue/orange/purple
 * masks; the dominant colour wins. Purple-dominant blobs are dropped (they
 * belong to the "Level" distractor). Pure TS — uses ops.connectedComponents.
 */

import { connectedComponents } from "./ops";
import type { Mask, PlotRegion, SeriesMasks } from "./types";
import type { WorkImage } from "./preprocess";

const MIN_WHITE_AREA = 4;
const MAX_WHITE_AREA = 360;
const MIN_BLOB_ASPECT = 0.35;
const MAX_BLOB_ASPECT = 2.6;
const BG_BRIGHTNESS = 225;
const RING_RADII = [3, 4, 5, 6, 7, 8, 9, 11, 14];
const RING_SAMPLES = 32;
const MIN_RING_COVERAGE = 0.3;

export type SeriesKey = "temp" | "lh";

export interface MarkerCenter {
  cx: number;
  cy: number;
  score: number;
  radius: number;
  kind: "ring" | "filled";
}

export interface Marker extends MarkerCenter {
  series: SeriesKey;
  cellIdx: number;
}

function whiteMask(work: WorkImage, plot: PlotRegion): Mask {
  const { rgba, width, height } = work;
  const data = new Uint8Array(width * height);
  for (let y = plot.y0; y < plot.y1; y++) {
    for (let x = plot.x0; x < plot.x1; x++) {
      const i = y * width + x;
      const r = rgba[i * 4], g = rgba[i * 4 + 1], b = rgba[i * 4 + 2];
      // BT.601 luma ≈ cv2 BGR2GRAY
      const gray = (r * 0.299 + g * 0.587 + b * 0.114) | 0;
      if (gray >= BG_BRIGHTNESS) data[i] = 255;
    }
  }
  return { data, width, height };
}

function ringCoverage(cx: number, cy: number, mask: Mask, radius: number): number {
  const { data, width, height } = mask;
  let hits = 0;
  for (let k = 0; k < RING_SAMPLES; k++) {
    const a = (2 * Math.PI * k) / RING_SAMPLES;
    const x = Math.round(cx + radius * Math.cos(a));
    const y = Math.round(cy + radius * Math.sin(a));
    if (x < 0 || y < 0 || x >= width || y >= height) continue;
    if (data[y * width + x] > 0) hits++;
  }
  return hits / RING_SAMPLES;
}

function bestRing(cx: number, cy: number, mask: Mask): { cov: number; r: number } {
  let cov = 0, r = RING_RADII[0];
  for (const rr of RING_RADII) {
    const c = ringCoverage(cx, cy, mask, rr);
    if (c > cov) { cov = c; r = rr; }
  }
  return { cov, r };
}

export function detectCenters(
  work: WorkImage,
  masks: SeriesMasks,
  plot: PlotRegion,
): { bbt: MarkerCenter[]; lh: MarkerCenter[] } {
  const white = whiteMask(work, plot);
  const { stats } = connectedComponents(white, 8);
  const bbt: MarkerCenter[] = [];
  const lh: MarkerCenter[] = [];
  for (const s of stats) {
    if (s.area < MIN_WHITE_AREA || s.area > MAX_WHITE_AREA) continue;
    const w = s.x1 - s.x0 + 1, h = s.y1 - s.y0 + 1;
    if (w < 3 || h < 3 || w > 26 || h > 26) continue;
    const aspect = w / Math.max(1, h);
    if (aspect < MIN_BLOB_ASPECT || aspect > MAX_BLOB_ASPECT) continue;
    const b = bestRing(s.cx, s.cy, masks.blue);
    const o = bestRing(s.cx, s.cy, masks.orange);
    const p = bestRing(s.cx, s.cy, masks.purple);
    if (Math.max(b.cov, o.cov, p.cov) < MIN_RING_COVERAGE) continue;
    if (p.cov > b.cov && p.cov > o.cov) continue; // Level distractor
    const kind = s.area > 60 ? "filled" : "ring";
    if (b.cov >= o.cov) bbt.push({ cx: s.cx, cy: s.cy, score: b.cov, radius: b.r, kind });
    else lh.push({ cx: s.cx, cy: s.cy, score: o.cov, radius: o.r, kind });
  }
  return { bbt, lh };
}

/** Snap centers to the nearest day cell (within 0.65·cellPx). Port of _snap. */
export function snapToCells(
  centers: MarkerCenter[],
  cells: number[],
  cellPx: number,
  series: SeriesKey,
): Map<number, Marker> {
  const half = cellPx * 0.65;
  const out = new Map<number, Marker>();
  for (const c of centers) {
    let bestIdx = -1, bestD = Infinity;
    for (let i = 0; i < cells.length; i++) {
      const d = Math.abs(c.cx - cells[i]);
      if (d < bestD) { bestD = d; bestIdx = i; }
    }
    if (bestIdx < 0 || bestD > half) continue;
    const m: Marker = { ...c, series, cellIdx: bestIdx };
    const prev = out.get(bestIdx);
    if (!prev || c.score > prev.score) out.set(bestIdx, m);
  }
  return out;
}
