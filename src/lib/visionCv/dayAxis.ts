/**
 * Day-cell grid — port of day_axis.py (autocorrelation + marker-spacing path).
 *
 * Autocorrelation of the per-column ink-density profile is the primary
 * cell-pitch signal (Python uses FFT; we compute it directly — the lag range
 * is small). The grid is then placed centered on the leftmost detected marker,
 * with a sub-pixel offset sweep, and tiled to N_DAYS cells.
 *
 * The date-row and vertical-gridline detectors from the Python `detect_grid`
 * are deferred (they need OCR / morphology); autocorrelation is the robust
 * default across the test screens.
 */

import { N_DAYS } from "./constants";
import type { PlotRegion, SeriesMasks } from "./types";

export interface DayGrid {
  cells: number[]; // x-centers, left→right
  cellPx: number;
  source: string;
}

/** Per-column count of (blue ∪ orange) ink within the plot region. */
function columnDensity(masks: SeriesMasks, plot: PlotRegion): number[] {
  const { width } = masks.blue;
  const b = masks.blue.data, o = masks.orange.data;
  const w = plot.x1 - plot.x0;
  const col = new Array<number>(Math.max(0, w)).fill(0);
  for (let y = plot.y0; y < plot.y1; y++) {
    const row = y * width;
    for (let x = plot.x0; x < plot.x1; x++) {
      if (b[row + x] || o[row + x]) col[x - plot.x0]++;
    }
  }
  return col;
}

/** First autocorrelation peak of the column density → cell pitch (px). */
export function autocorrCellPx(masks: SeriesMasks, plot: PlotRegion): number | null {
  const plotW = plot.x1 - plot.x0;
  if (plotW < 60) return null;
  const col = columnDensity(masks, plot);
  const n = col.length;
  const mean = col.reduce((a, b) => a + b, 0) / n;
  const x = col.map((v) => v - mean);
  const variance = x.reduce((a, b) => a + b * b, 0);
  if (variance < 0.5) return null;
  const minLag = Math.max(15, Math.floor(plotW / 80));
  const maxLag = Math.min(n - 1, Math.max(minLag + 1, Math.floor(plotW / 6)));
  if (maxLag <= minLag) return null;
  let bestLag = minLag, bestVal = -Infinity;
  for (let lag = minLag; lag < maxLag; lag++) {
    let acc = 0;
    for (let i = 0; i + lag < n; i++) acc += x[i] * x[i + lag];
    if (acc > bestVal) { bestVal = acc; bestLag = lag; }
  }
  return bestLag;
}

function modeDelta(sortedXs: number[]): number {
  const deltas: number[] = [];
  for (let i = 1; i < sortedXs.length; i++) {
    const d = sortedXs[i] - sortedXs[i - 1];
    if (d >= 8) deltas.push(d);
  }
  if (!deltas.length) return 0;
  const maxD = Math.max(...deltas);
  const nb = Math.max(6, Math.ceil(maxD / 2));
  const hist = new Array<number>(nb).fill(0);
  const bw = (maxD + 2) / nb;
  for (const d of deltas) hist[Math.min(nb - 1, Math.floor(d / bw))]++;
  let peak = 0;
  for (let i = 1; i < nb; i++) if (hist[i] > hist[peak]) peak = i;
  return (peak + 0.5) * bw;
}

/** Port of _grid_from_marker_spacing: pitch + grid placement from marker xs. */
export function gridFromMarkerSpacing(
  markerXs: number[],
  plot: PlotRegion,
  cellPxHint: number | null,
): DayGrid {
  const plotW = plot.x1 - plot.x0;
  const sorted = [...markerXs].sort((a, b) => a - b);
  let cellPx: number;
  if (cellPxHint && cellPxHint > 0) cellPx = cellPxHint;
  else if (sorted.length >= 2) cellPx = modeDelta(sorted) || plotW / N_DAYS;
  else cellPx = plotW / N_DAYS;

  const lo = Math.max(15, plotW / 80);
  const hi = Math.max(lo + 1, plotW / 6);
  cellPx = Math.min(hi, Math.max(lo, cellPx));

  // Cell CENTERS are c0 + k·cellPx. Anchor c0 on the leftmost marker, then
  // refine by minimizing each marker's distance to its nearest cell CENTER
  // (the residual and the cell positions must share one origin, else markers
  // land half a cell off the centers).
  let c0: number;
  if (sorted.length >= 2) {
    c0 = sorted[0];
    const step = Math.max(0.5, cellPx / 60);
    let bestOff = 0, bestErr = Infinity;
    for (let off = -cellPx / 2; off <= cellPx / 2 + step; off += step) {
      const a = c0 + off;
      let err = 0;
      for (const mx of sorted) {
        const rel = (((mx - a) % cellPx) + cellPx) % cellPx;
        err += Math.min(rel, cellPx - rel);
      }
      if (err < bestErr) { bestErr = err; bestOff = off; }
    }
    c0 += bestOff;
  } else {
    c0 = plot.x0 + cellPx / 2;
  }
  const cells: number[] = [];
  for (let k = 0; k < N_DAYS; k++) cells.push(c0 + k * cellPx);
  return { cells, cellPx, source: cellPxHint != null ? "autocorr" : "marker-spacing" };
}

export function detectGrid(
  masks: SeriesMasks,
  plot: PlotRegion,
  markerXs: number[],
): DayGrid {
  const hint = autocorrCellPx(masks, plot);
  return gridFromMarkerSpacing(markerXs, plot, hint);
}
