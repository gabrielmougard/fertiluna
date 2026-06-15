/**
 * Top-level in-browser CV pipeline — port of pipeline.py (calibration path).
 *
 * Wired stages (pure-TS + injected OCR):
 *   preprocess → colorSegmentation → plotRegion → axisColumns (OCR) →
 *   resolveAxes → refine plot.y from axis span → dayAxis grid →
 *   markerDetection → snap → calibrated value/present (real units →
 *   normalized) → guardrail interpolation → quality.
 *
 * `ocr` is injected (async). Without it, axis calibration is skipped and the
 * pipeline falls back to plot-extent y-fraction with scale defaulting to °F —
 * the same fallback the Python pipeline uses when no axis is read.
 *
 * Still NOT ported: bottom-table extraction (table_extract.py) — a bonus
 * feature, not part of the core value/present/scale/confidence contract.
 */

import { BBT_SCALES, LH_RANGE, N_DAYS, N_SERIES } from "./constants";
import { segment } from "./colorSegmentation";
import { detectPlotRegion } from "./plotRegion";
import { detectCenters, snapToCells, type Marker } from "./markerDetection";
import { detectGrid } from "./dayAxis";
import { toWorkCanvas } from "./preprocess";
import {
  detectAxisColumns, resolveAxes, type AxisColumn, type OcrFn, type ResolvedAxes,
} from "./axisColumns";
import { snapAxisColumn, interpolateSeries } from "./guardrails";
import { assessQuality } from "./quality";
import { extractTable } from "./tableExtract";
import type {
  ChartResultCv, CvDetections, CvMarkerDetection, PlotRegion,
} from "./types";

function zeros(rows: number, cols: number): number[][] {
  return Array.from({ length: rows }, () => new Array<number>(cols).fill(0));
}
function clamp01(x: number): number { return Math.max(0, Math.min(1, x)); }

/** Value via a calibrated axis column (real units → normalized), else
 *  plot-extent y-fraction (the Python no-axis fallback). */
function fillSeries(
  byDay: Map<number, Marker>,
  plot: PlotRegion,
  axis: AxisColumn | null,
  lo: number,
  hi: number,
): { value: number[]; present: number[] } {
  const value = new Array<number>(N_DAYS).fill(0);
  const present = new Array<number>(N_DAYS).fill(0);
  const span = Math.max(1e-6, hi - lo);
  for (const [d, m] of byDay) {
    if (d < 0 || d >= N_DAYS) continue;
    if (axis) value[d] = clamp01((axis.valueAt(m.cy) - lo) / span);
    else value[d] = clamp01((plot.y1 - m.cy) / Math.max(1, plot.y1 - plot.y0));
    present[d] = 1;
  }
  return { value, present };
}

export async function runPipelineCv(
  source: CanvasImageSource,
  ocr?: OcrFn,
): Promise<ChartResultCv> {
  const work = toWorkCanvas(source);
  const masks = segment(work);
  let plot = detectPlotRegion(masks);

  // ── axis calibration (when OCR is available) ──────────────────────────────
  let axes: ResolvedAxes | null = null;
  let scaleIdx = 1; // default °F until an axis is read
  let scaleConfidence = 0;
  if (ocr) {
    const columns = await detectAxisColumns(work, plot, ocr);
    axes = resolveAxes(columns);
    for (const c of [axes.bbt, axes.ratio, axes.level]) {
      if (c && c.nFit >= 2) snapAxisColumn(c);
    }
    if (axes.scaleIdx != null) { scaleIdx = axes.scaleIdx; scaleConfidence = axes.scaleConfidence; }
    // refine plot.y from the BBT axis tick span when trustworthy
    const anchor = axes.bbt ?? axes.level ?? axes.ratio;
    if (anchor && anchor.nFit >= 4) {
      const ys = anchor.boxes.filter((b) => b.value != null).map((b) => b.cy);
      const top = Math.min(...ys), bot = Math.max(...ys);
      const coarseH = Math.max(1, plot.y1 - plot.y0);
      if (bot - top >= 0.55 * coarseH) {
        plot = { ...plot, y0: Math.round(top), y1: Math.round(bot), method: plot.method + "+axiscol" };
      }
    }
  }

  // ── grid + markers ────────────────────────────────────────────────────────
  const centers = detectCenters(work, masks, plot);
  const markerXs = [...centers.bbt, ...centers.lh].map((c) => c.cx);
  const grid = detectGrid(masks, plot, markerXs);
  const bbtByDay = snapToCells(centers.bbt, grid.cells, grid.cellPx, "temp");
  const lhByDay = snapToCells(centers.lh, grid.cells, grid.cellPx, "lh");

  // left-pack to day 0 = leftmost detected cell across both series
  const allDays = [...new Set([...bbtByDay.keys(), ...lhByDay.keys()])].sort((a, b) => a - b);
  const d0 = allDays.length ? allDays[0] : 0;
  const repack = (m: Map<number, Marker>) => {
    const out = new Map<number, Marker>();
    for (const [d, mk] of m) if (d - d0 < N_DAYS) out.set(d - d0, mk);
    return out;
  };

  const [bbtLo, bbtHi] = [BBT_SCALES[scaleIdx].min, BBT_SCALES[scaleIdx].max];
  const bbtPacked = repack(bbtByDay);
  const lhPacked = repack(lhByDay);
  const value = zeros(N_SERIES, N_DAYS);
  const present = zeros(N_SERIES, N_DAYS);
  const temp = fillSeries(bbtPacked, plot, axes?.bbt ?? null, bbtLo, bbtHi);
  const lh = fillSeries(lhPacked, plot, axes?.ratio ?? null, LH_RANGE.min, LH_RANGE.max);
  value[0] = temp.value; present[0] = temp.present;
  value[1] = lh.value; present[1] = lh.present;

  // ── guardrail interpolation (separate mask) ───────────────────────────────
  const interpolated = zeros(N_SERIES, N_DAYS);
  for (let s = 0; s < N_SERIES; s++) {
    const r = interpolateSeries(value[s], present[s], 3);
    value[s] = r.value; interpolated[s] = r.interpolated;
  }

  // ── bottom table (when OCR available) ─────────────────────────────────────
  let table;
  if (ocr) {
    try {
      table = (await extractTable(work, plot, grid, ocr)) ?? undefined;
    } catch {
      table = undefined; // table is a bonus — never fail the whole pipeline
    }
  }

  // ── quality ───────────────────────────────────────────────────────────────
  const visibleDays = allDays.length;
  const q = assessQuality({
    present, scaleConfidence, axes,
    gridSource: grid.source, plotMethod: plot.method,
    visibleDays, truncated: visibleDays > N_DAYS,
  });

  // ── detection geometry for the UI overlay (work-canvas pixels) ────────────
  const markers: CvMarkerDetection[] = [];
  const pushMarkers = (
    packed: Map<number, Marker>, series: "temp" | "lh",
    detectedValue: number[], lo: number, hi: number,
  ) => {
    for (const [d, mk] of packed) {
      if (d < 0 || d >= N_DAYS) continue;
      markers.push({
        series, cx: mk.cx, cy: mk.cy, radius: mk.radius, day: d + 1,
        score: mk.score, valueReal: lo + detectedValue[d] * (hi - lo),
      });
    }
  };
  pushMarkers(bbtPacked, "temp", temp.value, bbtLo, bbtHi);
  pushMarkers(lhPacked, "lh", lh.value, LH_RANGE.min, LH_RANGE.max);

  const axisColumns: CvDetections["axisColumns"] = [];
  for (const c of [axes?.bbt, axes?.ratio, axes?.level]) {
    if (!c) continue;
    const labels = c.boxes
      .filter((b) => b.value != null)
      .map((b) => ({ bbox: b.bbox, value: b.value, text: b.text }));
    if (labels.length) axisColumns.push({ kind: c.kind, side: c.side, labels });
  }

  const tableRows: CvDetections["tableRows"] = (table?.rows ?? []).map((r) => ({
    name: r.name,
    labelBbox: r.labelBbox,
    cells: r.cells.map((c) => ({ bbox: c.bbox, text: c.text, kind: c.kind })),
  }));

  const detections: CvDetections = {
    work: { width: work.width, height: work.height },
    plot: { x0: plot.x0, y0: plot.y0, x1: plot.x1, y1: plot.y1, method: plot.method },
    markers, axisColumns, tableRows,
  };

  return {
    value, present, interpolated,
    scaleIdx, scaleLabel: BBT_SCALES[scaleIdx].label,
    confidence: q.confidence, status: q.status,
    visibleDays, truncated: visibleDays > N_DAYS,
    table, detections,
  };
}
