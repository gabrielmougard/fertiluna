/**
 * Bottom-of-screen table extractor — port of table_extract.py (core).
 *
 * Rows: calendar / CD / DPO / Sex / CM / Symptoms / hCG. Row positions come
 * from the LEFT-side labels (always present even when a row's cells are
 * sparse); COLUMNS reuse the chart day grid (the data point above a cell and
 * the cell describe the same day — the stated contract). Per-row extractors:
 *   calendar/CD/DPO → digit OCR
 *   Sex             → warm-colour heart icon (HSV)
 *   CM              → purple circle (HSV)
 *   Symptoms/hCG    → presence (any ink) + best-effort OCR
 *
 * Simplification vs Python: the polarity-independent calendar-text mask
 * (white-on-coloured-pill dates) and a separate calendar column detector are
 * omitted — we use the chart grid columns directly. Async (OCR).
 */

import { connectedComponents } from "./ops";
import { __test } from "./colorSegmentation";
import type { PlotRegion } from "./types";
import type { WorkImage } from "./preprocess";
import type { DayGrid } from "./dayAxis";
import type { OcrFn } from "./axisColumns";
import { N_DAYS } from "./constants";

const { rgbToHsv } = __test;

const CANONICAL_ROWS = ["calendar", "CD", "DPO", "Sex", "CM", "Symptoms", "hCG"];
const LABEL_KEYWORDS: [string, string][] = [
  ["cd", "CD"], ["dpo", "DPO"], ["sex", "Sex"], ["cm", "CM"],
  ["symptoms", "Symptoms"], ["symptom", "Symptoms"], ["hcg", "hCG"],
];
const MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"];
const DARK = 200;
// Sex heart (warm coral/red) — two bands wrapping the hue origin.
const HEART_BANDS: [number, number, number, number][] = [[0, 25, 90, 80], [170, 179, 90, 80]];
const PURPLE_CIRCLE: [number, number, number, number] = [125, 160, 50, 90];

export interface TableCell {
  text: string | null;
  bbox: [number, number, number, number];
  kind: "text" | "heart" | "circle" | "icon";
}
export interface TableRow {
  name: string;
  labelText: string;
  labelBbox: [number, number, number, number];
  yTop: number;
  yBottom: number;
  cells: TableCell[];
}
export interface TableData {
  rows: TableRow[];
  byName: Record<string, TableRow>;
}

function grayAt(work: WorkImage, i: number): number {
  return (work.rgba[i * 4] * 0.299 + work.rgba[i * 4 + 1] * 0.587 + work.rgba[i * 4 + 2] * 0.114) | 0;
}

function cropCanvas(work: WorkImage, x0: number, y0: number, x1: number, y1: number): HTMLCanvasElement {
  const w = Math.max(1, x1 - x0), h = Math.max(1, y1 - y0);
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  const ctx = c.getContext("2d")!;
  const out = ctx.createImageData(w, h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const si = ((y0 + y) * work.width + (x0 + x)) * 4, di = (y * w + x) * 4;
    out.data[di] = work.rgba[si]; out.data[di + 1] = work.rgba[si + 1];
    out.data[di + 2] = work.rgba[si + 2]; out.data[di + 3] = 255;
  }
  ctx.putImageData(out, 0, 0);
  return c;
}

function approxMatch(needle: string, hay: string, maxDist = 1): boolean {
  const n = needle.length, h = hay.length;
  if (n === 0 || h < n - maxDist) return false;
  for (let start = 0; start <= h - n + maxDist; start++) {
    for (let ln = Math.max(1, n - maxDist); ln <= n + maxDist; ln++) {
      const win = hay.slice(start, start + ln);
      if (Math.abs(win.length - n) > maxDist) continue;
      let d = Math.abs(win.length - n);
      for (let i = 0; i < Math.min(win.length, n); i++) if (win[i] !== needle[i]) d++;
      if (d <= maxDist) return true;
    }
  }
  return false;
}

function identifyRow(labelText: string, idx: number): string {
  const t = (labelText || "").trim().toLowerCase().replace(/[.\s]/g, "");
  if (!t) return idx < CANONICAL_ROWS.length ? CANONICAL_ROWS[idx] : `row${idx}`;
  for (const [kw, name] of LABEL_KEYWORDS) if (t.includes(kw)) return name;
  for (const [kw, name] of LABEL_KEYWORDS) if (kw.length >= 5 && approxMatch(kw, t)) return name;
  for (const m of MONTHS) if (t.startsWith(m)) return "calendar";
  return idx < CANONICAL_ROWS.length ? CANONICAL_ROWS[idx] : `row${idx}`;
}

function rowCentersFromLabels(
  work: WorkImage, regionTop: number, regionBottom: number, plot: PlotRegion,
): { centers: { cy: number; bbox: [number, number, number, number] }[]; typicalH: number } {
  const { width } = work;
  if (regionBottom - regionTop < 20) return { centers: [], typicalH: 0 };
  const xEnd = Math.min(width, Math.max(Math.round(width * 0.18), plot.x0 + Math.round(width * 0.06)));
  const mask = { data: new Uint8Array(width * work.height), width, height: work.height };
  for (let y = regionTop; y < regionBottom; y++)
    for (let x = 0; x < xEnd; x++)
      if (grayAt(work, y * width + x) <= DARK) mask.data[y * width + x] = 255;
  const { stats } = connectedComponents(mask, 8);
  const comps = stats
    .filter((s) => { const w = s.x1 - s.x0 + 1, h = s.y1 - s.y0 + 1; return s.area >= 6 && w >= 2 && h >= 4 && w <= 100 && h <= 40; })
    .map((s) => ({ cy: s.cy, x0: s.x0, y0: s.y0, x1: s.x1 + 1, y1: s.y1 + 1 }))
    .sort((a, b) => a.cy - b.cy);
  if (comps.length < 2) return { centers: [], typicalH: 0 };
  const deltas = comps.slice(1).map((c, i) => c.cy - comps[i].cy).sort((a, b) => a - b);
  const median = deltas[deltas.length >> 1];
  const large = deltas.filter((d) => d > median);
  const rowSpacing = large.length ? large[large.length >> 1] : median * 2;
  const gapThresh = Math.max(8, rowSpacing * 0.7);
  const clusters: (typeof comps)[] = [[comps[0]]];
  for (const c of comps.slice(1)) {
    if (c.cy - clusters[clusters.length - 1][clusters[clusters.length - 1].length - 1].cy < gapThresh)
      clusters[clusters.length - 1].push(c);
    else clusters.push([c]);
  }
  const centers = clusters.map((cl) => ({
    cy: cl.reduce((a, c) => a + c.cy, 0) / cl.length,
    bbox: [Math.min(...cl.map((c) => c.x0)), Math.min(...cl.map((c) => c.y0)),
      Math.max(...cl.map((c) => c.x1)), Math.max(...cl.map((c) => c.y1))] as [number, number, number, number],
  }));
  const heights = centers.map((c) => c.bbox[3] - c.bbox[1]).sort((a, b) => a - b);
  return { centers, typicalH: Math.max(12, heights[heights.length >> 1]) };
}

function countColorPixels(work: WorkImage, bb: [number, number, number, number], bands: [number, number, number, number][]): number {
  let cnt = 0;
  for (let y = bb[1]; y < bb[3]; y++) for (let x = bb[0]; x < bb[2]; x++) {
    const i = (y * work.width + x) * 4;
    const [h, s, v] = rgbToHsv(work.rgba[i], work.rgba[i + 1], work.rgba[i + 2]);
    for (const [hLo, hHi, sLo, vLo] of bands) if (h >= hLo && h <= hHi && s >= sLo && v >= vLo) { cnt++; break; }
  }
  return cnt;
}

async function extractCell(
  work: WorkImage, bb: [number, number, number, number], rowName: string, ocr: OcrFn,
): Promise<TableCell> {
  const area = Math.max(1, (bb[2] - bb[0]) * (bb[3] - bb[1]));
  const iconThresh = Math.max(6, area / 60);
  if (rowName === "Sex") {
    return { text: countColorPixels(work, bb, HEART_BANDS) >= iconThresh ? "♥" : null, bbox: bb, kind: "heart" };
  }
  if (rowName === "CM") {
    return { text: countColorPixels(work, bb, [PURPLE_CIRCLE]) >= iconThresh ? "●" : null, bbox: bb, kind: "circle" };
  }
  // text / presence rows: count dark ink, OCR if present
  let dark = 0;
  for (let y = bb[1]; y < bb[3]; y++) for (let x = bb[0]; x < bb[2]; x++) if (grayAt(work, y * work.width + x) < DARK) dark++;
  if (dark < 5) return { text: null, bbox: bb, kind: rowName === "Symptoms" || rowName === "hCG" ? "icon" : "text" };
  const txt = (await ocr(cropCanvas(work, bb[0], bb[1], bb[2], bb[3]))).trim();
  if (rowName === "Symptoms" || rowName === "hCG") return { text: txt || "•", bbox: bb, kind: "icon" };
  return { text: txt || null, bbox: bb, kind: "text" };
}

export async function extractTable(
  work: WorkImage, plot: PlotRegion, grid: DayGrid, ocr: OcrFn,
): Promise<TableData | null> {
  const regionTop = Math.min(work.height - 1, plot.y1 + 4);
  if (work.height - regionTop < 30) return null;
  const { centers, typicalH } = rowCentersFromLabels(work, regionTop, work.height, plot);
  if (centers.length < 2) return null;
  const halfH = Math.max(Math.round(typicalH * 0.85), 12);
  const half = Math.max(4, grid.cellPx * 0.45);
  const cols = grid.cells.slice(0, N_DAYS);

  const rows: TableRow[] = [];
  const byName: Record<string, TableRow> = {};
  for (let idx = 0; idx < Math.min(centers.length, CANONICAL_ROWS.length); idx++) {
    const { cy, bbox } = centers[idx];
    const yTop = Math.max(0, Math.round(cy - halfH));
    const yBot = Math.min(work.height, Math.round(cy + halfH));
    const labelText = (await ocr(cropCanvas(work, bbox[0], bbox[1], bbox[2], bbox[3]))).trim();
    const name = identifyRow(labelText, idx);
    const cells: TableCell[] = [];
    for (const cx of cols) {
      const x0 = Math.max(plot.x0, Math.round(cx - half));
      const x1 = Math.min(plot.x1, Math.round(cx + half));
      cells.push(await extractCell(work, [x0, yTop, x1, yBot], name, ocr));
    }
    while (cells.length < N_DAYS) cells.push({ text: null, bbox: [0, yTop, 0, yBot], kind: "text" });
    // dedup by name: keep the row with more populated cells
    const row: TableRow = { name, labelText, labelBbox: bbox, yTop, yBottom: yBot, cells };
    const prev = byName[name];
    if (prev) {
      const pop = (r: TableRow) => r.cells.filter((c) => c.text != null).length;
      if (pop(row) > pop(prev)) { rows[rows.indexOf(prev)] = row; byName[name] = row; }
      continue;
    }
    rows.push(row);
    byName[name] = row;
  }
  return { rows, byName };
}
