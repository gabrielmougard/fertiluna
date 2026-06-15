/**
 * Multi-axis y-label reader — port of axis_columns.py.
 *
 * Detects each numeric y-axis COLUMN (left: Ratio/Level, right: BBT °F/°C),
 * OCRs the labels, robustly fits value = a + b·y per column, and classifies
 * each by its value range. This is what calibrates marker-y → real units and
 * reads the °C/°F scale off the data (not a leading-digit guess).
 *
 * Async because OCR (PaddleOCR via ORT-Web) is async. Uses the pure-TS
 * connected-components + dilation in ops; OCR is injected as a function so the
 * module stays decoupled from the recognizer.
 */

import { connectedComponents, dilate } from "./ops";
import type { Mask, PlotRegion } from "./types";
import type { WorkImage } from "./preprocess";

export type Side = "left" | "right";
export type AxisKind = "ratio" | "level" | "bbt_f" | "bbt_c" | "unknown";

const AXIS_PROFILES: Record<Exclude<AxisKind, "unknown">, [number, number]> = {
  ratio: [1.9, 0.1],
  level: [95.0, 5.0],
  bbt_f: [99.5, 95.0],
  bbt_c: [37.4, 35.6],
};

export interface LabelBox {
  cx: number;
  cy: number;
  bbox: [number, number, number, number]; // x0,y0,x1,y1
  text: string;
  value: number | null;
}

export interface AxisColumn {
  side: Side;
  xCenter: number;
  boxes: LabelBox[];
  a: number;
  b: number;
  rmse: number;
  kind: AxisKind;
  nFit: number;
  valueAt(y: number): number;
}

/** OCR a cropped region → text. Injected so the recognizer stays decoupled. */
export type OcrFn = (canvas: HTMLCanvasElement) => Promise<string>;

function cropCanvas(work: WorkImage, x0: number, y0: number, x1: number, y1: number): HTMLCanvasElement {
  const w = Math.max(1, x1 - x0), h = Math.max(1, y1 - y0);
  const canvas = document.createElement("canvas");
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext("2d")!;
  const out = ctx.createImageData(w, h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const si = ((y0 + y) * work.width + (x0 + x)) * 4;
      const di = (y * w + x) * 4;
      out.data[di] = work.rgba[si];
      out.data[di + 1] = work.rgba[si + 1];
      out.data[di + 2] = work.rgba[si + 2];
      out.data[di + 3] = 255;
    }
  }
  ctx.putImageData(out, 0, 0);
  return canvas;
}

/** dark (gray≤200) AND low-saturation (≤60) text mask in the margin band. */
function wordBoxes(work: WorkImage, side: Side, plot: PlotRegion): LabelBox[] {
  const { rgba, width, height } = work;
  const bandW = Math.max(120, Math.round(width * 0.2));
  const x0 = side === "right" ? Math.max(0, width - bandW) : 0;
  const x1 = side === "right" ? width : Math.min(width, bandW);
  if (x1 - x0 < 30) return [];
  const y1b = Math.min(height, Math.round(height * 0.85));
  const mask: Mask = { data: new Uint8Array(width * height), width, height };
  for (let y = 0; y < y1b; y++) {
    for (let x = x0; x < x1; x++) {
      const i = (y * width + x) * 4;
      const r = rgba[i], g = rgba[i + 1], b = rgba[i + 2];
      const gray = (r * 0.299 + g * 0.587 + b * 0.114) | 0;
      const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
      const sat = mx === 0 ? 0 : ((mx - mn) / mx) * 255;
      if (gray <= 200 && sat <= 60) mask.data[y * width + x] = 255;
    }
  }
  // join digits of one number horizontally (not across the column gap)
  const merged = dilate(mask, 9, 3);
  const { stats } = connectedComponents(merged, 8);
  const out: LabelBox[] = [];
  for (const s of stats) {
    const w = s.x1 - s.x0 + 1, h = s.y1 - s.y0 + 1;
    if (s.area < 10 || w < 4 || h < 6 || h > 40 || w > 130) continue;
    out.push({ cx: s.cx, cy: s.cy, bbox: [s.x0, s.y0, s.x1 + 1, s.y1 + 1], text: "", value: null });
  }
  return out;
}

function clusterColumns(boxes: LabelBox[], tol = 40): LabelBox[][] {
  const cols: LabelBox[][] = [];
  for (const b of [...boxes].sort((p, q) => p.cx - q.cx)) {
    let placed = false;
    for (const c of cols) {
      const mean = c.reduce((a, z) => a + z.cx, 0) / c.length;
      if (Math.abs(b.cx - mean) < tol) { c.push(b); placed = true; break; }
    }
    if (!placed) cols.push([b]);
  }
  return cols;
}

const NUM_RX = /-?\d*\.?\d+/;
function parseToken(text: string): number | null {
  const t = (text || "").trim().replace(/^[><≥≤=~ ]+/, "");
  if (!t) return null;
  const m = t.replace(/,/g, ".").match(NUM_RX);
  if (!m) return null;
  const body = m[0];
  const v = parseFloat(body.startsWith(".") ? "0" + body : body);
  return Number.isNaN(v) ? null : v;
}

async function ocrAndValue(col: LabelBox[], work: WorkImage, ocr: OcrFn): Promise<void> {
  col.sort((a, b) => a.cy - b.cy);
  for (const b of col) {
    const [x0, y0, x1, y1] = b.bbox;
    const canvas = cropCanvas(work, Math.max(0, x0 - 2), Math.max(0, y0 - 2),
      Math.min(work.width, x1 + 2), Math.min(work.height, y1 + 2));
    b.text = (await ocr(canvas)) || "";
    b.value = parseToken(b.text);
  }
}

function largestApSubsetY(boxes: LabelBox[]): LabelBox[] {
  const valued = boxes.filter((b) => b.value != null).sort((a, b) => a.cy - b.cy);
  const n = valued.length;
  if (n < 3) return valued;
  const ys = valued.map((b) => b.cy);
  const gaps: number[] = [];
  for (let i = 1; i < n; i++) { const d = ys[i] - ys[i - 1]; if (d > 4) gaps.push(d); }
  if (!gaps.length) return valued;
  const maxG = Math.max(...gaps);
  const nb = Math.max(6, Math.min(20, gaps.length));
  const hist = new Array<number>(nb).fill(0);
  const bw = maxG / nb;
  for (const g of gaps) hist[Math.min(nb - 1, Math.floor(g / bw))]++;
  let peak = 0; for (let i = 1; i < nb; i++) if (hist[i] > hist[peak]) peak = i;
  const modal = (peak + 0.5) * bw;
  if (modal < 8) return valued;
  const tol = Math.max(6, 0.25 * modal);
  let best: number[] = [];
  for (let start = 0; start < n; start++) {
    const chain = [start];
    for (let k = start + 1; k < n; k++) {
      const d = ys[k] - ys[chain[chain.length - 1]];
      const steps = Math.round(d / modal);
      if (steps >= 1 && Math.abs(d - steps * modal) <= tol) chain.push(k);
    }
    if (chain.length > best.length) best = chain;
  }
  return best.length < 3 ? valued : best.map((i) => valued[i]);
}

function robustFit(pts: LabelBox[]): { a: number; b: number; rms: number; n: number } | null {
  if (pts.length < 2) return null;
  const fit = (arr: LabelBox[]) => {
    const n = arr.length;
    let sy = 0, sv = 0, syy = 0, syv = 0;
    for (const p of arr) { const y = p.cy, v = p.value!; sy += y; sv += v; syy += y * y; syv += y * v; }
    const denom = n * syy - sy * sy;
    if (Math.abs(denom) < 1e-9) return null;
    const b = (n * syv - sy * sv) / denom;
    const a = (sv - b * sy) / n;
    let se = 0; for (const p of arr) { const e = p.value! - (a + b * p.cy); se += e * e; }
    return { a, b, rms: Math.sqrt(se / n), n };
  };
  let r = fit(pts);
  if (!r) return null;
  if (r.rms > 1e-6 && pts.length >= 3) {
    const keep = pts.filter((p) => Math.abs(p.value! - (r!.a + r!.b * p.cy)) <= 1.5 * r!.rms);
    if (keep.length >= 2) { const r2 = fit(keep); if (r2) r = r2; }
  }
  return r;
}

function fitColumn(col: AxisColumn): boolean {
  const valued = col.boxes.filter((b) => b.value != null);
  if (valued.length < 3) return false;
  const pts = [...valued].sort((a, b) => a.cy - b.cy);
  // RANSAC over all valued boxes: pick the pair whose line has the most inliers.
  let bestInliers: LabelBox[] = [];
  for (let i = 0; i < pts.length; i++) {
    for (let j = i + 1; j < pts.length; j++) {
      const yi = pts[i].cy, vi = pts[i].value!, yj = pts[j].cy, vj = pts[j].value!;
      if (Math.abs(yi - yj) < 8) continue;
      const b = (vi - vj) / (yi - yj);
      const a = vi - b * yi;
      if (b >= 0) continue; // axis decreases as y grows
      const span = Math.abs((a + b * pts[0].cy) - (a + b * pts[pts.length - 1].cy));
      const tol = Math.max(0.15, (0.3 * span) / Math.max(1, pts.length - 1));
      const inl = pts.filter((p) => Math.abs(p.value! - (a + b * p.cy)) <= tol);
      if (inl.length > bestInliers.length) bestInliers = inl;
    }
  }
  const r = robustFit(bestInliers.length >= 3 ? bestInliers : valued);
  if (!r || r.b >= 0) return false;
  col.a = r.a; col.b = r.b; col.rmse = r.rms; col.nFit = r.n;
  return true;
}

function classify(col: AxisColumn): AxisKind {
  const valued = col.boxes.filter((b) => b.value != null);
  if (valued.length < 3) return "unknown";
  const ys = valued.map((b) => b.cy);
  const vTop = col.valueAt(Math.min(...ys));
  const vBot = col.valueAt(Math.max(...ys));
  const obsHi = Math.max(vTop, vBot), obsLo = Math.min(vTop, vBot);
  let bestKind: AxisKind = "unknown", bestErr = Infinity;
  for (const [kind, [top, bot]] of Object.entries(AXIS_PROFILES)) {
    const lo = Math.min(top, bot), hi = Math.max(top, bot);
    const err = (Math.abs(obsHi - hi) + Math.abs(obsLo - lo)) / (hi - lo);
    if (err < bestErr) { bestErr = err; bestKind = kind as AxisKind; }
  }
  return bestErr < 0.4 ? bestKind : "unknown";
}

function makeColumn(side: Side, group: LabelBox[]): AxisColumn {
  const col: AxisColumn = {
    side,
    xCenter: group.reduce((a, b) => a + b.cx, 0) / group.length,
    boxes: group,
    a: 0, b: 0, rmse: 0, kind: "unknown", nFit: 0,
    valueAt(y: number) { return this.a + this.b * y; },
  };
  return col;
}

export async function detectAxisColumns(
  work: WorkImage,
  plot: PlotRegion,
  ocr: OcrFn,
  minBoxes = 3,
): Promise<AxisColumn[]> {
  const out: AxisColumn[] = [];
  for (const side of ["left", "right"] as Side[]) {
    const boxes = wordBoxes(work, side, plot);
    for (const group of clusterColumns(boxes)) {
      if (group.length < minBoxes) continue;
      const col = makeColumn(side, group);
      await ocrAndValue(col.boxes, work, ocr);
      if (col.boxes.filter((b) => b.value != null).length < minBoxes) continue;
      col.boxes = largestApSubsetY(col.boxes);
      if (col.boxes.filter((b) => b.value != null).length < minBoxes) continue;
      if (!fitColumn(col)) continue;
      col.kind = classify(col);
      out.push(col);
    }
  }
  out.sort((a, b) => a.xCenter - b.xCenter);
  return out;
}

export interface ResolvedAxes {
  bbt: AxisColumn | null;
  scaleIdx: number | null; // 0=celsius, 1=fahrenheit
  scaleConfidence: number;
  ratio: AxisColumn | null;
  level: AxisColumn | null;
  columns: AxisColumn[];
}

export function resolveAxes(columns: AxisColumn[]): ResolvedAxes {
  const best = (kind: AxisKind): AxisColumn | null => {
    const c = columns.filter((x) => x.kind === kind);
    if (!c.length) return null;
    return c.sort((p, q) => q.nFit - p.nFit || p.rmse - q.rmse)[0];
  };
  const bbtF = best("bbt_f"), bbtC = best("bbt_c");
  let bbt: AxisColumn | null, scaleIdx: number | null;
  if (bbtF && bbtC) {
    if (bbtC.nFit > bbtF.nFit || (bbtC.nFit === bbtF.nFit && bbtC.rmse < bbtF.rmse)) {
      bbt = bbtC; scaleIdx = 0;
    } else { bbt = bbtF; scaleIdx = 1; }
  } else if (bbtF) { bbt = bbtF; scaleIdx = 1; }
  else if (bbtC) { bbt = bbtC; scaleIdx = 0; }
  else { bbt = null; scaleIdx = null; }
  let scaleConf = 0;
  if (bbt) scaleConf = Math.min(1, 0.5 + 0.1 * bbt.nFit) * (bbt.rmse < 0.3 ? 1 : 0.7);
  return {
    bbt, scaleIdx, scaleConfidence: scaleConf,
    ratio: best("ratio"), level: best("level"), columns,
  };
}
