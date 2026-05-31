/**
 * Browser chart-vision runtime — reads a chart screenshot and predicts the
 * per-day BBT + LH series, entirely on-device with onnxruntime-web.
 *
 * Pipeline:
 *   <img/canvas> ──draw to 384×224──▶ ImageNet-normalize ──▶ NCHW float32 tensor
 *                                          │
 *                                          ▼
 *                                 chart-vision-v1.onnx
 *                          (value logits, presence logits) [1,2,35]
 *                                          │
 *                          sigmoid + presence threshold
 *                                          ▼
 *                    VisionPrediction: per series, per day a normalized [0,1]
 *                    value (or null if absent) + a confidence
 *
 * The model outputs values NORMALIZED within each series' own y-range. To get
 * real units (°C / LH) the caller supplies that series' axis [min,max] and
 * de-normalizes — that's the only thing the user types (2 numbers per series),
 * which replaces the multi-click manual calibration.
 *
 * The model is fetched once and cached in IndexedDB (shared DexieJS cache).
 */

import * as ort from "onnxruntime-web/wasm";
import { CYCLE_MAX_DAYS } from "./constants";
import {
  fetchVisionManifest,
  visionModelFileUrl,
  VISION_MODEL_VERSION,
  type VisionManifest,
} from "./visionManifest";
import { getModelFile, type FetchProgress } from "./modelCache";

ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;

export type VisionSeriesKind = "temp" | "lh";

export interface VisionSeriesPrediction {
  kind: VisionSeriesKind;
  /** length CYCLE_MAX_DAYS; normalized [0,1] value or null where absent. */
  normalized: (number | null)[];
  /** length CYCLE_MAX_DAYS; presence probability per day. */
  confidence: number[];
  /** how many days the model thinks are present. */
  presentCount: number;
}

export interface VisionPrediction {
  temp: VisionSeriesPrediction;
  lh: VisionSeriesPrediction;
  /** Auto-detected BBT axis scale. */
  bbtScale: { index: number; label: string; min: number; max: number };
  /** Fixed LH axis range from the manifest. */
  lhRange: { min: number; max: number };
}

export interface VisionLoadProgress {
  phase: "manifest" | "model" | "ready";
  file?: FetchProgress;
}

let _session: ort.InferenceSession | null = null;
let _manifest: VisionManifest | null = null;
let _loading: Promise<void> | null = null;

export async function ensureVisionModelLoaded(
  onProgress?: (p: VisionLoadProgress) => void,
): Promise<void> {
  if (_session && _manifest) return;
  if (_loading) return _loading;

  _loading = (async () => {
    onProgress?.({ phase: "manifest" });
    const manifest = await fetchVisionManifest(VISION_MODEL_VERSION);
    _manifest = manifest;

    const entry = manifest.files.model;
    const buf = await getModelFile(
      `vision-${manifest.version}`,
      entry.path,
      visionModelFileUrl(entry.path),
      entry.sha256,
      entry.bytes,
      (file) => onProgress?.({ phase: "model", file }),
    );
    _session = await ort.InferenceSession.create(buf, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    onProgress?.({ phase: "ready" });
  })();

  try {
    await _loading;
  } finally {
    if (!(_session && _manifest)) _loading = null;
  }
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/**
 * Draw an image source onto an offscreen canvas at the model's input size and
 * pack it into a normalized NCHW Float32Array.
 */
function imageToTensor(
  source: CanvasImageSource,
  m: VisionManifest,
): ort.Tensor {
  const W = m.image.width;
  const H = m.image.height;
  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  // White background first (handles transparent PNGs cleanly).
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, W, H);
  ctx.drawImage(source, 0, 0, W, H);
  const { data } = ctx.getImageData(0, 0, W, H); // RGBA, row-major

  const [mr, mg, mb] = m.image.norm_mean;
  const [sr, sg, sb] = m.image.norm_std;

  // NCHW: plane R, then G, then B.
  const chw = new Float32Array(3 * H * W);
  const plane = H * W;
  for (let i = 0; i < plane; i++) {
    const r = data[i * 4] / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;
    chw[i] = (r - mr) / sr;
    chw[plane + i] = (g - mg) / sg;
    chw[2 * plane + i] = (b - mb) / sb;
  }
  return new ort.Tensor("float32", chw, [1, 3, H, W]);
}

/**
 * Run the model on an image source (HTMLImageElement / HTMLCanvasElement).
 * Returns normalized per-day predictions for both series.
 */
export async function predictChart(
  source: CanvasImageSource,
  onProgress?: (p: VisionLoadProgress) => void,
): Promise<VisionPrediction> {
  await ensureVisionModelLoaded(onProgress);
  if (!_session || !_manifest) throw new Error("Modèle visuel non chargé.");

  const input = imageToTensor(source, _manifest);
  const out = await _session.run({ image: input });
  const valueT = out["value"];
  const presentT = out["present"];
  const scaleT = out["scale"];
  if (!valueT || !presentT) throw new Error("Sorties du modèle introuvables.");

  const value = valueT.data as Float32Array; // [1,2,35] flattened
  const present = presentT.data as Float32Array;
  const nDays = _manifest.output.n_days;
  const thr = _manifest.output.presence_threshold;

  function series(idx: VisionSeriesKind, s: number): VisionSeriesPrediction {
    const normalized: (number | null)[] = new Array(CYCLE_MAX_DAYS).fill(null);
    const confidence: number[] = new Array(CYCLE_MAX_DAYS).fill(0);
    let count = 0;
    for (let d = 0; d < Math.min(nDays, CYCLE_MAX_DAYS); d++) {
      const flat = s * nDays + d;
      const p = sigmoid(present[flat]);
      confidence[d] = p;
      if (p >= thr) {
        // value is already a normalized position in [0,1] (soft-argmax); use
        // it directly, just clamp for safety. (No sigmoid — see manifest.)
        normalized[d] = Math.min(1, Math.max(0, value[flat]));
        count++;
      }
    }
    return { kind: idx, normalized, confidence, presentCount: count };
  }

  // ── auto-detect BBT axis scale (celsius / fahrenheit) ──
  const scales = _manifest.output.bbt_scales ?? [
    { label: "celsius", min: 35.6, max: 37.4 },
    { label: "fahrenheit", min: 95.0, max: 99.5 },
  ];
  let scaleIdx = 0;
  if (scaleT) {
    const sl = scaleT.data as Float32Array;
    for (let i = 1; i < sl.length && i < scales.length; i++) {
      if (sl[i] > sl[scaleIdx]) scaleIdx = i;
    }
  }
  const sc = scales[scaleIdx] ?? scales[0];
  const lhRange = _manifest.output.lh_range ?? { min: 0.1, max: 1.9 };

  return {
    temp: series("temp", 0),
    lh: series("lh", 1),
    bbtScale: { index: scaleIdx, label: sc.label, min: sc.min, max: sc.max },
    lhRange,
  };
}

/**
 * De-normalize a series' [0,1] predictions into real units using the axis
 * [min,max] the user supplied.
 *
 * The model's columns are positions across the plot region, so the data may
 * start mid-array (left offset) and have interior gaps. By default we COLLAPSE
 * the present columns left-to-right into consecutive table days (day 1, 2, 3…)
 * — what the user wants in the editable table. Pass `collapse: false` to keep
 * the original column positions (e.g. for the preview overlay).
 *
 * round defaults: temp → 0.01, lh → 0.1 (mirrors the manual digitizer).
 */
export function denormalizeSeries(
  pred: VisionSeriesPrediction,
  axisMin: number,
  axisMax: number,
  round?: (v: number) => number,
  collapse: boolean = true,
): (number | null)[] {
  const r =
    round ??
    (pred.kind === "temp"
      ? (v: number) => Math.round(v * 100) / 100
      : (v: number) => Math.round(v * 10) / 10);
  const span = axisMax - axisMin;
  const real = pred.normalized.map((n) =>
    n == null ? null : r(axisMin + n * span),
  );
  if (!collapse) return real;
  // Collapse present values to the left into consecutive days, pad with null.
  const present = real.filter((v): v is number => v != null);
  const out: (number | null)[] = new Array(real.length).fill(null);
  for (let i = 0; i < present.length; i++) out[i] = present[i];
  return out;
}

export function getVisionModelInfo(): VisionManifest | null {
  return _manifest;
}
