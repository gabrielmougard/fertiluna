/**
 * Browser PaddleOCR runtime — PP-OCRv3 English recognition only.
 *
 * Recognizes text given a CROP of one text region. Detection and
 * orientation are NOT in scope: the in-browser chart digitizer already
 * produces text bounding boxes (axis ticks, table cells), so PaddleOCR's
 * job is the leaf step: "what does this 30-px crop say?".
 *
 * Pipeline per OCR call:
 *   <crop> ──▶ resize to height=48, pad width to mult of 32 ──▶ RGB normalize
 *                                          │
 *                                          ▼
 *                              paddle-ocr-rec-v1.onnx
 *                                  (1, 3, 48, W)
 *                                          │
 *                              CTC greedy decode + vocab lookup
 *                                          ▼
 *                                  "97.5" / "MAR" / …
 *
 * The model + dict are fetched once and cached in IndexedDB via the shared
 * DexieJS cache — same versioned-by-sha256 pattern as the cycle classifier
 * and chart-vision models.
 */

import * as ort from "onnxruntime-web/wasm";
import {
  fetchPaddleOcrManifest,
  paddleOcrFileUrl,
  PADDLE_OCR_VERSION,
  type PaddleOcrManifest,
} from "./paddleOcrManifest";
import { getModelFile, type FetchProgress } from "./modelCache";

ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;

const REC_HEIGHT = 48;
const WIDTH_MULTIPLE = 32;
const MAX_WIDTH = 640;
// CTC blank class always occupies index 0.
const BLANK_INDEX = 0;

export interface PaddleOcrLoadProgress {
  phase: "manifest" | "model" | "dict" | "ready";
  file?: FetchProgress;
}

export interface PaddleOcrResult {
  text: string;
  /** Mean per-character softmax probability across the decoded sequence. */
  confidence: number;
}

let _session: ort.InferenceSession | null = null;
let _vocab: string[] | null = null;
let _manifest: PaddleOcrManifest | null = null;
let _loading: Promise<void> | null = null;

/**
 * Ensure the recognition model + character dict are downloaded and a
 * session is open. Safe to call repeatedly — only the first call does work.
 */
export async function ensurePaddleOcrLoaded(
  onProgress?: (p: PaddleOcrLoadProgress) => void,
): Promise<void> {
  if (_session && _vocab && _manifest) return;
  if (_loading) return _loading;

  _loading = (async () => {
    onProgress?.({ phase: "manifest" });
    const manifest = await fetchPaddleOcrManifest(PADDLE_OCR_VERSION);
    _manifest = manifest;

    // Model file
    const modelEntry = manifest.files.model;
    const modelBuf = await getModelFile(
      `paddleocr-${manifest.version}`,
      modelEntry.path,
      paddleOcrFileUrl(modelEntry.path),
      modelEntry.sha256,
      modelEntry.bytes,
      (file) => onProgress?.({ phase: "model", file }),
    );

    // Dict file — tiny but still cache-managed for sha256 validation.
    const dictEntry = manifest.files.dict;
    const dictBuf = await getModelFile(
      `paddleocr-${manifest.version}`,
      dictEntry.path,
      paddleOcrFileUrl(dictEntry.path),
      dictEntry.sha256,
      dictEntry.bytes,
      (file) => onProgress?.({ phase: "dict", file }),
    );
    const dictText = new TextDecoder("utf-8").decode(new Uint8Array(dictBuf));
    const dictChars = dictText.split("\n").filter((s) => s.length > 0);
    // PP-OCRv3 postprocess ALWAYS appends a trailing space class on top of
    // whatever dict is loaded. The vocab the model emits is [<blank>] +
    // dictChars + [" "]; total classes = dictChars.length + 2.
    _vocab = ["<blank>", ...dictChars, " "];

    _session = await ort.InferenceSession.create(modelBuf, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    onProgress?.({ phase: "ready" });
  })();

  try {
    await _loading;
  } finally {
    if (!(_session && _vocab && _manifest)) _loading = null;
  }
}

/**
 * Convert an ImageData-style RGBA crop into the model's NCHW float32 tensor.
 * - Resizes to fixed height = 48 keeping aspect ratio.
 * - Pads width to a multiple of 32 (model is fully-convolutional in width).
 * - Normalizes to [-1, 1].
 *
 * `source` can be anything drawable to a canvas (HTMLImageElement,
 * HTMLCanvasElement, ImageBitmap, OffscreenCanvas).
 */
function cropToTensor(source: CanvasImageSource): ort.Tensor {
  // Read source dimensions (works for HTMLImageElement, ImageBitmap, etc.).
  const srcW =
    "width" in source && typeof source.width === "number"
      ? source.width
      : (source as HTMLImageElement).naturalWidth;
  const srcH =
    "height" in source && typeof source.height === "number"
      ? source.height
      : (source as HTMLImageElement).naturalHeight;
  if (!srcW || !srcH) {
    // Degenerate input — emit a 32×48 white tensor so the session still runs.
    const data = new Float32Array(1 * 3 * REC_HEIGHT * WIDTH_MULTIPLE);
    data.fill(1.0); // (1 - 0.5)/0.5 = 1
    return new ort.Tensor("float32", data, [1, 3, REC_HEIGHT, WIDTH_MULTIPLE]);
  }
  // Scale so height = REC_HEIGHT; pick width = ceil( srcW * 48/srcH ) padded
  // up to WIDTH_MULTIPLE; clamp at MAX_WIDTH.
  let newW = Math.max(
    WIDTH_MULTIPLE,
    Math.round((srcW * REC_HEIGHT) / srcH),
  );
  newW = Math.min(MAX_WIDTH, newW);
  newW = Math.ceil(newW / WIDTH_MULTIPLE) * WIDTH_MULTIPLE;

  const canvas = document.createElement("canvas");
  canvas.width = newW;
  canvas.height = REC_HEIGHT;
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  // White background first so transparent-PNG crops don't bleed through.
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, newW, REC_HEIGHT);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(source, 0, 0, newW, REC_HEIGHT);
  const { data: rgba } = ctx.getImageData(0, 0, newW, REC_HEIGHT);

  const data = new Float32Array(1 * 3 * REC_HEIGHT * newW);
  const stride = REC_HEIGHT * newW;
  for (let y = 0; y < REC_HEIGHT; y++) {
    for (let x = 0; x < newW; x++) {
      const i = (y * newW + x) * 4;
      // RGB order. PaddleOCR rec normalize: (px/255 - 0.5)/0.5 = px/127.5 - 1
      const r = rgba[i] / 127.5 - 1.0;
      const g = rgba[i + 1] / 127.5 - 1.0;
      const b = rgba[i + 2] / 127.5 - 1.0;
      const idx = y * newW + x;
      data[idx] = r;                 // R channel
      data[stride + idx] = g;        // G channel
      data[2 * stride + idx] = b;    // B channel
    }
  }
  return new ort.Tensor("float32", data, [1, 3, REC_HEIGHT, newW]);
}

/**
 * Greedy CTC decode: per timestep argmax; collapse repeats; drop the blank
 * class. Returns (text, mean_softmax_confidence_over_kept_tokens).
 */
function ctcDecode(out: Float32Array, T: number, C: number): PaddleOcrResult {
  if (!_vocab) throw new Error("paddleOcr: vocab not loaded");
  const chars: string[] = [];
  const confs: number[] = [];
  let prev = -1;
  for (let t = 0; t < T; t++) {
    // argmax + softmax of this row
    const row = out.subarray(t * C, (t + 1) * C);
    let cls = 0;
    let max = row[0];
    for (let c = 1; c < C; c++) {
      if (row[c] > max) {
        max = row[c];
        cls = c;
      }
    }
    if (cls === prev) continue;
    prev = cls;
    if (cls === BLANK_INDEX || cls >= _vocab.length) continue;
    // softmax for this timestep's confidence
    let sum = 0;
    for (let c = 0; c < C; c++) sum += Math.exp(row[c] - max);
    confs.push(1 / sum);
    chars.push(_vocab[cls]);
  }
  const text = chars.join("").trim();
  const confidence =
    confs.length === 0 ? 0 : confs.reduce((a, b) => a + b, 0) / confs.length;
  return { text, confidence };
}

/**
 * Recognize text in a cropped image region.
 *
 * @param source any CanvasImageSource (HTMLImageElement, ImageBitmap,
 *               HTMLCanvasElement, OffscreenCanvas, …).
 */
export async function recognizeText(
  source: CanvasImageSource,
): Promise<PaddleOcrResult> {
  if (!_session || !_vocab) {
    throw new Error("paddleOcr: call ensurePaddleOcrLoaded() first");
  }
  const input = cropToTensor(source);
  const inputName = _session.inputNames[0];
  const outputs = await _session.run({ [inputName]: input });
  const outName = _session.outputNames[0];
  const out = outputs[outName];
  // PP-OCRv3 rec output shape: (1, T, C). Convert to flat row-major Float32Array.
  const dims = out.dims as number[];
  const T = dims[1];
  const C = dims[2];
  const data = out.data as Float32Array;
  return ctcDecode(data, T, C);
}

/** Convenience helper: recognize from raw RGBA pixels in a Uint8ClampedArray. */
export async function recognizeFromRGBA(
  pixels: Uint8ClampedArray,
  width: number,
  height: number,
): Promise<PaddleOcrResult> {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d")!;
  // Build ImageData then copy the pixels in — avoids the ImageData(view, …)
  // overload whose typing rejects a possibly-SharedArrayBuffer-backed view.
  const img = ctx.createImageData(width, height);
  img.data.set(pixels);
  ctx.putImageData(img, 0, 0);
  return recognizeText(canvas);
}
