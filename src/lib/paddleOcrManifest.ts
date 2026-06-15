/**
 * PaddleOCR (PP-OCRv3 recognition) manifest — typed shape of
 * paddle-ocr-manifest-<version>.json emitted by
 * model/scripts/build_paddleocr_onnx.py.
 *
 * Same versioned, sha256-validated caching pattern as the cycle classifier
 * and chart-vision models: the tiny manifest is always revalidated; the
 * ~9 MB .onnx + ~200 byte dict are cached in IndexedDB.
 *
 * This is the recognition-only flavor — we don't ship the detector or
 * classifier because the in-browser chart digitizer (curveDigitizer.ts +
 * the upcoming CV-pipeline port) already produces text bounding boxes.
 * PaddleOCR's job is just: "given this small crop of one text region,
 * what does it say?".
 */

import { MODEL_BASE_PATH } from "./manifest";

export const PADDLE_OCR_VERSION = "v1";

export interface PaddleOcrManifestFileEntry {
  path: string;
  sha256: string;
  bytes: number;
}

export interface PaddleOcrManifest {
  version: string;
  task: string;
  model: {
    name: string;
    input: {
      shape: string;
      height: number;
      channel_order: "RGB" | "BGR";
      normalize: { mean: [number, number, number]; std: [number, number, number] };
      scale: string;
    };
    output: {
      shape: string;
      decoder: "CTC";
    };
  };
  files: {
    model: PaddleOcrManifestFileEntry;
    dict: PaddleOcrManifestFileEntry;
  };
  source: {
    model_url: string;
    dict_url: string;
    opset: number;
  };
}

export function paddleOcrManifestUrl(version: string = PADDLE_OCR_VERSION): string {
  return `${MODEL_BASE_PATH}/paddle-ocr-manifest-${version}.json`;
}

export function paddleOcrFileUrl(fileName: string): string {
  return `${MODEL_BASE_PATH}/${fileName}`;
}

export async function fetchPaddleOcrManifest(
  version: string = PADDLE_OCR_VERSION,
): Promise<PaddleOcrManifest> {
  const res = await fetch(paddleOcrManifestUrl(version), { cache: "no-cache" });
  if (!res.ok) {
    throw new Error(`Failed to fetch PaddleOCR manifest (${res.status})`);
  }
  return (await res.json()) as PaddleOcrManifest;
}
