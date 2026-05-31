/**
 * Chart-vision model manifest — typed shape of chart-vision-manifest-<version>.json
 * emitted by model/fertiluna_vision/export_onnx.py.
 *
 * Versioned, sha256-validated caching identical to the cycle classifier: the
 * tiny manifest is always revalidated; the ~18 MB .onnx is cached in IndexedDB.
 */

import { MODEL_BASE_PATH } from "./manifest";

export const VISION_MODEL_VERSION = "v1";

export interface VisionManifestFileEntry {
  path: string;
  sha256: string;
  bytes: number;
}

export interface VisionManifest {
  version: string;
  task: string;
  image: {
    height: number;
    width: number;
    channels: number;
    norm_mean: [number, number, number];
    norm_std: [number, number, number];
    layout: string;
  };
  output: {
    n_series: number;
    series_names: string[]; // ["temp", "lh"]
    n_days: number;
    presence_threshold: number;
    // Axis-scale auto-detection: the model's `scale` output is a class index
    // into bbt_scales; the browser de-normalizes temp with the matching range.
    bbt_scales?: { label: string; min: number; max: number }[];
    lh_range?: { min: number; max: number };
  };
  files: { model: VisionManifestFileEntry };
  metrics: Record<string, unknown>;
  parity: Record<string, number>;
}

export function visionManifestUrl(version: string = VISION_MODEL_VERSION): string {
  return `${MODEL_BASE_PATH}/chart-vision-manifest-${version}.json`;
}

export function visionModelFileUrl(fileName: string): string {
  return `${MODEL_BASE_PATH}/${fileName}`;
}

export async function fetchVisionManifest(
  version: string = VISION_MODEL_VERSION,
): Promise<VisionManifest> {
  const res = await fetch(visionManifestUrl(version), { cache: "no-cache" });
  if (!res.ok) {
    throw new Error(
      `Échec du chargement du manifeste du modèle visuel (${res.status})`,
    );
  }
  return (await res.json()) as VisionManifest;
}
