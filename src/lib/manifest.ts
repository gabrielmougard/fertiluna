/**
 * Model manifest — the typed shape of model-manifest-<version>.json emitted by
 * model/fertiluna/export_onnx.py. Used for versioned caching: the browser
 * fetches the manifest (tiny, never cached aggressively), compares the version
 * + sha256 against what's stored in IndexedDB, and only re-downloads the heavy
 * .onnx files when they actually change.
 */

import { MODEL_VERSION } from "./constants";

export interface ManifestFileEntry {
  path: string;
  sha256: string;
  bytes: number;
}

export interface ModelManifest {
  version: string;
  labels: string[];
  n_features: number;
  feature_means: number[];
  feature_stds: number[];
  confidence_threshold: number;
  iforest_score_percentiles: { p5: number; p50: number; p95: number };
  files: {
    classifier: ManifestFileEntry;
    iforest: ManifestFileEntry;
  };
  metrics: {
    accuracy: number;
    log_loss: number;
    [k: string]: unknown;
  };
  parity: Record<string, number>;
}

/** Where the static assets live (served by the Cloudflare Worker from /public). */
export const MODEL_BASE_PATH = "/models";

export function manifestUrl(version: string = MODEL_VERSION): string {
  return `${MODEL_BASE_PATH}/model-manifest-${version}.json`;
}

export function modelFileUrl(fileName: string): string {
  return `${MODEL_BASE_PATH}/${fileName}`;
}

export async function fetchManifest(
  version: string = MODEL_VERSION,
): Promise<ModelManifest> {
  // cache: "no-cache" → always revalidate the small manifest so version bumps
  // are picked up immediately; the heavy .onnx files are cached in IndexedDB.
  const res = await fetch(manifestUrl(version), { cache: "no-cache" });
  if (!res.ok) {
    throw new Error(`Échec du chargement du manifeste modèle (${res.status})`);
  }
  return (await res.json()) as ModelManifest;
}
