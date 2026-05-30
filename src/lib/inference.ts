/**
 * Browser ML runtime — loads the ONNX models (from the DexieJS cache) and runs
 * inference entirely client-side with onnxruntime-web (WASM backend).
 *
 * Pipeline for Outil 1:
 *   raw curves ──extractFeatures──▶ (30,) vector
 *                                    │
 *                       ┌───────────┴────────────┐
 *                       ▼                         ▼
 *               classifier.onnx            iforest.onnx
 *           (calibrated probabilities)   (OOD anomaly score)
 *                       │                         │
 *                       └──────────┬──────────────┘
 *                                  ▼
 *                          CycleAnalysis (label, confidence,
 *                          per-class probs, ovulation day,
 *                          OOD percentile, derived metrics)
 *
 * Nothing here touches the network except the one-time model download, and
 * nothing is ever uploaded. All computation is on-device.
 */

import * as ort from "onnxruntime-web/wasm";
import {
  CONFIDENCE_THRESHOLD,
  LABELS,
  type CycleLabel,
  N_FEATURES,
  MODEL_VERSION,
} from "./constants";
import { extractFeatures } from "./features";
import {
  fetchManifest,
  modelFileUrl,
  type ModelManifest,
} from "./manifest";
import { getModelFile, type FetchProgress } from "./modelCache";

// Point ORT-web at the wasm assets bundled by Vite. The package ships its own
// wasm; setting the path lets the worker/CDN serve them. We keep threads off
// (no SharedArrayBuffer / COOP-COEP headers needed) and use a single proxy-free
// session for predictability.
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
// Note: we intentionally do NOT set ort.env.wasm.wasmPaths. We use the bundle
// build of onnxruntime-web/wasm, whose glue resolves the `.wasm` via a Vite-
// rewritten `new URL(...)` reference. Everything stays same-origin under the
// Cloudflare Worker; nothing is fetched from a CDN.

export interface DerivedMetrics {
  estimatedOvulationDay: number | null; // 1-indexed cycle day, or null
  follicularLength: number | null;
  lutealLength: number | null;
  thermalRiseAmplitude: number;
  lhPeakDay: number | null;
  lhPeakCount: number;
}

export interface CycleAnalysis {
  label: CycleLabel;
  /** Max calibrated class probability. */
  confidence: number;
  /** Whether confidence cleared the roadmap's 0.6 gate. */
  confident: boolean;
  /** Per-class calibrated probabilities, keyed by label. */
  probabilities: Record<CycleLabel, number>;
  /** Raw feature vector (for debugging / scrollytelling). */
  features: Float32Array;
  derived: DerivedMetrics;
  /** 0..100 — how unusual this curve is vs. the training distribution. */
  oodPercentile: number;
}

export interface LoadProgress {
  phase: "manifest" | "classifier" | "iforest" | "ready";
  file?: FetchProgress;
}

let _classifier: ort.InferenceSession | null = null;
let _iforest: ort.InferenceSession | null = null;
let _manifest: ModelManifest | null = null;
let _loadingPromise: Promise<void> | null = null;

/**
 * Ensure the models are loaded (idempotent, deduped). Safe to call repeatedly;
 * the heavy work happens once.
 */
export async function ensureModelLoaded(
  onProgress?: (p: LoadProgress) => void,
): Promise<void> {
  if (_classifier && _iforest && _manifest) return;
  if (_loadingPromise) return _loadingPromise;

  _loadingPromise = (async () => {
    onProgress?.({ phase: "manifest" });
    const manifest = await fetchManifest(MODEL_VERSION);
    _manifest = manifest;

    const clfEntry = manifest.files.classifier;
    const clfBuf = await getModelFile(
      manifest.version,
      clfEntry.path,
      modelFileUrl(clfEntry.path),
      clfEntry.sha256,
      clfEntry.bytes,
      (file) => onProgress?.({ phase: "classifier", file }),
    );
    _classifier = await ort.InferenceSession.create(clfBuf, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });

    const ifEntry = manifest.files.iforest;
    const ifBuf = await getModelFile(
      manifest.version,
      ifEntry.path,
      modelFileUrl(ifEntry.path),
      ifEntry.sha256,
      ifEntry.bytes,
      (file) => onProgress?.({ phase: "iforest", file }),
    );
    _iforest = await ort.InferenceSession.create(ifBuf, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });

    onProgress?.({ phase: "ready" });
  })();

  try {
    await _loadingPromise;
  } finally {
    // allow re-attempt on failure
    if (!(_classifier && _iforest && _manifest)) _loadingPromise = null;
  }
}

function softmaxAlreadyApplied(_probs: Float32Array): void {
  /* The classifier outputs calibrated probabilities directly (zipmap=false). */
}

/** Map a raw iforest decision_function score to a 0..100 "unusualness" scale. */
function oodPercentile(score: number, m: ModelManifest): number {
  const { p5, p50, p95 } = m.iforest_score_percentiles;
  // Lower score ⇒ more anomalous. Map [p5..p95] → [100..0] with p50 ≈ 50.
  if (score <= p5) return 100;
  if (score >= p95) return 0;
  if (score <= p50) {
    // p5..p50 → 100..50
    const t = (score - p5) / (p50 - p5 || 1);
    return Math.round(100 - t * 50);
  }
  // p50..p95 → 50..0
  const t = (score - p50) / (p95 - p50 || 1);
  return Math.round(50 - t * 50);
}

/**
 * Run the full Outil 1 analysis on raw curves.
 *
 * @param temps Float64Array(35) with NaN for missing days
 * @param lh    Float64Array(35) with NaN for missing days
 */
export async function analyzeCycle(
  temps: Float64Array,
  lh: Float64Array,
): Promise<CycleAnalysis> {
  await ensureModelLoaded();
  if (!_classifier || !_iforest || !_manifest) {
    throw new Error("Le modèle n'est pas chargé.");
  }

  const features = extractFeatures(temps, lh);
  if (features.length !== N_FEATURES) {
    throw new Error("Vecteur de features invalide.");
  }

  const input = new ort.Tensor("float32", features, [1, N_FEATURES]);

  // --- classifier ---
  const clfOut = await _classifier.run({ input });
  const probTensor = clfOut["probabilities"] ?? clfOut["output_probability"];
  if (!probTensor) throw new Error("Sortie 'probabilities' introuvable.");
  const probs = probTensor.data as Float32Array;
  softmaxAlreadyApplied(probs);

  let bestIdx = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[bestIdx]) bestIdx = i;
  const confidence = probs[bestIdx];

  const probabilities = {} as Record<CycleLabel, number>;
  for (let i = 0; i < LABELS.length; i++) probabilities[LABELS[i]] = probs[i];

  // Confidence gate (roadmap §4: if max proba < 0.6 → "données insuffisantes").
  let label: CycleLabel = LABELS[bestIdx];
  const confident = confidence >= CONFIDENCE_THRESHOLD;
  if (!confident) label = "donnees_insuffisantes";

  // --- iforest (advisory OOD) ---
  let ood = 0;
  try {
    const ifOut = await _iforest.run({ input });
    const scoreTensor = ifOut["scores"] ?? ifOut["scores_output"];
    if (scoreTensor) {
      const s = (scoreTensor.data as Float32Array)[0];
      ood = oodPercentile(s, _manifest);
    }
  } catch {
    ood = 0;
  }

  return {
    label,
    confidence,
    confident,
    probabilities,
    features,
    derived: deriveMetrics(features),
    oodPercentile: ood,
  };
}

/** Pull human-meaningful numbers out of the feature vector for the result UI. */
function deriveMetrics(f: Float32Array): DerivedMetrics {
  const ovDay = f[9] > 0 ? Math.round(f[9]) : null;
  const follLen = f[23] > 0 ? Math.round(f[23]) : null;
  const lutLen = f[24] > 0 ? Math.round(f[24]) : null;
  const lhPeakDay = f[19] > 0 ? Math.round(f[19]) : null;
  return {
    estimatedOvulationDay: ovDay,
    follicularLength: follLen,
    lutealLength: lutLen,
    thermalRiseAmplitude: f[14],
    lhPeakDay,
    lhPeakCount: Math.round(f[21]),
  };
}

/** Expose manifest metrics for an "À propos du modèle" panel. */
export function getModelInfo(): ModelManifest | null {
  return _manifest;
}
