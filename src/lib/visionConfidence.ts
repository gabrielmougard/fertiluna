/**
 * Browser-side confidence + status for the on-device ONNX vision path —
 * the failure-UX signal that decides whether to trust an auto-extraction,
 * prompt the user to verify, or reject the image.
 *
 * Mirrors the Python CV pipeline's `quality.py` so both backends speak the
 * same vocabulary ("extracted" / "low_confidence" / "not_a_chart"). The
 * signals available on the ONNX path differ from the CV pipeline (no axis-fit
 * residual), so we derive confidence from what the model does expose:
 *   - how many data points were detected (coverage),
 *   - how crisp the presence head is on the detected days,
 *   - whether a temperature series exists at all (needed to calibrate °C/°F).
 */

import type { VisionPrediction, VisionSeriesPrediction } from "./visionInference";

export type VisionStatus = "extracted" | "low_confidence" | "not_a_chart";

export interface VisionQuality {
  confidence: number; // [0,1]
  status: VisionStatus;
  reasons: string[];
}

const LOW_CONF = 0.45;

/** Mean presence probability over the days the model marked present. */
function meanPresentProb(s: VisionSeriesPrediction): number {
  const vals: number[] = [];
  for (let i = 0; i < s.normalized.length; i++) {
    if (s.normalized[i] != null) vals.push(s.confidence[i] ?? 0);
  }
  if (!vals.length) return 0;
  return vals.reduce((a, b) => a + b, 0) / vals.length;
}

export function assessVisionQuality(p: VisionPrediction): VisionQuality {
  const reasons: string[] = [];
  const tN = p.temp.presentCount;
  const lN = p.lh.presentCount;
  const best = Math.max(tN, lN);

  // coverage: ≥10 detected points = fully confident on coverage
  const markerScore = Math.min(1, best / 10);
  // crispness: how sure the presence head is on the points it kept
  const probScore = Math.max(meanPresentProb(p.temp), meanPresentProb(p.lh));
  // a temperature series is needed to anchor the °C/°F scale
  const tempPresent = tN >= 2;

  if (best === 0) reasons.push("aucun point de donnée détecté");
  else if (best < 5) reasons.push(`peu de points détectés (${best})`);
  if (!tempPresent) reasons.push("pas de courbe de température nette");

  let confidence =
    0.5 * markerScore + 0.4 * probScore + 0.1 * (tempPresent ? 1 : 0);
  confidence = Math.max(0, Math.min(1, confidence));

  let status: VisionStatus;
  if (best < 2) {
    status = "not_a_chart";
    confidence = Math.min(confidence, 0.15);
    reasons.push("aucune courbe exploitable — ce n'est probablement pas un graphique de cycle");
  } else if (confidence < LOW_CONF) {
    status = "low_confidence";
  } else {
    status = "extracted";
  }

  return { confidence: Math.round(confidence * 100) / 100, status, reasons };
}
