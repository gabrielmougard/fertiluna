/**
 * Pipeline confidence + status — port of quality.py (CV-path version).
 * A readable temperature axis is a GATE: without it values can't be
 * calibrated, so the result can't be "extracted".
 */

import type { ResolvedAxes } from "./axisColumns";
import type { ChartStatus } from "./types";

const W = { axis: 0.25, scale: 0.15, markers: 0.3, grid: 0.2, plot: 0.1 };
const LOW_CONF = 0.45;

function gridScore(source: string): number {
  return ({ "date-row": 1, gridlines: 0.95, autocorr: 0.8, "marker-spacing": 0.6, uniform: 0.3 } as Record<string, number>)[source] ?? 0.4;
}
function plotScore(method: string): number {
  if (method.includes("axiscol") || method.includes("ticks")) return 1;
  if (method.includes("ink+grid")) return 0.8;
  if (method.includes("ink")) return 0.6;
  return 0.3;
}

export interface QualityReport {
  confidence: number;
  status: ChartStatus;
  components: Record<string, number>;
  reasons: string[];
}

export function assessQuality(args: {
  present: number[][];
  scaleConfidence: number;
  axes: ResolvedAxes | null;
  gridSource: string;
  plotMethod: string;
  visibleDays: number;
  truncated: boolean;
}): QualityReport {
  const reasons: string[] = [];
  const bbt = args.axes?.bbt ?? null;
  let axisScore = 0;
  if (bbt && bbt.nFit >= 2) {
    axisScore = Math.min(1, bbt.nFit / 8) * Math.exp(-Math.min(bbt.rmse, 3));
    if (bbt.nFit < 5) reasons.push(`BBT axis fit on only ${bbt.nFit} ticks`);
  } else reasons.push("no BBT temperature axis detected");

  const scaleScore = Math.max(0, Math.min(1, args.scaleConfidence));
  const bbtN = args.present[0].reduce((a, b) => a + b, 0);
  const lhN = args.present[1].reduce((a, b) => a + b, 0);
  const best = Math.max(bbtN, lhN);
  const markerScore = Math.min(1, best / 10);
  if (best === 0) reasons.push("no data points recovered");
  else if (best < 5) reasons.push(`few data points recovered (${best})`);

  const gScore = gridScore(args.gridSource);
  const pScore = plotScore(args.plotMethod);

  const axisPresent = axisScore > 0;
  const effMarker = markerScore * (axisPresent ? 1 : 0.25);
  let confidence =
    W.axis * axisScore + W.scale * scaleScore + W.markers * effMarker +
    W.grid * gScore + W.plot * pScore;
  confidence = Math.max(0, Math.min(1, confidence));

  let status: ChartStatus;
  if (!axisPresent && best < 2) {
    status = "not_a_chart";
    confidence = Math.min(confidence, 0.15);
    reasons.push("no axis and no data points — likely not a cycle chart");
  } else if (!axisPresent) {
    status = "low_confidence";
    confidence = Math.min(confidence, LOW_CONF - 0.01);
    reasons.push("no readable temperature axis — values not calibrated");
  } else if (confidence < LOW_CONF) {
    status = "low_confidence";
  } else {
    status = "extracted";
  }
  if (args.truncated) reasons.push(`chart shows ~${args.visibleDays} days; only the first 35 were kept`);

  return {
    confidence: Math.round(confidence * 100) / 100,
    status,
    components: {
      axis: +axisScore.toFixed(3), scale: +scaleScore.toFixed(3),
      markers: +effMarker.toFixed(3), grid: +gScore.toFixed(3),
      plot: +pScore.toFixed(3), bbtPoints: bbtN, lhPoints: lhN,
    },
    reasons,
  };
}
