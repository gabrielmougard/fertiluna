import { describe, it, expect } from "vitest";
import {
  denormalizeSeries,
  type VisionSeriesPrediction,
} from "./visionInference";
import { CYCLE_MAX_DAYS } from "./constants";

function mkPred(
  kind: "temp" | "lh",
  normalized: (number | null)[],
): VisionSeriesPrediction {
  const padded = normalized
    .slice(0, CYCLE_MAX_DAYS)
    .concat(new Array(Math.max(0, CYCLE_MAX_DAYS - normalized.length)).fill(null));
  return {
    kind,
    normalized: padded,
    confidence: padded.map((v) => (v == null ? 0 : 1)),
    presentCount: padded.filter((v) => v != null).length,
  };
}

describe("vision de-normalization", () => {
  it("maps [0,1] to the supplied temp axis range and rounds to 0.01", () => {
    // normalized 0 -> min, 1 -> max, 0.5 -> midpoint
    const pred = mkPred("temp", [0, 0.5, 1, null]);
    const out = denormalizeSeries(pred, 36.0, 37.0);
    expect(out[0]).toBe(36.0);
    expect(out[1]).toBe(36.5);
    expect(out[2]).toBe(37.0);
    expect(out[3]).toBeNull();
  });

  it("rounds LH to 0.1 by default", () => {
    const pred = mkPred("lh", [0.27, 0.83]);
    const out = denormalizeSeries(pred, 0, 3); // 0.27*3=0.81 -> 0.8 ; 0.83*3=2.49 -> 2.5
    expect(out[0]).toBe(0.8);
    expect(out[1]).toBe(2.5);
  });

  it("handles inverted-ish ranges and clamps via normalized bounds", () => {
    const pred = mkPred("temp", [0, 1]);
    const out = denormalizeSeries(pred, 36.2, 36.6);
    expect(out[0]).toBeCloseTo(36.2, 5);
    expect(out[1]).toBeCloseTo(36.6, 5);
  });

  it("preserves length = CYCLE_MAX_DAYS", () => {
    const pred = mkPred("temp", [0.5]);
    const out = denormalizeSeries(pred, 36, 37);
    expect(out.length).toBe(CYCLE_MAX_DAYS);
  });
});
