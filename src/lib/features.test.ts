import { describe, it, expect } from "vitest";
import fixtures from "./__fixtures__/feature-fixtures.json";
import { extractFeatures, toCurveArray } from "./features";
import { N_FEATURES } from "./constants";

interface Fixture {
  archetype: string;
  k: number;
  temps: (number | null)[];
  lh: (number | null)[];
  expected_features: number[];
}

const cases = fixtures as Fixture[];

describe("feature extraction parity (TS vs Python)", () => {
  it("has fixtures to test", () => {
    expect(cases.length).toBeGreaterThan(0);
  });

  for (const c of cases) {
    it(`matches Python for ${c.archetype}#${c.k}`, () => {
      const temps = toCurveArray(c.temps);
      const lh = toCurveArray(c.lh);
      const got = extractFeatures(temps, lh);

      expect(got.length).toBe(N_FEATURES);
      expect(c.expected_features.length).toBe(N_FEATURES);

      for (let i = 0; i < N_FEATURES; i++) {
        const a = got[i];
        const b = c.expected_features[i];
        // float32 tolerance. Both sides cast to float32; allow a small abs+rel eps.
        const tol = 1e-4 + 1e-4 * Math.abs(b);
        expect(
          Math.abs(a - b),
          `feature[${i}] got=${a} expected=${b}`,
        ).toBeLessThanOrEqual(tol);
      }
    });
  }
});
