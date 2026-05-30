/**
 * Feature extraction — the browser-side port of model/fertiluna/features.py.
 *
 * ⚠️ PARITY CONTRACT: this must produce identical output to the Python
 * implementation for the same input. The Python test suite emits
 * `feature-fixtures.json`; `features.test.ts` loads it and asserts equality
 * within float32 tolerance. If you change one side, change both and re-run
 * the parity test.
 *
 * Conventions (identical to Python):
 *   - A missing day is represented by `NaN`, never 0.
 *   - All features are scalars; output is a fixed Float32Array of length N_FEATURES.
 *   - `estimated_ovulation_day` uses the SENSIPLAN 3-over-6 rule.
 */

import { CYCLE_MAX_DAYS, N_FEATURES } from "./constants";

function nanCount(a: Float64Array | number[]): number {
  let c = 0;
  for (let i = 0; i < a.length; i++) if (!Number.isNaN(a[i])) c++;
  return c;
}

function nanMean(a: Float64Array | number[]): number {
  let sum = 0;
  let n = 0;
  for (let i = 0; i < a.length; i++) {
    if (!Number.isNaN(a[i])) {
      sum += a[i];
      n++;
    }
  }
  return n === 0 ? 0 : sum / n;
}

/** Population std (ddof=0), matching numpy's default. Returns 0 if < 2 values. */
function nanStd(a: Float64Array | number[]): number {
  const vals: number[] = [];
  for (let i = 0; i < a.length; i++) if (!Number.isNaN(a[i])) vals.push(a[i]);
  if (vals.length < 2) return 0;
  const mean = vals.reduce((s, v) => s + v, 0) / vals.length;
  let acc = 0;
  for (const v of vals) acc += (v - mean) * (v - mean);
  return Math.sqrt(acc / vals.length);
}

function nanMin(a: Float64Array | number[]): number {
  let m = Infinity;
  for (let i = 0; i < a.length; i++)
    if (!Number.isNaN(a[i]) && a[i] < m) m = a[i];
  return m === Infinity ? 0 : m;
}

function nanMax(a: Float64Array | number[]): number {
  let m = -Infinity;
  for (let i = 0; i < a.length; i++)
    if (!Number.isNaN(a[i]) && a[i] > m) m = a[i];
  return m === -Infinity ? 0 : m;
}

function nanMedian(a: Float64Array | number[]): number {
  const vals: number[] = [];
  for (let i = 0; i < a.length; i++) if (!Number.isNaN(a[i])) vals.push(a[i]);
  if (vals.length === 0) return 0;
  vals.sort((x, y) => x - y);
  const mid = Math.floor(vals.length / 2);
  return vals.length % 2 === 1 ? vals[mid] : (vals[mid - 1] + vals[mid]) / 2;
}

function slice(a: Float64Array, start: number, end: number): Float64Array {
  return a.subarray(Math.max(0, start), Math.min(a.length, end));
}

/**
 * SENSIPLAN 3-over-6 rule. Find the first day d such that temps[d..d+2] are all
 * strictly greater than max(temps[d-6..d-1]), with the 3rd day ≥ 0.20 above that
 * max. Returns the index of the LAST follicular day (one before the rise), or -1.
 */
function detectOvulationDaySensiplan(temps: Float64Array): number {
  const n = temps.length;
  for (let d = 6; d < n - 2; d++) {
    const prior = slice(temps, d - 6, d);
    if (nanCount(prior) < 4) continue;
    const rising = slice(temps, d, d + 3);
    if (nanCount(rising) < 3) continue;
    const priorMax = nanMax(prior);
    let allAbove = true;
    for (let i = 0; i < rising.length; i++) {
      if (!(rising[i] > priorMax)) {
        allAbove = false;
        break;
      }
    }
    if (!allAbove) continue;
    if (rising[2] - priorMax < 0.2) continue;
    return d - 1;
  }
  return -1;
}

function longestRun(mask: boolean[]): number {
  let longest = 0;
  let current = 0;
  for (const v of mask) {
    if (v) {
      current++;
      if (current > longest) longest = current;
    } else {
      current = 0;
    }
  }
  return longest;
}

function findLocalMaxima(arr: Float64Array, threshold: number): number[] {
  const out: number[] = [];
  const n = arr.length;
  for (let i = 0; i < n; i++) {
    const v = arr[i];
    if (Number.isNaN(v) || v <= threshold) continue;
    let left = i > 0 ? arr[i - 1] : -Infinity;
    let right = i < n - 1 ? arr[i + 1] : -Infinity;
    if (Number.isNaN(left)) left = -Infinity;
    if (Number.isNaN(right)) right = -Infinity;
    if (v >= left && v >= right) out.push(i);
  }
  return out;
}

/** Least-squares slope over NaN-filtered (dayIndex, value). 0 if < 2 valid. */
function slope(arr: Float64Array): number {
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < arr.length; i++) {
    if (!Number.isNaN(arr[i])) {
      xs.push(i);
      ys.push(arr[i]);
    }
  }
  if (xs.length < 2) return 0;
  const xMean = xs.reduce((s, v) => s + v, 0) / xs.length;
  const yMean = ys.reduce((s, v) => s + v, 0) / ys.length;
  let num = 0;
  let den = 0;
  for (let i = 0; i < xs.length; i++) {
    num += (xs[i] - xMean) * (ys[i] - yMean);
    den += (xs[i] - xMean) * (xs[i] - xMean);
  }
  if (den === 0) return 0;
  return num / den;
}

/**
 * Extract the fixed-length feature vector from raw curves.
 *
 * @param temps length CYCLE_MAX_DAYS, °C or NaN for missing
 * @param lh    length CYCLE_MAX_DAYS, relative units or NaN for missing
 * @returns Float32Array of length N_FEATURES
 */
export function extractFeatures(
  temps: Float64Array,
  lh: Float64Array,
): Float32Array {
  if (temps.length !== CYCLE_MAX_DAYS || lh.length !== CYCLE_MAX_DAYS) {
    throw new Error(
      `extractFeatures expects arrays of length ${CYCLE_MAX_DAYS}`,
    );
  }
  const f = new Float32Array(N_FEATURES);

  const nTemp = nanCount(temps);
  const nLh = nanCount(lh);

  f[0] = nTemp;
  f[1] = nLh;
  f[2] = 1 - nTemp / CYCLE_MAX_DAYS;
  f[3] = 1 - nLh / CYCLE_MAX_DAYS;

  f[4] = nanMean(temps);
  f[5] = nanStd(temps);
  f[6] = nTemp > 0 ? nanMin(temps) : 0;
  f[7] = nTemp > 0 ? nanMax(temps) : 0;
  f[8] = f[7] - f[6];

  const ov = detectOvulationDaySensiplan(temps);
  f[9] = ov >= 0 ? ov + 1 : 0;

  if (ov >= 0) {
    const follicular = slice(temps, 0, ov + 1);
    const luteal = slice(temps, ov + 1, CYCLE_MAX_DAYS);
    f[10] = nanMean(follicular);
    f[11] = nanStd(follicular);
    f[12] = nanMean(luteal);
    f[13] = nanStd(luteal);
    f[14] = Math.max(0, f[12] - f[10]);

    let steepness = 0;
    const lo = Math.max(0, ov - 1);
    const hi = Math.min(CYCLE_MAX_DAYS - 1, ov + 3);
    for (let d = lo; d < hi; d++) {
      if (!(Number.isNaN(temps[d]) || Number.isNaN(temps[d + 1]))) {
        steepness = Math.max(steepness, temps[d + 1] - temps[d]);
      }
    }
    f[15] = steepness;

    const plateauThreshold = f[10] + 0.15;
    const postVals: number[] = [];
    for (let i = 0; i < luteal.length; i++)
      if (!Number.isNaN(luteal[i])) postVals.push(luteal[i]);

    let plateauCount = 0;
    for (const v of postVals) if (v > plateauThreshold) plateauCount++;
    f[16] = plateauCount;

    const postMaskFull: boolean[] = [];
    for (let i = 0; i < luteal.length; i++) {
      postMaskFull.push(!Number.isNaN(luteal[i]) && luteal[i] > plateauThreshold);
    }
    f[17] = longestRun(postMaskFull);

    const follMax = nanCount(follicular) > 0 ? nanMax(follicular) : 0;
    let dips = 0;
    for (const v of postVals) if (v < follMax) dips++;
    f[18] = dips;

    f[23] = ov + 1;
    let lutealObserved = 0;
    for (let i = 0; i < luteal.length; i++)
      if (!Number.isNaN(luteal[i])) lutealObserved++;
    f[24] = lutealObserved;
  } else {
    f[10] = nanMean(temps);
    f[11] = nanStd(temps);
    f[12] = 0;
    f[13] = 0;
    f[14] = 0;
    f[15] = 0;
    f[16] = 0;
    f[17] = 0;
    f[18] = 0;
    f[23] = 0;
    f[24] = 0;
  }

  // LH features
  if (nLh > 0) {
    const lhBaseline = nanMedian(lh);
    const peaks = findLocalMaxima(lh, Math.max(1.3, lhBaseline * 1.5));
    if (peaks.length > 0) {
      let best = peaks[0];
      for (const p of peaks) if (lh[p] > lh[best]) best = p;
      f[19] = best + 1;
      f[20] = lh[best];
      f[21] = peaks.length;
      f[22] = ov >= 0 ? ov + 1 - (best + 1) : 0;
    } else {
      f[19] = 0;
      f[20] = nanMax(lh);
      f[21] = 0;
      f[22] = 0;
    }
  } else {
    f[19] = 0;
    f[20] = 0;
    f[21] = 0;
    f[22] = 0;
  }

  // Slopes around (estimated) ovulation
  if (ov >= 0) {
    f[25] = slope(slice(temps, Math.max(0, ov - 4), ov + 1));
    f[26] = slope(slice(temps, ov + 1, Math.min(CYCLE_MAX_DAYS, ov + 5)));
  } else {
    f[25] = slope(slice(temps, 0, 10));
    f[26] = slope(slice(temps, 10, 20));
  }

  if (nTemp > 0) {
    const overallMean = f[4];
    let below = 0;
    const aboveMask: boolean[] = [];
    const belowMask: boolean[] = [];
    for (let i = 0; i < temps.length; i++) {
      const valid = !Number.isNaN(temps[i]);
      const isBelow = valid && temps[i] < overallMean;
      const isAbove = valid && temps[i] >= overallMean;
      if (isBelow) below++;
      aboveMask.push(isAbove);
      belowMask.push(isBelow);
    }
    f[27] = below / Math.max(1, nTemp);
    f[28] = longestRun(aboveMask);
    f[29] = longestRun(belowMask);
  } else {
    f[27] = 0;
    f[28] = 0;
    f[29] = 0;
  }

  return f;
}

/**
 * Convert a per-day input (numbers or null for missing) into a NaN-filled
 * Float64Array of length CYCLE_MAX_DAYS, the format extractFeatures expects.
 */
export function toCurveArray(days: (number | null)[]): Float64Array {
  const out = new Float64Array(CYCLE_MAX_DAYS).fill(NaN);
  for (let i = 0; i < Math.min(days.length, CYCLE_MAX_DAYS); i++) {
    const v = days[i];
    if (v !== null && v !== undefined && Number.isFinite(v)) out[i] = v;
  }
  return out;
}
