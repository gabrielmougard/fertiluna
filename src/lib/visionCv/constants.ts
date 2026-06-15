/**
 * Constants for the in-browser CV digitizer — a TypeScript port of the Python
 * reference in model/fertiluna_vision_cv/constants.py. Kept byte-for-byte in
 * step with it: the HSV bands, working-canvas band, and day count MUST match
 * so the TS port reproduces the Python pipeline's output.
 */

import { CYCLE_MAX_DAYS } from "../constants";

export const N_DAYS = CYCLE_MAX_DAYS; // 35
export const N_SERIES = 2; // [temp (BBT), lh]
export const SERIES_NAMES = ["temp", "lh"] as const;

// BBT axis conventions (index → [min,max] in real units). Matches BBT_SCALES.
export const BBT_SCALES: { label: string; min: number; max: number }[] = [
  { label: "celsius", min: 35.6, max: 37.4 },
  { label: "fahrenheit", min: 95.0, max: 99.5 },
];
export const LH_RANGE = { min: 0.1, max: 1.9 };

// Working-canvas width band. Don't aggressively downscale phone screenshots
// (markers go sub-pixel) nor upscale tiny images past 2×.
export const WORK_W_MIN = 1200;
export const WORK_W_MAX = 2400;

/**
 * HSV bands in OpenCV's scale — H ∈ [0,179], S/V ∈ [0,255]. The TS HSV
 * conversion below MUST produce H in [0,179] to reuse these thresholds.
 * Each band: [hueLo, hueHi, satLo, valLo]; hue upper bound, S/V upper = max.
 * Non-overlapping hue ranges so purple "Level" never leaks into blue.
 */
export const HSV_BLUE: [number, number, number, number] = [95, 124, 12, 100];
export const HSV_ORANGE: [number, number, number, number] = [0, 18, 30, 120];
export const HSV_PURPLE: [number, number, number, number] = [128, 162, 35, 100];

export const PRESENCE_THRESHOLD = 0.5;
