/**
 * HSV colour segmentation — port of color_segmentation.py.
 *
 * Produces the blue (BBT) / orange (LH-Ratio) / purple (LH-Level) masks.
 * The HSV conversion reproduces OpenCV's exact scale (H ∈ [0,179],
 * S,V ∈ [0,255]) so the Python thresholds in constants transfer unchanged.
 *
 * Pure TypeScript over the RGBA pixel buffer — no OpenCV.js needed for this
 * stage. (cv2's small CLOSE morphology that bridges 1-2px line gaps is
 * deferred to the ops layer; the raw masks are already usable downstream.)
 */

import { HSV_BLUE, HSV_ORANGE, HSV_PURPLE } from "./constants";
import type { Mask, SeriesMasks } from "./types";
import type { WorkImage } from "./preprocess";

/**
 * RGB → HSV in OpenCV's 8-bit convention.
 * H ∈ [0,179] (degrees/2), S ∈ [0,255], V ∈ [0,255]. Matches cv2.cvtColor
 * BGR2HSV rounding closely enough for thresholding.
 */
function rgbToHsv(r: number, g: number, b: number): [number, number, number] {
  const rf = r / 255, gf = g / 255, bf = b / 255;
  const max = Math.max(rf, gf, bf);
  const min = Math.min(rf, gf, bf);
  const d = max - min;
  let h = 0;
  if (d !== 0) {
    if (max === rf) h = ((gf - bf) / d) % 6;
    else if (max === gf) h = (bf - rf) / d + 2;
    else h = (rf - gf) / d + 4;
    h *= 60;
    if (h < 0) h += 360;
  }
  const s = max === 0 ? 0 : d / max;
  // OpenCV: H/2 (so 0..179), S*255, V*255
  return [Math.round(h / 2), Math.round(s * 255), Math.round(max * 255)];
}

function emptyMask(width: number, height: number): Mask {
  return { data: new Uint8Array(width * height), width, height };
}

function inBand(
  h: number, s: number, v: number,
  band: [number, number, number, number],
): boolean {
  const [hLo, hHi, sLo, vLo] = band;
  return h >= hLo && h <= hHi && s >= sLo && v >= vLo;
}

export function segment(img: WorkImage): SeriesMasks {
  const { rgba, width, height } = img;
  const blue = emptyMask(width, height);
  const orange = emptyMask(width, height);
  const purple = emptyMask(width, height);
  const n = width * height;
  for (let i = 0; i < n; i++) {
    const r = rgba[i * 4];
    const g = rgba[i * 4 + 1];
    const b = rgba[i * 4 + 2];
    const [h, s, v] = rgbToHsv(r, g, b);
    if (inBand(h, s, v, HSV_BLUE)) blue.data[i] = 255;
    else if (inBand(h, s, v, HSV_ORANGE)) orange.data[i] = 255;
    else if (inBand(h, s, v, HSV_PURPLE)) purple.data[i] = 255;
  }
  return { blue, orange, purple };
}

// exported for unit testing the HSV conversion against known colours.
export const __test = { rgbToHsv, inBand };
