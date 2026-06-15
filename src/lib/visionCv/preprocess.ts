/**
 * Image → working-canvas RGBA, port of preprocess.py.
 *
 * Resizes the source into the [WORK_W_MIN, WORK_W_MAX] width band preserving
 * aspect ratio (skips the resize when already in range), so every downstream
 * threshold operates at a consistent scale. Pure canvas — no OpenCV.js.
 */

import { WORK_W_MAX, WORK_W_MIN } from "./constants";

export interface WorkImage {
  rgba: Uint8ClampedArray; // length = width*height*4
  width: number;
  height: number;
  scale: number; // applied resize factor vs source (1 = unchanged)
}

export function toWorkCanvas(source: CanvasImageSource): WorkImage {
  const srcW =
    "naturalWidth" in source && (source as HTMLImageElement).naturalWidth
      ? (source as HTMLImageElement).naturalWidth
      : (source as { width: number }).width;
  const srcH =
    "naturalHeight" in source && (source as HTMLImageElement).naturalHeight
      ? (source as HTMLImageElement).naturalHeight
      : (source as { height: number }).height;

  let scale = 1;
  if (srcW < WORK_W_MIN) scale = WORK_W_MIN / srcW;
  else if (srcW > WORK_W_MAX) scale = WORK_W_MAX / srcW;
  const width = Math.max(1, Math.round(srcW * scale));
  const height = Math.max(1, Math.round(srcH * scale));

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(source, 0, 0, width, height);
  const { data } = ctx.getImageData(0, 0, width, height);
  return { rgba: data, width, height, scale };
}
