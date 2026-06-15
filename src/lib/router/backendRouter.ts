/**
 * Hybrid backend router (client side).
 *
 * Decides per-request whether to digitize ON-DEVICE (the CV TS port — private,
 * the default) or via the CLOUD LLM proxy (`/api/digitize`). The cloud path is
 * gated by the "hybrid + consent" rules:
 *   - a remote kill-switch (`cloudEnabled`) can force everything on-device,
 *   - the cloud path runs ONLY with explicit user consent,
 *   - when both hold, a configurable share (default 90 %) goes to cloud.
 * Any cloud failure transparently falls back to on-device, so the user always
 * gets a result.
 */

import { runPipelineCv } from "../visionCv/pipeline";
import { ensurePaddleOcrLoaded, recognizeText } from "../paddleOcr";
import type { Locale } from "../i18n";
import {
  DEFAULT_ROUTER_CONFIG,
  type Backend,
  type DigitizeResult,
  type RouterConfig,
} from "./types";

/** Pure decision — exported for testing. */
export function chooseBackend(
  cfg: RouterConfig,
  rand: () => number = Math.random,
): Backend {
  if (!cfg.cloudEnabled || !cfg.consentGiven) return "on-device";
  return rand() < cfg.cloudShare ? "cloud" : "on-device";
}

function toDataUrl(source: CanvasImageSource): string {
  const w =
    (source as HTMLImageElement).naturalWidth ||
    (source as { width: number }).width;
  const h =
    (source as HTMLImageElement).naturalHeight ||
    (source as { height: number }).height;
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  canvas.getContext("2d")!.drawImage(source, 0, 0, w, h);
  // JPEG keeps the upload small; charts tolerate mild compression.
  return canvas.toDataURL("image/jpeg", 0.9);
}

/** On-device OCR for axis calibration — PaddleOCR via ORT-Web, stays local. */
async function deviceOcr(canvas: HTMLCanvasElement): Promise<string> {
  await ensurePaddleOcrLoaded();
  return (await recognizeText(canvas)).text;
}

export async function digitizeOnDevice(source: CanvasImageSource): Promise<DigitizeResult> {
  const r = await runPipelineCv(source, deviceOcr);
  return { ...r, backend: "on-device" };
}

export async function digitizeCloud(
  source: CanvasImageSource,
  locale: Locale,
): Promise<DigitizeResult> {
  const res = await fetch("/api/digitize", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ image: toDataUrl(source), lang: locale }),
  });
  if (!res.ok) throw new Error(`cloud digitize failed: ${res.status}`);
  return (await res.json()) as DigitizeResult;
}

/**
 * Digitize a chart image via the chosen backend, falling back to on-device on
 * any cloud error. `cfg` defaults to cloud-off (safe baseline).
 */
export async function digitize(
  source: CanvasImageSource,
  locale: Locale = "fr",
  cfg: RouterConfig = DEFAULT_ROUTER_CONFIG,
): Promise<DigitizeResult> {
  const backend = chooseBackend(cfg);
  if (backend === "cloud") {
    try {
      return await digitizeCloud(source, locale);
    } catch (e) {
      // Transparent fallback: the user still gets an on-device result.
      console.warn("[router] cloud path failed, falling back on-device:", e);
      return digitizeOnDevice(source);
    }
  }
  return digitizeOnDevice(source);
}
