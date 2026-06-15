/**
 * Hybrid-router types. The router decides whether a digitization runs
 * ON-DEVICE (ONNX / CV port — private, default) or via the CLOUD LLM
 * (OpenRouter, behind explicit consent). Both produce a `DigitizeResult` so
 * the rest of the app is backend-agnostic.
 */

import type { ChartResultCv } from "../visionCv/types";
import type { Locale } from "../i18n";

export type Backend = "on-device" | "cloud";

export interface DigitizeResult extends ChartResultCv {
  backend: Backend;
  /** Short human-readable summary in the user's language (cloud path only). */
  summary?: string;
}

export interface RouterConfig {
  /** Remote kill-switch: when false, EVERYTHING stays on-device. */
  cloudEnabled: boolean;
  /** Fraction routed to cloud WHEN cloud is enabled AND consent was given. */
  cloudShare: number; // 0..1
  /** Explicit user opt-in (the "hybrid + consent" decision). */
  consentGiven: boolean;
}

export const DEFAULT_ROUTER_CONFIG: RouterConfig = {
  // Default OFF until the OpenRouter key is configured + consent UX is wired.
  cloudEnabled: false,
  cloudShare: 0.9, // the "90 % LLM / 10 % own models" target, once enabled
  consentGiven: false,
};

export interface CloudRequest {
  image: string; // data URL (base64) — sent to the server proxy only
  lang: Locale;
}
