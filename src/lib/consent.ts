/**
 * Cloud-analysis consent — persisted opt-in for the hybrid router.
 *
 * The cloud (LLM) path sends a chart image off-device, so per the
 * "hybrid + consent" decision it is OFF by default and only runs after an
 * explicit, remembered opt-in. This module persists that choice and builds
 * the RouterConfig the backend router consumes.
 *
 * `cloudEnabled` (the kill-switch / "is the cloud backend available at all")
 * is a separate, deployment-level flag — exposed here as a constant so it can
 * be flipped on once the OpenRouter key is configured. Consent alone does not
 * route to cloud unless the backend is enabled.
 */

import { DEFAULT_ROUTER_CONFIG, type RouterConfig } from "./router/types";

const CONSENT_KEY = "fertiluna.cloudConsent.v1";

// Flip to true once OPENROUTER_API_KEY is configured + you want the cloud path
// live. Until then the router stays 100 % on-device regardless of consent.
// NOTE: the cloud path only actually runs if /api/digitize has an
// OPENROUTER_API_KEY (set it in .dev.vars locally, or as a wrangler secret in
// prod). Without the key the endpoint returns 503 and the client falls back to
// the on-device CV pipeline, so enabling this flag is always safe.
export const CLOUD_BACKEND_ENABLED = true;

export function getConsent(): boolean {
  try {
    return localStorage.getItem(CONSENT_KEY) === "1";
  } catch {
    return false;
  }
}

export function setConsent(value: boolean): void {
  try {
    localStorage.setItem(CONSENT_KEY, value ? "1" : "0");
  } catch {
    /* storage unavailable (private mode) — consent simply won't persist */
  }
}

/** RouterConfig reflecting the current consent + deployment flag. */
export function getRouterConfig(): RouterConfig {
  return {
    ...DEFAULT_ROUTER_CONFIG,
    cloudEnabled: CLOUD_BACKEND_ENABLED,
    consentGiven: getConsent(),
  };
}
