/**
 * Ad configuration — read from the Cloudflare Worker env at request time.
 *
 * The whole ad layer is env-gated: with no `ADSENSE_PUB_ID` configured (the
 * default, and always in `DEV_MODE`), every <AdSlot> renders a labelled,
 * layout-stable placeholder instead of a real AdSense unit. This keeps the site
 * running ad-free locally and on a not-yet-approved domain, while reserving the
 * exact pixel space so there is zero layout shift (CLS) once real ads turn on.
 *
 * Slot IDs map to the ad units described in docs/AD_SETUP_GUIDE.md. They are
 * optional: a placement with no matching slot id still renders the placeholder.
 */

export interface AdConfig {
  /** AdSense publisher id, e.g. `ca-pub-XXXXXXXXXXXXXXXX`. */
  pubId: string | null;
  /** When true, force placeholders even if a pubId is present (local dev). */
  devMode: boolean;
  /** Per-placement AdSense slot ids (10-ish digit strings). */
  slots: {
    top: string | null;
    inline: string | null;
    rail: string | null;
    bottom: string | null;
  };
}

type RuntimeEnv = Record<string, string | undefined>;

function readEnv(locals: unknown): RuntimeEnv {
  // Cloudflare adapter exposes worker vars/secrets at locals.runtime.env.
  const env = (locals as { runtime?: { env?: RuntimeEnv } })?.runtime?.env;
  return env ?? {};
}

/** Resolve the ad configuration for the current request. */
export function getAdConfig(locals: unknown): AdConfig {
  const env = readEnv(locals);
  const pubId = (env.ADSENSE_PUB_ID || "").trim() || null;
  const devMode = (env.DEV_MODE || "").toLowerCase() === "true";
  return {
    pubId,
    devMode,
    slots: {
      top: (env.ADSENSE_SLOT_TOP || "").trim() || null,
      inline: (env.ADSENSE_SLOT_INLINE || "").trim() || null,
      rail: (env.ADSENSE_SLOT_RAIL || "").trim() || null,
      bottom: (env.ADSENSE_SLOT_BOTTOM || "").trim() || null,
    },
  };
}

/** Whether real AdSense markup (vs. a placeholder) should be emitted. */
export function adsLive(cfg: AdConfig): boolean {
  return !!cfg.pubId && !cfg.devMode;
}
