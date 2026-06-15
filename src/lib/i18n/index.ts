/**
 * Lightweight i18n — locale detection + a tiny `t()` over typed catalogs.
 *
 * Two jobs:
 *   1. pick the UI locale (FR primary, EN for the alias domain / EN traffic),
 *   2. provide the target language code to the LLM router so cloud results
 *      come back in the trafficked user's language.
 *
 * No framework — Astro pages can call `t(key, locale)` and the router/consent
 * code shares the same catalogs. Detection works both at the edge (Cloudflare
 * `request.cf.country` + Accept-Language) and on the client (navigator.language).
 */

import { fr } from "./fr";
import { en } from "./en";

export type Locale = "fr" | "en";
export const DEFAULT_LOCALE: Locale = "fr"; // French-first (TikTok audience)
export const SUPPORTED: Locale[] = ["fr", "en"];

export type MessageKey = keyof typeof fr;
// Values are plain strings (each locale differs); the KEY set is what's
// shared + type-checked across catalogs.
export type Catalog = Record<MessageKey, string>;

const CATALOGS: Record<Locale, Catalog> = { fr, en };

/** Translate a key for a locale, falling back to the default locale then key. */
export function t(key: MessageKey, locale: Locale = DEFAULT_LOCALE): string {
  return CATALOGS[locale]?.[key] ?? CATALOGS[DEFAULT_LOCALE][key] ?? key;
}

/** Substitute `{name}` placeholders in a string with values from `vars`. */
export function format(
  template: string,
  vars: Record<string, string | number>,
): string {
  return template.replace(/\{(\w+)\}/g, (m, k) =>
    k in vars ? String(vars[k]) : m,
  );
}

/** Translate + interpolate in one call. */
export function tf(
  key: MessageKey,
  locale: Locale,
  vars: Record<string, string | number>,
): string {
  return format(t(key, locale), vars);
}

/** Normalize an arbitrary language tag (e.g. "en-US", "fr-CA") to a Locale. */
export function toLocale(tag: string | null | undefined): Locale {
  if (!tag) return DEFAULT_LOCALE;
  const base = tag.toLowerCase().split("-")[0];
  return (SUPPORTED as string[]).includes(base) ? (base as Locale) : DEFAULT_LOCALE;
}

/**
 * Pick a locale from request signals (edge or client). Order of preference:
 *   1. explicit override (e.g. ?lang= or a cookie) — caller passes it,
 *   2. country (FR/BE/CH-fr lean French; most others → English baseline),
 *   3. Accept-Language / navigator.language,
 *   4. default.
 * Country is a soft hint only; language headers win when present.
 */
export function detectLocale(opts: {
  override?: string | null;
  country?: string | null;
  acceptLanguage?: string | null;
}): Locale {
  if (opts.override) return toLocale(opts.override);
  if (opts.acceptLanguage) {
    // first tag in the Accept-Language list
    const first = opts.acceptLanguage.split(",")[0]?.trim();
    if (first) return toLocale(first);
  }
  const frCountries = ["FR", "BE", "LU", "MC"];
  if (opts.country && frCountries.includes(opts.country.toUpperCase())) {
    return "fr";
  }
  return DEFAULT_LOCALE;
}

/** Human language name for an LLM "respond in X" instruction. */
export function languageName(locale: Locale): string {
  return locale === "fr" ? "French" : "English";
}

/** Cookie name persisting an explicit FR/EN choice across navigation. */
export const LOCALE_COOKIE = "lang";

/**
 * Resolve the UI locale for an SSR request. Preference:
 *   1. explicit ?lang override (the header FR/EN toggle),
 *   2. a persisted cookie (a previous explicit pick),
 *   3. the trafficked user's signals (Accept-Language, then country),
 *   4. default locale.
 */
export function resolveRequestLocale(input: {
  override?: string | null;
  cookie?: string | null;
  acceptLanguage?: string | null;
  country?: string | null;
}): Locale {
  if (input.override) return toLocale(input.override);
  if (input.cookie) return toLocale(input.cookie);
  return detectLocale({
    acceptLanguage: input.acceptLanguage,
    country: input.country,
  });
}
