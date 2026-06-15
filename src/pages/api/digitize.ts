/**
 * Server proxy for the CLOUD (LLM) digitization path.
 *
 * Runs on the Cloudflare Worker (Astro SSR). The OpenRouter API key lives ONLY
 * here as a server secret (`OPENROUTER_API_KEY`) — it is never shipped to the
 * browser. The client posts a consented image + target language; we call
 * OpenRouter's vision chat completion, normalize the result into the shared
 * schema, and return it.
 *
 * Privacy: this endpoint is the deliberate, CONSENTED exception to the
 * otherwise-on-device design. It is only ever called after the user opts in
 * (see router/backendRouter.ts). We do not log image content. If no key is
 * configured the endpoint returns 503 so the client falls back to on-device.
 *
 * Secrets:
 *   dev  — put OPENROUTER_API_KEY (+ optional OPENROUTER_MODEL) in .dev.vars
 *   prod — `wrangler secret put OPENROUTER_API_KEY`
 */

import type { APIRoute } from "astro";
import { BBT_SCALES, LH_RANGE, N_DAYS, N_SERIES } from "../../lib/visionCv/constants";
import { buildMessages, parseLlmJson } from "../../lib/router/openrouterPrompt";
import { languageName, toLocale } from "../../lib/i18n";

export const prerender = false;

const OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions";
// Smoke-tested default: gemini-3.5-flash read accurate per-day temp+LH (97.3,
// 97.25, 97.35 … matching the CV pipeline) where gpt-4o-mini returned flat
// values and 2.5-flash returned out-of-range ones. It's a "thinking" model,
// so it needs a generous max_tokens (reasoning eats the budget) and the
// json_object response format to stay fence-free + complete.
const DEFAULT_MODEL = "google/gemini-3.5-flash";

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function clamp01(x: number): number {
  return Math.max(0, Math.min(1, x));
}

function normalizeSeries(
  vals: (number | null)[], lo: number, hi: number,
): { value: number[]; present: number[] } {
  const span = Math.max(1e-6, hi - lo);
  const value = new Array<number>(N_DAYS).fill(0);
  const present = new Array<number>(N_DAYS).fill(0);
  for (let d = 0; d < N_DAYS; d++) {
    const v = vals[d];
    if (v == null || Number.isNaN(v)) continue;
    value[d] = clamp01((v - lo) / span);
    present[d] = 1;
  }
  return { value, present };
}

export const POST: APIRoute = async (context) => {
  const env =
    ((context.locals as Record<string, any>)?.runtime?.env as
      | Record<string, string>
      | undefined) ?? {};
  const key = env.OPENROUTER_API_KEY ?? import.meta.env.OPENROUTER_API_KEY;
  if (!key) {
    // No key configured → tell the client to use on-device.
    return json({ error: "cloud_unavailable" }, 503);
  }
  const model = env.OPENROUTER_MODEL ?? DEFAULT_MODEL;

  let payload: { image?: string; lang?: string };
  try {
    payload = await context.request.json();
  } catch {
    return json({ error: "bad_request" }, 400);
  }
  if (!payload.image || !payload.image.startsWith("data:image/")) {
    return json({ error: "missing_image" }, 400);
  }
  const locale = toLocale(payload.lang);

  let content: string;
  try {
    const resp = await fetch(OPENROUTER_URL, {
      method: "POST",
      headers: {
        authorization: `Bearer ${key}`,
        "content-type": "application/json",
        // OpenRouter attribution headers (optional but recommended).
        "x-title": "FertiLuna",
      },
      body: JSON.stringify({
        model,
        messages: buildMessages(payload.image, languageName(locale)),
        temperature: 0,
        // Generous budget: "thinking" vision models spend tokens on reasoning
        // before emitting the 35+35 arrays; too small truncates the JSON.
        max_tokens: 4096,
        // Keep the output a single fence-free JSON object.
        response_format: { type: "json_object" },
      }),
    });
    if (!resp.ok) {
      return json({ error: "upstream_error", status: resp.status }, 502);
    }
    const data = (await resp.json()) as {
      choices?: { message?: { content?: string } }[];
    };
    content = data?.choices?.[0]?.message?.content ?? "";
  } catch {
    return json({ error: "upstream_unreachable" }, 502);
  }

  let parsed;
  try {
    parsed = parseLlmJson(content);
  } catch {
    return json({ error: "unparseable_model_output" }, 502);
  }

  const scaleIdx = parsed.scale === "fahrenheit" ? 1 : 0;
  const [bbtLo, bbtHi] = [BBT_SCALES[scaleIdx].min, BBT_SCALES[scaleIdx].max];
  const temp = normalizeSeries(parsed.temp, bbtLo, bbtHi);
  const lh = normalizeSeries(parsed.lh, LH_RANGE.min, LH_RANGE.max);

  const value = new Array(N_SERIES);
  const present = new Array(N_SERIES);
  value[0] = temp.value; present[0] = temp.present;
  value[1] = lh.value; present[1] = lh.present;
  const interpolated = Array.from({ length: N_SERIES }, () =>
    new Array<number>(N_DAYS).fill(0),
  );

  const confidence = clamp01(parsed.confidence);
  const bbtN = present[0].reduce((a: number, b: number) => a + b, 0);
  const lhN = present[1].reduce((a: number, b: number) => a + b, 0);
  const best = Math.max(bbtN, lhN);
  const status =
    best < 2 ? "not_a_chart" : confidence < 0.45 ? "low_confidence" : "extracted";

  return json({
    backend: "cloud",
    value,
    present,
    interpolated,
    scaleIdx,
    scaleLabel: BBT_SCALES[scaleIdx].label,
    confidence,
    status,
    visibleDays: best,
    truncated: false,
    summary: parsed.summary,
  });
};
