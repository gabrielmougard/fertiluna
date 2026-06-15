/**
 * Prompt construction for the OpenRouter vision call. Kept separate from the
 * API route so it's unit-testable and easy to tune. The model is asked for
 * STRICT JSON in real units (°C/°F + LH ratio) — natural for an LLM — which
 * the server then normalizes into the shared [0,1] schema.
 */

import { N_DAYS } from "../visionCv/constants";

export interface LlmChartJson {
  scale: "celsius" | "fahrenheit";
  /** real-unit temperature per day (length N_DAYS), null where no reading. */
  temp: (number | null)[];
  /** LH test ratio per day (length N_DAYS), null where no reading. */
  lh: (number | null)[];
  /** short summary in the requested language. */
  summary: string;
  /** model's own confidence in the extraction, 0..1. */
  confidence: number;
}

const SYSTEM = (langName: string) =>
  `You are a careful medical-chart digitizer. You read a fertility cycle `
  + `chart (basal body temperature + LH/ovulation tests) from an image and `
  + `return STRICT JSON only — no prose, no code fences.\n`
  + `Rules:\n`
  + `- Output exactly ${N_DAYS} entries for "temp" and "lh", left-aligned to `
  + `the first visible cycle day; use null for days with no data point.\n`
  + `- "temp" values are real temperatures in the chart's unit; "scale" is `
  + `"celsius" or "fahrenheit" read from the axis.\n`
  + `- "lh" values are the LH test ratio (typically 0.1–1.9); null if absent.\n`
  + `- Do NOT invent data points: if a day has no marker, use null.\n`
  + `- "summary" must be written in ${langName}.\n`
  + `- "confidence" is your own 0..1 estimate of extraction reliability.`;

const USER_TEXT =
  "Digitize this fertility chart. Return JSON with keys: scale, temp, lh, "
  + "summary, confidence.";

export function buildMessages(dataUrl: string, langName: string): unknown[] {
  return [
    { role: "system", content: SYSTEM(langName) },
    {
      role: "user",
      content: [
        { type: "text", text: USER_TEXT },
        { type: "image_url", image_url: { url: dataUrl } },
      ],
    },
  ];
}

/** Parse the model's message content (tolerating code fences) into LlmChartJson. */
export function parseLlmJson(content: string): LlmChartJson {
  let s = content.trim();
  // Strip opening/closing code fences INDEPENDENTLY so a response that lost
  // its closing ``` to truncation still parses.
  s = s.replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, "").trim();
  const obj = JSON.parse(s);
  return {
    scale: obj.scale === "fahrenheit" ? "fahrenheit" : "celsius",
    temp: Array.isArray(obj.temp) ? obj.temp : [],
    lh: Array.isArray(obj.lh) ? obj.lh : [],
    summary: typeof obj.summary === "string" ? obj.summary : "",
    confidence: typeof obj.confidence === "number" ? obj.confidence : 0.7,
  };
}
