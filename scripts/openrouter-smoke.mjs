/**
 * One-shot OpenRouter smoke test for the cloud digitize path.
 *
 * Sends ONE real screenshot to OpenRouter with the SAME prompt the server
 * proxy uses, and prints the parsed structured JSON. Validates: the key works,
 * the model is reachable, and the prompt yields parseable per-day arrays.
 *
 * The key is read from env OPENROUTER_API_KEY (never hard-coded / printed).
 *   OPENROUTER_API_KEY="$(cat OPENROUTER_API_KEY.txt)" node scripts/openrouter-smoke.mjs <image> [model]
 */
import { readFileSync } from "node:fs";

const N_DAYS = 35;
const key = process.env.OPENROUTER_API_KEY;
if (!key) {
  console.error("OPENROUTER_API_KEY not set in env");
  process.exit(2);
}
const imgPath = process.argv[2];
const model = process.argv[3] || "openai/gpt-4o-mini";
if (!imgPath) {
  console.error("usage: node openrouter-smoke.mjs <image.png> [model]");
  process.exit(2);
}

const b64 = readFileSync(imgPath).toString("base64");
const dataUrl = `data:image/png;base64,${b64}`;
const langName = "French";

const SYSTEM =
  `You are a careful medical-chart digitizer. You read a fertility cycle ` +
  `chart (basal body temperature + LH/ovulation tests) from an image and ` +
  `return STRICT JSON only — no prose, no code fences.\nRules:\n` +
  `- Output exactly ${N_DAYS} entries for "temp" and "lh", left-aligned to ` +
  `the first visible cycle day; use null for days with no data point.\n` +
  `- "temp" values are real temperatures in the chart's unit; "scale" is ` +
  `"celsius" or "fahrenheit" read from the axis.\n` +
  `- "lh" values are the LH test ratio (typically 0.1–1.9); null if absent.\n` +
  `- Do NOT invent data points: if a day has no marker, use null.\n` +
  `- "summary" must be written in ${langName}.\n` +
  `- "confidence" is your own 0..1 estimate of extraction reliability.`;

const body = {
  model,
  messages: [
    { role: "system", content: SYSTEM },
    {
      role: "user",
      content: [
        { type: "text", text: "Digitize this fertility chart. Return JSON with keys: scale, temp, lh, summary, confidence." },
        { type: "image_url", image_url: { url: dataUrl } },
      ],
    },
  ],
  temperature: 0,
  max_tokens: 4096,
  // gemini 2.5/3.x flash are "thinking" models — reasoning eats the token
  // budget before the content. json_object format keeps output fence-free.
  response_format: { type: "json_object" },
};

const t0 = Date.now();
const resp = await fetch("https://openrouter.ai/api/v1/chat/completions", {
  method: "POST",
  headers: {
    authorization: `Bearer ${key}`,
    "content-type": "application/json",
    "x-title": "FertiLuna",
  },
  body: JSON.stringify(body),
});
console.log(`HTTP ${resp.status} in ${Date.now() - t0}ms  (model ${model})`);
if (!resp.ok) {
  console.error("upstream error:", (await resp.text()).slice(0, 400));
  process.exit(1);
}
const data = await resp.json();
const content = data?.choices?.[0]?.message?.content ?? "";
console.log("\n--- raw content (first 600 chars) ---");
console.log(content.slice(0, 600));

// parse like the server does — strip opening/closing fences independently so a
// truncated response (missing the closing ```) still parses.
let s = content.trim();
s = s.replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, "").trim();
try {
  const obj = JSON.parse(s);
  const present = (a) => (Array.isArray(a) ? a.filter((v) => v != null).length : 0);
  console.log("\n--- parsed ---");
  console.log("scale:", obj.scale, "| confidence:", obj.confidence);
  console.log("temp present days:", present(obj.temp), "/", (obj.temp || []).length);
  console.log("lh present days:", present(obj.lh), "/", (obj.lh || []).length);
  console.log("temp[0..12]:", (obj.temp || []).slice(0, 13));
  console.log("summary:", (obj.summary || "").slice(0, 160));
  console.log("\nPARSE: OK");
} catch (e) {
  console.error("\nPARSE FAILED:", e.message);
  process.exit(1);
}
