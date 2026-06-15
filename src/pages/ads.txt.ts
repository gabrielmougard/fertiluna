import type { APIRoute } from "astro";

/**
 * ads.txt — authorizes ad networks to sell inventory on this domain (required
 * by AdSense). The single line is provided by the worker env `ADS_TXT` so it
 * stays out of source control. With no value set, we return an empty 200 (a
 * valid, if empty, ads.txt) rather than a 404.
 */
export const GET: APIRoute = ({ locals }) => {
  const env = (locals as { runtime?: { env?: Record<string, string | undefined> } })
    ?.runtime?.env;
  const body = (env?.ADS_TXT ?? "").trim();

  return new Response(body + (body ? "\n" : ""), {
    headers: {
      "content-type": "text/plain; charset=utf-8",
      "cache-control": "public, max-age=86400",
    },
  });
};
