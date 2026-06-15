import type { APIRoute } from "astro";

// robots.txt — allow everything, point crawlers at the sitemap. The API path is
// disallowed (it is a JSON endpoint, not indexable content).
export const GET: APIRoute = ({ site }) => {
  const origin = site?.origin ?? "https://fertiluna.com";
  const body = [
    "User-agent: *",
    "Allow: /",
    "Disallow: /api/",
    "",
    `Sitemap: ${origin}/sitemap.xml`,
    "",
  ].join("\n");

  return new Response(body, {
    headers: {
      "content-type": "text/plain; charset=utf-8",
      "cache-control": "public, max-age=86400",
    },
  });
};
