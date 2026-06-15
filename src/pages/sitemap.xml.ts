import type { APIRoute } from "astro";

/**
 * Hand-rolled sitemap. The site is small (a handful of
 * routes) so we avoid an extra build dependency. Each entry advertises its FR
 * and EN equivalents so search engines serve the right language per market.
 */
interface Entry {
  loc: string;
  changefreq: string;
  priority: string;
  /** hreflang alternates for this URL. */
  alternates?: { hreflang: string; href: string }[];
}

export const GET: APIRoute = ({ site }) => {
  const origin = (site?.origin ?? "https://fertiluna.com").replace(/\/$/, "");

  const landingAlts = [
    { hreflang: "fr", href: `${origin}/` },
    { hreflang: "en", href: `${origin}/en/` },
    { hreflang: "x-default", href: `${origin}/` },
  ];
  const toolAlts = [
    { hreflang: "fr", href: `${origin}/outils/analyse-courbe` },
    { hreflang: "en", href: `${origin}/outils/analyse-courbe?lang=en` },
    { hreflang: "x-default", href: `${origin}/outils/analyse-courbe` },
  ];
  const privacyAlts = [
    { hreflang: "fr", href: `${origin}/confidentialite` },
    { hreflang: "en", href: `${origin}/confidentialite?lang=en` },
    { hreflang: "x-default", href: `${origin}/confidentialite` },
  ];

  const entries: Entry[] = [
    { loc: `${origin}/`, changefreq: "weekly", priority: "1.0", alternates: landingAlts },
    { loc: `${origin}/en/`, changefreq: "weekly", priority: "0.9", alternates: landingAlts },
    {
      loc: `${origin}/outils/analyse-courbe`,
      changefreq: "weekly",
      priority: "0.9",
      alternates: toolAlts,
    },
    { loc: `${origin}/a-propos`, changefreq: "monthly", priority: "0.5" },
    {
      loc: `${origin}/confidentialite`,
      changefreq: "yearly",
      priority: "0.3",
      alternates: privacyAlts,
    },
  ];

  const xml = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" xmlns:xhtml="http://www.w3.org/1999/xhtml">
${entries
  .map((e) => {
    const alts = (e.alternates ?? [])
      .map(
        (a) =>
          `    <xhtml:link rel="alternate" hreflang="${a.hreflang}" href="${a.href}"/>`,
      )
      .join("\n");
    return `  <url>
    <loc>${e.loc}</loc>
    <changefreq>${e.changefreq}</changefreq>
    <priority>${e.priority}</priority>
${alts}
  </url>`;
  })
  .join("\n")}
</urlset>
`;

  return new Response(xml, {
    headers: {
      "content-type": "application/xml; charset=utf-8",
      "cache-control": "public, max-age=3600",
    },
  });
};
