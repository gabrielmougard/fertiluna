// @ts-check
import { defineConfig } from "astro/config";
import cloudflare from "@astrojs/cloudflare";

// FertiLuna runs as a Cloudflare Worker (SSR for SEO on the marketing/tool
// shell) while ALL machine-learning inference happens client-side in the
// browser via onnxruntime-web. No user data ever leaves the device — the model
// is fetched once as a static asset and cached in IndexedDB (DexieJS).
export default defineConfig({
  site: "https://fertil-luna.fr",
  output: "server",
  adapter: cloudflare({
    platformProxy: { enabled: true },
  }),
  vite: {
    // We import `onnxruntime-web/wasm` (its default "bundle" build): the small
    // JS glue is inlined into our bundle, and the multi-MB `.wasm` binary is
    // referenced via `new URL("...wasm", import.meta.url)`. We let Vite handle
    // that URL natively — it emits the wasm into _astro/ with a content hash and
    // rewrites the reference. This works in dev and prod with zero extra config,
    // so we do NOT override ort.env.wasm.wasmPaths.
    optimizeDeps: {
      exclude: ["onnxruntime-web"],
    },
    build: {
      chunkSizeWarningLimit: 2000,
    },
  },
});
