#!/usr/bin/env node
/**
 * sync-assets.mjs — copy the model artifacts and the ONNX Runtime Web WASM
 * runtime into public/ so the Cloudflare Worker serves them as static assets.
 *
 * Runs automatically before `build` and `dev` (see package.json). Keeping this
 * as a script (rather than committing copies by hand) means re-exporting the
 * model and bumping onnxruntime-web stay in sync with one command.
 */
import { cpSync, mkdirSync, existsSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = join(__dirname, "..");

const MODEL_VERSION = "v1";

function copy(from, to) {
  if (!existsSync(from)) {
    console.warn(`[sync-assets] WARN missing: ${from}`);
    return false;
  }
  mkdirSync(dirname(to), { recursive: true });
  cpSync(from, to);
  console.log(`[sync-assets] ${from} -> ${to}`);
  return true;
}

// 1. Model artifacts: model/artifacts/*-v1.onnx + manifest -> public/models/
const artifacts = join(root, "model", "artifacts");
const modelsOut = join(root, "public", "models");
const modelFiles = [
  `cycle-classifier-${MODEL_VERSION}.onnx`,
  `cycle-iforest-${MODEL_VERSION}.onnx`,
  `model-manifest-${MODEL_VERSION}.json`,
];
let modelOk = true;
for (const f of modelFiles) {
  modelOk = copy(join(artifacts, f), join(modelsOut, f)) && modelOk;
}
if (!modelOk) {
  console.warn(
    "[sync-assets] Some model files are missing. Run the Python export:\n" +
      "  cd model && .venv/bin/python -m scripts.train_and_export --out artifacts --version v1",
  );
}

// 2. ORT WASM runtime: NOTHING to copy.
//    We import `onnxruntime-web/wasm` (bundle build); Vite emits the `.wasm`
//    into _astro/ with a content hash and rewrites the reference natively, so
//    we no longer stage it under public/ort.


console.log("[sync-assets] done.");
