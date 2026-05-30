# FertiLuna

> Comprendre son cycle. Prendre les bonnes décisions.

An account-free web tool that analyses menstrual-cycle data (basal body
temperature + LH tests) and explains it in plain language. All machine-learning
inference runs **100 % in the browser** — no account, no server-side data, no
upload. The model is downloaded once, cached in IndexedDB, and versioned.

This repo contains two halves:

| Path        | What it is                                                            |
| ----------- | --------------------------------------------------------------------- |
| `model/`    | Python package: synthetic data, training, ONNX export (scikit-learn). |
| `src/`      | Astro app (Cloudflare Worker SSR) + browser ML runtime.               |
| `public/`   | Static assets: the exported `.onnx` models + ONNX Runtime Web wasm.   |
| `scripts/`  | `sync-assets.mjs` copies model artifacts + ORT wasm into `public/`.   |

## Architecture

```
                 ┌─────────────────────── model/ (Python) ───────────────────────┐
                 │  synthetic.py → features.py → train.py → export_onnx.py        │
                 │  50k synthetic cycles · Random Forest + Platt calibration      │
                 │  + Isolation Forest OOD backstop → ONNX (skl2onnx)             │
                 └───────────────────────────────┬───────────────────────────────┘
                                                 │  artifacts/*.onnx + manifest
                              scripts/sync-assets.mjs │
                                                 ▼
   ┌──────────────────────────── Cloudflare Worker (Astro SSR) ─────────────────────────┐
   │  /                      marketing home (SEO)                                        │
   │  /outils/analyse-courbe Outil 1 — tabular input + result                           │
   │  /models/*.onnx         static assets (served once, cached client-side)            │
   │  /ort/*.wasm            ONNX Runtime Web (WASM backend)                             │
   └───────────────────────────────────────────┬────────────────────────────────────────┘
                                                ▼  browser
   ┌──────────────────────────────── src/lib (browser runtime) ─────────────────────────┐
   │  features.ts       parity port of features.py (verified against fixtures)           │
   │  modelCache.ts     DexieJS / IndexedDB versioned model cache (sha256-validated)     │
   │  inference.ts      onnxruntime-web: classifier + iforest → CycleAnalysis            │
   │  curveChart.ts     animated SVG curve with ovulation band + LH peaks                │
   │  curveDigitizer.ts screenshot → per-day values (chart digitization, in-browser)     │
   │  digitizerUI.ts    interactive calibration controller for the digitizer             │
   └────────────────────────────────────────────────────────────────────────────────────┘
```

## Importing a screenshot (chart digitization)

Most users won't type a table — they'll have a screenshot from Flo/Premom/
Clearblue or a paper chart. `CurveDigitizer.astro` recovers the data
**entirely in the browser** (WebPlotDigitizer-style), no upload, no server:

1. Upload the image (drawn to a `<canvas>`; the file never leaves the device).
2. Calibrate the axes with 4 clicks: two day reference points + two temperature
   reference lines. Human-supplied calibration is what makes extraction reliable
   across every app's chart style.
3. Pick the curve colour (one click; auto-suggested via saturation heuristic).
4. A column-scan finds the curve pixels per x-column, takes their vertical
   centroid, maps pixel→(day, °C) via the calibration, and resamples to one
   value per cycle day.
5. The recovered values land in the **editable table**, where the user
   verifies/corrects them before analysis (human-in-the-loop).

The extraction math (`src/lib/curveDigitizer.ts`) is pure and unit-tested
(`curveDigitizer.test.ts`) against a synthetically-rendered chart — it recovers
per-day temperatures within 0.05 °C.

## The model (SOTA rationale)

- **Synthetic data, not real.** No public BBT/LH dataset exists at scale, and
  the "ground truth" for cycle classification is itself a deterministic clinical
  rule (SENSIPLAN 3-over-6). Physiologically-grounded synthetic generation with
  realistic noise gives full edge-case coverage and a tractable supervised target.
- **Random Forest, not an LLM.** Interpretable, reproducible, ~5 ms inference,
  and exports cleanly to ONNX. (Roadmap §4.)
- **Single prefit-calibrated forest.** `CalibratedClassifierCV(cv=5)` would train
  five forests → ~250 MB ONNX. We train ONE forest then Platt-calibrate it on a
  held-out split (`FrozenEstimator`) → **~5.6 MB** at **~89 %** accuracy.
- **Calibration matters.** The roadmap's "max proba < 0.6 → données
  insuffisantes" gate is only honest with calibrated probabilities.
- **Isolation Forest OOD backstop.** Flags curves unlike anything in training so
  the UI can say "this is unusual" instead of over-confidently mislabelling.

Outputs: `ovulation_confirmee`, `ovulation_douteuse`, `anovulation`,
`phase_luteale_courte`, `donnees_insuffisantes`.

## Quickstart

### 1. Train + export the model (Python, via `uv`)

```bash
cd model
uv venv --python 3.12
uv pip install -e . pytest matplotlib
.venv/bin/pytest -q                                   # 9 tests
.venv/bin/python -m scripts.train_and_export \
    --out artifacts --n-samples 50000 --version v1
```

This writes `model/artifacts/cycle-classifier-v1.onnx` (~5.6 MB),
`cycle-iforest-v1.onnx` (~1.3 MB), `model-manifest-v1.json`, and
`feature-fixtures.json` (used by the TS parity test).

### 2. Run the app (Astro + Cloudflare)

```bash
npm install
npm run dev            # sync-assets runs automatically, then astro dev
# → http://localhost:4321
```

`npm run build` produces a Cloudflare Worker in `dist/`; `npm run deploy` builds
and deploys via wrangler.

### 3. Tests

```bash
npm test               # vitest: TS↔Python feature-extraction parity
```

The parity test (`src/lib/features.test.ts`) loads the Python-generated fixtures
and asserts the TypeScript feature extractor produces identical output within
float32 tolerance — the contract that lets the browser reproduce Python's input.

## Privacy / RGPD

No account, no cookies for tracking, no server-side storage. Cycle data is
processed entirely on-device; only the (public, non-personal) model files are
ever fetched. Matches the roadmap's "traitement 100 % en mémoire" requirement —
here, in the browser.

## Status / roadmap

- [x] Python model package (synthetic, features, train, ONNX export)
- [x] Browser ML runtime (onnxruntime-web) + DexieJS versioned cache
- [x] Outil 1 — analyse simple de courbe (animated result, doctor questions, glossary)
- [x] Screenshot import — in-browser chart digitization with axis calibration
- [ ] Ad integration (slots are reserved in the layout; see `docs/AD_SETUP_GUIDE.md`)
- [ ] Outils 2–5 (multi-cycle, grossesse, bilan hormonal)
- [ ] SENSIPLAN expert validation on real annotated curves
