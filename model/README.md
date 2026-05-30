# FertiLuna — Model

Python package for training and exporting the models used by the FertiLuna
Astro app. Two models live here:

1. **Cycle classifier** (`fertiluna/`) — Random Forest that labels a cycle
   (ovulation confirmée / douteuse / anovulation / phase lutéale courte /
   données insuffisantes) from extracted features. ~5.6 MB ONNX.
2. **Chart-vision model** (`fertiluna_vision/`) — a small CNN that reads a chart
   *screenshot* and outputs per-day BBT + LH (normalized value + presence),
   removing most of the manual axis-calibration clicking. ~19 MB ONNX (<100M
   params). Trained on synthetic charts we render ourselves.

## What's here

```
fertiluna/                 # cycle classifier
  constants.py     # CYCLE_MAX_DAYS, LABELS, FEATURE_NAMES — synced with src/lib/constants.ts
  synthetic.py     # physiologically-grounded cycle generator (5 archetypes)
  features.py      # NaN-safe feature extraction; canonical implementation
  train.py         # RF + Platt calibration + Isolation Forest backstop
  export_onnx.py   # sklearn -> ONNX + manifest with checksums
fertiluna_vision/          # chart-screenshot -> per-day series
  constants.py     # IMG_H/IMG_W, N_DAYS, N_SERIES — synced with src/lib/visionInference.ts
  render.py        # synthetic chart renderer (matplotlib): dual-axis, bands, 2 lines, noise
  dataset.py       # torch Dataset (+ cached .npz dataset)
  model.py         # compact depthwise-separable CNN, value + presence heads
  train.py         # masked losses, CUDA/AMP, MPS, checkpointing
  export_onnx.py   # torch -> single-file ONNX + manifest + parity check
scripts/
  train_and_export.py            # cycle classifier
  build_vision_dataset.py        # pre-render charts to .npz
  train_and_export_vision.py     # vision model (see its header for the best recipe)
  export_vision_from_ckpt.py     # recover ONNX from a checkpoint
tests/
artifacts/         # gitignored; outputs land here
data/              # gitignored; pre-rendered chart datasets (.npz)
```

## Quickstart

```bash
cd model
uv venv --python 3.12
uv pip install -e . pytest matplotlib
.venv/bin/pytest -q
.venv/bin/python -m scripts.train_and_export --out artifacts --n-samples 50000 --version v1
```

Outputs in `artifacts/`:
- `cycle-classifier-v1.onnx` — main model (~5.6MB; single calibrated forest)
- `cycle-iforest-v1.onnx` — anomaly backstop (~1.3MB)
- `model-manifest-v1.json` — version + checksums + calibration metadata
- `feature-fixtures.json` — used by the TS parity test

The model artifacts are copied into the Astro app's `public/models/` directory
to be served as Worker static assets and cached client-side in IndexedDB
(DexieJS). See `../README.md` for the export → browser wiring.

## Chart-vision model (screenshot → per-day values)

```bash
cd model
uv pip install -e ".[vision]"          # adds torch, pillow, matplotlib, onnxscript
# CUDA GPU? install the CUDA torch wheel instead:
#   uv pip install --upgrade "torch>=2.2" --index-url https://download.pytorch.org/whl/cu121

# 1) pre-render synthetic charts once (reused every epoch)
.venv/bin/python -m scripts.build_vision_dataset --out data --n 120000 --seed 1  --workers 16
.venv/bin/python -m scripts.build_vision_dataset --out data --n 10000  --seed 99 --workers 16

# 2) train the best model + export (see the script header for the full recipe)
.venv/bin/python -m scripts.train_and_export_vision \
    --train-npz data/charts-120000-seed1.npz --val-npz data/charts-10000-seed99.npz \
    --width 3.0 --epochs 40 --batch-size 128 --num-workers 16 \
    --out artifacts --version v1
```

The script header in `scripts/train_and_export_vision.py` documents the
recommended "best model" recipe in full (compute time aside). Training
auto-uses CUDA + AMP when available, then MPS, then CPU. Best weights are
checkpointed to `artifacts/chart-vision-v1.ckpt` every improvement; recover with
`scripts/export_vision_from_ckpt.py` if export didn't run.

Outputs: `chart-vision-v1.onnx` (single self-contained file, ~19 MB at width
3.0) + `chart-vision-manifest-v1.json`. The model outputs **normalized** values
in [0,1] per series — the browser asks the user only for each axis' min/max to
de-normalize (far fewer interactions than full manual calibration).

## ONNX IO contract (browser must match)

| Model | input | outputs |
|-------|-------|---------|
| classifier | `input` float32 `[N, 30]` | `label` `[N]`, `probabilities` `[N, 5]` |
| iforest | `input` float32 `[N, 30]` | `label` `[N, 1]`, `scores` `[N, 1]` (decision_function orientation; negative ⇒ anomalous) |
| chart-vision | `image` float32 `[N, 3, 224, 384]` (ImageNet-normalized NCHW) | `value` `[N, 2, 35]`, `present` `[N, 2, 35]` (both RAW logits; apply sigmoid in browser). Series order: `[temp, lh]` |


## SOTA approach (short version)

- **Why synthetic data:** No public BBT/LH dataset at scale exists. The
  "ground truth" for cycle classification is itself a deterministic rule
  (SENSIPLAN 3-over-6), so physiologically-grounded generation with realistic
  noise gives the model a tractable supervised target with full edge-case
  coverage.
- **Why Random Forest:** interpretable, robust to missing data via NaN-safe
  features, exports cleanly to ONNX, ~5ms inference in the browser. Modern
  gradient boosters give marginal accuracy gains here for higher export complexity.
- **Why a single prefit-calibrated forest:** `CalibratedClassifierCV(cv=5)`
  trains five forests, which serialise to a ~250MB ONNX TreeEnsemble — far too
  large to ship to a browser. We instead train ONE forest on a fit split, then
  fit Platt scaling on a held-out calibration split (`FrozenEstimator`), giving
  a ~5.6MB graph at ~89% accuracy.
- **Why Platt calibration:** raw RF probabilities are pushed toward 0/1. To make
  the "< 0.6 → données insuffisantes" gate statistically honest, we wrap the RF
  in `CalibratedClassifierCV(method="sigmoid")`.
- **Why Isolation Forest backstop:** the classifier always outputs *some* class.
  When the feature vector is far from anything seen in training, we want the UI
  to say "this curve is unusual" — that's what the iforest score is for.

## Browser parity contract

`fertiluna/features.py` and `src/lib/features.ts` MUST produce identical
output for the same inputs. The `feature-fixtures.json` artifact lets the TS
test in the Astro project verify parity.
