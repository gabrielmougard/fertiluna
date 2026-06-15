# `fertiluna_vision_cv` — classical-CV chart digitizer

Pure-OpenCV chart digitizer designed for **browser/WASM deployment**. Reads
the Premom-family fertility-chart screenshots into the same series schema
the CNN emits (BBT + LH per day) plus a structured table parse
(calendar / CD / DPO / Sex / CM / Symptoms / hCG with per-cell bboxes).
No torch, no transformers, no Tesseract — just OpenCV + numpy.

## Pipeline

```
image
 └─> preprocess           load + EXIF + resize into [1200, 2400] width
 └─> color_segmentation   HSV masks (blue BBT / orange LH / purple Level)
                          with NON-OVERLAPPING hue bands so purple "Level"
                          markers can't pollute blue / orange.
 └─> plot_region          ink bbox (cover-line filtered) ∪ gridline band;
                          falls back to ink-only when gridline morphology
                          collapses (alpha-blended bands break the pale
                          stripes on some screens).
 └─> axis_columns         MULTI-AXIS reader. Detect text WORD-boxes in the
                          left/right margins, cluster them into vertical
                          x-COLUMNS (one per axis: Ratio / Level / BBT-°F /
                          BBT-°C), OCR each box with PaddleOCR-rec, RANSAC a
                          line per column (robust to half-tick ".5" misreads
                          by anchoring on the reliable integer ticks),
                          classify each column by its value range. The °C/°F
                          SCALE is read directly from which BBT column exists
                          — no leading-digit guess — and dual C+F axes are
                          handled by keeping both columns. Plot vertical
                          bounds are refined to the BBT axis tick span.
 └─> day_axis             cell grid — autocorrelation of column-density as
                          PRIMARY signal (robust across all 4 test screens);
                          date-row digits & vertical gridlines accepted only
                          when their pitch matches the autocorr estimate.
 └─> marker_detection     small white blob + colored ring coverage per color
                          (blue / orange / purple). The LH series is chosen
                          at runtime (orange "Ratio" vs purple "Level") by
                          marker density + line continuity. Dense continuous
                          curves that defeat discrete detection are read by
                          per-cell LINE-SAMPLING; the BBT cover-line is
                          detected and excluded from sampling.
 └─> table_extract        bottom table: row centers from LEFT-side labels;
                          column centers from the calendar row's own date
                          numbers (not the chart day grid); per-row
                          extractor — text rows OCR, Sex/CM detect colored
                          icons (♥, ●), Symptoms/hCG detect presence.
 └─> ChartResult(value, present, scale_idx, table, lh_source,
                 visible_days, truncated)
```

## Output schema

Same `value`, `present`, `scale_idx` as the CNN's ONNX outputs, **plus**:

```json
{
  "scale": {"label": "fahrenheit", "bbt_range": [95.0, 99.5], "lh_range": [0.1, 1.9]},
  "value":   [[…35 floats…], […35 floats…]],
  "present": [[…35 0/1…],     […35 0/1…]],
  "decoded": {
    "temp": [97.26, 97.24, …, null, 97.93, …],
    "lh":   [0.34, 0.46, 1.76, 0.88, …]
  },
  "table": {
    "calendar": {"label": "MAR", "label_bbox": […], "cells": ["12","13",…,null], "cell_bboxes":[…]},
    "CD":       {"label": "CD",  "cells": ["24","25",…], "cell_bboxes":[…]},
    "DPO":      { … },
    "Sex":      { "cells": [null,"♥",null,"♥",…] },
    "CM":       { "cells": [null,null,"●","●",…] },
    "Symptoms": { … },
    "hCG":      { … }
  }
}
```

Missing rows keep their keys but their `cells` array stays all-null.

## Install + run

```bash
cd model
uv pip install -e ".[vision_cv]"     # only opencv-python + pillow + numpy

# Assess: render annotated overlays + JSON for visual review
python -m fertiluna_vision_cv.cli assess \
    --images real-screen-*.png --out /tmp/cv-assess

# Inference: print decoded series for one image
python -m fertiluna_vision_cv.cli infer --image real-screen-1.png
```

## Browser / WASM deployment

The pipeline is portable to **OpenCV.js** (the OpenCV WASM build). All
primitives used here exist there:

| Operator | OpenCV.js |
|---|---|
| `cvtColor`, `inRange`, `bitwise_*` | ✓ |
| `morphologyEx`, `getStructuringElement` | ✓ |
| `connectedComponentsWithStats`, `findContours` | ✓ |
| `matchTemplate`, `putText` (Hershey only) | ✓ |
| `boxFilter`, `Sobel`, `GaussianBlur` | ✓ |

A custom OpenCV.js build with only `imgproc` + `imgcodecs` ships at **~5-8 MB
WASM**, comfortably under any browser bundle budget. The TS port mirrors
`pipeline.py` one-to-one — about 400 lines of straightforward code.

The Python implementation in this package is the **reference algorithm**
the WASM port targets.

## OCR

Three browser-friendly backends:

| Backend | Bundle cost | Quality | When to use |
|---|---|---|---|
| `template` (default) | 0 extra deps | best-effort | Lightest path; numeric filter for axis ticks; no real cell-text reading |
| `paddle`             | ~9 MB ONNX + tiny dict | accurate on real fonts | Production — reads axis ticks AND table cells |
| `none`               | 0 deps        | skip OCR | Fastest; emit cell bboxes without text |

### `paddle` backend — PP-OCRv3 recognition

Recognition-only (skip detection/classification): the CV pipeline already
locates text bboxes, so PaddleOCR just answers "what does this 30-px crop
say?". One ONNX file, one CTC decode, no extra Paddle install needed at
runtime (only at build time).

**Build the ONNX** (run once, ships into `public/models`):

```bash
cd model
uv pip install paddle2onnx paddlepaddle
python -m scripts.build_paddleocr_onnx --out ../public/models
# Produces:
#   ../public/models/paddle-ocr-rec-v1.onnx        (~9 MB)
#   ../public/models/paddle-ocr-dict-v1.txt        (190 B)
#   ../public/models/paddle-ocr-manifest-v1.json
```

**Use it from Python**:

```bash
python -m fertiluna_vision_cv.cli assess \
    --images real-screen-*.png --out /tmp/cv-assess --ocr paddle
```

**Use it from the browser** ([src/lib/paddleOcr.ts](../../src/lib/paddleOcr.ts)):

```ts
import { ensurePaddleOcrLoaded, recognizeText } from "./paddleOcr";

await ensurePaddleOcrLoaded((p) => console.log("loading", p));
const { text, confidence } = await recognizeText(cropCanvas);
// → "95.5"   confidence 0.97
```

The TS runner mirrors `visionInference.ts` exactly: ORT-Web session, DexieJS
model cache, sha256-validated manifest. Total cold-start: ~9 MB download
(cached in IndexedDB after the first run).

Verified output on real-screen-1 with `--ocr paddle`:

```
right ticks: 99.5 | 99 | 5 | 98 | 5 | 97 | 5 | 96 | 5 | 95
calendar   : Mar | 8 | 9 | 10 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | …
CD         : CD  | 25 | 26 | 27 | 29 | 30 | 31 | 32 | 33 | 34 | 35 | 36 | 37 | 38 | …
DPO        : DPO | · | · | · | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | …
Sex        : · | ♥ | · | ♥ | · | · | …
```

## Detection quality (real screenshots in the repo)

| Screen | BBT | LH | Scale | LH source | Notes |
|---|---|---|---|---|---|
| real-screen-1 (English Premom) | 16/35 | 15/35 | F (auto) | orange "Ratio" | Reference target |
| real-screen-2 (German Premom)  | 19/35 | 18/35 | C (auto)  | orange "Ratio" | Celsius now auto-detected (scale-flip rebuilds the axis mapping) |
| real-screen-3 (dense, ~58 days, LH Peak) | 34/35 | 35/35 | F | **purple "Level"** | LH read off the dense purple line via per-cell line-sampling; `truncated=true` |
| real-screen-4 (sparse markers + LH Peak) | 31/35 | 35/35 | F | **purple "Level"** | Same; `truncated=true` |

### LH series is auto-selected, not hardcoded

The Premom family draws TWO non-blue curves:

* **"Ratio"** (orange/coral) — the LH:PdG ratio, on the 0.1-1.9 axis.
* **"Level"** (purple/violet) — the LH level, on the 5-≥95 axis.

Either may be the densely-sampled per-day line depending on the chart
variant. `marker_detection.resolve_lh_color` picks the LH source from BOTH
the discrete-marker count AND line continuity (column ink-span), so:

- screens 1/2 → orange is the dense line → `lh_source="orange"`, values
  normalized on the Ratio (0.1-1.9) tick mapping;
- screens 3/4 → purple is the dense per-day line (orange shows only the
  LH-Peak super-marker) → `lh_source="purple"`, values normalized by
  plot-extent (the Ratio tick mapping doesn't apply to the Level axis).

This replaced the old behavior that hardcoded orange→LH and DROPPED purple
as a "distractor" — which returned LH=1 (just the peak) on screens 3/4 even
though the dense per-day LH data was right there in purple.

### Line-sampling for dense continuous curves

When a series is a smooth dense line whose per-day open circles fuse into
the stroke (screen-3's ~50 purple markers collapse into one connected
component), discrete white-center-blob detection under-counts badly. The
pipeline falls back to **per-cell line-sampling**: for each cell column,
take the ink centroid directly off the line. Triggered only when the line
is continuous (≥70% column coverage) and discrete markers leave it
under-sampled, so genuinely sparse few-point lines are still read
discretely. The BBT cover-line (a flat horizontal threshold stroke) is
detected and excluded before sampling blue, so empty columns don't read it
as data.

### Output additions

The `ChartResult` / decoded JSON now carry:

- `lh_source` — `"orange"` or `"purple"` (which line was read as LH).
- `visible_days` — estimated day count the chart displays.
- `truncated` — `true` when `visible_days > N_DAYS` (35) so the consumer
  knows the rightmost days were dropped from the fixed-width tensors
  instead of silently trusting a full window.

### Remaining honest limits

- **Axis reading** now uses PaddleOCR recognition on per-column text boxes
  with a RANSAC line fit anchored on the reliable integer ticks, so it reads
  the separate Ratio / Level / BBT-°F / BBT-°C scales correctly and reads
  the °C/°F unit straight off the axis (verified on all 4 screens). It can
  still fail on charts whose tick fonts are too small/blurred to OCR; in that
  case a column yields no fit and the series falls back to plot-extent
  normalization (`value` is an honest [0,1] fraction, real units approximate).
- **LH real-units when `lh_source="purple"`** are advisory: the value is
  normalized by the detected Level-axis range, not the Ratio axis the
  schema's `lh_range` (0.1-1.9) describes.
- **Table column grid** uses the calendar row's own date-number positions
  when they form a regular grid (screens 1-2); on dense screens where the
  dates can't be cleanly clustered it falls back to the chart day-grid
  (screens 3-4), which can drift a little. A table-structure detector
  (PaddleOCR-Structure-style ruling-line detection) would make this exact.
- **Table cell text** is PaddleOCR best-effort. Use the bboxes for UI
  rendering and let the user confirm digit-heavy cells.

## Iteration workflow

```bash
# 1. Run on your screenshots
python -m fertiluna_vision_cv.cli assess \
    --images path/to/*.png --out /tmp/cv-assess

# 2. Open the gallery + see what's drawn wrong
open /tmp/cv-assess/index.html

# 3. Tweak the constant most relevant to the failure mode:
#    constants.py        — HSV bands, working canvas size
#    marker_detection.py — area / ring coverage thresholds
#    day_axis.py         — autocorrelation lag range
#    plot_region.py      — fallback ratios

# 4. Re-run assess and compare overlays
```
