# `visionCv` — in-browser TypeScript port of the CV digitizer

Port of the Python reference (`model/fertiluna_vision_cv/`) so the classical-CV
pipeline can run fully client-side, with no server, as the on-device default
in the hybrid router. The Python package stays the source of truth; this
mirrors it stage-for-stage.

## Status

| Stage | Python | TS module | Status |
|---|---|---|---|
| preprocess → working canvas | `preprocess.py` | `preprocess.ts` | ✅ ported (canvas) |
| HSV colour segmentation | `color_segmentation.py` | `colorSegmentation.ts` | ✅ ported (pure TS, tested) |
| connected components + dilation | cv2 | `ops.ts` | ✅ ported (pure TS) |
| plot region (line-only ink bbox) | `plot_region.py` | `plotRegion.ts` | ✅ ported (tested) |
| day grid (autocorr) | `day_axis.py` | `dayAxis.ts` | ✅ ported (direct autocorr, tested) |
| marker detection (white-blob + ring) | `marker_detection.py` | `markerDetection.ts` | ✅ ported |
| axis columns + OCR | `axis_columns.py` | `axisColumns.ts` | ✅ ported (async, uses `../paddleOcr.ts`) |
| guardrails (axis repair, interpolate) | `guardrails.py` | `guardrails.ts` | ✅ ported |
| quality / confidence | `quality.py` | `quality.ts` | ✅ ported |
| top-level pipeline (calibration) | `pipeline.py` | `pipeline.ts` | ✅ ported (async, OCR-driven) |
| table extraction | `table_extract.py` | `tableExtract.ts` | ✅ ported (core) |

**The full pipeline is ported.** `tableExtract.ts` covers the core table
parse (row centres from left labels, chart-grid columns, per-row text/heart/
circle/presence extractors). One simplification vs Python: the polarity-
independent calendar-text mask (white-on-coloured-pill dates) and a separate
calendar column detector are omitted — the chart day grid supplies the
columns, matching the "data point ↔ table cell" contract.

## OpenCV.js — not needed

Every cv2 op the pipeline uses was implementable in pure TypeScript:
connected-components and dilation live in `ops.ts`, and everything else is
math (autocorrelation, ring sampling, robust linear fits, RANSAC). So the TS
port pulls **no OpenCV.js / WASM** — the bundle stays tiny and the only model
asset is the PaddleOCR ONNX already used for OCR.

## OCR

Already ported: `../paddleOcr.ts` (PP-OCRv3 recognition via ORT-Web) +
`../paddleOcrManifest.ts`. `axisColumns.ts` and `tableExtract.ts` will call
`recognizeText()` exactly as the Python stages call the PaddleOCR backend.

## Output contract

`ChartResultCv` (`types.ts`) mirrors the Python `ChartResult` / CLI JSON —
`value`, `present`, `interpolated`, `scaleIdx`, `confidence`, `status`,
`visibleDays`, `truncated` — so a caller can swap NN ↔ CV ↔ (LLM) behind the
router without caring which produced the result.

## Parity testing

Port stages are unit-tested against the Python thresholds (see
`colorSegmentation.test.ts`). The end-goal parity test: run both the Python
pipeline and this TS port on the same fixtures and assert the `value`/`present`
tensors match within tolerance.
