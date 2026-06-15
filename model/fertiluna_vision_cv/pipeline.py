"""End-to-end CV pipeline. Same output schema as fertiluna_vision (the CNN).

Returned `ChartResult` contains:
    value   : (N_SERIES, N_DAYS) float32 in [0,1], normalized within each
              series' axis range (Premom LH range is fixed; BBT range follows
              the classified scale).
    present : (N_SERIES, N_DAYS) float32 in {0,1}.
    scale   : 0=celsius, 1=fahrenheit  (index into BBT_SCALES).
    debug   : per-stage intermediates kept around for the overlay / CLI;
              callers that only want the inference tensors can ignore it.

Flow (matches the "imaginary lines, read the dots" idea):

    image
     └─> preprocess           load + EXIF-rotate + resize to WORK_W
     └─> color_segmentation   HSV masks (blue/orange/purple)
     └─> plot_region          plot bbox from ink + horizontal gridlines
     └─> day_axis.detect_grid LOCK day cells from vertical gridlines (or
                              fall back to marker-spacing inference)
     └─> marker.extract_per_cell  for each cell, scan colored ink column
                                  → y center if a marker run is present
     └─> axis_calibration     classify °C vs °F (deterministic, no OCR engine)
     └─> ChartResult(value, present, scale_idx)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import axis_calibration as axis_cal
from . import axis_columns
from . import color_segmentation as colors
from . import day_axis
from . import guardrails
from . import marker_detection as md
from . import ocr_backend
from . import quality as quality_mod
from . import plot_region as plot_reg
from . import preprocess
from . import table_extract
from .axis_columns import AxisColumn
from .constants import BBT_SCALES, LH_RANGE, N_DAYS, N_SERIES
from .marker_detection import Marker
from .plot_region import PlotRegion


@dataclass
class ChartResult:
    value: np.ndarray
    present: np.ndarray
    scale_idx: int
    scale_label: str
    scale_confidence: float
    # Bottom-of-screen table parsed alongside the chart. Empty if the
    # screenshot doesn't include one. Each row is the same length as
    # `value`/`present` along the day axis (N_DAYS), with None entries for
    # empty cells. `bbox` arrays mirror the cell layout so downstream
    # rendering can show the original cell location.
    table: dict = field(default_factory=dict)
    # Day-window bookkeeping. The output tensors are fixed at N_DAYS columns
    # (the CNN contract). When the chart shows MORE than N_DAYS days of data
    # (real-screen-3 has ~58), the rightmost days are dropped. `visible_days`
    # is our best estimate of how many days the chart actually displays;
    # `truncated` is True when that exceeds N_DAYS so the consumer KNOWS data
    # was lost rather than silently trusting a full 35-day window.
    visible_days: int = 0
    truncated: bool = False
    # Production quality signal. `confidence` in [0,1]; `status` is one of
    # "extracted" | "low_confidence" | "not_a_chart". `quality` carries the
    # per-signal breakdown + human-readable reasons. The consumer thresholds
    # on these to decide: trust / prompt-verify / reject-and-ask-again.
    confidence: float = 0.0
    status: str = "extracted"
    quality: dict = field(default_factory=dict)
    # Guardrail interpolation: (N_SERIES, N_DAYS) float32 in {0,1}. 1.0 marks
    # a day whose `value` was LINEARLY INTERPOLATED across a short gap between
    # measured days — NOT measured. Kept strictly separate from `present`
    # (which stays 0 on those days) so a synthesised BBT is never mistaken
    # for a real measurement. `value` carries the interpolated number so the
    # curve can be drawn continuously; the consumer decides whether to trust
    # interpolated days for any downstream computation.
    interpolated: np.ndarray | None = None
    debug: dict = field(default_factory=dict)


def _y_to_fraction(y_px: float, plot: PlotRegion) -> float:
    """Legacy fallback: linear mapping from plot.y0/y1.
    y_px=plot.y0 (top) -> 1.0,  y_px=plot.y1 (bottom) -> 0.0."""
    span = max(1.0, plot.y1 - plot.y0)
    return float(np.clip((plot.y1 - y_px) / span, 0.0, 1.0))


def _y_to_value_via_mapping(
    y_px: float, mapping: AxisColumn | None, plot: PlotRegion,
    axis_lo: float, axis_hi: float,
) -> float:
    """Map a y-pixel to a NORMALIZED value in [0,1] within [axis_lo, axis_hi].

    Uses the axis-tick fitted mapping when available — that anchors the
    mapping on the actual labels the app printed. Falls back to the plot
    region's y0/y1 if the mapping wasn't fittable (e.g. labels couldn't be
    OCR'd reliably).
    """
    if mapping is not None:
        real = mapping.value_at(y_px)
        return float(np.clip((real - axis_lo) / max(1e-6, axis_hi - axis_lo),
                             0.0, 1.0))
    return _y_to_fraction(y_px, plot)


def _fill_series(
    markers_by_day: dict[int, Marker],
    plot: PlotRegion,
    mapping: AxisColumn | None,
    axis_lo: float, axis_hi: float,
) -> tuple[np.ndarray, np.ndarray]:
    value = np.zeros(N_DAYS, dtype=np.float32)
    present = np.zeros(N_DAYS, dtype=np.float32)
    for d, m in markers_by_day.items():
        if 0 <= d < N_DAYS:
            value[d] = _y_to_value_via_mapping(m.cy, mapping, plot,
                                               axis_lo, axis_hi)
            present[d] = 1.0
    return value, present


def _pack_left(
    markers_by_day: dict[int, Marker],
) -> dict[int, Marker]:
    """Collapse sparse cell indices left-to-right so consumers see a tight
    [0, n_data) range.

    The downstream browser flattens by `present` mask anyway, so the absolute
    cell index doesn't matter to it — but packing makes the output easier to
    read and aligns the two series on the same offset when they both share a
    data window (the user's typical case: BBT and LH cover the same days).
    """
    if not markers_by_day:
        return {}
    ordered = sorted(markers_by_day.items())
    return {i: m for i, (_, m) in enumerate(ordered)}


def run_pipeline(image: str | Path | np.ndarray) -> ChartResult:
    """Full pipeline. `image` is either a path or a pre-loaded BGR array.

    OCR is hardcoded to PaddleOCR (PP-OCRv3 recognition, ~9 MB ONNX). The
    model is loaded once per process from `<repo>/public/models/` and reused
    for every axis-tick label and table cell. Run `python -m
    scripts.build_paddleocr_onnx --out ../public/models` first if the ONNX
    file isn't built yet — PaddleOCRBackend will raise otherwise.

    Output schema is the CNN's ONNX outputs PLUS a `table` dict carrying
    the parsed bottom-of-screen rows.
    """
    if isinstance(image, (str, Path)):
        raw = preprocess.load_bgr(image)
    else:
        raw = image
    work, work_scale = preprocess.to_work_canvas(raw)
    ocr_be = ocr_backend.PaddleOCRBackend()

    masks = colors.segment(work)
    plot_coarse = plot_reg.detect(work, masks)
    cal = axis_cal.classify(work, plot_coarse)
    # Read tick labels and use them to ANCHOR the chart's vertical extent.
    # The bottommost numeric tick = chart axis bottom (95°F / 0.1 LH); the
    # topmost = axis top. The table sits BELOW this, so refining plot.y1
    # here keeps every later step (grid detection, marker scan, value
    # mapping) anchored on the actual chart and not on table UI below it.
    #
    # MULTI-AXIS reader: detect every numeric y-axis column separately
    # (Ratio / Level / BBT-°F / BBT-°C) instead of the old y-only row merge
    # that glued adjacent columns into one bogus scale. Scale (°C vs °F) is
    # read DIRECTLY from which BBT column exists — no leading-digit guess.
    columns = axis_columns.detect_axis_columns(work, plot_coarse, ocr=ocr_be)
    axes = axis_columns.resolve_axes(columns)
    # GUARDRAIL: snap each fitted axis column's labels back onto its inferred
    # arithmetic grid. The RANSAC fit already gives a robust (a,b) by ignoring
    # half-tick OCR misreads; this rewrites those misread/blank labels to the
    # true on-grid value ([36, "5", 37] → [36, 36.5, 37]) so the label set is
    # self-consistent for the overlay and any downstream label consumer.
    for _col in (axes.bbt, axes.ratio, axes.level):
        if _col is not None and _col.n_fit >= 2:
            guardrails.snap_axis_column(_col)
    if axes.scale_idx is not None:
        cal.scale_idx = axes.scale_idx
        cal.scale_label = BBT_SCALES[axes.scale_idx][0]
        cal.scale_confidence = max(cal.scale_confidence, axes.scale_confidence)

    # Refine plot.y0/y1 from the BBT axis tick span when it's trustworthy
    # (a real axis spans most of the chart height). This anchors grid /
    # marker / value steps on the actual chart, not the table UI below.
    coarse_h = max(1, plot_coarse.y1 - plot_coarse.y0)
    plot = plot_coarse
    anchor_axis = axes.bbt or axes.level or axes.ratio
    if anchor_axis is not None and anchor_axis.n_fit >= 4:
        ays = [b.cy for b in anchor_axis.boxes if b.value is not None]
        tick_top, tick_bot = int(min(ays)), int(max(ays))
        if (tick_bot - tick_top) >= 0.55 * coarse_h:
            plot = PlotRegion(
                x0=plot_coarse.x0, y0=tick_top,
                x1=plot_coarse.x1, y1=tick_bot,
                method=plot_coarse.method + "+axiscol",
            )

    grid = day_axis.detect_grid(work, plot, masks)
    marker_meta: dict = {}
    bbt_by_day, lh_by_day = md.extract_per_cell(
        masks.blue, masks.orange, plot, grid.cells, grid.cell_px,
        bgr=work, masks=masks, out_meta=marker_meta,
    )
    lh_color = marker_meta.get("lh_color", "orange")

    # Bottom-of-screen table. Starts just below plot.y1. May be absent
    # entirely if the screenshot is chart-only.
    table = table_extract.extract_table(
        work, plot, grid, ocr_be,
    )

    # Pack so day 0 = leftmost detected cell across BOTH series, preserving
    # relative offsets between series. The browser collapses `present` anyway,
    # but packing keeps the overlay readable and matches the CNN's "left
    # padding via left_pad" contract for the Premom partial-width case.
    all_days = sorted(set(bbt_by_day.keys()) | set(lh_by_day.keys()))
    if all_days:
        d0 = all_days[0]
        bbt_packed = {d - d0: m for d, m in bbt_by_day.items() if d - d0 < N_DAYS}
        lh_packed = {d - d0: m for d, m in lh_by_day.items() if d - d0 < N_DAYS}
    else:
        bbt_packed, lh_packed = {}, {}

    bbt_lo, bbt_hi = BBT_SCALES[cal.scale_idx][1]
    lh_lo, lh_hi = LH_RANGE
    value = np.zeros((N_SERIES, N_DAYS), dtype=np.float32)
    present = np.zeros((N_SERIES, N_DAYS), dtype=np.float32)
    # BBT: use the detected temperature axis column (its `value_at` returns
    # real °F/°C, normalized into the scale's range). The column's kind
    # already matches cal.scale_idx, so lo/hi are consistent. Falls back to
    # plot-extent when no BBT axis was readable.
    value[0], present[0] = _fill_series(bbt_packed, plot, axes.bbt,
                                        bbt_lo, bbt_hi)
    # LH axis selection — keyed off which colored line is the LH series:
    #   orange "Ratio" → the Ratio axis column (0.1-1.9);
    #   purple "Level" → the Level axis column (5-95), normalized by ITS own
    #                    detected range (the schema's lh_range describes the
    #                    Ratio axis, so a Level reading is normalized by the
    #                    Level column's own lo/hi to stay an honest [0,1]).
    if lh_color == "orange":
        lh_axis = axes.ratio
        lh_norm_lo, lh_norm_hi = lh_lo, lh_hi
    else:
        lh_axis = axes.level
        if lh_axis is not None:
            lvals = [b.value for b in lh_axis.boxes if b.value is not None]
            lh_norm_lo, lh_norm_hi = (min(lvals), max(lvals)) if lvals else (5.0, 95.0)
        else:
            lh_norm_lo, lh_norm_hi = 5.0, 95.0
    value[1], present[1] = _fill_series(lh_packed, plot, lh_axis,
                                        lh_norm_lo, lh_norm_hi)

    # GUARDRAIL: interpolate short gaps so the curve can be drawn continuously.
    # The interpolated days are tracked in a SEPARATE mask and never folded
    # into `present` — measured ≠ synthesised. This matters medically: a
    # fabricated BBT presented as measured could skew an ovulation estimate.
    interpolated = np.zeros((N_SERIES, N_DAYS), dtype=np.float32)
    for s in range(N_SERIES):
        filled, interp = guardrails.interpolate_series(
            value[s], present[s], max_gap=3,
        )
        value[s] = filled
        interpolated[s] = interp

    # ── visible-day estimate + truncation flag ─────────────────────────────
    # Estimate how many day-cells the chart actually shows by dividing the
    # widest data-line column-span by the cell pitch. When a series is a
    # continuous line it spans the full data width, so this recovers the true
    # day count even past N_DAYS (where the grid was capped). If it exceeds
    # N_DAYS, the rightmost days were dropped from the fixed-width tensors.
    def _line_cell_span(mask) -> int:
        H, W = mask.shape
        x0, x1 = max(0, plot.x0), min(W, plot.x1 + 1)
        y0, y1 = max(0, plot.y0), min(H, plot.y1 + 1)
        strip = mask[y0:y1, x0:x1]
        if strip.size == 0:
            return 0
        cols = np.where((strip > 0).any(axis=0))[0]
        if cols.size < 2:
            return 0
        width_px = float(cols.max() - cols.min())
        if grid.cell_px <= 0:
            return 0
        return int(round(width_px / grid.cell_px)) + 1

    visible_days = max(
        len(all_days),
        _line_cell_span(masks.blue),
        _line_cell_span(masks.orange if lh_color == "orange" else masks.purple),
    )
    truncated = visible_days > N_DAYS

    # ── quality / confidence / status ──────────────────────────────────────
    qr = quality_mod.assess_quality(
        present=present,
        scale_confidence=cal.scale_confidence,
        axes=axes,
        grid_source=grid.source,
        plot_method=plot.method,
        visible_days=visible_days,
        truncated=truncated,
    )

    return ChartResult(
        value=value,
        present=present,
        scale_idx=cal.scale_idx,
        scale_label=cal.scale_label,
        scale_confidence=cal.scale_confidence,
        table=table,
        visible_days=visible_days,
        truncated=truncated,
        interpolated=interpolated,
        confidence=qr.confidence,
        status=qr.status,
        quality={"components": qr.components, "reasons": qr.reasons},
        debug={
            "work_image": work,
            "work_scale": work_scale,
            "masks": masks,
            "plot": plot,
            "tick_roi": cal.tick_roi,
            "grid": grid,
            "bbt_by_day_raw": bbt_by_day,    # before left-packing
            "lh_by_day_raw": lh_by_day,
            "bbt_by_day": bbt_packed,
            "lh_by_day": lh_packed,
            "left_offset": (all_days[0] if all_days else 0),
            "axis_columns": columns,
            "axes": axes,
            "bbt_axis": axes.bbt,
            "ratio_axis": axes.ratio,
            "level_axis": axes.level,
            "lh_color": lh_color,
            "marker_meta": marker_meta,
        },
    )
