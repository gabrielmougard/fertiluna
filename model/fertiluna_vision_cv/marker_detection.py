"""Marker detection via WHITE-CENTER blobs with COLORED-RING verification.

A Premom marker is one of:
    * Open circle: colored ring around a white interior fill.
    * Filled super-marker: colored disk with a white glyph inside (the
      LH-Peak "P", the BBT cover-line start "B"). The glyph is still a
      small WHITE blob; the surrounding ring is still colored.

So the universal signature is:

        a SMALL WHITE BLOB whose neighborhood is dominated by ONE color.

Detection pipeline:
    1. Find small white-ish blobs inside the plot (connected components of
       gray > BG_BRIGHTNESS, filtered by area + aspect ratio).
    2. For each blob, sample a ring of pixels at several candidate radii
       around the blob center, in BOTH the blue mask and the orange mask.
    3. The ring with the higher color-coverage fraction (subject to a
       minimum) wins, and the blob becomes a marker for that series.

Why this beats v2 (column-scan) and v3 (hole-from-contour-hierarchy):
    * No fragile hole-via-contour-hierarchy step — that pipeline picked up
      stray micro-holes in axis text and produced spurious low-y markers.
    * No threshold tuning of "run length > line stroke" — we explicitly
      verify the COLOR signature surrounding each candidate.
    * Handles open-rings and filled-disks-with-glyph identically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from .color_segmentation import SeriesMasks
from .plot_region import PlotRegion


SeriesKey = Literal["temp", "lh"]


@dataclass
class Marker:
    series: SeriesKey
    cell_idx: int
    cx: float
    cy: float
    radius: float
    kind: str          # "ring" | "filled"
    score: float       # color-ring coverage fraction ∈ [0,1]


# ── tunables (working canvas WIDTH in [1200, 2400]) ────────────────────────
# White-interior area: at the working-canvas range, an open-circle marker's
# white fill spans diameter ~4-12 px → area ~12-115. Bottom edge dropped to
# 4 to catch high-res downscaled markers; top stays generous for filled
# super-markers (LH-Peak white "P" or BBT "B" glyphs).
MIN_WHITE_AREA = 4
MAX_WHITE_AREA = 360
MIN_BLOB_ASPECT = 0.35
MAX_BLOB_ASPECT = 2.6
# Background-brightness threshold for "white-ish" pixels. Chart background
# is one huge component (filtered by MAX_WHITE_AREA), leaving only the small
# isolated white markers/glyphs to be tested.
BG_BRIGHTNESS = 225
# Candidate ring radii spanning small-on-downscale-screens (r=3) to large-
# filled-super-markers (r=14). The detector keeps the best-fitting radius.
RING_RADII = (3, 4, 5, 6, 7, 8, 9, 11, 14)
RING_SAMPLES = 32
MIN_RING_COVERAGE = 0.30           # lowered from 0.45 — real screenshots
                                   # have anti-aliased pale rings whose
                                   # color-mask coverage rarely exceeds 60%
                                   # at any single radius. Purple-ring
                                   # rejection in detect_centers() keeps
                                   # Level markers out.


def _white_blobs(
    bgr: np.ndarray, plot: PlotRegion,
) -> list[tuple[float, float, float]]:
    """Small white-ish connected components inside plot. Returns (cx, cy, area).

    Uses BOTH brightness AND low-saturation criteria — a pure white marker
    fill has sat=0, but alpha-blended fertile / period band fills are also
    bright (gray ~234) yet have sat 25-50. Without the sat gate the band
    pixels merge with white marker centers into one huge component that
    gets size-filtered out, so markers on top of bands disappear.
    """
    plot_region = bgr[plot.y0:plot.y1, plot.x0:plot.x1]
    if plot_region.size == 0:
        return []
    gray = cv2.cvtColor(plot_region, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(plot_region, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    white = cv2.inRange(gray, BG_BRIGHTNESS, 255)
    n, _, stats, cents = cv2.connectedComponentsWithStats(white, connectivity=8)
    out: list[tuple[float, float, float]] = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not (MIN_WHITE_AREA <= area <= MAX_WHITE_AREA):
            continue
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if w < 3 or h < 3 or w > 26 or h > 26:
            continue
        aspect = w / max(1, h)
        if not (MIN_BLOB_ASPECT <= aspect <= MAX_BLOB_ASPECT):
            continue
        out.append((
            float(cents[i, 0]) + plot.x0,
            float(cents[i, 1]) + plot.y0,
            float(area),
        ))
    return out


def _ring_coverage(
    cx: float, cy: float, mask: np.ndarray, radius: int,
) -> float:
    """Fraction of `RING_SAMPLES` angular samples at `radius` hitting the mask."""
    H, W = mask.shape
    if radius < 1:
        return 0.0
    angles = np.linspace(0.0, 2.0 * np.pi, RING_SAMPLES, endpoint=False)
    xs = np.clip(np.round(cx + radius * np.cos(angles)).astype(int), 0, W - 1)
    ys = np.clip(np.round(cy + radius * np.sin(angles)).astype(int), 0, H - 1)
    hits = int((mask[ys, xs] > 0).sum())
    return hits / RING_SAMPLES


def _best_ring_coverage(
    cx: float, cy: float, mask: np.ndarray,
) -> tuple[float, int]:
    """Sweep `RING_RADII` and return (best_coverage, best_radius)."""
    best_cov = 0.0
    best_r = RING_RADII[0]
    for r in RING_RADII:
        cov = _ring_coverage(cx, cy, mask, r)
        if cov > best_cov:
            best_cov = cov
            best_r = r
    return best_cov, best_r


Center = tuple[float, float, float, int, str]


def detect_centers_by_color(
    bgr: np.ndarray, masks: SeriesMasks, plot: PlotRegion,
) -> dict[str, list[Center]]:
    """Returns {"blue": [...], "orange": [...], "purple": [...]}.

    Each white-blob candidate is assigned to the SINGLE color whose ring
    coverage around it is highest (above MIN_RING_COVERAGE). Unlike the old
    detector this does NOT drop purple — on some chart variants the purple
    "Level" line is the real per-day LH measurement (dense open circles)
    while orange "Ratio" carries only the LH-Peak super-marker. The pipeline
    decides at a higher level which colored line plays the LH role; the
    detector's job is just to faithfully report every marker per color.

    Each entry = (cx, cy, score, radius, kind).
    """
    blobs = _white_blobs(bgr, plot)
    out: dict[str, list[Center]] = {"blue": [], "orange": [], "purple": []}
    color_masks = {
        "blue": masks.blue,
        "orange": masks.orange,
        "purple": masks.purple,
    }
    for cx, cy, area in blobs:
        covs: dict[str, tuple[float, int]] = {}
        for name, m in color_masks.items():
            covs[name] = _best_ring_coverage(cx, cy, m)
        best_color = max(covs, key=lambda k: covs[k][0])
        best_cov, best_r = covs[best_color]
        if best_cov < MIN_RING_COVERAGE:
            continue
        kind = "filled" if area > 60 else "ring"
        out[best_color].append((cx, cy, best_cov, best_r, kind))
    return out


def detect_centers(
    bgr: np.ndarray, masks: SeriesMasks, plot: PlotRegion,
) -> tuple[list[Center], list[Center]]:
    """Back-compat shim: returns (bbt_centers, lh_centers).

    BBT is always blue. The LH centers are the UNION of orange and purple
    candidates so callers that only need rough x-positions for grid
    inference see every marker regardless of which line is the real LH line.
    Series-role disambiguation (which of orange/purple is LH) happens in the
    pipeline via `resolve_lh_color`.
    """
    by_color = detect_centers_by_color(bgr, masks, plot)
    return by_color["blue"], by_color["orange"] + by_color["purple"]


def resolve_lh_color(
    by_color: dict[str, list[Center]],
    masks: SeriesMasks | None = None,
    plot: PlotRegion | None = None,
) -> str:
    """Decide which colored line carries the per-day LH data.

    The Premom family draws TWO non-blue curves:
      * "Ratio"  (orange/coral) — LH:PdG ratio
      * "Level"  (purple/violet) — LH level

    Depending on the chart variant, EITHER may be the densely-sampled
    per-day line while the other shows only the LH-Peak super-marker:
      - real-screen-1/2: orange is dense, purple is absent/sparse  → LH=orange
      - real-screen-3/4: purple is dense (open circle every day),
                         orange shows only the peak                → LH=purple

    Two signals decide it:
      1) discrete marker count per color (works when markers are isolable —
         real-screen-4 has 29 purple vs 1 orange);
      2) line CONTINUITY (column ink-span fraction) — needed for
         real-screen-3 where the dense purple circles fuse into one blob so
         the discrete count collapses to ~2, yet the purple line covers
         nearly every column while orange covers only the peak.

    Purple wins when it is clearly the denser/longer line; ties and the
    common single-orange-line case favor orange (historical default).
    """
    n_orange = len(by_color.get("orange", []))
    n_purple = len(by_color.get("purple", []))
    # Signal 1: discrete markers.
    if n_purple >= 4 and n_purple > n_orange + 2:
        return "purple"
    # Signal 2: continuity. When neither color isolates many discrete
    # markers, compare how much of the chart width each line spans.
    if masks is not None and plot is not None:
        span_orange = _ink_span_fraction(masks.orange, plot)
        span_purple = _ink_span_fraction(masks.purple, plot)
        # Purple must be a genuinely continuous line (covers most columns)
        # AND substantially longer than orange to override the default.
        if span_purple >= 0.70 and span_purple > span_orange + 0.25:
            return "purple"
    return "orange"


def _snap(
    centers: list[tuple[float, float, float, int, str]],
    cells: list[float],
    cell_px: float,
    series: SeriesKey,
) -> dict[int, Marker]:
    """Assign each detected marker to a day cell.

    Naive independent nearest-cell snapping loses markers two ways on dense
    charts (real-screen-4: 36 markers → only 27 cells):
      * COLLISIONS — two adjacent markers round to the same cell when the
        grid phase is slightly off, so one is silently dropped;
      * DRIFT — accumulated grid-offset error pushes late markers just past
        the ±half window of their true cell.

    Markers are physically ordered left-to-right and their cell indices must
    therefore be strictly increasing. We exploit that: sort markers by x,
    walk them in order, and assign each to the nearest cell that is > the
    previous assignment. When the nearest free cell is taken, step to the
    next one (markers are ~one cell pitch apart, so the next cell is the
    right home). This resolves collisions instead of discarding, and tolerates
    drift because each marker only needs to beat its neighbor, not hit an
    absolute window.
    """
    if not cells:
        return {}
    half = cell_px * 0.75
    ordered = sorted(centers, key=lambda c: c[0])
    out: dict[int, Marker] = {}
    last_idx = -1
    n_cells = len(cells)
    for cx, cy, score, radius, kind in ordered:
        # nearest cell at or after last_idx + 1
        best_idx, best_d = -1, float("inf")
        for i in range(max(0, last_idx + 1), n_cells):
            d = abs(cx - cells[i])
            if d < best_d:
                best_d, best_idx = d, i
            elif cells[i] > cx and d > best_d:
                # moving away past the marker — no closer cell ahead
                break
        if best_idx == -1:
            continue
        # Drift tolerance: accept if within `half`, OR if this is the only
        # plausible home (nearest forward cell) and within 1.4·pitch — dense
        # uniform marker runs drift but stay ~one pitch apart.
        if best_d > half and best_d > 1.4 * cell_px:
            continue
        out[best_idx] = Marker(
            series=series, cell_idx=best_idx, cx=cx, cy=cy,
            radius=float(radius), kind=kind, score=float(score),
        )
        last_idx = best_idx
    return out


# ── line sampling (dense continuous curves) ────────────────────────────────
# Some chart variants draw a series as a SMOOTH dense curve whose per-day open
# circles are too small / too merged with the stroke to isolate as discrete
# white-center blobs (real-screen-3's purple "Level" line: ~50 markers fuse
# into one connected component). For those we don't need discrete markers at
# all — the line itself carries one y-value per cell column. We sample the
# ink centroid in a narrow window around each cell center.

# Fraction of cell_px sampled on each side of a cell center. Narrow enough
# that on a steep slope we read the value AT the cell, not a smear of
# neighbors; wide enough to survive a 1-2px grid-offset error.
_SAMPLE_HALF_FRAC = 0.30
# A cell column must carry at least this many ink pixels to count as present
# (rejects gaps where the line is absent / behind a band edge).
_MIN_COL_INK = 4


def _detect_cover_line_y(mask: np.ndarray, plot: PlotRegion) -> float | None:
    """Detect the y of a flat horizontal cover-line in `mask`, or None.

    The Premom BBT chart draws a horizontal "cover line" (the temperature
    threshold) as a single flat stroke spanning most of the chart width at a
    constant y. It is NOT per-day data, so when we line-sample the blue
    series we must exclude it — otherwise every empty column reads the
    cover-line y and we invent points across the whole grid.

    Signature: a row band (a few px tall) whose horizontal ink extent covers
    a large fraction of the plot width AND is much flatter (wider) than the
    data line at any single y. We scan row-wise ink counts after a horizontal
    opening that keeps only long flat runs.
    """
    H, W = mask.shape
    x0, x1 = max(0, plot.x0), min(W, plot.x1 + 1)
    y0, y1 = max(0, plot.y0), min(H, plot.y1 + 1)
    region = mask[y0:y1, x0:x1]
    if region.size == 0:
        return None
    w = region.shape[1]
    # Keep only horizontal runs at least ~40% of plot width — the data line
    # is never flat for that long, but the cover-line is.
    kw = max(20, int(w * 0.40))
    horiz_k = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
    horiz = cv2.morphologyEx(region, cv2.MORPH_OPEN, horiz_k)
    row_ink = (horiz > 0).sum(axis=1).astype(np.float32)
    if row_ink.max() < kw:
        return None
    # The cover-line's row(s) dominate. Take the ink-weighted centroid of the
    # rows that survived the long-horizontal opening.
    rows = np.where(row_ink >= 0.5 * row_ink.max())[0]
    if rows.size == 0:
        return None
    return float(rows.mean()) + y0


def sample_line_per_cell(
    mask: np.ndarray,
    plot: PlotRegion,
    cells: list[float],
    cell_px: float,
    series: SeriesKey,
    exclude_y: float | None = None,
    exclude_half: float = 6.0,
) -> dict[int, Marker]:
    """Read one y-value per cell directly off a continuous color line.

    For each cell center, take the ink pixels of `mask` within a narrow
    x-window, restricted to the plot's y-range. If the column has enough ink
    AND its vertical spread is bounded (a single line crossing, not a tall
    band artifact), record the centroid y as that cell's value.

    `exclude_y`: y of a flat cover-line to ignore. Pixels within
    ±`exclude_half` of it are dropped BEFORE deciding presence, so columns
    that contain ONLY the cover-line (no data point) are correctly skipped
    rather than reading the cover-line as data.

    Returns the same `dict[cell_idx -> Marker]` shape as `_snap` so the rest
    of the pipeline treats line-sampled points and discrete markers
    identically. `kind` is tagged "line" for debugging / overlay.
    """
    H, W = mask.shape
    y_lo = max(0, plot.y0)
    y_hi = min(H, plot.y1 + 1)
    half = max(2, int(round(cell_px * _SAMPLE_HALF_FRAC)))
    # A real line crossing spans at most a handful of px vertically within a
    # narrow x-window; a band-fill edge or a fat junction would span much
    # more. Cap the allowed vertical spread at ~one cell pitch so we don't
    # average across a near-vertical segment into a meaningless midpoint.
    max_spread = max(12.0, cell_px * 1.2)
    out: dict[int, Marker] = {}
    for i, ccx in enumerate(cells):
        x_lo = max(0, int(round(ccx - half)))
        x_hi = min(W, int(round(ccx + half)) + 1)
        if x_hi - x_lo < 1:
            continue
        window = mask[y_lo:y_hi, x_lo:x_hi]
        ys, _ = np.where(window > 0)
        if ys.size < _MIN_COL_INK:
            continue
        ys_real = ys + y_lo
        if exclude_y is not None:
            # Drop cover-line pixels. If the column has data ABOVE/BELOW the
            # cover-line we keep it; if it's only the cover-line we skip.
            keep = np.abs(ys_real - exclude_y) > exclude_half
            ys_real = ys_real[keep]
            if ys_real.size < _MIN_COL_INK:
                continue
        # Use the median y (robust to a few stray band-edge pixels) and check
        # the inlier spread around it.
        med = float(np.median(ys_real))
        inliers = ys_real[np.abs(ys_real - med) <= max_spread / 2.0]
        if inliers.size < _MIN_COL_INK:
            continue
        cy = float(np.mean(inliers))
        out[i] = Marker(
            series=series, cell_idx=i, cx=float(ccx), cy=cy,
            radius=float(cell_px * 0.15), kind="line",
            score=float(min(1.0, inliers.size / 20.0)),
        )
    return out


def _ink_span_fraction(mask: np.ndarray, plot: PlotRegion) -> float:
    """Fraction of plot-width columns that carry any ink for this mask.

    A dense continuous line covers nearly every column; a sparse
    few-marker line covers very few. Used to decide whether line-sampling
    is appropriate (continuous) vs. would invent data (sparse).
    """
    H, W = mask.shape
    x0 = max(0, plot.x0)
    x1 = min(W, plot.x1 + 1)
    y0 = max(0, plot.y0)
    y1 = min(H, plot.y1 + 1)
    strip = mask[y0:y1, x0:x1]
    if strip.size == 0:
        return 0.0
    col_has_ink = (strip > 0).any(axis=0)
    return float(col_has_ink.mean())


def extract_per_cell(
    blue_mask: np.ndarray,
    orange_mask: np.ndarray,
    plot: PlotRegion,
    cells: list[float],
    cell_px: float,
    bgr: np.ndarray | None = None,
    masks: SeriesMasks | None = None,
    out_meta: dict | None = None,
) -> tuple[dict[int, Marker], dict[int, Marker]]:
    """Extract markers per cell.

    The new detector needs the full BGR image (for white-blob detection) and
    the SeriesMasks struct (for ring-coverage on both colors). Pipeline.py
    passes them via the optional kwargs; the legacy positional arguments are
    kept so the function signature is backwards-compatible with anything
    that still calls it with just the masks.

    `out_meta`: optional dict the caller passes in to receive detection
    metadata — currently `lh_color` ("orange"|"purple"), `n_orange`,
    `n_purple`. The pipeline uses `lh_color` to pick the correct axis
    normalization (orange "Ratio" uses the 0.1-1.9 axis; purple "Level"
    uses the 5-95 axis, so its values are normalized by plot extent rather
    than the Ratio tick mapping).
    """
    if bgr is None or masks is None:
        # Legacy path: rebuild a SeriesMasks-like view from the two masks.
        # Used only when extract_per_cell is called without the new kwargs;
        # in that case white-blob detection isn't possible — return empty.
        return {}, {}
    by_color = detect_centers_by_color(bgr, masks, plot)
    lh_color = resolve_lh_color(by_color, masks=masks, plot=plot)
    bbt_centers = by_color["blue"]
    lh_centers = by_color[lh_color]

    n_cells = max(1, len(cells))
    bbt = _snap(bbt_centers, cells, cell_px, "temp")
    lh = _snap(lh_centers, cells, cell_px, "lh")

    # Line-sampling fallback for DENSE CONTINUOUS curves.
    #
    # When a series is drawn as a smooth line whose per-day open circles fuse
    # into the stroke (real-screen-3: ~50 purple "Level" markers collapse
    # into one connected component; real-screen-4: discrete detection snaps
    # 29 markers onto only 17 cells due to collisions), discrete detection
    # under-counts. The line itself carries one y per cell, so we line-sample.
    #
    # Trigger (per series): line-sample when the line is CONTINUOUS enough
    # that one value exists at (nearly) every cell column, AND that recovers
    # more cells than discrete snapping. Two regimes:
    #   * span ≥ 0.88  → near-fully-continuous line: a value at essentially
    #     every cell, so line-sampling reads the COMPLETE per-day series.
    #     Prefer it whenever it beats discrete (real-screen-4 purple: 36
    #     discrete markers snap to ~26 cells, but the line covers all 35).
    #   * 0.70 ≤ span < 0.88 → only switch when discrete clearly under-samples
    #     the line (< 70% of the line's column-span in cells), so we don't
    #     override an already-complete discrete read.
    # The continuity gate keeps us off genuinely sparse few-point lines
    # (real-screen-1 orange: isolated peak + sparse rest) where discrete is
    # correct and line-sampling would invent data between points.
    #
    # BBT (blue) carries the flat horizontal COVER-LINE which is NOT per-day
    # data, so before sampling blue we locate the cover-line y and exclude
    # it — otherwise empty columns would all read the cover-line and invent
    # points past the real series. LH lines (orange/purple) have no cover-line.
    lh_color_mask = (masks.orange if lh_color == "orange" else masks.purple)
    bbt_used_line = False
    lh_used_line = False

    def _should_line_sample(span: float, n_disc: int) -> bool:
        if span >= 0.88:
            return True
        return span >= 0.70 and n_disc < 0.70 * span * n_cells

    blue_span = _ink_span_fraction(masks.blue, plot)
    if _should_line_sample(blue_span, len(bbt)):
        cover_y = _detect_cover_line_y(masks.blue, plot)
        sampled = sample_line_per_cell(
            masks.blue, plot, cells, cell_px, "temp", exclude_y=cover_y,
        )
        if len(sampled) > len(bbt):
            bbt = sampled
            bbt_used_line = True

    lh_span = _ink_span_fraction(lh_color_mask, plot)
    if _should_line_sample(lh_span, len(lh)):
        sampled = sample_line_per_cell(lh_color_mask, plot, cells, cell_px, "lh")
        if len(sampled) > len(lh):
            lh = sampled
            lh_used_line = True

    if out_meta is not None:
        out_meta["lh_color"] = lh_color
        out_meta["n_orange"] = len(by_color["orange"])
        out_meta["n_purple"] = len(by_color["purple"])
        out_meta["bbt_method"] = "line" if bbt_used_line else "markers"
        out_meta["lh_method"] = "line" if lh_used_line else "markers"
    return bbt, lh
