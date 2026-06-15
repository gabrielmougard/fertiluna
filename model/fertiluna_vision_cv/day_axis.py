"""Day-cell grid detection.

The chart shows K evenly-spaced day cells. We need their pixel x-centers so
the marker-extraction step can scan one cell column at a time — the
"draw imaginary lines, read the dots against the scales" approach.

Detection strategies, tried in order:

  1. **Vertical gridlines** — Premom-style charts draw a faint pale-gray
     vertical line at each cell boundary (`axvline(d - 0.5, ...)` in the
     renderer). Morphology on the pale-gray-but-not-white band, opened with
     a tall vertical kernel, gives column-sum peaks at gridline x-positions.
     Cell centers = midpoints between consecutive gridlines.

  2. **Marker-spacing inference** — if gridlines fail (too pale, banded
     backgrounds, missing rendering), fall back to: pool rough marker
     x-positions from the color masks, take the median consecutive Δx as
     cell pitch, then slide the offset along that pitch to maximize
     marker-to-cell-center alignment.

  3. **Uniform fallback** — plot.width / N_DAYS, used only if both above
     fail (e.g., near-empty chart).

The grid is independent of WHICH cells contain data: it covers the visible
x-range, and the per-cell scan decides presence.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .color_segmentation import SeriesMasks
from .constants import N_DAYS
from .plot_region import PlotRegion


@dataclass
class DayGrid:
    cells: list[float]   # x-centers in working-canvas pixels, left-to-right
    cell_px: float
    source: str          # "date-row" | "gridlines" | "marker-spacing" | "uniform"


def _detect_vertical_gridlines(bgr: np.ndarray, plot: PlotRegion) -> list[float] | None:
    """Find vertical pale-gray gridlines within the plot region.

    Returns a sorted list of x-pixel positions in working-canvas coords.

    Pitfall: alpha-blended band fills (violet fertile, pink period) end up in
    the same brightness band as gridlines. We exclude them by requiring NEAR-
    NEUTRAL color (low saturation) — real gridlines are R≈G≈B, tinted bands
    are not.
    """
    region = bgr[plot.y0:plot.y1, plot.x0:plot.x1]
    h, w = region.shape[:2]
    if h < 30 or w < 100:
        return None
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    # Pale brightness window. Real Premom grid (#efeff4) sits around 239.
    pale_v = cv2.inRange(gray, 228, 247)
    # Neutral hue: tinted bands carry saturation; gridlines do not.
    neutral = cv2.inRange(sat, 0, 18)
    pale = cv2.bitwise_and(pale_v, neutral)
    kh = max(20, h // 3)
    vert_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kh))
    vert = cv2.morphologyEx(pale, cv2.MORPH_OPEN, vert_k)
    col_sums = vert.sum(axis=0).astype(np.float32)
    if col_sums.max() < 255 * 5:
        return None
    # Find peak columns: local maxima above 40% of max.
    thresh = 0.4 * col_sums.max()
    peaks: list[float] = []
    in_run = False
    run_start = 0
    for i, c in enumerate(col_sums):
        if c >= thresh and not in_run:
            run_start = i
            in_run = True
        elif c < thresh and in_run:
            peaks.append((run_start + i - 1) / 2)
            in_run = False
    if in_run:
        peaks.append((run_start + len(col_sums) - 1) / 2)
    if len(peaks) < 5:
        return None
    if len(peaks) < 5:
        return None
    spacings = np.diff(peaks)
    med = float(np.median(spacings))
    if med < 8:
        return None
    keep = (spacings >= 0.6 * med) & (spacings <= 1.4 * med)
    if keep.sum() < 0.5 * len(spacings):
        return None
    return [p + plot.x0 for p in peaks]


def _detect_date_row(bgr: np.ndarray, plot: PlotRegion) -> list[float] | None:
    """Cluster centroids of dark digit-shaped blobs in the row below the plot.

    Premom renders a calendar-date number in each cell directly below the
    plot. The text is high-contrast and one cluster per cell, which makes
    this the most reliable cell-position signal when gridlines are too pale
    or polluted by band fills.
    """
    H, W = bgr.shape[:2]
    y0 = min(H - 1, plot.y1 + 4)
    y1 = min(H, plot.y1 + max(40, int(plot.height * 0.18)))
    if y1 - y0 < 18:
        return None
    band = bgr[y0:y1, plot.x0:plot.x1]
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    _, dark = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, _, stats, cents = cv2.connectedComponentsWithStats(dark, connectivity=8)
    xs: list[float] = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        ww = int(stats[i, cv2.CC_STAT_WIDTH])
        hh = int(stats[i, cv2.CC_STAT_HEIGHT])
        # digit-sized component filter (rejects long pill backgrounds and noise)
        if 12 <= area <= 600 and 3 <= ww <= 40 and 6 <= hh <= 36:
            xs.append(float(cents[i, 0]))
    if len(xs) < 5:
        return None
    xs.sort()
    # Merge multi-digit numbers ("12" → two components within ~8 px).
    clusters: list[list[float]] = [[xs[0]]]
    # rough cell pitch from raw spacings: clusters merge below 30% of median
    spacings = np.diff(xs)
    med = float(np.median(spacings)) if spacings.size else 20.0
    merge_thr = max(4.0, med * 0.30)
    for x in xs[1:]:
        if x - clusters[-1][-1] < merge_thr:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    centers = [sum(c) / len(c) + plot.x0 for c in clusters]
    # filter by inter-cluster consistency (same idea as gridlines)
    if len(centers) < 5:
        return None
    diffs = np.diff(centers)
    med2 = float(np.median(diffs))
    if med2 < 10:
        return None
    kept = [centers[0]]
    for c in centers[1:]:
        d = c - kept[-1]
        if 0.55 * med2 <= d <= 1.45 * med2:
            kept.append(c)
    if len(kept) < 5:
        return None
    return kept


def _rough_marker_xs(
    bgr: np.ndarray, masks: SeriesMasks, plot: PlotRegion,
) -> list[float]:
    """Marker x-centers via the SAME white-blob+ring detector the main
    extractor uses, so grid inference can't disagree with cell snapping.
    """
    from .marker_detection import detect_centers

    bbt, lh = detect_centers(bgr, masks, plot)
    return sorted([c[0] for c in bbt] + [c[0] for c in lh])


def _autocorr_cell_px(masks: SeriesMasks, plot: PlotRegion) -> float | None:
    """Estimate cell pitch via autocorrelation of the column-density profile.

    Markers repeat at the cell pitch, so the per-column color-pixel sum is
    quasi-periodic with period = cell_px. Autocorrelation has its first peak
    (after the main lobe) at that period. Robust to noise, band fills, and
    sparse missing markers.
    """
    plot_w = plot.x1 - plot.x0
    if plot_w < 60:
        return None
    combined = cv2.bitwise_or(masks.blue, masks.orange)
    strip = combined[plot.y0:plot.y1, plot.x0:plot.x1]
    if strip.size == 0:
        return None
    col_sum = (strip > 0).sum(axis=0).astype(np.float32)
    if col_sum.std() < 0.5:
        return None
    col_sum = col_sum - col_sum.mean()
    n = col_sum.size
    pad = 1
    while pad < 2 * n:
        pad <<= 1
    f = np.fft.rfft(col_sum, pad)
    acorr = np.fft.irfft(f * np.conj(f), pad)[:n]
    # Skip the main lobe and unreasonably large lags. Real cell pitches sit
    # in [plot_w / 40, plot_w / 8] for charts of 8..40 visible cells.
    # min_lag bounds the SMALLEST plausible cell pitch. plot_w/80 lets us
    # detect dense layouts like real-screen-3 where markers sit every
    # ~35-40 px (≈55 visible cells across the chart). A larger floor would
    # force autocorrelation onto the first harmonic and quadruple the
    # marker-collision rate at cell-snap time.
    min_lag = max(15, plot_w // 80)
    max_lag = min(n - 1, max(min_lag + 1, plot_w // 6))
    if max_lag <= min_lag:
        return None
    sub = acorr[min_lag:max_lag]
    if sub.size == 0:
        return None
    return float(int(np.argmax(sub)) + min_lag)


def _grid_from_marker_spacing(
    marker_xs: list[float], plot: PlotRegion,
    cell_px_hint: float | None = None,
) -> tuple[list[float], float, str]:
    """Estimate cell pitch + offset, then tile cells across plot.

    `cell_px_hint`: if provided (e.g. from autocorrelation), trust it for the
    pitch and only solve for the offset. Otherwise fall back to inter-marker
    spacing histogram.
    """
    if len(marker_xs) < 2 and cell_px_hint is None:
        cell_px = plot.width / N_DAYS
        cells = [plot.x0 + (k + 0.5) * cell_px for k in range(N_DAYS)]
        return cells, cell_px, "uniform"
    sorted_x = sorted(marker_xs)
    if cell_px_hint is not None and cell_px_hint > 0:
        cell_px = float(cell_px_hint)
    elif len(sorted_x) >= 2:
        deltas = np.diff(np.array(sorted_x))
        deltas = deltas[deltas >= 8.0]   # drop near-duplicate peaks
        if deltas.size == 0:
            cell_px = plot.width / N_DAYS
        else:
            # mode of small deltas: 1-cell hops dominate, n-cell gaps are tail.
            max_d = float(deltas.max())
            bin_w = 2.0
            n_bins = max(1, int(np.ceil(max_d / bin_w)))
            hist, edges = np.histogram(deltas, bins=n_bins,
                                       range=(0, max_d + bin_w))
            cutoff = max(1, int(0.8 * n_bins))
            peak = int(np.argmax(hist[:cutoff])) if hist[:cutoff].any() else 0
            cell_px = float((edges[peak] + edges[peak + 1]) / 2)
            if cell_px < 8.0:
                cell_px = float(np.median(deltas))
    else:
        cell_px = plot.width / N_DAYS
    # Safety clamp: cell pitches are at most plot.width/6 (≥ 6 cells visible)
    # and at least plot.width/80 (allows dense layouts like real-screen-3 with
    # ~55 days of markers per chart). A tighter lower bound would force the
    # autocorrelation answer onto its first harmonic and collapse pairs of
    # adjacent markers into the same cell at snap time.
    lo = max(15.0, plot.width / 80)
    hi = max(lo + 1.0, plot.width / 6)
    if not (lo <= cell_px <= hi):
        cell_px = float(np.clip(cell_px, lo, hi))
    # Place the grid CENTERED ON THE OBSERVED MARKER DISTRIBUTION rather
    # than anchored at plot.x0. Charts with more than N_DAYS visible days
    # (real-screen-3 has ~58) would otherwise have their rightmost markers
    # fall past cell[34] and be lost. We pick a cell-center that minimizes
    # marker→nearest-cell residuals, then tile N_DAYS cells around it.
    marker_arr = np.array(sorted_x, dtype=np.float64)
    if len(marker_arr) >= 2:
        first_m = float(marker_arr[0])
        # Anchor at the LEFTMOST marker so the grid captures cells 0..N_DAYS-1
        # starting there. Charts longer than N_DAYS days (real-screen-3 has
        # ~58) drop their rightmost markers — the cycle window the schema
        # represents is the FIRST 35 days of visible data.
        grid_first = first_m - cell_px / 2
        # Refine the offset by minimizing the marker-to-cell-center error
        # over a one-cell-wide sweep.
        step = max(0.5, cell_px / 60)
        best_off, best_err = 0.0, float("inf")
        for off in np.arange(-cell_px / 2, cell_px / 2 + step, step):
            anchor = grid_first + off
            rel = (marker_arr - anchor) % cell_px
            dist = np.minimum(rel, cell_px - rel)
            err = float(dist.sum())
            if err < best_err:
                best_err = err
                best_off = float(off)
        grid_first = grid_first + best_off
    else:
        grid_first = plot.x0
    # Tile N_DAYS cells, clipping at plot bounds.
    cells = [grid_first + (k + 0.5) * cell_px for k in range(N_DAYS)]
    return cells, cell_px, "marker-spacing"


def detect_grid(bgr: np.ndarray, plot: PlotRegion, masks: SeriesMasks) -> DayGrid:
    """Detect day-cell centers + cell pitch.

    Autocorrelation of the column-density profile is the PRIMARY signal —
    it's robust to noisy/missing gridlines and to varying chart palettes.
    Other detectors (date-row clustering, vertical gridline morphology)
    are accepted only when they pass a sanity-check against the
    autocorrelation pitch.
    """
    # Autocorrelation: most robust because it doesn't depend on detecting
    # individual features, only on the periodicity of marker spacing.
    cell_hint = _autocorr_cell_px(masks, plot)
    plot_w = plot.x1 - plot.x0
    # Plausible cell-pitch band: 8-40 visible cells across the plot.
    pitch_lo = plot_w / 80   # up to ~80 cells (dense Premom layouts)
    pitch_hi = plot_w / 8    # at least ~8 cells visible

    # Try date-row digit clustering — most accurate when it works.
    # Coverage sanity: the cluster count must be close to plot_w / cell_px
    # (charts with date numbers SOME of which are obscured by colored bands
    # only contribute partial clusters; we reject those and fall through).
    date_xs = _detect_date_row(bgr, plot)
    if date_xs is not None and len(date_xs) >= 8:
        diffs = np.diff(date_xs)
        cell_px = float(np.median(diffs))
        expected_n = plot_w / cell_px if cell_px > 0 else 0
        coverage = len(date_xs) / expected_n if expected_n > 0 else 0
        if (pitch_lo <= cell_px <= pitch_hi
                and coverage >= 0.6
                and (cell_hint is None
                     or 0.7 * cell_hint <= cell_px <= 1.3 * cell_hint)):
            return DayGrid(cells=date_xs[:N_DAYS], cell_px=cell_px,
                           source="date-row")

    # Vertical gridline morphology — same coverage sanity check.
    grid_xs = _detect_vertical_gridlines(bgr, plot)
    if grid_xs is not None and len(grid_xs) >= 5:
        diffs = np.diff(grid_xs)
        cell_px = float(np.median(diffs))
        expected_n = plot_w / cell_px if cell_px > 0 else 0
        coverage = len(grid_xs) / expected_n if expected_n > 0 else 0
        if (pitch_lo <= cell_px <= pitch_hi
                and coverage >= 0.6
                and (cell_hint is None
                     or 0.7 * cell_hint <= cell_px <= 1.3 * cell_hint)):
            cells = [(grid_xs[i] + grid_xs[i + 1]) / 2.0
                     for i in range(len(grid_xs) - 1)]
            return DayGrid(cells=cells[:N_DAYS], cell_px=cell_px,
                           source="gridlines")

    # Autocorr + marker-spacing cross-check.
    #
    # Autocorrelation peaks at the cell pitch — but ALSO at sub-harmonics
    # (½ pitch, ⅓ pitch) when the signal isn't perfectly periodic. The
    # marker-spacing histogram from the actual detected white blobs is a
    # second opinion grounded in real markers; we prefer it when it
    # disagrees with autocorr by more than 25%.
    marker_xs = _rough_marker_xs(bgr, masks, plot)
    spacing_pitch: float | None = None
    if len(marker_xs) >= 4:
        deltas = np.diff(np.array(sorted(marker_xs)))
        deltas = deltas[(deltas >= pitch_lo) & (deltas <= pitch_hi)]
        if deltas.size >= 3:
            # mode via histogram with bin width = 2 px
            mx = float(deltas.max())
            n_bins = max(6, int(np.ceil(mx / 2.0)))
            hist, edges = np.histogram(deltas, bins=n_bins,
                                       range=(0.0, mx + 2.0))
            peak = int(np.argmax(hist))
            spacing_pitch = float((edges[peak] + edges[peak + 1]) / 2)
    chosen_hint = cell_hint
    if (cell_hint is not None and spacing_pitch is not None
            and abs(cell_hint - spacing_pitch) / max(cell_hint,
                                                     spacing_pitch) > 0.25):
        # disagree → trust the marker-grounded histogram.
        chosen_hint = spacing_pitch
    elif cell_hint is None and spacing_pitch is not None:
        chosen_hint = spacing_pitch
    cells, cell_px, src = _grid_from_marker_spacing(marker_xs, plot,
                                                    cell_px_hint=chosen_hint)
    if chosen_hint is not None:
        src = "autocorr" if chosen_hint == cell_hint else "marker-spacing"
    return DayGrid(cells=cells, cell_px=cell_px, source=src)
