"""Bottom-of-screen table extractor.

Premom-family chart screenshots have a TABLE sitting BELOW the plot. Top-to-
bottom it contains (when present):

    * calendar  — starting month (3-letter) + day-of-month numbers
    * CD        — cycle day
    * DPO       — days past ovulation (blank pre-ovulation)
    * Sex       — heart icon per day a sexual encounter was logged
    * CM        — cervical-mucus indicator dot (purple circle when present)
    * Symptoms  — symptom annotation per day
    * hCG       — pregnancy-test result per day

Why this version is a rewrite:

  v1 used the CHART'S day-cell grid as table column boundaries. That over-
  segments because the calendar's date numbers don't sit at exactly the same
  pitch — they have their OWN x-positions. Result: "12" read as "0 1" because
  the cell straddled the end of one number and the start of the next.

  v1 also only detected rows that contained DARK TEXT, so rows whose only
  content is colored icons (Sex hearts, CM purple dots) were missed entirely.

v2 fixes both:

  1. The CALENDAR row drives the table's column grid. We find dark-text
     clusters in that row, group them into date-number clusters via a
     bimodal-gap split (intra-digit gaps are small, inter-cell gaps are
     large), and the cluster centers become the table column centers.
  2. We extrapolate ALL canonical rows downward at the calendar→CD row
     spacing, so empty / icon-only rows get cells anyway.
  3. Each row gets a content extractor matched to its TYPE:
       * "calendar" / "CD" / "DPO" → digit OCR
       * "Sex"     → heart-icon presence (warm-color pixels)
       * "CM"      → purple-circle presence (HSV purple band)
       * "Symptoms" / "hCG" → generic non-empty detection (any dark pixel)
     Each extractor returns a string (the cell's interpretation) or None.

When the screenshot is chart-only (no table) every row stays empty and
the `table` dict is empty.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .constants import N_DAYS
from .day_axis import DayGrid
from .plot_region import PlotRegion


_CANONICAL_ROW_ORDER = (
    "calendar", "CD", "DPO", "Sex", "CM", "Symptoms", "hCG",
)

# Row-name detection keywords (case-insensitive) found in left-of-row labels.
_LABEL_KEYWORDS = (
    ("cd", "CD"), ("dpo", "DPO"), ("sex", "Sex"),
    ("cm", "CM"), ("symptoms", "Symptoms"), ("symptom", "Symptoms"),
    ("hcg", "hCG"),
)
_MONTH_ABBREVS = ("jan", "feb", "mar", "apr", "may", "jun",
                  "jul", "aug", "sep", "oct", "nov", "dec")

# Icon-row color thresholds (HSV; H 0..179 in OpenCV).
# Sex heart in Premom renders as warm orange / coral (#e8825f); some apps
# use red. We accept a generous red-to-orange band.
_HSV_HEART = ((0, 25, 90, 80), (170, 179, 90, 80))   # two bands wrapping red
# CM cue is purple — same hue family as the LH-peak band, distinct from
# axis/data colors.
_HSV_PURPLE_CIRCLE = (125, 160, 50, 90)
# Generic "non-background dark" pixels for Symptoms / hCG presence checks.
_DARK_THRESHOLD = 200   # Premom-family apps render label text in PALE GRAY
                        # (RGB ≈ 150-180). A tighter 170 misses real labels
                        # on screens 2/3/4; 200 + a saturation gate (applied
                        # where it matters) catches them without pulling in
                        # tinted band fills.


@dataclass
class TableCell:
    text: str | None                    # OCR'd content, or "♥" / "●" markers,
                                        # or None if visually empty
    bbox: tuple[int, int, int, int]     # (x0, y0, x1, y1) in working canvas
    kind: str = "text"                  # "text" | "heart" | "circle" | "icon"


@dataclass
class TableRow:
    name: str                           # canonical row name
    label_bbox: tuple[int, int, int, int]
    label_text: str
    y_top: int
    y_bottom: int
    cells: list[TableCell] = field(default_factory=list)


# ── row centers from LEFT-SIDE row labels ──────────────────────────────────
def _row_centers_from_labels(
    bgr: np.ndarray, region_top: int, region_bottom: int, plot: PlotRegion,
) -> tuple[list[tuple[float, tuple[int, int, int, int]]], int]:
    """Find table row centers from the LEFT-side labels themselves.

    Each row in the Premom table has a short label on the left
    (Mar / CD / DPO / Sex / CM / Symptoms / hCG). Detecting dark text in
    the left strip and clustering by y gives one cluster per row, which
    is the single most reliable row-position signal — bands of cell
    CONTENT (text or icons) can be missing for sparse rows but the LABEL
    is always there when the row exists.

    Returns ([(row_center_y, label_bbox), …], typical_row_height).
    """
    H, W = bgr.shape[:2]
    if region_bottom - region_top < 20:
        return [], 0
    x_search_end = max(int(W * 0.18), plot.x0 + int(W * 0.06))
    x_search_end = min(W, x_search_end)
    strip = bgr[region_top:region_bottom, 0:x_search_end]
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    dark = cv2.inRange(gray, 0, _DARK_THRESHOLD)
    n, _, stats, cents = cv2.connectedComponentsWithStats(dark, connectivity=8)
    comps: list[tuple[float, int, int, int, int]] = []
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if a < 6 or w < 2 or h < 4 or w > 100 or h > 40:
            continue
        cy = float(cents[i, 1]) + region_top
        x0 = int(stats[i, cv2.CC_STAT_LEFT])
        y0 = int(stats[i, cv2.CC_STAT_TOP]) + region_top
        comps.append((cy, x0, y0, w, h))
    if len(comps) < 2:
        return [], 0
    comps.sort(key=lambda c: c[0])
    # Cluster by y: a new cluster starts when the y-gap exceeds half the
    # typical row spacing (estimated from the median y-delta).
    deltas = sorted(comps[i + 1][0] - comps[i][0] for i in range(len(comps) - 1))
    if not deltas:
        return [], 0
    median = deltas[len(deltas) // 2]
    # Row labels span few px vertically; intra-row deltas are small,
    # inter-row deltas large.  Bimodal split via "above-median is inter".
    large = [d for d in deltas if d > median]
    if large:
        row_spacing_est = float(np.median(large))
    else:
        row_spacing_est = float(median * 2)
    # 0.7 of the typical inter-row spacing — tight enough to keep adjacent
    # rows separate but loose enough that a row's own anti-aliased baseline
    # (two components separated by ~20px) merges back into one cluster.
    gap_thresh = max(8.0, row_spacing_est * 0.7)
    clusters: list[list[tuple[float, int, int, int, int]]] = [[comps[0]]]
    for c in comps[1:]:
        if c[0] - clusters[-1][-1][0] < gap_thresh:
            clusters[-1].append(c)
        else:
            clusters.append([c])
    # Per cluster: y center + bounding box around its components.
    out: list[tuple[float, tuple[int, int, int, int]]] = []
    for cl in clusters:
        cy = float(np.mean([c[0] for c in cl]))
        x_min = min(c[1] for c in cl)
        y_min = min(c[2] for c in cl)
        x_max = max(c[1] + c[3] for c in cl)
        y_max = max(c[2] + c[4] for c in cl)
        out.append((cy, (x_min, y_min, x_max, y_max)))
    typical_h = int(np.median([cl_bb[1][3] - cl_bb[1][1] for cl_bb in out]))
    return out, max(12, typical_h)


# ── horizontal band detection (text OR icon presence) ───────────────────────
def _detect_nonempty_bands(
    bgr: np.ndarray, region_top: int, region_bottom: int,
    x0: int, x1: int,
) -> list[tuple[int, int]]:
    """Detect any row that has *something* (text OR colored icon).

    Uses BOTH dark-text rows AND saturated-color rows so icon-only rows
    (Sex hearts, CM purple dots) get picked up.
    """
    if region_bottom - region_top < 10 or x1 - x0 < 50:
        return []
    region = bgr[region_top:region_bottom, x0:x1]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    # combine: pixel is "interesting" if either dark (text) or saturated (icon)
    interesting = cv2.bitwise_or(
        cv2.inRange(gray, 0, _DARK_THRESHOLD),
        cv2.inRange(sat, 60, 255),
    )
    row_sums = (interesting > 0).sum(axis=1).astype(np.float32)
    if row_sums.max() < 2:
        return []
    row_sums = cv2.boxFilter(row_sums.reshape(1, -1), -1, (1, 3)).ravel()
    thr = max(3.0, 0.08 * row_sums.max())
    active = row_sums >= thr
    bands: list[tuple[int, int]] = []
    in_run, run_start = False, 0
    for i, a in enumerate(active):
        if a and not in_run:
            run_start = i
            in_run = True
        elif not a and in_run:
            bands.append((region_top + run_start, region_top + i - 1))
            in_run = False
    if in_run:
        bands.append((region_top + run_start, region_top + len(active) - 1))
    # filter: tiny bands are noise / single-baseline strokes; merge bands
    # whose vertical gap is small (typography baselines split a single row).
    bands = [b for b in bands if (b[1] - b[0]) >= 4]
    merged: list[tuple[int, int]] = []
    for b in bands:
        if merged and (b[0] - merged[-1][1]) <= 5:
            merged[-1] = (merged[-1][0], b[1])
        else:
            merged.append(b)
    return merged


# ── calendar row → table column grid ───────────────────────────────────────
def _calendar_column_centers(
    bgr: np.ndarray, band: tuple[int, int],
    x_lo: int, x_hi: int,
) -> tuple[list[float], float]:
    """Find the x-centers of date number clusters in the calendar row.

    Approach: connected components of dark text within the band, then group
    adjacent components into one date number by the bimodal-gap rule
    (intra-number digit gaps are small; inter-number cell gaps are large).

    Returns (column_centers, cell_pitch_px).
    """
    y0, y1 = band
    strip = bgr[y0:y1 + 1, x_lo:x_hi]
    if strip.size == 0:
        return [], 0.0
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    dark = cv2.inRange(gray, 0, _DARK_THRESHOLD)
    n, _, stats, cents = cv2.connectedComponentsWithStats(dark, connectivity=8)
    comp_xs: list[tuple[float, float, float]] = []   # (cx, left, right)
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if a < 4 or w < 2 or h < 4 or w > 50 or h > 40:
            continue
        cx = float(cents[i, 0]) + x_lo
        left = int(stats[i, cv2.CC_STAT_LEFT]) + x_lo
        right = left + w
        comp_xs.append((cx, float(left), float(right)))
    if len(comp_xs) < 2:
        return [], 0.0
    comp_xs.sort(key=lambda c: c[0])
    # Gap from one component's RIGHT edge to next component's LEFT edge —
    # this is cleaner than centroid-to-centroid for variable-width digits.
    gaps = [comp_xs[i + 1][1] - comp_xs[i][2] for i in range(len(comp_xs) - 1)]
    if not gaps:
        return [comp_xs[0][0]], 0.0
    # Bimodal split: anything bigger than the median gap is an "inter-number"
    # gap (new cell); anything ≤ median is "intra-number" (same cell).
    gaps_sorted = sorted(gaps)
    median = gaps_sorted[len(gaps_sorted) // 2]
    # Threshold = halfway between intra-mean and inter-mean approximations.
    small = [g for g in gaps if g <= median]
    large = [g for g in gaps if g > median]
    if not small or not large:
        threshold = max(8.0, median)
    else:
        threshold = (np.mean(small) + np.mean(large)) / 2.0
    threshold = max(8.0, float(threshold))

    groups: list[list[tuple[float, float, float]]] = [[comp_xs[0]]]
    for i, c in enumerate(comp_xs[1:], start=1):
        if (c[1] - groups[-1][-1][2]) <= threshold:
            groups[-1].append(c)
        else:
            groups.append([c])
    centers = [float(np.mean([c[0] for c in g])) for g in groups]
    if len(centers) >= 2:
        pitches = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
        cell_pitch = float(np.median(pitches))
    else:
        cell_pitch = float(x_hi - x_lo) / max(1, len(centers))
    return centers, cell_pitch


# ── table-structure detector (polarity-independent, autocorr pitch) ─────────
def _calendar_text_mask(strip_bgr: np.ndarray) -> np.ndarray:
    """Binary mask of date-number GLYPHS in the calendar band, regardless of
    polarity.

    Premom highlights period / fertile days by wrapping the date numbers in a
    SOLID colored pill (pink/purple) with WHITE text. A plain dark threshold
    misses those white-on-color digits and only finds the plain
    dark-on-white ones, so the column detector sees a sparse, irregular set.

    We combine two signals so a glyph is captured in EITHER rendering:
      * dark-on-light: adaptive threshold (handles the pale-gray text on
        white background);
      * light-on-color: bright, saturated-background pixels whose local
        neighborhood is colored — i.e. white text sitting on a pill. We get
        this as bright pixels minus the (smoothed) colored-pill body, leaving
        the glyph strokes.
    Morphological opening removes single-pixel noise; the result is a clean
    per-glyph ink mask usable for column-density profiling.
    """
    gray = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    # dark text on light background
    dark = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 6,
    )
    # white text on a colored pill: very bright AND low-saturation pixels
    # (the digit strokes) sitting where the pill is saturated nearby.
    bright = cv2.inRange(val, 225, 255)
    low_sat = cv2.inRange(sat, 0, 60)
    white_glyph = cv2.bitwise_and(bright, low_sat)
    # only keep those white pixels that are INSIDE a colored pill: dilate the
    # saturated-pill mask and intersect.
    pill = cv2.inRange(sat, 60, 255)
    pill_dil = cv2.dilate(pill, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    white_on_pill = cv2.bitwise_and(white_glyph, pill_dil)
    ink = cv2.bitwise_or(dark, white_on_pill)
    # drop the pill outline itself (long thin runs) — keep compact glyphs
    ink = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
    )
    return ink


def _autocorr_pitch(profile: np.ndarray, lo: int, hi: int) -> float | None:
    """First autocorrelation peak of a 1-D density profile within [lo, hi]."""
    x = profile.astype(np.float64)
    if x.std() < 1e-6:
        return None
    x = x - x.mean()
    n = x.size
    pad = 1
    while pad < 2 * n:
        pad <<= 1
    f = np.fft.rfft(x, pad)
    ac = np.fft.irfft(f * np.conj(f), pad)[:n]
    lo = max(2, lo)
    hi = min(n - 1, hi)
    if hi <= lo:
        return None
    sub = ac[lo:hi]
    if sub.size == 0:
        return None
    return float(int(np.argmax(sub)) + lo)


def _calendar_grid_robust(
    bgr: np.ndarray, band: tuple[int, int], x_lo: int, x_hi: int,
    n_days: int,
) -> tuple[list[float], float] | None:
    """Detect the calendar column grid as a regular lattice.

    Polarity-independent glyph mask → column-density profile → dominant
    pitch via autocorrelation (robust to date numbers obscured inside
    colored pills, which break per-component clustering). Phase is fixed by
    cross-correlating a comb of the estimated pitch against the profile.

    Returns (column_centers, pitch) or None if no regular pitch emerges.
    """
    y0, y1 = band
    strip = bgr[y0:y1 + 1, x_lo:x_hi]
    if strip.size == 0 or strip.shape[1] < 60:
        return None
    ink = _calendar_text_mask(strip)
    profile = (ink > 0).sum(axis=0).astype(np.float64)
    if profile.sum() < 20:
        return None
    w = profile.size
    # Plausible date pitch: 12..60 visible cells across the table width.
    pitch = _autocorr_pitch(profile, lo=w // 60, hi=w // 12)
    if pitch is None or pitch < 8:
        return None
    # Fix phase: slide a unit comb (spikes at k*pitch) and maximize overlap
    # with the profile.
    best_phase, best_score = 0.0, -1.0
    step = max(0.5, pitch / 40)
    phases = np.arange(0.0, pitch, step)
    idx = np.arange(w)
    for ph in phases:
        # distance of each column to the nearest comb tooth
        rel = (idx - ph) % pitch
        d = np.minimum(rel, pitch - rel)
        weight = np.maximum(0.0, 1.0 - d / (0.25 * pitch))
        score = float((profile * weight).sum())
        if score > best_score:
            best_score, best_phase = score, float(ph)
    # Build centers across the full width, keep those inside the strip.
    centers = []
    k = 0
    while True:
        c = best_phase + k * pitch
        if c > w:
            break
        if c >= 0:
            centers.append(c + x_lo)
        k += 1
        if k > n_days + 30:
            break
    if len(centers) < 8:
        return None
    return centers[:n_days + 5], float(pitch)


def _columns_are_regular(centers: list[float], pitch: float) -> bool:
    """True when the detected calendar columns are evenly spaced.

    The date numbers are printed at a constant pitch; if most consecutive
    gaps are an integer multiple of the pitch (±25%), the detection is a
    real grid and safe to use directly. A noisy / partial detection (dates
    obscured by colored bands, mis-split digits) fails this and we fall back
    to the chart grid.
    """
    if len(centers) < 8 or pitch <= 0:
        return False
    gaps = np.diff(np.array(sorted(centers), dtype=np.float64))
    ok = 0
    for g in gaps:
        steps = round(g / pitch)
        if steps >= 1 and abs(g - steps * pitch) <= 0.25 * pitch:
            ok += 1
    return ok >= 0.7 * len(gaps)


def _extend_columns(
    centers: list[float], pitch: float, x_hi: int, n_days: int,
) -> list[float]:
    """Tile a regular grid at `pitch` covering the detected calendar columns.

    The detected date numbers may stop short of N_DAYS (later cells empty) or
    have internal gaps (a date hidden behind a band). We anchor on the median
    phase of the detected centers and lay down n_days columns left-to-right,
    starting at the leftmost detected center, so every day gets a bbox.
    """
    if not centers:
        return []
    centers = sorted(centers)
    start = centers[0]
    cols = [start + k * pitch for k in range(n_days)]
    # don't run past the plot's right edge
    return [c for c in cols if c <= x_hi + pitch]


# ── per-row label discovery + canonical-name mapping ───────────────────────
def _approx_match(needle: str, haystack: str, max_dist: int = 2) -> bool:
    """Tiny Levenshtein-like check used for fuzzy row-label matching.

    TrOCR mis-OCRs row labels frequently ("SYMPTONS" for "SYMPTOMS",
    "ODA" for some single-glyph cell). We accept any sliding-window
    substring with up to `max_dist` character differences.
    """
    n, h = len(needle), len(haystack)
    if n == 0 or h < n - max_dist:
        return False
    for start in range(max(0, h - n + 1)):
        end = min(h, start + n + max_dist)
        for ln in range(max(1, n - max_dist), n + max_dist + 1):
            window = haystack[start:start + ln]
            if abs(len(window) - n) > max_dist:
                continue
            d = sum(1 for x, y in zip(needle, window) if x != y) + \
                abs(len(window) - n)
            if d <= max_dist:
                return True
    return False


def _identify_row(label_text: str, idx: int) -> str:
    """Canonical row name from OCR'd label text. Tries keyword substring
    match first, then approximate match (Levenshtein-tolerant for OCR
    misreads like SYMPTONS→Symptoms), then 3-letter month → calendar,
    then positional fallback as last resort."""
    t = (label_text or "").strip().lower().replace(".", "").replace(" ", "")
    if not t:
        return _CANONICAL_ROW_ORDER[idx] if idx < len(_CANONICAL_ROW_ORDER) \
            else f"row{idx}"
    for kw, name in _LABEL_KEYWORDS:
        if kw in t:
            return name
    # Approximate matching for OCR misreads — but only on long keywords.
    # On a 2-char needle like "cm", 1-character tolerance accepts any
    # 2-letter window with 1 matching letter (e.g. "ym" in "symptons"),
    # which would mis-route "SYMPTONS" to CM. Long keywords stay safe.
    for kw, name in _LABEL_KEYWORDS:
        if len(kw) >= 5 and _approx_match(kw, t, max_dist=1):
            return name
    for m in _MONTH_ABBREVS:
        if t.startswith(m):
            return "calendar"
    return _CANONICAL_ROW_ORDER[idx] if idx < len(_CANONICAL_ROW_ORDER) \
        else f"row{idx}"


def _row_label_bbox(
    bgr: np.ndarray, y_top: int, y_bot: int, plot: PlotRegion, ocr,
) -> tuple[tuple[int, int, int, int], str]:
    """Find the row label sitting at the left edge of this row's band.

    Premom puts row labels at figure x ≈ 0.095 (~228 px on a 2400-wide
    canvas). When plot region detection drags plot.x0 down to the image
    edge, those labels end up INSIDE plot.x0, so we widen the search to
    the leftmost 15% of the image regardless of plot.x0.
    """
    H, W = bgr.shape[:2]
    y0 = max(0, y_top - 2)
    y1 = min(H, y_bot + 2)
    x_search_end = max(int(W * 0.15), plot.x0 + int(W * 0.05))
    x_search_end = min(W, x_search_end)
    if x_search_end < 20:
        return (0, y_top, 0, y_bot), ""
    crop = bgr[y0:y1, 0:x_search_end]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    dark = cv2.inRange(gray, 0, _DARK_THRESHOLD)
    n, _, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    if n <= 1:
        return (0, y_top, 0, y_bot), ""
    boxes: list[tuple[int, int, int, int]] = []
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if a < 6 or w < 2 or h < 4 or w > 90 or h > 40:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP]) + y0
        boxes.append((x, x + w, y, y + h))
    if not boxes:
        return (0, y_top, 0, y_bot), ""
    boxes.sort(key=lambda b: b[0])
    cluster = [boxes[0]]
    for b in boxes[1:]:
        if b[0] - cluster[-1][1] <= 10:
            cluster.append(b)
        else:
            break
    x0_c = min(b[0] for b in cluster)
    x1_c = max(b[1] for b in cluster)
    y0_c = min(b[2] for b in cluster)
    y1_c = max(b[3] for b in cluster)
    label_crop = bgr[y0_c:y1_c, x0_c:x1_c]
    text = (ocr.ocr(label_crop) if ocr is not None else "").strip()
    return (x0_c, y0_c, x1_c, y1_c), text


# ── row-type-specific cell extractors ──────────────────────────────────────
def _extract_text(crop: np.ndarray, ocr) -> str | None:
    """OCR a cell that's expected to contain digits or short text."""
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    if (gray < _DARK_THRESHOLD).sum() < 5:
        return None
    text = (ocr.ocr(crop) if ocr is not None else "").strip()
    return text if text else None


def _extract_heart(crop: np.ndarray) -> str | None:
    """Detect a warm-colored (heart) icon in this cell. Returns '♥' or None."""
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h_lo1, h_hi1, s_lo1, v_lo1 = _HSV_HEART[0]
    h_lo2, h_hi2, s_lo2, v_lo2 = _HSV_HEART[1]
    mask = cv2.bitwise_or(
        cv2.inRange(hsv,
                    np.array([h_lo1, s_lo1, v_lo1], dtype=np.uint8),
                    np.array([h_hi1, 255, 255], dtype=np.uint8)),
        cv2.inRange(hsv,
                    np.array([h_lo2, s_lo2, v_lo2], dtype=np.uint8),
                    np.array([h_hi2, 255, 255], dtype=np.uint8)),
    )
    if int(mask.sum() // 255) >= max(6, mask.size // 60):
        return "♥"
    return None


def _extract_purple_circle(crop: np.ndarray) -> str | None:
    """Detect a purple circle (CM indicator). Returns '●' or None."""
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h_lo, h_hi, s_lo, v_lo = _HSV_PURPLE_CIRCLE
    mask = cv2.inRange(hsv,
                       np.array([h_lo, s_lo, v_lo], dtype=np.uint8),
                       np.array([h_hi, 255, 255], dtype=np.uint8))
    if int(mask.sum() // 255) >= max(6, mask.size // 60):
        return "●"
    return None


def _extract_presence(crop: np.ndarray, ocr) -> str | None:
    """Generic: detect any non-background content (Symptoms / hCG).

    Tries OCR first; falls back to a presence marker if pixels are there
    but text isn't readable.
    """
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV) if crop.ndim == 3 else None
    sat = hsv[:, :, 1] if hsv is not None else np.zeros_like(gray)
    interesting = cv2.bitwise_or(
        cv2.inRange(gray, 0, _DARK_THRESHOLD),
        cv2.inRange(sat, 60, 255),
    )
    if (interesting > 0).sum() < 5:
        return None
    text = (ocr.ocr(crop) if ocr is not None else "").strip()
    return text if text else "•"


def _extract_cell(
    bgr: np.ndarray, bbox: tuple[int, int, int, int], row_name: str, ocr,
) -> tuple[str | None, str]:
    """Extract one cell using a row-type-specific strategy. Returns
    (content_string_or_None, content_kind)."""
    x0, y0, x1, y1 = bbox
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None, "text"
    crop = bgr[max(0, y0):y1, max(0, x0):x1]
    if row_name in ("calendar", "CD", "DPO"):
        return _extract_text(crop, ocr), "text"
    if row_name == "Sex":
        return _extract_heart(crop), "heart"
    if row_name == "CM":
        return _extract_purple_circle(crop), "circle"
    return _extract_presence(crop, ocr), "icon"


# ── public entrypoint ──────────────────────────────────────────────────────
def extract_table(
    bgr: np.ndarray, plot: PlotRegion, grid: DayGrid, ocr,
) -> dict:
    H = bgr.shape[0]
    region_top = min(H - 1, plot.y1 + 4)
    if H - region_top < 30:
        return {}
    bands = _detect_nonempty_bands(bgr, region_top, H, plot.x0, plot.x1)
    if not bands:
        return {}
    cal_band = bands[0]
    # Table column grid: PREFER the table's OWN structure (where the app
    # printed the date numbers), NOT where chart markers happen to sit —
    # using the chart day-grid slides cells out of registration whenever the
    # data is sparser/denser than the printed calendar (real-screen-2).
    #
    # Two structure detectors, then a fallback:
    #   1) _calendar_grid_robust — polarity-independent glyph mask +
    #      autocorrelation pitch. Handles dates obscured inside colored
    #      period/fertile PILLS (white-on-color), which defeat the simpler
    #      per-component clustering (real-screen-3/4).
    #   2) _calendar_column_centers — per-component dark-text clustering;
    #      still preferred when it yields a dense, regular grid (it's exact
    #      when dates are plain dark-on-white).
    #   3) chart day grid — last resort, offset-refined to any date numbers.
    chart_pitch = grid.cell_px or (plot.width / N_DAYS)
    cal_cols, cal_pitch = _calendar_column_centers(
        bgr, cal_band, plot.x0, plot.x1,
    )
    col_centers: list[float]
    col_pitch: float
    col_source: str

    cal_is_good = (
        len(cal_cols) >= max(8, N_DAYS // 3)
        and cal_pitch > 6
        and _columns_are_regular(cal_cols, cal_pitch)
    )
    robust = _calendar_grid_robust(bgr, cal_band, plot.x0, plot.x1, N_DAYS)

    if cal_is_good:
        # Plain dark-on-white dates cleanly clustered → use them directly.
        col_centers = _extend_columns(cal_cols, cal_pitch, plot.x1, N_DAYS)
        col_pitch = cal_pitch
        col_source = "calendar"
    elif (robust is not None
          and chart_pitch > 0
          and 0.6 * chart_pitch <= robust[1] <= 1.7 * chart_pitch):
        # Structure detector found a regular lattice whose pitch is
        # consistent with the chart's day pitch (sanity check against
        # latching onto a harmonic). Use the table's own lattice.
        col_centers, col_pitch = robust[0][:N_DAYS], robust[1]
        col_source = "calendar-structure"
    else:
        # Fallback: chart day grid, offset-refined toward any date numbers.
        col_pitch = chart_pitch
        col_centers = list(grid.cells[:N_DAYS])
        if (len(cal_cols) >= max(8, N_DAYS // 3) and col_pitch > 0
                and 0.85 * col_pitch <= cal_pitch <= 1.15 * col_pitch):
            chart_arr = np.array(col_centers, dtype=np.float64)
            cal_arr = np.array(cal_cols, dtype=np.float64)
            step = max(0.5, col_pitch / 60)
            best_shift, best_err = 0.0, float("inf")
            for s in np.arange(-col_pitch / 2, col_pitch / 2 + step, step):
                err = float(np.min(np.abs(cal_arr[:, None] -
                                          (chart_arr + s)[None, :]),
                                   axis=1).sum())
                if err < best_err:
                    best_err = err
                    best_shift = float(s)
            col_centers = [c + best_shift for c in col_centers]
        col_source = "chart-grid"
    cal_h = cal_band[1] - cal_band[0]

    # Row centers from the left-side ROW LABELS — most reliable signal
    # because every row has a label even when its cells are mostly empty.
    label_rows, typical_label_h = _row_centers_from_labels(
        bgr, region_top, H, plot,
    )
    # Fallback: derive centers from content bands if the label search
    # didn't find at least the calendar + one more row.
    if len(label_rows) < 2:
        band_centers = [(b[0] + b[1]) // 2 for b in bands]
        cal_center = (cal_band[0] + cal_band[1]) // 2
        distinct_centers: list[int] = [cal_center]
        min_gap = max(cal_h, 8)
        for c in band_centers[1:]:
            if c - distinct_centers[-1] >= min_gap:
                distinct_centers.append(c)
        # extrapolate downward to up to 7 rows
        if len(distinct_centers) >= 2:
            spacings = [distinct_centers[i + 1] - distinct_centers[i]
                        for i in range(len(distinct_centers) - 1)]
            row_spacing = int(np.median(spacings))
        else:
            row_spacing = int(cal_h * 1.6)
        while len(distinct_centers) < len(_CANONICAL_ROW_ORDER):
            next_y = distinct_centers[-1] + row_spacing
            if next_y >= H:
                break
            distinct_centers.append(next_y)
        center_label_pairs: list[tuple[float, tuple[int, int, int, int] | None]] = [
            (float(c), None) for c in distinct_centers[:len(_CANONICAL_ROW_ORDER)]
        ]
    else:
        center_label_pairs = [(cy, bb) for cy, bb in label_rows[:len(_CANONICAL_ROW_ORDER)]]

    rows: list[TableRow] = []
    by_name: dict[str, TableRow] = {}
    half_cell_h = max(int(max(cal_h, typical_label_h) * 0.85), 12)
    for idx, (center_y, label_bbox_hint) in enumerate(center_label_pairs):
        y_top = max(0, int(center_y - half_cell_h))
        y_bot = min(H, int(center_y + half_cell_h))
        if y_top >= H:
            break
        if label_bbox_hint is not None:
            label_bbox = label_bbox_hint
            label_crop = bgr[label_bbox[1]:label_bbox[3],
                             label_bbox[0]:label_bbox[2]]
            label_text = (ocr.ocr(label_crop) if ocr is not None else "").strip()
        else:
            label_bbox, label_text = _row_label_bbox(bgr, y_top, y_bot, plot, ocr)
        name = _identify_row(label_text, idx)
        half = max(4.0, col_pitch * 0.45)
        cells: list[TableCell] = []
        for cx in col_centers:
            x0 = max(plot.x0, int(round(cx - half)))
            x1 = min(plot.x1, int(round(cx + half)))
            content, kind = _extract_cell(
                bgr, (x0, y_top, x1, y_bot), name, ocr,
            )
            cells.append(TableCell(text=content, bbox=(x0, y_top, x1, y_bot),
                                   kind=kind))
        while len(cells) < N_DAYS:
            cells.append(TableCell(text=None, bbox=(0, y_top, 0, y_bot),
                                   kind="text"))
        # Stop emitting if two consecutive rows are completely empty —
        # we've extrapolated past the actual table.
        any_content = any(c.text is not None for c in cells) or bool(label_text)
        if (not any_content and rows
                and not any(c.text for c in rows[-1].cells)
                and not rows[-1].label_text):
            break
        row = TableRow(
            name=name, label_bbox=label_bbox, label_text=label_text,
            y_top=y_top, y_bottom=y_bot, cells=cells,
        )
        # Deduplicate: if this row's positionally-inferred name collides
        # with a previously-kept row of the same name, prefer the one with
        # ACTUAL CONTENT (more non-null cells). This catches anti-aliased
        # baselines that look like a separate row at the wrong y.
        if name in by_name:
            prev = by_name[name]
            this_pop = sum(1 for c in cells if c.text is not None)
            prev_pop = sum(1 for c in prev.cells if c.text is not None)
            if this_pop > prev_pop:
                rows.remove(prev)
                rows.append(row)
                by_name[name] = row
            continue
        rows.append(row)
        by_name[name] = row
    return {"rows": rows, "by_name": by_name, "col_source": col_source,
            "col_centers": col_centers}
