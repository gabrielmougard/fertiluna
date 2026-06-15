"""Axis tick-label detection + y→value linear fit.

Replaces the brittle "plot.y0/plot.y1 + BBT_SCALES range" mapping with one
anchored on the actual tick label positions printed by the app.

Pipeline:

  1. Detect dark-text rows in a vertical strip near the plot edge (right
     of plot.x1 for BBT, left of plot.x0 for LH). The strip OVERLAPS the
     plot edge because plot region detection routinely puts plot.x1 past
     the axis labels.
  2. Group connected components by y into label rows.
  3. Find the largest subset of rows whose y values form an arithmetic
     progression — that's the evenly-spaced tick column. Non-label dark
     text (LH-peak badge, app chrome) doesn't satisfy this property and is
     dropped.
  4. Assume the topmost row sits at the axis MAX and the bottommost at the
     axis MIN — this holds for all Premom-family layouts because their
     extreme labels carry "≤"/"≥" prefixes meaning the axis ends there.
  5. Linear fit value = a + b * y_pixel across the resolved anchors.

OCR is intentionally NOT done in this version: rendering cv2.putText
templates against real-app sans-serif fonts is unreliable enough that the
position-only signal beats it on accuracy. The TickLabel.text field is
left empty; a future pass could OCR each row to detect partial scales
(e.g. a chart cropped to 96-99°F instead of 95-99.5°F).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

from .plot_region import PlotRegion


Side = Literal["right", "left"]


@dataclass
class TickLabel:
    y: float                                # center y in working canvas
    bbox: tuple[int, int, int, int]         # (x0, y0, x1, y1) in working canvas
    text: str                               # OCR'd text after digit-by-digit matching
    value: float | None = None              # interpreted numeric value


@dataclass
class AxisMapping:
    side: Side
    a: float                                # value = a + b * y_pixel
    b: float
    anchors: list[TickLabel] = field(default_factory=list)
    rmse: float = 0.0
    source: str = "ocr"                     # "ocr" | "count-based" | "none"

    def value_at(self, y: float) -> float:
        return self.a + self.b * y

    def y_at(self, value: float) -> float:
        if abs(self.b) < 1e-9:
            return 0.0
        return (value - self.a) / self.b


# ── digit template bank ────────────────────────────────────────────────────
_DIGITS = "0123456789."
_FONTS = (
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
    cv2.FONT_HERSHEY_COMPLEX_SMALL,
)
_TEMPLATE_CACHE: dict[int, list[tuple[str, np.ndarray]]] = {}


def _render(d: str, scale: float, font: int) -> np.ndarray:
    thick = max(1, int(round(scale * 1.6)))
    (w, h), _ = cv2.getTextSize(d, font, scale, thick)
    img = np.zeros((h + 4, w + 4), dtype=np.uint8)
    cv2.putText(img, d, (2, h + 1), font, scale, 255, thick, cv2.LINE_AA)
    return img


def _templates_near(target_h: int) -> list[tuple[str, np.ndarray]]:
    key = max(8, target_h)
    cached = _TEMPLATE_CACHE.get(key)
    if cached is not None:
        return cached
    out: list[tuple[str, np.ndarray]] = []
    # Try scales spanning the target height since real fonts and Hershey
    # render at different per-scale heights.
    for ref_h in (28, 24, 22, 20, 18, 16):
        s = key / ref_h
        if not 0.2 <= s <= 3.0:
            continue
        for d in _DIGITS:
            for f in _FONTS:
                g = _render(d, s, f)
                if 6 < g.shape[0] < 80 and 4 < g.shape[1] < 80:
                    out.append((d, g))
    _TEMPLATE_CACHE[key] = out
    return out


def _ocr_glyph(glyph: np.ndarray) -> tuple[str, float]:
    """Best-match digit / '.' for a single glyph image (binary, white=ink)."""
    h, w = glyph.shape
    if h < 4 or w < 2:
        return "?", 0.0
    # Period detection — small, round, near the bottom — special-case it
    # because Hershey periods are too tiny to match reliably.
    if h <= 6 and w <= 6:
        return ".", 0.8
    templates = _templates_near(h)
    best_score, best_char = -1.0, "?"
    for d, t in templates:
        if t.shape[0] > glyph.shape[0] + 4 or t.shape[1] > glyph.shape[1] + 4:
            continue
        # match smaller dimension to fit inside glyph
        if t.shape[0] > h:
            t = cv2.resize(t, (max(1, int(t.shape[1] * h / t.shape[0])), h))
        if t.shape[1] > w:
            t = cv2.resize(t, (w, max(1, int(t.shape[0] * w / t.shape[1]))))
        if t.shape[0] > h or t.shape[1] > w:
            continue
        res = cv2.matchTemplate(glyph, t, cv2.TM_CCOEFF_NORMED)
        s = float(res.max()) if res.size > 0 else -1.0
        if s > best_score:
            best_score = s
            best_char = d
    return best_char, best_score


def _ocr_label_image(label_img: np.ndarray) -> str:
    """Segment label into left-to-right glyphs, OCR each, concatenate."""
    if label_img.size == 0:
        return ""
    gray = (label_img if label_img.ndim == 2
            else cv2.cvtColor(label_img, cv2.COLOR_BGR2GRAY))
    _, dark = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, _, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    glyphs: list[tuple[int, int, int, int]] = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 4:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if w > 40 or h > 40:
            continue
        glyphs.append((x, y, w, h))
    glyphs.sort(key=lambda g: g[0])
    chars: list[str] = []
    for x, y, w, h in glyphs:
        glyph = dark[y:y + h, x:x + w]
        pad = 2
        g = cv2.copyMakeBorder(glyph, pad, pad, pad, pad,
                               cv2.BORDER_CONSTANT, value=0)
        ch, score = _ocr_glyph(g)
        if score > 0.20:
            chars.append(ch)
    return "".join(chars)


# ── label row detection ────────────────────────────────────────────────────
def _label_rows(
    bgr: np.ndarray, plot: PlotRegion, side: Side,
) -> list[tuple[float, int, int, int, int]]:
    """Detect tick label bounding boxes in the strip near the plot edge.

    The search window OVERLAPS the plot edge in both directions: plot region
    detection routinely puts plot.x1 at the BBT-axis label x (because the
    cover-line/markers stretch the ink bbox), so the labels themselves end
    up INSIDE plot.x1. Markers are colored (gray > 160) and don't survive
    the dark-text threshold so the overlap doesn't pollute the search.
    """
    H, W = bgr.shape[:2]
    # Search the OUTER image margin (rightmost or leftmost 20%) for tick
    # labels, IGNORING plot.x1/x0. Plot region detection routinely puts
    # the box's right edge well inside the actual axis label column —
    # screen-3 has plot.x1 at 78% of image width while the BBT axis labels
    # live at 95%, so a search anchored on plot.x1 misses them entirely.
    # Searching the image edge is adaptive and works for every layout.
    band_w = max(120, int(W * 0.20))
    if side == "right":
        x0 = max(0, W - band_w)
        x1 = W
    else:
        x0 = 0
        x1 = min(W, band_w)
    if x1 - x0 < 30:
        return []
    # Vertical search range: use a GENEROUS window — the coarse plot region
    # can be off by hundreds of pixels (gridline morphology collapses when
    # alpha-blended bands break the pale stripes), so anchoring tick search
    # on it would miss the real labels. Search the whole top half of the
    # image where chart axes always live, then let the AP filter discard
    # non-progression rows.
    y0 = 0
    y1 = min(H, int(H * 0.80))
    col = bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(col, cv2.COLOR_BGR2GRAY)
    # Premom-family apps render axis tick labels in PALE GRAY (RGB ≈ 150-180),
    # NOT black. A strict dark threshold (≤160) misses them entirely on
    # screen-2/3/4 — only the bolder UI labels survive. Use 200 + a low-sat
    # gate so we don't accidentally pull in colored band-fill edges.
    hsv = cv2.cvtColor(col, cv2.COLOR_BGR2HSV)
    dark = cv2.bitwise_and(
        cv2.inRange(gray, 0, 200),
        cv2.inRange(hsv[:, :, 1], 0, 60),
    )
    n, _, stats, cents = cv2.connectedComponentsWithStats(dark, connectivity=8)
    cmps: list[tuple[float, float, int, int, int, int, int, int]] = []
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < 6 or a > 800:
            continue
        cw = int(stats[i, cv2.CC_STAT_WIDTH])
        ch = int(stats[i, cv2.CC_STAT_HEIGHT])
        if cw < 2 or ch < 4 or cw > 36 or ch > 36:
            continue
        cmps.append((
            float(cents[i, 0]) + x0,
            float(cents[i, 1]) + y0,
            int(stats[i, cv2.CC_STAT_LEFT]) + x0,
            int(stats[i, cv2.CC_STAT_TOP]) + y0,
            int(stats[i, cv2.CC_STAT_LEFT]) + cw + x0,
            int(stats[i, cv2.CC_STAT_TOP]) + ch + y0,
            cw, ch,
        ))
    if len(cmps) < 2:
        return []
    cmps.sort(key=lambda c: c[1])
    typical_h = int(np.median([c[7] for c in cmps]))
    merge_thr = max(5, int(typical_h * 0.9))
    clusters: list[list[tuple]] = [[cmps[0]]]
    for c in cmps[1:]:
        if abs(c[1] - clusters[-1][-1][1]) <= merge_thr:
            clusters[-1].append(c)
        else:
            clusters.append([c])
    rows: list[tuple[float, int, int, int, int]] = []
    for cl in clusters:
        x_min = min(c[2] for c in cl)
        y_min = min(c[3] for c in cl)
        x_max = max(c[4] for c in cl)
        y_max = max(c[5] for c in cl)
        # tighter y-center: mean of component y-centroids in the cluster
        y_center = float(np.mean([c[1] for c in cl]))
        rows.append((y_center, x_min, y_min, x_max, y_max))
    return rows


# ── value interpretation ───────────────────────────────────────────────────
_DIGIT_RX = re.compile(r"[\d.]+")


def _parse_text(text: str) -> tuple[float | None, bool]:
    """Parse a raw OCR string. Returns (value_or_None, is_fragment).

    `is_fragment` is True for strings like ".5", ".6" that need a neighbor's
    integer part to resolve to a value.
    """
    cleaned = _DIGIT_RX.findall(text)
    if not cleaned:
        return None, False
    body = "".join(cleaned)
    if not body or body == ".":
        return None, False
    # ".5" / ".X" → fragment: needs the integer from the row below it.
    if body.startswith(".") and body.count(".") == 1:
        try:
            return float("0" + body), True
        except ValueError:
            return None, False
    # otherwise try direct float parse
    try:
        return float(body), False
    except ValueError:
        return None, False


def _interpret_bbt(labels: list[TickLabel]) -> list[TickLabel]:
    """Resolve BBT-axis labels. Top-most label = highest value; bottommost
    = lowest. Walk bottom-up keeping track of the latest integer label so
    ".5" fragments above it can be resolved by adding 0.5.
    """
    sorted_lbl = sorted(labels, key=lambda l: -l.y)   # bottommost first
    last_integer: int | None = None
    for lbl in sorted_lbl:
        v, is_frag = _parse_text(lbl.text)
        if v is None:
            lbl.value = None
            continue
        if is_frag:
            if last_integer is not None:
                lbl.value = float(last_integer) + v   # e.g. 96 + 0.5 = 96.5
            else:
                lbl.value = None
        else:
            lbl.value = v
            if abs(v - round(v)) < 1e-6:
                last_integer = int(round(v))
    return labels


def _interpret_lh(labels: list[TickLabel]) -> list[TickLabel]:
    """Resolve LH-axis labels. Premom prints them as full floats "0.1".."1.9"."""
    for lbl in labels:
        v, _ = _parse_text(lbl.text)
        lbl.value = v
    return labels


# ── linear fit ─────────────────────────────────────────────────────────────
def _fit(labels: list[TickLabel], side: Side) -> AxisMapping | None:
    pts = [(l.y, l.value) for l in labels if l.value is not None]
    if len(pts) < 2:
        return None
    ys = np.array([p[0] for p in pts], dtype=np.float64)
    vs = np.array([p[1] for p in pts], dtype=np.float64)
    # Simple OLS with one outlier-rejection pass: keep points within 1.5 RMS.
    b, a = np.polyfit(ys, vs, 1)
    resid = vs - (a + b * ys)
    rms = float(np.sqrt(np.mean(resid * resid)))
    if rms > 0.05:
        keep = np.abs(resid) <= 1.5 * rms
        if keep.sum() >= 2:
            b, a = np.polyfit(ys[keep], vs[keep], 1)
            resid = vs[keep] - (a + b * ys[keep])
            rms = float(np.sqrt(np.mean(resid * resid)))
    return AxisMapping(
        side=side, a=float(a), b=float(b),
        anchors=[l for l in labels if l.value is not None],
        rmse=rms, source="ocr",
    )


def _count_based_fallback(
    rows: list[tuple[float, int, int, int, int]],
    side: Side,
    scale_idx: int | None,
) -> AxisMapping | None:
    """If OCR is unreliable, assume the detected rows correspond to the
    expected evenly-spaced tick positions for the given scale.

    For BBT: 10 ticks at 95.0..99.5 every 0.5°F (Fahrenheit) or 95.0..99.5
    if scale_idx=1; 35.6..37.4 every 0.2°C if scale_idx=0.
    For LH:  10 ticks at 0.1..1.9 every 0.2.
    """
    if len(rows) < 2:
        return None
    ys = sorted(r[0] for r in rows)
    n = len(ys)
    if side == "right":
        if scale_idx == 1:
            v_top, v_bot, step = 99.5, 95.0, 0.5
        else:
            v_top, v_bot, step = 37.4, 35.6, 0.2
        n_expected = int(round((v_top - v_bot) / step)) + 1
    else:
        v_top, v_bot, step = 1.9, 0.1, 0.2
        n_expected = int(round((v_top - v_bot) / step)) + 1
    # Even if n != n_expected, assume the FIRST detected row is at v_bot and
    # the LAST is at v_top — linearly interpolate the rest. This works as
    # long as the top and bottom rows are correctly detected.
    a = (v_bot * ys[-1] - v_top * ys[0]) / (ys[-1] - ys[0])
    b = (v_top - v_bot) / (ys[0] - ys[-1])
    return AxisMapping(
        side=side, a=float(a), b=float(b),
        anchors=[], rmse=0.0, source="count-based",
    )


# ── arithmetic-progression filter ──────────────────────────────────────────
def _largest_ap_subset(ys: list[float]) -> tuple[list[int], float]:
    """Find the largest subset of `ys` forming an arithmetic progression.

    Strategy: pick the modal delta first (the dominant inter-row spacing in
    the strip), THEN walk through every starting point with that fixed
    delta and keep the longest chain. Using a fixed modal delta avoids the
    pair-based RANSAC pitfall of latching onto a tiny spurious delta.

    Returns (indices_into_ys_sorted_ascending, modal_delta).
    """
    n = len(ys)
    if n < 3:
        return list(range(n)), (float(ys[-1] - ys[0]) if n == 2 else 0.0)
    sorted_pairs = sorted(enumerate(ys), key=lambda p: p[1])
    sorted_idx = [p[0] for p in sorted_pairs]
    sorted_ys = [p[1] for p in sorted_pairs]
    deltas = np.diff(np.array(sorted_ys))
    deltas = deltas[deltas > 5]
    if deltas.size == 0:
        return list(range(n)), 0.0
    n_bins = max(6, min(24, deltas.size))
    bins, edges = np.histogram(deltas, bins=n_bins)
    peak = int(np.argmax(bins))
    modal = float((edges[peak] + edges[peak + 1]) / 2)
    if modal < 10:
        return list(range(n)), modal
    # Tighter tolerance (10% of modal) than v1's 20% — y-axis ticks render
    # with sub-pixel precision so a 10% drift is already wider than real
    # variance, but tight enough to reject the first table-row label below
    # the last axis tick (which sits at ~70% of one full tick spacing).
    tol = 0.10 * modal
    best: list[int] = []
    for start in range(len(sorted_ys)):
        chain = [start]
        for k in range(start + 1, len(sorted_ys)):
            d = sorted_ys[k] - sorted_ys[chain[-1]]
            n_steps = int(round(d / modal))
            if n_steps >= 1 and abs(d - n_steps * modal) <= tol:
                chain.append(k)
        if len(chain) > len(best):
            best = chain
    if not best:
        return list(range(n)), modal
    return [sorted_idx[i] for i in best], modal


_NUMERIC_CHARS = set("0123456789.")


def _is_numeric_row(bgr: np.ndarray, bbox: tuple[int, int, int, int],
                    ocr) -> tuple[bool, str]:
    """Decide whether the label image is a NUMBER (y-axis tick) vs TEXT
    (table row label like CD/DPO/Sex/CM/Symptoms/hCG/Mar).

    Uses the active OCR backend. A label counts as numeric when its
    OCR'd text contains at least one digit AND at most one non-numeric
    character (allowing the ≤/≥ prefix which often OCRs as garbage).
    """
    x0, y0, x1, y1 = bbox
    crop = bgr[max(0, y0 - 1):y1 + 1, max(0, x0 - 1):x1 + 1]
    if crop.size == 0:
        return False, ""
    text = ocr.ocr(crop) if ocr is not None else ""
    if not text:
        # Fallback: width/height heuristic. A row label like "Symptoms" is
        # much wider than a number like "95". We don't really know without
        # OCR, so default to ACCEPTING — the AP filter will weed out
        # non-progression rows anyway.
        return True, ""
    cleaned = text.strip()
    digits = sum(1 for c in cleaned if c in _NUMERIC_CHARS)
    others = sum(1 for c in cleaned if c.isalpha())
    if digits == 0:
        return False, cleaned
    # Allow ≤/≥ prefix + at most one stray letter (OCR noise on the prefix).
    return others <= 1, cleaned


def _detect_anchors(
    bgr: np.ndarray, plot: PlotRegion, side: Side, ocr=None,
) -> list[TickLabel]:
    """Return label rows in arithmetic progression, top-to-bottom.

    The AP filter alone is a STRONG structural signal — axis tick labels
    are evenly spaced, UI chrome and other text usually isn't. We trust it
    to do the filtering and skip an OCR-based numeric pre-filter (Paddle
    returns empty text on tiny 9×14 px tick crops, which would drop most
    real labels). OCR is still run after AP, on each surviving row, to
    populate `text` for scale-classification downstream.
    """
    rows = _label_rows(bgr, plot, side)
    if len(rows) < 2:
        return []
    # Soft pre-filter: OCR each row's crop; DROP rows whose text is
    # clearly alphabetic (≥2 letters AND no digits). Keep empty-OCR rows
    # because tiny axis-tick crops often OCR to "" — those are still
    # candidates and the AP filter will decide. This keeps UI chrome
    # ("Ratio", "CD", "Einstellung") out of the candidate set without
    # dropping real but-too-small tick labels.
    candidates: list[tuple[float, int, int, int, int, str]] = []
    for r in rows:
        text = ""
        if ocr is not None:
            x0, y0, x1, y1 = r[1], r[2], r[3], r[4]
            crop = bgr[max(0, y0 - 1):y1 + 1, max(0, x0 - 1):x1 + 1]
            if crop.size > 0:
                text = (ocr.ocr(crop) or "").strip()
        # clearly alphabetic = ≥2 letters and no digits at all
        digits = sum(1 for c in text if c.isdigit())
        letters = sum(1 for c in text if c.isalpha())
        if digits == 0 and letters >= 2:
            continue
        candidates.append((r[0], r[1], r[2], r[3], r[4], text))
    if len(candidates) < 2:
        return []
    rows_sorted = sorted(candidates, key=lambda r: r[0])
    ys = [r[0] for r in rows_sorted]
    kept_idx, _ = _largest_ap_subset(ys)
    out: list[TickLabel] = []
    for i in kept_idx:
        r = rows_sorted[i]
        out.append(TickLabel(y=r[0],
                             bbox=(r[1], r[2], r[3], r[4]),
                             text=r[5]))
    return out


def _mapping_from_extremes(
    anchors: list[TickLabel], side: Side, scale_idx: int | None,
) -> AxisMapping | None:
    """Assume topmost anchor = scale max, bottommost = scale min."""
    if len(anchors) < 2:
        return None
    # anchors come pre-sorted ascending y (top → bottom in screen coords)
    y_top = anchors[0].y
    y_bot = anchors[-1].y
    if abs(y_bot - y_top) < 4:
        return None
    if side == "right":
        if scale_idx == 1:
            v_top, v_bot = 99.5, 95.0
        elif scale_idx == 0:
            v_top, v_bot = 37.4, 35.6
        else:
            v_top, v_bot = 99.5, 95.0
    else:
        v_top, v_bot = 1.9, 0.1
    b = (v_top - v_bot) / (y_top - y_bot)
    a = v_top - b * y_top
    # tag anchors with their inferred values (evenly-spaced between extremes)
    n = len(anchors)
    if n >= 2:
        for k, lbl in enumerate(anchors):
            lbl.value = v_top + (v_bot - v_top) * (k / (n - 1))
    return AxisMapping(
        side=side, a=float(a), b=float(b),
        anchors=anchors, rmse=0.0, source="positions",
    )


# ── public entrypoint ──────────────────────────────────────────────────────
def detect_axis_mappings(
    bgr: np.ndarray, plot: PlotRegion, scale_idx: int | None = None,
    ocr=None,
) -> tuple[AxisMapping | None, AxisMapping | None, list[TickLabel], list[TickLabel]]:
    """Detect (right=BBT, left=LH) axis mappings.

    Each may be None if the label strip has too few or non-progressive rows.
    `ocr`: an OCRBackend used to verify each candidate label is numeric.
    """
    # Local import avoids a circular import at module load (guardrails imports
    # from this module).
    from .guardrails import mapping_from_labels, repair_axis_labels

    out: dict[Side, AxisMapping | None] = {"right": None, "left": None}
    labels: dict[Side, list[TickLabel]] = {"right": [], "left": []}
    for side in ("right", "left"):  # type: ignore[assignment]
        anchors = _detect_anchors(bgr, plot, side, ocr=ocr)  # type: ignore[arg-type]
        # GUARDRAIL: repair the OCR'd labels by snapping onto the axis's
        # arithmetic grid (fills blanks, fixes ".5"-style fragments), then
        # fit the mapping through the repaired set. Fall back to the
        # endpoint-only mapping if too few labels survived to repair.
        repaired, _rms = repair_axis_labels(anchors, side)  # type: ignore[arg-type]
        labels[side] = repaired  # type: ignore[index]
        mapping = mapping_from_labels(repaired, side)  # type: ignore[arg-type]
        if mapping is None:
            mapping = _mapping_from_extremes(anchors, side, scale_idx)  # type: ignore[arg-type]
        out[side] = mapping
    return out["right"], out["left"], labels["right"], labels["left"]
