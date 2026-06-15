"""Multi-axis y-label reader (the professional approach).

The old `axis_ticks._label_rows` clustered text components by Y ONLY, which
glued together the SEPARATE stacked y-axes a fertility chart prints side by
side: on the left "Ratio" (0.1-1.9) and "Level" (5-95) sit in adjacent
columns; on the right the BBT axis sometimes prints °F AND °C in two columns.
Y-only clustering concatenated them ("0.5"+"25" → "0.525"), so a single
"axis" carried two interleaved scales and the fitted mapping was garbage.

This module does what table/figure-digitizing tools (PaddleOCR-Structure,
Textract, img2table) do for axes:

  1. detect text WORD-boxes (merge the digits of ONE number with a small
     horizontal dilation, but NOT across the gap between columns);
  2. cluster the boxes into vertical COLUMNS by x-centre — each column is one
     candidate axis;
  3. OCR every box with the real recognizer (PaddleOCR-rec), parse to a
     numeric value, STITCH ".5"-style fragments onto the integer above;
  4. for each column, robustly fit value = a + b·y;
  5. classify each column by its value range (Ratio / Level / BBT-°F /
     BBT-°C) so the pipeline can pick the right axis per series and read the
     temperature scale off the DATA instead of a leading-digit guess.

Everything here uses only OpenCV + numpy + the existing OCR backend, so it
ports to OpenCV.js / ORT-Web exactly like the rest of the package.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

from .plot_region import PlotRegion

Side = Literal["right", "left"]
AxisKind = Literal["ratio", "level", "bbt_f", "bbt_c", "unknown"]


# Canonical axis ranges (value at top of axis, value at bottom). Premom-family
# layouts; used both to CLASSIFY a detected column and to sanity-check its fit.
AXIS_PROFILES: dict[AxisKind, tuple[float, float]] = {
    "ratio": (1.9, 0.1),
    "level": (95.0, 5.0),
    "bbt_f": (99.5, 95.0),
    "bbt_c": (37.4, 35.6),
}


@dataclass
class LabelBox:
    cx: float
    cy: float
    bbox: tuple[int, int, int, int]   # x0, y0, x1, y1 (working-canvas px)
    text: str = ""
    value: float | None = None


@dataclass
class AxisColumn:
    side: Side
    x_center: float
    boxes: list[LabelBox]
    a: float = 0.0                    # value = a + b * y_pixel
    b: float = 0.0
    rmse: float = 0.0
    kind: AxisKind = "unknown"
    n_fit: int = 0                    # how many boxes contributed to the fit

    def value_at(self, y: float) -> float:
        return self.a + self.b * y


# ── word-box detection ──────────────────────────────────────────────────────
def _word_boxes(bgr: np.ndarray, side: Side, plot: PlotRegion) -> list[LabelBox]:
    """Detect axis-label word boxes in the outer margin on `side`.

    A small HORIZONTAL dilation joins the digits of one number ("9","9",".","5"
    → one box) without bridging the gap to the next column (the columns are
    >~60 px apart, the dilation is ~9 px). Components are filtered to
    digit-text size, and to the chart's vertical extent so the calendar /
    table date numbers BELOW the plot don't get clustered into an axis.
    """
    H, W = bgr.shape[:2]
    band_w = max(120, int(W * 0.20))
    if side == "right":
        x0, x1 = max(0, W - band_w), W
    else:
        x0, x1 = 0, min(W, band_w)
    if x1 - x0 < 30:
        return []
    # Search the generous top region; the per-column AP filter (in
    # detect_axis_columns) discards the calendar/table date boxes that sit
    # below the chart, so we don't need a tight plot.y clip here (the
    # refined plot.y can collapse to just the data band on some screens,
    # which would clip the real axis ticks).
    _ = plot  # kept for signature symmetry / future use
    y0, y1 = 0, min(H, int(H * 0.85))
    sub = bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    # Premom tick labels are pale gray, low saturation. Same gate the old
    # detector used — it works; only the CLUSTERING was wrong.
    dark = cv2.bitwise_and(
        cv2.inRange(gray, 0, 200),
        cv2.inRange(hsv[:, :, 1], 0, 60),
    )
    # Join the glyphs of one number horizontally; keep columns separate.
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    merged = cv2.dilate(dark, k)
    n, _, stats, cents = cv2.connectedComponentsWithStats(merged, connectivity=8)
    boxes: list[LabelBox] = []
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        cw = int(stats[i, cv2.CC_STAT_WIDTH])
        ch = int(stats[i, cv2.CC_STAT_HEIGHT])
        if a < 10 or cw < 4 or ch < 6 or ch > 40 or cw > 130:
            continue
        bx0 = int(stats[i, cv2.CC_STAT_LEFT]) + x0
        by0 = int(stats[i, cv2.CC_STAT_TOP]) + y0
        boxes.append(LabelBox(
            cx=float(cents[i, 0]) + x0,
            cy=float(cents[i, 1]) + y0,
            bbox=(bx0, by0, bx0 + cw, by0 + ch),
        ))
    return boxes


def _cluster_columns(boxes: list[LabelBox], tol: float = 40.0) -> list[list[LabelBox]]:
    """Group boxes into vertical columns by x-centre (single-link, `tol` px)."""
    cols: list[list[LabelBox]] = []
    for b in sorted(boxes, key=lambda z: z.cx):
        placed = False
        for c in cols:
            if abs(b.cx - float(np.mean([z.cx for z in c]))) < tol:
                c.append(b)
                placed = True
                break
        if not placed:
            cols.append([b])
    return cols


# ── value parsing + fragment stitching ──────────────────────────────────────
_NUM_RX = re.compile(r"-?\d*\.?\d+")


def _parse_token(text: str) -> tuple[float | None, bool]:
    """Parse one OCR token. Returns (value, is_fragment).

    A "fragment" is a bare ".5"/".6" tick that needs the integer printed above
    it (PP-OCRv3 reads the half-degree ticks on the BBT axis as ".5").
    """
    t = (text or "").strip().lstrip("><≥≤=~ ")
    if not t:
        return None, False
    mm = _NUM_RX.findall(t.replace(",", "."))
    if not mm:
        return None, False
    body = mm[0]
    if body.startswith("."):
        try:
            return float("0" + body), True
        except ValueError:
            return None, False
    try:
        return float(body), False
    except ValueError:
        return None, False


def _ocr_and_value(col: list[LabelBox], bgr: np.ndarray, ocr) -> None:
    """OCR every box top-to-bottom and parse a numeric value per box.

    We do NOT try to stitch ".5" half-tick fragments here: PP-OCRv3 reads the
    tiny half-degree ticks inconsistently ("5", ".5", ".2", ".8" — several of
    which are outright misreads). Instead `_fit_column` keeps only the
    high-confidence INTEGER ticks (99, 98, 97 … / 37, 36, 35), which OCR
    reliably and which fully determine the linear axis. Half-tick noise is
    discarded by the fit, not trusted.
    """
    col.sort(key=lambda b: b.cy)
    for b in col:
        x0, y0, x1, y1 = b.bbox
        crop = bgr[max(0, y0 - 2):y1 + 2, max(0, x0 - 2):x1 + 2]
        b.text = (ocr.ocr(crop) if (ocr is not None and crop.size) else "") or ""
        v, _ = _parse_token(b.text)
        b.value = v


# ── per-column robust linear fit ────────────────────────────────────────────
def _largest_ap_subset_y(boxes: list[LabelBox]) -> list[LabelBox]:
    """Keep the largest subset of valued boxes whose Y positions form an
    arithmetic progression (the evenly-spaced axis ticks).

    Axis ticks are printed at a constant y-pitch; stray calendar/table date
    boxes that leaked into the column sit at a different (usually larger) y
    far below, breaking the progression. We pick the modal y-gap and keep the
    longest chain consistent with it.
    """
    valued = sorted((b for b in boxes if b.value is not None), key=lambda b: b.cy)
    n = len(valued)
    if n < 3:
        return valued
    ys = [b.cy for b in valued]
    gaps = np.diff(np.array(ys))
    gaps = gaps[gaps > 4]
    if gaps.size == 0:
        return valued
    nb = max(6, min(20, gaps.size))
    hist, edges = np.histogram(gaps, bins=nb)
    peak = int(np.argmax(hist))
    modal = float((edges[peak] + edges[peak + 1]) / 2)
    if modal < 8:
        return valued
    tol = max(6.0, 0.25 * modal)
    best: list[int] = []
    for start in range(n):
        chain = [start]
        for k in range(start + 1, n):
            d = ys[k] - ys[chain[-1]]
            steps = round(d / modal)
            if steps >= 1 and abs(d - steps * modal) <= tol:
                chain.append(k)
        if len(chain) > len(best):
            best = chain
    if len(best) < 3:
        return valued
    return [valued[i] for i in best]


def _fit_column(col: AxisColumn) -> bool:
    """Robust value = a + b·y fit, driven by the reliable INTEGER ticks.

    The column's boxes alternate integer ticks (99, 98, …) with half-degree
    ticks whose OCR is unreliable. Rather than trust every value, we:
      1. take integer-valued boxes as the primary anchors (they read
         reliably and are evenly spaced);
      2. RANSAC a line over them — for each candidate pair, count how many
         integers agree with the implied (a, b) within tolerance;
      3. fit the best inlier set. If too few integers exist, fall back to a
         plain robust fit over all valued boxes.

    Enforces b < 0 (axis values decrease as y grows). Returns True on success.
    """
    valued = [b for b in col.boxes if b.value is not None]
    if len(valued) < 3:
        return False

    def _robust(pts: list[LabelBox]) -> tuple[float, float, float, int] | None:
        if len(pts) < 2:
            return None
        ys = np.array([p.cy for p in pts], dtype=np.float64)
        vs = np.array([p.value for p in pts], dtype=np.float64)
        b1, a0 = np.polyfit(ys, vs, 1)
        resid = vs - (a0 + b1 * ys)
        rms = float(np.sqrt(np.mean(resid * resid)))
        if rms > 1e-6 and len(pts) >= 3:
            keep = np.abs(resid) <= 1.5 * rms
            if keep.sum() >= 2:
                b1, a0 = np.polyfit(ys[keep], vs[keep], 1)
                resid = vs[keep] - (a0 + b1 * ys[keep])
                rms = float(np.sqrt(np.mean(resid * resid)))
                ys = ys[keep]
        return float(a0), float(b1), rms, int(len(ys))

    # RANSAC over ALL valued boxes: for each candidate pair, count how many
    # boxes lie on the implied line. The correct ticks (integers AND the
    # correctly-read extremes like 37.4 / 35.6) are collinear; the half-tick
    # misreads (".2"→0.2, ".8"→0.8) are scattered outliers. We DON'T restrict
    # to integers because some axes (°C: 37.4 … 35.6) have too few integer
    # ticks but several correct decimal reads.
    pts = sorted(valued, key=lambda b: b.cy)
    # tolerance scales with the per-tick value step so it adapts to °C
    # (0.2/tick) vs °F (0.5/tick) vs Level (10/tick).
    best_inliers: list[LabelBox] = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            yi, vi = pts[i].cy, pts[i].value
            yj, vj = pts[j].cy, pts[j].value
            if abs(yi - yj) < 8:
                continue
            b1 = (vi - vj) / (yi - yj)
            a0 = vi - b1 * yi
            if b1 >= 0:
                continue
            # per-tick step implied by this pair (value change over the modal
            # y-gap); tolerance = 60% of one step, min 0.15.
            est_span = abs(a0 + b1 * pts[0].cy - (a0 + b1 * pts[-1].cy))
            tol = max(0.15, 0.30 * est_span / max(1, len(pts) - 1))
            inl = [b for b in pts if abs(b.value - (a0 + b1 * b.cy)) <= tol]
            if len(inl) > len(best_inliers):
                best_inliers = inl
    best = _robust(best_inliers) if len(best_inliers) >= 3 else _robust(valued)
    if best is None:
        return False
    a0, b1, rms, n = best
    if b1 >= 0:
        return False
    col.a, col.b, col.rmse, col.n_fit = a0, b1, rms, n
    return True


def _classify(col: AxisColumn) -> AxisKind:
    """Label the column by which canonical axis its FIT best matches.

    We evaluate the fitted line at the top-most and bottom-most VALUED box y
    (robust to a couple of outliers because the fit already rejected them)
    and compare that predicted span to each profile. Using the fit rather
    than raw min/max keeps stray calendar-date boxes that slipped into the
    column from corrupting the decision.
    """
    valued = [b for b in col.boxes if b.value is not None]
    if len(valued) < 3:
        return "unknown"
    ys = [b.cy for b in valued]
    y_top, y_bot = min(ys), max(ys)
    v_top = col.value_at(y_top)
    v_bot = col.value_at(y_bot)
    obs_hi, obs_lo = max(v_top, v_bot), min(v_top, v_bot)
    best_kind: AxisKind = "unknown"
    best_err = float("inf")
    for kind, (top, bot) in AXIS_PROFILES.items():
        lo, hi = min(top, bot), max(top, bot)
        span = hi - lo
        err = (abs(obs_hi - hi) + abs(obs_lo - lo)) / span
        if err < best_err:
            best_err = err
            best_kind = kind
    return best_kind if best_err < 0.4 else "unknown"


# ── public entrypoint ───────────────────────────────────────────────────────
def detect_axis_columns(
    bgr: np.ndarray, plot: PlotRegion, ocr=None,
    min_boxes: int = 3,
) -> list[AxisColumn]:
    """Detect ALL numeric y-axis columns on both sides of the chart.

    Returns a list of fitted, classified `AxisColumn`s (left + right),
    sorted left-to-right by x_center. Columns that don't yield ≥`min_boxes`
    numeric labels or a monotonic fit are dropped.
    """
    out: list[AxisColumn] = []
    for side in ("left", "right"):  # type: ignore[assignment]
        boxes = _word_boxes(bgr, side, plot)  # type: ignore[arg-type]
        for group in _cluster_columns(boxes):
            if len(group) < min_boxes:
                continue
            col = AxisColumn(
                side=side, x_center=float(np.mean([b.cx for b in group])),
                boxes=group,
            )
            _ocr_and_value(col.boxes, bgr, ocr)
            n_valued = sum(1 for b in col.boxes if b.value is not None)
            if n_valued < min_boxes:
                continue
            # Keep only the evenly-spaced tick subset (drops calendar/table
            # date boxes that leaked into the column).
            col.boxes = _largest_ap_subset_y(col.boxes)
            if sum(1 for b in col.boxes if b.value is not None) < min_boxes:
                continue
            if not _fit_column(col):
                continue
            col.kind = _classify(col)
            out.append(col)
    out.sort(key=lambda c: c.x_center)
    return out


@dataclass
class ResolvedAxes:
    """The axes the pipeline actually consumes, picked from detected columns."""
    bbt: AxisColumn | None          # temperature axis (bbt_f or bbt_c)
    scale_idx: int | None           # 0=celsius, 1=fahrenheit (from bbt.kind)
    scale_confidence: float
    ratio: AxisColumn | None        # orange "Ratio" LH axis (0.1-1.9)
    level: AxisColumn | None        # purple "Level" LH axis (5-95)
    columns: list[AxisColumn] = field(default_factory=list)


def resolve_axes(columns: list[AxisColumn]) -> ResolvedAxes:
    """Pick the BBT / Ratio / Level axes from all detected columns.

    Scale (°C vs °F) is read DIRECTLY from which BBT column was found, not a
    leading-digit guess. When a chart prints BOTH °C and °F columns (some
    layouts do), we prefer the one with the lower fit residual / more ticks.
    """
    def _best(kind: AxisKind) -> AxisColumn | None:
        cands = [c for c in columns if c.kind == kind]
        if not cands:
            return None
        # prefer more fitted ticks, then lower rmse
        return sorted(cands, key=lambda c: (-c.n_fit, c.rmse))[0]

    bbt_f = _best("bbt_f")
    bbt_c = _best("bbt_c")
    # If both units are present pick the better-supported fit; ties favor °F
    # (the more common export). Otherwise take whichever exists.
    bbt: AxisColumn | None
    scale_idx: int | None
    if bbt_f and bbt_c:
        if (bbt_c.n_fit, -bbt_c.rmse) > (bbt_f.n_fit, -bbt_f.rmse):
            bbt, scale_idx = bbt_c, 0
        else:
            bbt, scale_idx = bbt_f, 1
    elif bbt_f:
        bbt, scale_idx = bbt_f, 1
    elif bbt_c:
        bbt, scale_idx = bbt_c, 0
    else:
        bbt, scale_idx = None, None

    scale_conf = 0.0
    if bbt is not None:
        # confidence from fit quality + tick count
        scale_conf = float(min(1.0, 0.5 + 0.1 * bbt.n_fit) *
                           (1.0 if bbt.rmse < 0.3 else 0.7))

    return ResolvedAxes(
        bbt=bbt, scale_idx=scale_idx, scale_confidence=scale_conf,
        ratio=_best("ratio"), level=_best("level"), columns=columns,
    )
