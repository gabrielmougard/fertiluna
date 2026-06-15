"""BBT-axis scale classification (Celsius vs Fahrenheit) — no OCR engine.

Why this is needed: per-day VALUES are normalized in [0,1] within each series'
axis range. For BBT the range is one of two known options:
    celsius     (35.6, 37.4)
    fahrenheit  (95.0, 99.5)
Downstream code (browser, cycle model) needs to know WHICH to de-normalize.

Two complementary cues are combined:

  A. Template matching against the rightmost tick-label column. We render
     small synthetic digits with multiple cv2 Hershey fonts × multiple sizes
     and keep the best NCC score per digit. Fahrenheit labels include "95",
     "96", "97", "98", "99" — all start with "9". Celsius labels include
     "35", "36", "37" — all start with "3".

  B. Structural cue: digit "9" encloses a single background "hole" (the loop
     at the top); digits "3", "5", "7" do not enclose any. We binarize the
     tick column, find dark connected components (≈ glyphs), and count the
     enclosed background regions per glyph. The DOMINANT hole-count over the
     visible labels discriminates F (mostly 1-hole glyphs from "9X") from C
     (mostly 0-hole glyphs from "3X"/"5X"/"7X").

The final classification picks the cue with higher confidence; if both are
weak we default to fahrenheit (US-market dominant convention).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .constants import BBT_SCALES
from .plot_region import PlotRegion


@dataclass
class AxisCalibration:
    scale_idx: int
    scale_label: str
    scale_confidence: float
    tick_roi: tuple[int, int, int, int]
    detail: dict


# ── template bank ───────────────────────────────────────────────────────────
_FONTS = (
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
)
_DIGITS_F = ("9",)   # fahrenheit labels start with 9
_DIGITS_C = ("3",)   # celsius labels start with 3


def _render_digit(d: str, height_px: int, font: int) -> np.ndarray:
    target_h = max(8, int(height_px))
    scale = target_h / 22.0
    thick = max(1, int(round(scale * 1.6)))
    (w, h), _ = cv2.getTextSize(d, font, scale, thick)
    img = np.zeros((h + 6, w + 6), dtype=np.uint8)
    cv2.putText(img, d, (3, h + 2), font, scale, 255, thick, cv2.LINE_AA)
    return img


def _tick_column_roi(bgr: np.ndarray, plot: PlotRegion) -> tuple[int, int, int, int]:
    """Narrow ROI containing the right-edge BBT axis labels.

    Tick labels live in the rightmost ~50-70 px of the chart (just inside
    plot.x1 when plot.x1 is at the image edge, just outside otherwise).
    Keep the strip narrow so chart data ink (markers, line strokes) inside
    plot.x1 doesn't pollute the hole-count and template features.
    """
    H, W = bgr.shape[:2]
    strip_w = 70
    # Place the strip centered on plot.x1, so labels fall inside whether
    # plot.x1 is inside the image or right at the edge.
    cx = min(W - strip_w // 2 - 1, plot.x1)
    x0 = max(0, cx - strip_w // 2)
    x1 = min(W, cx + strip_w // 2)
    y0 = max(0, plot.y0 - 4)
    y1 = min(H, plot.y1 + 4)
    return x0, y0, x1, y1


def _best_template_score(strip: np.ndarray, digit: str) -> float:
    """Best NCC score across (font × size) variations."""
    if strip.size == 0:
        return 0.0
    strip_h = strip.shape[0]
    sizes = [max(10, strip_h // h) for h in (40, 32, 26, 22, 18, 14)]
    best = 0.0
    for size in sizes:
        for font in _FONTS:
            glyph = _render_digit(digit, size, font)
            if (glyph.shape[0] >= strip.shape[0] or
                    glyph.shape[1] >= strip.shape[1]):
                continue
            res = cv2.matchTemplate(strip, glyph, cv2.TM_CCOEFF_NORMED)
            if res.size > 0:
                best = max(best, float(res.max()))
    return best


def _hole_count_score(text: np.ndarray) -> tuple[float, dict]:
    """Heuristic: fraction of dark glyphs that enclose a background hole.

    Returns (score_in_[-1,1], detail).
      * +1.0 → strong "Fahrenheit": all glyphs have ≥1 hole (e.g., "9", "0")
      * -1.0 → strong "Celsius":   no glyphs have holes (e.g., "3", "5", "7")
      *  0.0 → ambiguous / no glyphs found.

    Method: connected components on the dark glyph mask gives one component
    per glyph. For each, examine the background pixels INSIDE its bounding
    box: if a background blob there is fully enclosed (i.e., not touching
    the bbox border), it's an interior hole.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(text, connectivity=8)
    if n <= 1:
        return 0.0, {"glyphs": 0}
    glyphs_with_holes = 0
    glyphs_considered = 0
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 20 or w < 4 or h < 6 or w > text.shape[1] // 2:
            continue  # skip noise / band fragments
        glyphs_considered += 1
        # crop the component's bbox, mask to this component, look at interior bg
        comp = (labels[y:y + h, x:x + w] == i).astype(np.uint8) * 255
        # invert: interior background = 255
        bg = cv2.bitwise_not(comp)
        # flood fill from a corner to mark "outside" background
        ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.floodFill(bg.copy(), ff_mask, (0, 0), 128)
        # any remaining 255 in bg-after-flood = enclosed hole
        bg2 = bg.copy()
        cv2.floodFill(bg2, np.zeros((h + 2, w + 2), dtype=np.uint8),
                      (0, 0), 128)
        # but our copy was modified by floodFill — re-check on a fresh copy:
        bg3 = bg.copy()
        cv2.floodFill(bg3, np.zeros((h + 2, w + 2), dtype=np.uint8),
                      (0, 0), 0)  # paint outside black
        # enclosed: white pixels still surviving after killing the outside
        enclosed = (bg3 > 0).sum()
        if enclosed > 4:
            glyphs_with_holes += 1
    if glyphs_considered == 0:
        return 0.0, {"glyphs": 0}
    frac = glyphs_with_holes / glyphs_considered
    # score: +1 if all have holes (F), -1 if none (C). Linear in between.
    return (frac * 2.0 - 1.0,
            {"glyphs": glyphs_considered, "with_holes": glyphs_with_holes})


def classify(bgr: np.ndarray, plot: PlotRegion) -> AxisCalibration:
    x0, y0, x1, y1 = _tick_column_roi(bgr, plot)
    col = bgr[y0:y1, x0:x1]
    detail: dict = {"tick_roi": (x0, y0, x1, y1)}
    if col.size == 0:
        return AxisCalibration(1, BBT_SCALES[1][0], 0.0, (x0, y0, x1, y1), detail)
    gray = cv2.cvtColor(col, cv2.COLOR_BGR2GRAY)
    _, text = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Cue A — template matching
    s9 = _best_template_score(text, "9")
    s3 = _best_template_score(text, "3")
    margin_t = s9 - s3
    detail["template"] = {"9": s9, "3": s3, "margin": margin_t}

    # Cue B — structural hole count
    hole_score, hole_detail = _hole_count_score(text)
    detail["holes"] = hole_detail | {"score": hole_score}

    # Combine: template margin is in [-1,1] (scaled), hole_score is in [-1,1].
    # Weight templates a bit higher when both glyphs score reasonably (≥0.3).
    template_conf = min(max(s9, s3), 1.0)
    if template_conf >= 0.30:
        combined = 0.6 * margin_t * 3.0 + 0.4 * hole_score
    else:
        combined = hole_score  # template signal too weak, trust structural cue
    combined = float(np.clip(combined, -1.0, 1.0))

    if abs(combined) < 0.08:
        # Truly ambiguous → default to fahrenheit, low confidence.
        idx = 1
        conf = 0.0
    else:
        idx = 1 if combined > 0 else 0
        conf = float(abs(combined))
    detail["combined"] = combined
    return AxisCalibration(
        scale_idx=idx,
        scale_label=BBT_SCALES[idx][0],
        scale_confidence=conf,
        tick_roi=(x0, y0, x1, y1),
        detail=detail,
    )
