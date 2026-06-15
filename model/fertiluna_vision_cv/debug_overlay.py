"""Visual assessment overlay.

Renders a side-by-side preview of the algorithm's view of a chart:
    * green rectangle      = detected plot region (`PlotRegion`)
    * yellow rectangle     = BBT-axis tick-label ROI (scale classifier)
    * gray dotted verticals = day-cell grid (from `DayGrid`)
    * blue circles + tag   = BBT marker per cell (column-scan result)
    * orange circles + tag = LH-ratio marker per cell
    * text annotation      = scale label + confidence + grid source + counts

The user inspects these overlays during development to spot:
    - off-by-one grid alignment,
    - mis-detected plot box,
    - colour bleed into the wrong series,
    - LH-peak halo eating a neighbour cell,
    - line-only columns falsely flagged as markers (would mean MIN_RUN_PX
      is set too low for this chart's stroke thickness).
"""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from .constants import BBT_SCALES, LH_RANGE, N_DAYS
from .marker_detection import Marker
from .pipeline import ChartResult


_GREEN = (60, 200, 90)
_YELLOW = (40, 220, 240)
_BLUE = (220, 130, 60)
_ORANGE = (60, 140, 240)
_GRAY = (200, 200, 200)
_PINK = (160, 120, 240)
_TEXT_BG = (30, 30, 30)
_TEXT_FG = (240, 240, 240)


def _draw_markers(img: np.ndarray, markers: Iterable[Marker], color):
    for m in markers:
        cv2.circle(img, (int(m.cx), int(m.cy)), 12, color, 2, cv2.LINE_AA)
        cv2.drawMarker(img, (int(m.cx), int(m.cy)), color, cv2.MARKER_CROSS,
                       8, 1, cv2.LINE_AA)


def _text(img: np.ndarray, x: int, y: int, text: str, color=_TEXT_FG):
    cv2.rectangle(img, (x - 4, y - 16), (x + 7 * len(text), y + 4), _TEXT_BG, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def render_overlay(result: ChartResult) -> np.ndarray:
    img = result.debug["work_image"].copy()
    plot = result.debug["plot"]
    grid = result.debug["grid"]

    # plot region
    cv2.rectangle(img, (plot.x0, plot.y0), (plot.x1, plot.y1), _GREEN, 2)
    _text(img, plot.x0 + 4, plot.y0 - 6, f"plot ({plot.method})", _GREEN)

    # tick ROI
    tx0, ty0, tx1, ty1 = result.debug["tick_roi"]
    cv2.rectangle(img, (tx0, ty0), (tx1, ty1), _YELLOW, 1)
    _text(img, tx0, ty0 - 6,
          f"scale={result.scale_label} ({result.scale_confidence:.2f})",
          _YELLOW)

    # day-cell vertical guides — from the LOCKED grid (not from markers)
    for d, cx in enumerate(grid.cells):
        x = int(round(cx))
        if plot.x0 <= x <= plot.x1:
            for y in range(plot.y0, plot.y1, 8):
                cv2.line(img, (x, y), (x, min(plot.y1, y + 4)), _GRAY, 1)
            # tiny cell-index label below plot
            cv2.putText(img, str(d + 1), (x - 6, plot.y1 + 14),
                        cv2.FONT_HERSHEY_PLAIN, 0.7, (140, 140, 140), 1,
                        cv2.LINE_AA)

    # markers (post-packing) — re-add the left_offset to draw at the actual cell
    left = result.debug["left_offset"]
    bbt_raw = list(result.debug["bbt_by_day_raw"].values())
    lh_raw = list(result.debug["lh_by_day_raw"].values())
    _draw_markers(img, bbt_raw, _BLUE)
    _draw_markers(img, lh_raw, _ORANGE)

    # axis columns — one colored box per detected tick, labeled with its
    # parsed value, plus the column's classified kind (ratio/level/bbt_f/
    # bbt_c) at the top. A faint horizontal line at each tick y lets you
    # check the marker y-positions against the inferred axis grid.
    _KIND_COLORS = {
        "ratio": (90, 140, 255),    # orange-ish (BGR)
        "level": (200, 90, 160),    # purple-ish
        "bbt_f": (230, 150, 60),    # blue-ish
        "bbt_c": (230, 150, 60),
        "unknown": (180, 180, 180),
    }
    for col in result.debug.get("axis_columns", []):
        c = _KIND_COLORS.get(col.kind, (180, 180, 180))
        valued = [b for b in col.boxes if b.value is not None]
        for b in valued:
            x0, y0, x1, y1 = b.bbox
            cv2.rectangle(img, (x0 - 1, y0 - 1), (x1 + 1, y1 + 1), c, 1)
            cv2.putText(img, f"{b.value:.1f}",
                        (x1 + 3 if col.side == "right" else max(0, x0 - 40),
                         int((y0 + y1) / 2) + 4),
                        cv2.FONT_HERSHEY_PLAIN, 0.7, c, 1, cv2.LINE_AA)
        if valued:
            ytop = min(b.bbox[1] for b in valued)
            cv2.putText(img, f"{col.kind}",
                        (int(col.x_center) - 20, max(12, ytop - 6)),
                        cv2.FONT_HERSHEY_PLAIN, 0.9, c, 1, cv2.LINE_AA)

    bbt_n = int(result.present[0].sum())
    lh_n = int(result.present[1].sum())
    _text(img, 8, 18,
          f"BBT: {bbt_n}/{N_DAYS}   LH: {lh_n}/{N_DAYS}   "
          f"left_off={left}   grid={grid.source}")
    _text(img, 8, 38,
          f"plot={plot.method}   cell_px={grid.cell_px:.1f}   "
          f"n_cells={len(grid.cells)}")

    # Bottom table: draw each row's label bbox + every cell bbox.
    table = getattr(result, "table", None) or {}
    rows = table.get("rows", [])
    row_color_map = {
        "calendar": (180, 180, 60),
        "CD": (200, 100, 200),
        "DPO": (60, 200, 200),
        "Sex": (60, 60, 220),
        "CM": (200, 200, 60),
        "Symptoms": (180, 120, 60),
        "hCG": (120, 80, 200),
    }
    for row in rows:
        rc = row_color_map.get(row.name, (120, 120, 120))
        # row label box (left of plot)
        lx0, ly0, lx1, ly1 = row.label_bbox
        if lx1 > lx0:
            cv2.rectangle(img, (lx0 - 1, ly0 - 1), (lx1 + 1, ly1 + 1), rc, 1)
            _text(img, max(2, lx0), max(14, ly0 - 4), row.name, rc)
        # per-cell boxes
        for c in row.cells:
            x0, y0, x1, y1 = c.bbox
            if x1 - x0 < 2:
                continue
            cv2.rectangle(img, (x0, y0), (x1, y1), rc, 1)
            if c.text:
                cv2.putText(img, c.text[:4],
                            (x0 + 2, int((y0 + y1) / 2) + 4),
                            cv2.FONT_HERSHEY_PLAIN, 0.7, rc, 1, cv2.LINE_AA)
    return img


def render_value_strip(result: ChartResult) -> np.ndarray:
    """Compact per-day strip visualizing decoded series in REAL UNITS.

    Two rows of N_DAYS cells. Filled = present; cell brightness encodes the
    normalized value (top=1.0 white) but the printed number is the
    DE-NORMALIZED real value — °F or °C for BBT (per the detected scale),
    the LH ratio (0.1-1.9) for LH. The row label shows the scale + range so
    you can sanity-check the axis assignment at a glance.
    """
    bbt_lo, bbt_hi = BBT_SCALES[result.scale_idx][1]
    lh_lo, lh_hi = LH_RANGE
    unit = "°F" if result.scale_idx == 1 else "°C"
    rows = [
        (_BLUE, f"BBT {unit} ({bbt_lo}..{bbt_hi})", (bbt_lo, bbt_hi), "{:.2f}"),
        (_ORANGE, f"LH ratio ({lh_lo}..{lh_hi})", (lh_lo, lh_hi), "{:.2f}"),
    ]
    cell_w = 38
    cell_h = 28
    label_w = 150
    pad = 6
    strip = np.full(
        (2 * cell_h + 3 * pad + 24,
         label_w + N_DAYS * cell_w + 2 * pad, 3),
        255, dtype=np.uint8,
    )
    for s, (color, name, (lo, hi), fmt) in enumerate(rows):
        y0 = pad + 24 + s * (cell_h + pad)
        cv2.putText(strip, name, (4, y0 + cell_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        for d in range(N_DAYS):
            x0 = label_w + d * cell_w
            x1 = x0 + cell_w - 2
            y1 = y0 + cell_h - 2
            if result.present[s, d] > 0.5:
                v = float(result.value[s, d])
                real = lo + v * (hi - lo)
                shade = int(40 + 200 * (1.0 - v))
                cv2.rectangle(strip, (x0, y0), (x1, y1),
                              (shade, shade, shade), -1)
                cv2.rectangle(strip, (x0, y0), (x1, y1), color, 1)
                cv2.putText(strip, fmt.format(real), (x0 + 2, y0 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255),
                            1, cv2.LINE_AA)
            else:
                cv2.rectangle(strip, (x0, y0), (x1, y1), (235, 235, 235), 1)
    for d in range(N_DAYS):
        x0 = label_w + d * cell_w
        cv2.putText(strip, str(d + 1), (x0 + 8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (90, 90, 90), 1, cv2.LINE_AA)
    return strip


def compose_assessment(result: ChartResult) -> np.ndarray:
    overlay = render_overlay(result)
    strip = render_value_strip(result)
    h1, w1 = overlay.shape[:2]
    h2, w2 = strip.shape[:2]
    w = max(w1, w2)
    canvas = np.full((h1 + h2 + 8, w, 3), 255, dtype=np.uint8)
    canvas[:h1, :w1] = overlay
    canvas[h1 + 8:h1 + 8 + h2, :w2] = strip
    return canvas
