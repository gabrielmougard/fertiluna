"""Plot-region (chart bounding-box) detection.

Strategy:
  1. Get the bounding box of all "data ink" — the union of BBT-blue and
     LH-orange pixels (purple Level is ALSO useful for boundary estimation
     even though we don't predict it, so we include it here too). Inflate the
     box by a small margin to recover the plot box that contains it.
  2. Detect horizontal gridlines via morphology — the longest horizontal pale
     strokes in the image. Use their min/max y as top/bottom of plot.
  3. Reconcile (1) and (2): take the gridline y-extent when it's plausible,
     else fall back to the ink bbox y-extent.
  4. As a last resort, use FALLBACK_PLOT_RATIOS.

The detected box is in WORKING-CANVAS pixel coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .color_segmentation import SeriesMasks
from .constants import FALLBACK_PLOT_RATIOS


@dataclass
class PlotRegion:
    x0: int  # left, inclusive
    y0: int  # top, inclusive
    x1: int  # right, inclusive
    y1: int  # bottom, inclusive
    method: str  # "ink+grid" | "ink" | "grid" | "fallback" — for debugging

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def _ink_bbox(masks: SeriesMasks) -> tuple[int, int, int, int] | None:
    """Bounding box of CHART-LINE ink (the BBT/LH/Level curves), excluding:
        - long thin horizontal components (cover lines, axis frames),
        - isolated icon-sized components below the chart (Sex hearts, CM
          dots, Symptoms icons drawn in the bottom table area).

    Principle: chart-LINE ink contains LONG horizontal runs of color
    (markers plus the line between them). Icons in the table area are
    short, isolated blobs. Open the mask with a wide horizontal kernel
    to keep only the line content; the icons collapse to nothing.
    """
    # Use ONLY chart-series ink (BBT blue + LH orange). Purple ink is
    # contaminated by the Level distractor AND by alpha-blended period /
    # fertile band fills that extend INTO the table area on some screens
    # (real-screen-4 has the period band reaching ~y=800). Blue + orange
    # are reliably constrained to the chart region.
    union = cv2.bitwise_or(masks.blue, masks.orange)
    H, W = union.shape
    n, labels, stats, cents = cv2.connectedComponentsWithStats(union, connectivity=8)
    if n <= 1:
        return None

    # Two-stage filtering:
    #   1) Drop long-thin horizontal components (BBT cover-line, frames).
    #   2) Keep only LARGE connected components — the chart line+markers
    #      form a single big blob (often >500 px²); table icons (Sex
    #      hearts, CM dots) are small isolated blobs of ~30-80 px².
    #
    # This works generally regardless of marker spacing, line pitch, or
    # icon shape — large vs small is a stable separator across screens.
    keep = np.zeros_like(union)
    big_areas: list[int] = []
    for i in range(1, n):
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        a = int(stats[i, cv2.CC_STAT_AREA])
        if w > 200 and h < max(6, w // 18):
            continue   # cover line / axis frame
        big_areas.append(a)
    if not big_areas:
        return None
    # Threshold: keep components whose area is at least 25% of the LARGEST
    # surviving component. The chart line dominates by area, so 25% catches
    # the line + any marker-rich large blobs while wiping isolated icons.
    max_area = max(big_areas)
    area_thresh = max(80, int(0.25 * max_area))
    for i in range(1, n):
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        a = int(stats[i, cv2.CC_STAT_AREA])
        if w > 200 and h < max(6, w // 18):
            continue
        if a < area_thresh:
            continue
        keep[labels == i] = 255
    ys, xs = np.where(keep > 0)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _gridline_band(bgr: np.ndarray) -> tuple[int, int] | None:
    """Return (y_top, y_bottom) of the horizontal gridline band, or None.

    Same neutral-saturation gate as the vertical-gridline detector: tinted
    capsule backgrounds in the date row below the chart fall in the same
    brightness window as gridlines and were dragging y_bottom down past the
    actual plot area.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    pale = cv2.inRange(gray, 228, 247)
    neutral = cv2.inRange(sat, 0, 18)
    pale = cv2.bitwise_and(pale, neutral)
    h, w = pale.shape
    kw = max(40, w // 8)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
    horiz = cv2.morphologyEx(pale, cv2.MORPH_OPEN, horiz_kernel)
    row_sums = horiz.sum(axis=1)
    if row_sums.max() == 0:
        return None
    thresh = 0.4 * row_sums.max()
    strong = np.where(row_sums >= thresh)[0]
    if strong.size < 2:
        return None
    return int(strong.min()), int(strong.max())


def detect(bgr: np.ndarray, masks: SeriesMasks) -> PlotRegion:
    H, W = bgr.shape[:2]
    ink = _ink_bbox(masks)
    grid = _gridline_band(bgr)

    if ink is not None and grid is not None:
        ix0, iy0, ix1, iy1 = ink
        gy0, gy1 = grid
        ink_h = max(40, iy1 - iy0)
        # When the gridline detector collapses to a small band (e.g.
        # alpha-blended Level / fertile bands interrupt the pale stripes
        # and only a thin top run survives), gy_height << ink_h. In that
        # case the gridline signal is unreliable, so fall back to the ink
        # y-extent for the plot vertical bounds.
        grid_h = gy1 - gy0
        if grid_h < 0.4 * ink_h:
            gy0, gy1 = iy0, iy1
        else:
            # Cap gridlines to a generous padding around ink so the bottom
            # UI region (date pills, dividers) can't drag y1 past the chart.
            gy0 = max(gy0, iy0 - ink_h)
            gy1 = min(gy1, iy1 + ink_h)
        margin_x = max(2, (ix1 - ix0) // 60)
        margin_y = max(2, (gy1 - gy0) // 60)
        return PlotRegion(
            x0=max(0, ix0 - margin_x),
            y0=max(0, gy0 - margin_y),
            x1=min(W - 1, ix1 + margin_x),
            y1=min(H - 1, gy1 + margin_y),
            method="ink+grid",
        )
    if ink is not None:
        ix0, iy0, ix1, iy1 = ink
        # without gridlines, pad the ink bbox slightly upwards so the topmost
        # marker doesn't sit on the box border (it would skew y_fraction).
        pad = max(3, (iy1 - iy0) // 30)
        return PlotRegion(
            x0=max(0, ix0 - pad),
            y0=max(0, iy0 - pad),
            x1=min(W - 1, ix1 + pad),
            y1=min(H - 1, iy1 + pad),
            method="ink",
        )
    if grid is not None:
        gy0, gy1 = grid
        return PlotRegion(
            x0=int(W * FALLBACK_PLOT_RATIOS[0]),
            y0=gy0,
            x1=int(W * FALLBACK_PLOT_RATIOS[2]),
            y1=gy1,
            method="grid",
        )
    # nothing — fallback ratios
    return PlotRegion(
        x0=int(W * FALLBACK_PLOT_RATIOS[0]),
        y0=int(H * FALLBACK_PLOT_RATIOS[1]),
        x1=int(W * FALLBACK_PLOT_RATIOS[2]),
        y1=int(H * FALLBACK_PLOT_RATIOS[3]),
        method="fallback",
    )
