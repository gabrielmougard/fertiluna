"""HSV color segmentation for chart-ink series.

Three masks are produced from a working-canvas BGR image:
    blue   — BBT line + markers   (always the `temp` series)
    orange — LH "Ratio" line + markers
    purple — LH "Level" line + markers

Both orange and purple are CANDIDATE LH lines: which one carries the
per-day LH data depends on the chart variant, and is chosen at runtime by
`marker_detection.resolve_lh_color` (purple is NOT a fixed distractor — on
dense-Level charts it IS the real LH measurement).

Saturation/value thresholds in constants.py are tuned to reject:
    - pale violet fertile band
    - pale pink period band
    - light gridlines, dark text, white markerfill
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .constants import HSV_BLUE, HSV_ORANGE, HSV_PURPLE


@dataclass
class SeriesMasks:
    blue: np.ndarray     # uint8 0/255 — BBT ink
    orange: np.ndarray   # uint8 0/255 — LH "Ratio" ink (candidate LH line)
    purple: np.ndarray   # uint8 0/255 — LH "Level" ink (candidate LH line)


def _band_mask(hsv: np.ndarray, band: tuple[int, int, int, int]) -> np.ndarray:
    h_lo, h_hi, s_lo, v_lo = band
    lo = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
    hi = np.array([h_hi, 255, 255], dtype=np.uint8)
    return cv2.inRange(hsv, lo, hi)


def segment(bgr: np.ndarray) -> SeriesMasks:
    """Return per-series binary masks in working-canvas coords."""
    # Slight pre-blur to bridge anti-aliased line edges into the H band.
    blurred = cv2.GaussianBlur(bgr, (3, 3), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    blue = _band_mask(hsv, HSV_BLUE)
    orange = _band_mask(hsv, HSV_ORANGE)
    purple = _band_mask(hsv, HSV_PURPLE)
    # close small gaps along the line stroke so connected components stay whole
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, k, iterations=1)
    orange = cv2.morphologyEx(orange, cv2.MORPH_CLOSE, k, iterations=1)
    purple = cv2.morphologyEx(purple, cv2.MORPH_CLOSE, k, iterations=1)
    return SeriesMasks(blue=blue, orange=orange, purple=purple)
