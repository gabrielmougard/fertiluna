"""Shared constants — re-export the CNN's constants so the two pipelines emit
identical output tensors and the downstream browser code doesn't care which one
produced the JSON.

Anything CV-only (HSV ranges, marker-size bounds, working-canvas size) lives
here; everything that defines the OUTPUT CONTRACT is imported from the CNN
package so they can never drift out of sync.
"""

from __future__ import annotations

from fertiluna_vision.constants import (  # re-exported, do not redefine
    BBT_SCALES,
    LH_RANGE,
    N_BBT_SCALES,
    N_DAYS,
    N_SERIES,
    PRESENCE_THRESHOLD,
    SERIES_NAMES,
)

# ── working canvas ──────────────────────────────────────────────────────────
# Internal resolution the CV pipeline runs at. Larger than the CNN canvas
# (224x384) because classical CV needs vertical resolution to separate the ~18
# BBT gridlines and to keep marker contours distinct from line segments.
WORK_W = 1600
# height is derived from aspect ratio (kept proportional during preprocess)

# ── HSV color thresholds (OpenCV: H ∈ [0,179], S/V ∈ [0,255]) ───────────────
# Tuned by sampling all 4 real screenshots in the repo. Each entry:
# (hue_lo, hue_hi, sat_lo, val_lo). val_lo keeps us out of dark text.
#
# CRITICAL: hue bands must NOT overlap, otherwise purple "Level" markers
# leak into the blue mask and get classified as BBT. Real-app palettes:
#   BBT light blue    ~ H 105-120   (#95aeff and lighter shades)
#   Level violet/purple ~ H 130-150 (#9e6fe3)
#   LH coral/orange   ~ H 0-15       (#ff9e8d and warmer)
#   Pink band fill    ~ H 165-180   (alpha-blended, low sat) — must NOT
#                                    leak into orange.
HSV_BLUE = (95, 124, 12, 100)     # sat low: real-screen-3 BBT line is
                                  # pale at sat≈8-25; can't reject it.
HSV_ORANGE = (0, 18, 30, 120)     # sat low: real-screen-3 LH thin line
                                  # has sat≈30-90 (peak marker is sat>100).
HSV_PURPLE = (128, 162, 35, 100)
# Pale fertile/period bands have low saturation so the sat floors above
# filter them out without a separate pass.

# ── column-scan marker tunables ────────────────────────────────────────────
# Open-circle markers in the Premom renderer are ~3-5 px radius at the rendered
# DPI; after upscaling to WORK_W they grow. Tunables live in
# `marker_detection.py` because they belong with the scanner logic.

# ── plot-region detection ───────────────────────────────────────────────────
# When color-ink and gridline cues conflict or fail, fall back to these
# Premom-empirical ratios. Used as a last-resort bbox.
FALLBACK_PLOT_RATIOS = (0.10, 0.10, 0.92, 0.72)  # (left, top, right, bottom)
