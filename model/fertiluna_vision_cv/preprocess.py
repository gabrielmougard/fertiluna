"""Image loading + working-canvas normalization.

Real screenshots arrive at wildly varying resolutions (1242x2208, 750x1334, …).
We resize to a fixed working WIDTH preserving aspect ratio so all downstream
thresholds (HSV ranges, area limits, kernel sizes) operate at a consistent
scale. We do NOT crop or pad.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .constants import WORK_W

# Don't aggressively downscale high-res phone screenshots — marker rings
# become sub-pixel-thin and the white center blob loses identifiability.
# Don't upscale small images past 2× either (sharpens noise without adding
# real information). The band keeps marker pixel sizes inside the detector's
# tuned range across all input resolutions.
WORK_W_MIN = 1200
WORK_W_MAX = 2400


def load_bgr(path: Path | str) -> np.ndarray:
    """Read an image as BGR (OpenCV's native order), respecting EXIF rotation."""
    pil = Image.open(path)
    pil = ImageOps.exif_transpose(pil).convert("RGB")
    rgb = np.asarray(pil)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def to_work_canvas(bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """Resize BGR image into the [WORK_W_MIN, WORK_W_MAX] width band,
    preserving aspect ratio. Skip the resize entirely when source width is
    already in range so marker pixel sizes stay close to native."""
    _ = WORK_W  # kept exported for back-compat
    h, w = bgr.shape[:2]
    if WORK_W_MIN <= w <= WORK_W_MAX:
        return bgr, 1.0
    target = WORK_W_MIN if w < WORK_W_MIN else WORK_W_MAX
    scale = target / w
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(bgr, (target, new_h), interpolation=interp), scale
