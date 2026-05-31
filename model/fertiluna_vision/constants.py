"""Shared constants for the chart-vision model.

Keep IMG_H/IMG_W, N_DAYS, and the output layout in sync with the TS browser
preprocessing (src/lib/visionInference.ts).
"""

# Cycle days the model resolves along the width axis. Matches the cycle model's
# CYCLE_MAX_DAYS so the digitized output drops straight into the same table.
N_DAYS = 35

# Two series: index 0 = BBT (temperature), index 1 = LH/hormone.
N_SERIES = 2
SERIES_NAMES = ["temp", "lh"]

# ── Axis-scale classification ────────────────────────────────────────────────
# Fertility apps use a small, FIXED set of axis conventions, so instead of OCR
# we classify which one the BBT axis uses. The model emits a `scale` head; the
# browser looks up the [min,max] to de-normalize — no user input needed.
#
# BBT_SCALES index -> (label, (axis_min, axis_max)) in real units.
BBT_SCALES = [
    ("celsius", (35.6, 37.4)),
    ("fahrenheit", (95.0, 99.5)),
]
N_BBT_SCALES = len(BBT_SCALES)
# LH "Ratio" axis is fixed in this style.
LH_RANGE = (0.1, 1.9)

# Fixed input canvas (RGB). Width > height because charts are landscape and the
# day axis (what we resolve at N_DAYS resolution) runs horizontally.
IMG_H = 224
IMG_W = 384

# ImageNet normalization (we use a pretrained-style backbone init).
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

# Output tensors (per image):
#   value:   (N_SERIES, N_DAYS)  float in [0,1]  — normalized within each
#                                                   series' own plot value range
#   present: (N_SERIES, N_DAYS)  float in [0,1]  — sigmoid logit -> probability
#                                                   a data point exists that day
PRESENCE_THRESHOLD = 0.5
