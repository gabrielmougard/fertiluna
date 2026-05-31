"""Run the chart-vision ONNX model on real screenshots and print the decoded
per-day BBT + LH series. Manual sanity-check tool.

Usage:
    python -m scripts.predict_vision_image \
        --onnx artifacts/chart-vision-v1.onnx \
        --image ../real-screen-1.png \
        [--temp-min 36.0 --temp-max 37.2 --lh-min 0 --lh-max 3]

Outputs, per series, the per-day normalized value + presence probability, and
(if axis min/max given) the de-normalized real values — so you can eyeball them
against the chart.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from fertiluna_vision.constants import (
    IMG_H,
    IMG_W,
    N_DAYS,
    NORM_MEAN,
    NORM_STD,
    PRESENCE_THRESHOLD,
    SERIES_NAMES,
)


def preprocess(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((IMG_W, IMG_H), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0  # H,W,3
    arr = (arr - np.array(NORM_MEAN)) / np.array(NORM_STD)
    chw = arr.transpose(2, 0, 1)[None].astype(np.float32)  # 1,3,H,W
    return chw


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", type=Path, default=Path("artifacts/chart-vision-v1.onnx"))
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--temp-min", type=float, default=None)
    p.add_argument("--temp-max", type=float, default=None)
    p.add_argument("--lh-min", type=float, default=None)
    p.add_argument("--lh-max", type=float, default=None)
    p.add_argument("--thr", type=float, default=PRESENCE_THRESHOLD)
    args = p.parse_args()

    x = preprocess(args.image)
    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    value_out, present_logit = sess.run(["value", "present"], {"image": x})
    value = value_out[0]                  # (S, D) already in [0,1]
    present = sigmoid(present_logit)[0]   # (S, D)

    ranges = {
        "temp": (args.temp_min, args.temp_max),
        "lh": (args.lh_min, args.lh_max),
    }

    print(f"\n=== {args.image.name} ===")
    for s, name in enumerate(SERIES_NAMES):
        lo, hi = ranges[name]
        present_days = int((present[s] >= args.thr).sum())
        print(f"\n--- series '{name}'  ({present_days}/{N_DAYS} days present) ---")
        header = "day:  " + " ".join(f"{d+1:>5}" for d in range(N_DAYS))
        print(header)
        # presence prob
        print("pres: " + " ".join(f"{present[s, d]:5.2f}" for d in range(N_DAYS)))
        # normalized value (only where present)
        nrow = []
        for d in range(N_DAYS):
            nrow.append(f"{value[s, d]:5.2f}" if present[s, d] >= args.thr else "  .  ")
        print("norm: " + " ".join(nrow))
        # de-normalized if range given
        if lo is not None and hi is not None:
            drow = []
            for d in range(N_DAYS):
                if present[s, d] >= args.thr:
                    real = lo + value[s, d] * (hi - lo)
                    drow.append(f"{real:5.2f}")
                else:
                    drow.append("  .  ")
            print(f"real: " + " ".join(drow) + f"   (axis {lo}..{hi})")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
