"""Run the chart-vision ONNX model on real screenshots and print the decoded
per-day BBT + LH series. Manual sanity-check tool.

Usage:
    # v2 (with scale classifier) — auto-detects celsius vs fahrenheit:
    python -m scripts.predict_vision_image \\
        --onnx artifacts/chart-vision-v2.onnx \\
        --image ../real-screen-1.png

    # Override the auto-picked BBT axis (or supply one for v1 which has no scale head):
    python -m scripts.predict_vision_image \\
        --onnx artifacts/chart-vision-v1.onnx \\
        --image ../real-screen-1.png \\
        --temp-min 36.0 --temp-max 37.2

Outputs, per series, the per-day normalized value + presence probability, and
the de-normalized real values — so you can eyeball them against the chart.
For v2: the BBT axis comes from the model's scale classifier (BBT_SCALES) and
the LH axis defaults to the Premom-style LH_RANGE. Both can be overridden.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from fertiluna_vision.constants import (
    BBT_SCALES,
    IMG_H,
    IMG_W,
    LH_RANGE,
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
    p.add_argument("--onnx", type=Path,
                   default=Path("artifacts/chart-vision-v2.onnx"))
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--temp-min", type=float, default=None,
                   help="Override auto-detected BBT axis min (else BBT_SCALES[scale][0]).")
    p.add_argument("--temp-max", type=float, default=None,
                   help="Override auto-detected BBT axis max.")
    p.add_argument("--lh-min", type=float, default=None,
                   help=f"Override LH axis min (else LH_RANGE[0]={LH_RANGE[0]}).")
    p.add_argument("--lh-max", type=float, default=None,
                   help=f"Override LH axis max (else LH_RANGE[1]={LH_RANGE[1]}).")
    p.add_argument("--thr", type=float, default=PRESENCE_THRESHOLD)
    args = p.parse_args()

    x = preprocess(args.image)
    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])

    # v2 adds a "scale" head (BBT axis classifier). v1 didn't have it — fall
    # back gracefully so the same script handles both.
    output_names = [o.name for o in sess.get_outputs()]
    has_scale = "scale" in output_names
    fetch = ["value", "present"] + (["scale"] if has_scale else [])
    outs = sess.run(fetch, {"image": x})
    value = outs[0][0]                  # (S, D) already in [0,1]
    present = sigmoid(outs[1])[0]       # (S, D)
    scale_logits = outs[2][0] if has_scale else None  # (N_BBT_SCALES,)

    # ---- pick axis ranges ------------------------------------------------
    # BBT axis: user override > model's scale classifier (v2) > none (v1).
    if args.temp_min is not None and args.temp_max is not None:
        temp_lo, temp_hi = args.temp_min, args.temp_max
        temp_source = "manual"
    elif scale_logits is not None:
        scale_idx = int(np.argmax(scale_logits))
        # softmax just for a readable confidence
        exps = np.exp(scale_logits - scale_logits.max())
        scale_probs = exps / exps.sum()
        scale_label, (temp_lo, temp_hi) = BBT_SCALES[scale_idx]
        temp_source = (f"auto({scale_label}, p="
                       f"{scale_probs[scale_idx]:.2f}; logits={scale_logits.tolist()})")
    else:
        temp_lo = temp_hi = None
        temp_source = "unknown (v1 model, pass --temp-min/--temp-max)"

    # LH axis: user override > Premom default (LH_RANGE).
    if args.lh_min is not None and args.lh_max is not None:
        lh_lo, lh_hi = args.lh_min, args.lh_max
        lh_source = "manual"
    else:
        lh_lo, lh_hi = LH_RANGE
        lh_source = f"default LH_RANGE={LH_RANGE}"

    ranges = {"temp": (temp_lo, temp_hi), "lh": (lh_lo, lh_hi)}

    # ---- print -----------------------------------------------------------
    print(f"\n=== {args.image.name} ===")
    print(f"  model:    {args.onnx.name}")
    print(f"  bbt axis: {temp_source}"
          + (f"  ->  [{temp_lo}, {temp_hi}]" if temp_lo is not None else ""))
    print(f"  lh axis:  {lh_source}  ->  [{lh_lo}, {lh_hi}]")

    for s, name in enumerate(SERIES_NAMES):
        lo, hi = ranges[name]
        present_days = int((present[s] >= args.thr).sum())
        print(f"\n--- series '{name}'  ({present_days}/{N_DAYS} days present) ---")
        header = "day:  " + " ".join(f"{d+1:>5}" for d in range(N_DAYS))
        print(header)
        print("pres: " + " ".join(f"{present[s, d]:5.2f}" for d in range(N_DAYS)))
        nrow = []
        for d in range(N_DAYS):
            nrow.append(f"{value[s, d]:5.2f}" if present[s, d] >= args.thr else "  .  ")
        print("norm: " + " ".join(nrow))
        if lo is not None and hi is not None:
            drow = []
            for d in range(N_DAYS):
                if present[s, d] >= args.thr:
                    real = lo + value[s, d] * (hi - lo)
                    drow.append(f"{real:5.2f}")
                else:
                    drow.append("  .  ")
            print("real: " + " ".join(drow) + f"   (axis {lo}..{hi})")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
