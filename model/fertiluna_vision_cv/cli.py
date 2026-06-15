"""Command-line tools for the CV pipeline.

    # 1. Print decoded series for a single image (mirrors predict_vision_image.py)
    python -m fertiluna_vision_cv.cli infer --image real-screen-1.png

    # 2. Render side-by-side assessment overlays for one or many images
    python -m fertiluna_vision_cv.cli assess \\
        --images real-screen-1.png real-screen-2.png \\
        --out /tmp/cv-assess

The `assess` sub-command writes per-image:
    {stem}_overlay.png   visual annotation (boxes + markers + day grid)
    {stem}_decoded.json  same JSON schema the ONNX model would emit
plus an `index.html` gallery for quick browser inspection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .constants import BBT_SCALES, LH_RANGE, N_DAYS, PRESENCE_THRESHOLD, SERIES_NAMES
from .debug_overlay import compose_assessment
from .pipeline import ChartResult, run_pipeline


def _result_to_dict(result: ChartResult, source: str | None = None) -> dict:
    bbt_lo, bbt_hi = BBT_SCALES[result.scale_idx][1]
    lh_lo, lh_hi = LH_RANGE
    out = {
        "source": source,
        "scale": {
            "idx": result.scale_idx,
            "label": result.scale_label,
            "confidence": result.scale_confidence,
            "bbt_range": [bbt_lo, bbt_hi],
            "lh_range": [lh_lo, lh_hi],
        },
        # Production quality signal: confidence in [0,1] + a coarse status
        # ("extracted" | "low_confidence" | "not_a_chart") the consumer
        # thresholds on, plus the per-signal breakdown and reasons.
        "confidence": result.confidence,
        "status": result.status,
        "quality": result.quality,
        # Day-window bookkeeping (see ChartResult). `visible_days` may exceed
        # the N_DAYS tensor width; `truncated` flags that data was dropped.
        "visible_days": result.visible_days,
        "truncated": result.truncated,
        # Which colored line was read as the LH series ("orange" = the
        # "Ratio" curve on the 0.1-1.9 axis; "purple" = the dense "Level"
        # curve on the 5-95 axis). When "purple", the normalized LH value is
        # a plot-extent fraction, NOT a Ratio reading, so `decoded.lh`
        # real-units below are advisory only.
        "lh_source": result.debug.get("lh_color", "orange"),
        "value": result.value.tolist(),
        "present": result.present.tolist(),
        # Guardrail: which days were synthesised by interpolation (1.0) vs
        # measured (0.0). `present` stays 0 on interpolated days — measured
        # and interpolated are distinct classes.
        "interpolated": (result.interpolated.tolist()
                         if result.interpolated is not None else None),
        # convenience: human-readable per-day real-units series
        "decoded": {},
        # bottom-of-screen table — empty dicts for rows the screenshot
        # didn't include, so the schema is always the same shape.
        "table": {
            "calendar": {"label": None, "cells": [None] * N_DAYS,
                         "cell_bboxes": [None] * N_DAYS, "label_bbox": None},
            "CD": {"label": None, "cells": [None] * N_DAYS,
                   "cell_bboxes": [None] * N_DAYS, "label_bbox": None},
            "DPO": {"label": None, "cells": [None] * N_DAYS,
                    "cell_bboxes": [None] * N_DAYS, "label_bbox": None},
            "Sex": {"label": None, "cells": [None] * N_DAYS,
                    "cell_bboxes": [None] * N_DAYS, "label_bbox": None},
            "CM": {"label": None, "cells": [None] * N_DAYS,
                   "cell_bboxes": [None] * N_DAYS, "label_bbox": None},
            "Symptoms": {"label": None, "cells": [None] * N_DAYS,
                         "cell_bboxes": [None] * N_DAYS, "label_bbox": None},
            "hCG": {"label": None, "cells": [None] * N_DAYS,
                    "cell_bboxes": [None] * N_DAYS, "label_bbox": None},
        },
    }
    for s, name in enumerate(SERIES_NAMES):
        lo, hi = (bbt_lo, bbt_hi) if name == "temp" else (lh_lo, lh_hi)
        out["decoded"][name] = [
            (lo + float(result.value[s, d]) * (hi - lo))
            if result.present[s, d] >= PRESENCE_THRESHOLD else None
            for d in range(N_DAYS)
        ]
    # populate table rows that the extractor actually found
    if result.table and "by_name" in result.table:
        for name, row in result.table["by_name"].items():
            if name not in out["table"]:
                continue
            out["table"][name]["label"] = row.label_text or None
            out["table"][name]["label_bbox"] = list(row.label_bbox)
            for d in range(min(N_DAYS, len(row.cells))):
                cell = row.cells[d]
                out["table"][name]["cells"][d] = cell.text
                out["table"][name]["cell_bboxes"][d] = list(cell.bbox)
    return out


def _print_table(image: Path, result: ChartResult, thr: float) -> None:
    bbt_lo, bbt_hi = BBT_SCALES[result.scale_idx][1]
    lh_lo, lh_hi = LH_RANGE
    print(f"\n=== {image.name} ===")
    print(f"  scale: {result.scale_label} ({result.scale_confidence:.2f})  "
          f"-> bbt[{bbt_lo}, {bbt_hi}]   lh[{lh_lo}, {lh_hi}]")
    for s, name in enumerate(SERIES_NAMES):
        lo, hi = (bbt_lo, bbt_hi) if name == "temp" else (lh_lo, lh_hi)
        present_days = int((result.present[s] >= thr).sum())
        print(f"\n--- series '{name}'  ({present_days}/{N_DAYS} days present) ---")
        print("day:  " + " ".join(f"{d + 1:>5}" for d in range(N_DAYS)))
        print("pres: " + " ".join(f"{result.present[s, d]:5.2f}"
                                  for d in range(N_DAYS)))
        nrow, drow = [], []
        for d in range(N_DAYS):
            if result.present[s, d] >= thr:
                nrow.append(f"{result.value[s, d]:5.2f}")
                drow.append(f"{lo + result.value[s, d] * (hi - lo):5.2f}")
            else:
                nrow.append("  .  ")
                drow.append("  .  ")
        print("norm: " + " ".join(nrow))
        print("real: " + " ".join(drow))


def _cmd_infer(args) -> int:
    result = run_pipeline(args.image)
    _print_table(args.image, result, args.thr)
    return 0


def _cmd_assess(args) -> int:
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    entries: list[tuple[Path, dict]] = []
    for img_path in args.images:
        result = run_pipeline(img_path)
        overlay = compose_assessment(result)
        stem = img_path.stem
        overlay_path = out_dir / f"{stem}_overlay.png"
        json_path = out_dir / f"{stem}_decoded.json"
        cv2.imwrite(str(overlay_path), overlay)
        payload = _result_to_dict(result, source=str(img_path))
        json_path.write_text(json.dumps(payload, indent=2))
        entries.append((img_path, payload))
        print(f"[assess] {img_path.name} -> {overlay_path.name}  "
              f"scale={result.scale_label} "
              f"bbt={int(result.present[0].sum())}/{N_DAYS}  "
              f"lh={int(result.present[1].sum())}/{N_DAYS}")

    # tiny gallery for browser preview
    html = ["<!doctype html><meta charset=utf-8><title>fertiluna_vision_cv assess</title>",
            "<style>body{font-family:sans-serif;background:#222;color:#eee;margin:24px;}"
            "section{margin-bottom:48px;border-bottom:1px solid #444;padding-bottom:24px;}"
            "img{max-width:100%;border:1px solid #444;}"
            "code{background:#000;padding:2px 6px;border-radius:4px;}</style>"]
    for img_path, payload in entries:
        s = payload["scale"]
        html.append(f"<section><h2>{img_path.name}</h2>")
        html.append(f"<p>scale: <code>{s['label']} (conf={s['confidence']:.2f})"
                    f" -> bbt[{s['bbt_range'][0]}, {s['bbt_range'][1]}]</code></p>")
        html.append(f"<img src='{img_path.stem}_overlay.png'></section>")
    (out_dir / "index.html").write_text("\n".join(html))
    print(f"\n[assess] gallery -> {out_dir / 'index.html'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="fertiluna_vision_cv")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_infer = sub.add_parser("infer", help="run pipeline + print decoded series")
    p_infer.add_argument("--image", type=Path, required=True)
    p_infer.add_argument("--thr", type=float, default=PRESENCE_THRESHOLD)
    p_infer.set_defaults(func=_cmd_infer)

    p_assess = sub.add_parser("assess", help="render annotated overlays + JSON")
    p_assess.add_argument("--images", type=Path, nargs="+", required=True)
    p_assess.add_argument("--out", type=Path, required=True)
    p_assess.set_defaults(func=_cmd_assess)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
