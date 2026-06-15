"""Model-assisted labeling for the REAL screenshot corpus.

Hand-labeling per-day BBT/LH values from scratch is tedious. Instead we
pre-fill: run the pipeline, dump its prediction as an EDITABLE ground-truth
JSON next to the image, plus the assessment overlay. A human then opens the
overlay, corrects the JSON where the pipeline got it wrong, and flips
`"reviewed": true`. Only reviewed labels count in `eval_corpus.py`.

This is the standard correct-the-machine labeling loop — far faster than
labeling blank, and it focuses human effort exactly where the model is weak.

Usage:
    # seed labels for every real screenshot into the corpus dir
    python -m scripts.label_assist --images ../real-screen-*.png --out corpus
    # then edit corpus/real-screen-1.labels.json, set reviewed=true
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from fertiluna_vision_cv import run_pipeline
from fertiluna_vision_cv.constants import BBT_SCALES
from fertiluna_vision_cv.debug_overlay import compose_assessment


def _labels_payload(result, source: str) -> dict:
    return {
        "source": source,
        # Flip to true once you've corrected the arrays below. Unreviewed
        # labels are IGNORED by eval_corpus (they'd just score the model
        # against its own prediction).
        "reviewed": False,
        "scale_idx": int(result.scale_idx),
        "scale_label": BBT_SCALES[result.scale_idx][0],
        # GT arrays, pre-filled from the prediction — CORRECT THESE.
        # value: (2, 35) normalized [0,1] within each series' axis range.
        # present: (2, 35) {0,1}. Row 0 = temp (BBT), row 1 = lh.
        "value": [[round(float(v), 4) for v in row] for row in result.value],
        "present": [[int(p > 0.5) for p in row] for row in result.present],
        "pipeline_confidence": result.confidence,
        "pipeline_status": result.status,
        "note": ("Pre-filled from pipeline prediction. Open the *_overlay.png, "
                 "fix value/present/scale_idx where wrong, then set "
                 "reviewed=true."),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True,
                    help="corpus directory for *.labels.json + *_overlay.png")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing labels (default: skip reviewed ones)")
    args = ap.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    for img_path in args.images:
        stem = img_path.stem
        labels_path = out / f"{stem}.labels.json"
        if labels_path.exists() and not args.force:
            existing = json.loads(labels_path.read_text())
            if existing.get("reviewed"):
                print(f"[label] {stem}: reviewed — skipping (use --force to overwrite)")
                continue
        result = run_pipeline(img_path)
        labels_path.write_text(json.dumps(_labels_payload(result, str(img_path)),
                                          indent=2))
        overlay = compose_assessment(result)
        cv2.imwrite(str(out / f"{stem}_overlay.png"), overlay)
        print(f"[label] {stem}: seeded {labels_path.name} "
              f"(status={result.status}, conf={result.confidence:.2f}) "
              f"— review + set reviewed=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
