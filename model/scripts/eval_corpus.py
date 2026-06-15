"""Regression eval on the REAL hand-labeled screenshot corpus.

Complements `eval_vision_cv.py` (synthetic): synthetic gives volume + free
GT but its renderer diverges from real apps; this measures the distribution
we actually deploy on. It scores the pipeline against the corrected
ground-truth JSONs produced via `label_assist.py` — counting ONLY labels
with `"reviewed": true`, so we never score the model against its own
pre-fill.

Shares alignment + metrics with the synthetic eval (eval_metrics).

Usage:
    python -m scripts.eval_corpus --corpus corpus
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fertiluna_vision_cv import run_pipeline
from fertiluna_vision_cv import eval_metrics as em
from fertiluna_vision_cv.constants import N_DAYS, N_SERIES


def _load_gt(labels_path: Path) -> dict | None:
    data = json.loads(labels_path.read_text())
    if not data.get("reviewed"):
        return None
    value = np.array(data["value"], dtype=np.float32)
    present = np.array(data["present"], dtype=np.float32)
    if value.shape != (N_SERIES, N_DAYS) or present.shape != (N_SERIES, N_DAYS):
        raise ValueError(f"{labels_path.name}: value/present must be "
                         f"{N_SERIES}x{N_DAYS}")
    return {
        "value": value,
        "present": present,
        "scale_idx": data.get("scale_idx"),
        "source": data.get("source", labels_path.stem),
    }


def _find_image(corpus: Path, stem: str) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        cand = corpus / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--min-bbt-f1", type=float, default=0.55)
    ap.add_argument("--max-value-mae", type=float, default=0.16)
    ap.add_argument("--min-scale-acc", type=float, default=0.80)
    args = ap.parse_args()

    corpus: Path = args.corpus
    label_files = sorted(corpus.glob("*.labels.json"))
    if not label_files:
        print(f"[corpus] no *.labels.json in {corpus}. Seed with "
              f"`python -m scripts.label_assist --images … --out {corpus}`.")
        return 0

    agg = em.new_accumulator()
    reviewed, skipped = 0, 0
    for lf in label_files:
        stem = lf.name[: -len(".labels.json")]
        gt = _load_gt(lf)
        if gt is None:
            skipped += 1
            print(f"  [skip] {stem} — not reviewed yet")
            continue
        img = _find_image(corpus, stem)
        if img is None:
            print(f"  [warn] {stem} — labels present but no image found")
            continue
        r = run_pipeline(img)
        em.accumulate(agg, r, gt["value"], gt["present"], gt["scale_idx"])
        reviewed += 1
        print(f"  {stem}: status={r.status} conf={r.confidence:.2f} "
              f"pred_scale={r.scale_idx} gt_scale={gt['scale_idx']}")

    print(f"\n[corpus] {reviewed} reviewed, {skipped} unreviewed (skipped)")
    if reviewed == 0:
        print("No reviewed labels yet — correct the seeded JSONs and set "
              "reviewed=true. Nothing to gate.")
        return 0

    m = em.summarize(agg)
    print("\n── real-corpus metrics ────────────────────────────────────────")
    print(f"  scale acc (all)   : {m['scale_acc']:.3f}")
    for s, nm in enumerate(["temp", "lh"]):
        print(f"  {nm:<4} presence F1  : {m['f1'][s]:.3f}   "
              f"value MAE : {m['mae'][s]:.3f}")
    print(f"  mean confidence   : {m['mean_confidence']:.3f}")
    print(f"  status spread     : {m['status']}")

    print("\n── gate ───────────────────────────────────────────────────────")
    checks = [
        ("bbt_presence_f1", m["f1"][0], args.min_bbt_f1, "ge"),
        ("bbt_value_mae", m["mae"][0], args.max_value_mae, "le"),
    ]
    # Real corpus: scale GT is human-verified, so we gate scale over ALL
    # reviewed samples (no axis-recall confound to discount here) once there
    # are enough to be meaningful.
    if agg["scale_total"] >= 8:
        checks.append(("scale_acc", m["scale_acc"], args.min_scale_acc, "ge"))
    else:
        print(f"  [skip] scale_acc — only {agg['scale_total']} labeled "
              f"samples (<8), not gated")
    ok = True
    for name, val, thr, op in checks:
        passed = (val >= thr) if op == "ge" else (val <= thr)
        ok = ok and passed
        sign = "≥" if op == "ge" else "≤"
        print(f"  [{'PASS' if passed else 'FAIL'}] {name} = {val:.3f} "
              f"(need {sign} {thr})")
    print("───────────────────────────────────────────────────────────────")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
