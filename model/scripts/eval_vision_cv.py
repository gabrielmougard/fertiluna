"""Regression eval for the classical-CV digitizer on SYNTHETIC charts.

We render Premom-style charts with the same generator the CNN trains on — so
we get EXACT ground-truth value / present / scale for free, at any volume.
The CV pipeline runs on each rendered image and is scored against that GT.

Metrics + alignment live in `fertiluna_vision_cv.eval_metrics` (shared with
the real-corpus eval in `eval_corpus.py`).

A regression GATE applies thresholds and exits non-zero on failure, so this
can run in CI. The hard gates are the ROBUST core metrics (BBT presence F1,
BBT value MAE). Scale accuracy is TRACKED and only hard-gated on the
"extracted" (trustworthy) subset once that subset is large enough — on
synthetic charts the all-samples scale number is confounded by axis-column
recall (when the BBT axis isn't read, scale defaults and the result is
already flagged low-confidence).

Usage:
    python -m scripts.eval_vision_cv --n 40 --seed 7 --dpi 200
"""

from __future__ import annotations

import argparse

import numpy as np

from fertiluna_vision_cv import run_pipeline
from fertiluna_vision_cv import eval_metrics as em


def run_synthetic(n: int, seed: int, dpi: int = 200) -> dict:
    # Local import so the eval only needs the vision (render) extra when run.
    from fertiluna_vision.render import render_premom_chart
    import cv2

    # Render at a DPI that puts the synthetic width in the same band as real
    # phone screenshots (~2000 px). At default DPI the synthetic charts are
    # ~800 px wide, where the tiny axis labels under-OCR and the eval would
    # understate real-world accuracy.
    rng = np.random.default_rng(seed)
    agg = em.new_accumulator()
    for i in range(n):
        sample = render_premom_chart(rng, dpi_override=dpi)
        bgr = cv2.cvtColor(np.asarray(sample.image), cv2.COLOR_RGB2BGR)
        r = run_pipeline(bgr)
        gt_scale = sample.bbt_scale if sample.bbt_scale in (0, 1) else None
        em.accumulate(agg, r, sample.value, sample.present, gt_scale)
        if (i + 1) % 10 == 0:
            print(f"  …{i + 1}/{n}")
    return agg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="synthetic samples")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dpi", type=int, default=200,
                    help="render DPI; ~200 matches real phone-screenshot width")
    ap.add_argument("--min-bbt-f1", type=float, default=0.55)
    ap.add_argument("--max-value-mae", type=float, default=0.16)
    ap.add_argument("--min-scale-acc-extracted", type=float, default=0.80)
    args = ap.parse_args()

    print(f"[eval] {args.n} synthetic Premom charts (seed {args.seed}, "
          f"dpi {args.dpi}) …")
    agg = run_synthetic(args.n, args.seed, dpi=args.dpi)
    m = em.summarize(agg)

    print("\n── synthetic metrics ──────────────────────────────────────────")
    print(f"  scale acc (all)        : {m['scale_acc']:.3f}")
    print(f"  scale acc (extracted)  : {m['scale_acc_extracted']:.3f}  "
          f"({agg['scale_correct_ex']}/{m['scale_total_extracted']})  ← user-facing")
    for s, nm in enumerate(["temp", "lh"]):
        print(f"  {nm:<4} presence F1        : {m['f1'][s]:.3f}   "
              f"value MAE : {m['mae'][s]:.3f}   "
              f"(tp={m['tp'][s]} fp={m['fp'][s]} fn={m['fn'][s]})")
    print(f"  mean confidence        : {m['mean_confidence']:.3f}")
    print(f"  status spread          : {m['status']}")

    print("\n── gate ───────────────────────────────────────────────────────")
    checks = [
        ("bbt_presence_f1", m["f1"][0], args.min_bbt_f1, "ge"),
        ("bbt_value_mae", m["mae"][0], args.max_value_mae, "le"),
    ]
    if m["scale_total_extracted"] >= 8:
        checks.append(("scale_acc_extracted", m["scale_acc_extracted"],
                       args.min_scale_acc_extracted, "ge"))
    else:
        print(f"  [skip] scale_acc_extracted — only "
              f"{m['scale_total_extracted']} extracted samples (<8), not gated")
    ok = True
    for name, val, thr, op in checks:
        passed = (val >= thr) if op == "ge" else (val <= thr)
        ok = ok and passed
        sign = "≥" if op == "ge" else "≤"
        print(f"  [{'PASS' if passed else 'FAIL'}] {name} = {val:.3f} "
              f"(need {sign} {thr})")
    print("───────────────────────────────────────────────────────────────")
    print("RESULT:", "PASS" if ok else "FAIL")
    print(f"\nTRACKED (not gated): scale_acc(all)={m['scale_acc']:.3f}, "
          f"lh_f1={m['f1'][1]:.3f}, lh_mae={m['mae'][1]:.3f} — known gaps: "
          "scale needs axis recall; LH lines are often markerless.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
