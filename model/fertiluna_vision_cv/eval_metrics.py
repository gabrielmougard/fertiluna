"""Shared scoring for the digitizer evals (synthetic + real corpus).

Both `scripts/eval_vision_cv.py` (synthetic GT) and `scripts/eval_corpus.py`
(hand-labeled real GT) compare a predicted (value, present) against a GT
(value, present) with the SAME logic, so it lives here once:

  * best-shift alignment — the pipeline left-packs its day window, so a
    constant day offset vs the GT window is expected and shouldn't be scored
    as error; we find the shift that maximises present-day overlap.
  * per-series tp/fp/fn → presence F1.
  * value MAE on co-present days (normalized [0,1] units).

All arrays are (N_SERIES, N_DAYS) float; present is {0,1}.
"""

from __future__ import annotations

import numpy as np

from .constants import N_DAYS, N_SERIES


def best_shift(gt_present: np.ndarray, pred_present: np.ndarray,
               max_shift: int = 8) -> int:
    """Day offset s (applied to pred) maximising present-day overlap with GT."""
    best_s, best_ov = 0, -1
    for s in range(-max_shift, max_shift + 1):
        ov = 0
        for d in range(N_DAYS):
            ds = d + s
            if 0 <= ds < N_DAYS:
                ov += int(((gt_present[:, d] > 0.5) &
                           (pred_present[:, ds] > 0.5)).sum())
        if ov > best_ov:
            best_ov, best_s = ov, s
    return best_s


def score_one(
    gt_value: np.ndarray, gt_present: np.ndarray,
    pred_value: np.ndarray, pred_present: np.ndarray,
) -> list[tuple[int, int, int, float, int]]:
    """Per-series (tp, fp, fn, abs_err_sum, co_present) after best-shift align."""
    s = best_shift(gt_present, pred_present)
    out: list[tuple[int, int, int, float, int]] = []
    for series in range(N_SERIES):
        tp = fp = fn = 0
        err_sum, co = 0.0, 0
        for d in range(N_DAYS):
            ds = d + s
            g = gt_present[series, d] > 0.5
            p = (0 <= ds < N_DAYS) and (pred_present[series, ds] > 0.5)
            if g and p:
                tp += 1
                err_sum += abs(float(gt_value[series, d]) -
                               float(pred_value[series, ds]))
                co += 1
            elif p and not g:
                fp += 1
            elif g and not p:
                fn += 1
        out.append((tp, fp, fn, err_sum, co))
    return out


def f1(tp: int, fp: int, fn: int) -> float:
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def new_accumulator() -> dict:
    return {
        "scale_correct": 0, "scale_total": 0,
        "scale_correct_ex": 0, "scale_total_ex": 0,
        "tp": [0] * N_SERIES, "fp": [0] * N_SERIES, "fn": [0] * N_SERIES,
        "err": [0.0] * N_SERIES, "co": [0] * N_SERIES,
        "status": {}, "conf_sum": 0.0, "n": 0,
    }


def accumulate(agg: dict, result, gt_value: np.ndarray, gt_present: np.ndarray,
               gt_scale: int | None) -> None:
    """Fold one (prediction `result`, GT) pair into the accumulator."""
    agg["n"] += 1
    agg["status"][result.status] = agg["status"].get(result.status, 0) + 1
    agg["conf_sum"] += result.confidence
    if gt_scale in (0, 1):
        agg["scale_total"] += 1
        correct = int(result.scale_idx == gt_scale)
        agg["scale_correct"] += correct
        if result.status == "extracted":
            agg["scale_total_ex"] += 1
            agg["scale_correct_ex"] += correct
    for s, (tp, fp, fn, err, co) in enumerate(
        score_one(gt_value, gt_present, result.value, result.present)
    ):
        agg["tp"][s] += tp
        agg["fp"][s] += fp
        agg["fn"][s] += fn
        agg["err"][s] += err
        agg["co"][s] += co


def summarize(agg: dict) -> dict:
    """Reduce an accumulator to the reported metrics."""
    scale_acc = (agg["scale_correct"] / agg["scale_total"]
                 if agg["scale_total"] else float("nan"))
    scale_acc_ex = (agg["scale_correct_ex"] / agg["scale_total_ex"]
                    if agg["scale_total_ex"] else float("nan"))
    f1s = [f1(agg["tp"][s], agg["fp"][s], agg["fn"][s])
           for s in range(N_SERIES)]
    mae = [(agg["err"][s] / agg["co"][s]) if agg["co"][s] else float("nan")
           for s in range(N_SERIES)]
    return {
        "scale_acc": scale_acc,
        "scale_acc_extracted": scale_acc_ex,
        "scale_total_extracted": agg["scale_total_ex"],
        "f1": f1s,
        "mae": mae,
        "mean_confidence": agg["conf_sum"] / max(1, agg["n"]),
        "status": agg["status"],
        "tp": agg["tp"], "fp": agg["fp"], "fn": agg["fn"],
    }
