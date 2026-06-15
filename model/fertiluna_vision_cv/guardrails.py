"""Post-OCR guardrails: repair axis labels, classify scale, interpolate gaps.

OCR on tiny tick crops is noisy ("99.5" → "29070", "36.5" → "5"). But an
axis is a STRICT ARITHMETIC PROGRESSION in pixel space, so we can recover
the true labels from a robust line fit through the confident reads and snap
everything onto the inferred grid. That fixes the canonical failure the
user flagged:

    OCR reads  [36, "5", 37, ...]   (the ".5" half-tick lost its "36")
    repaired   [36, 36.5, 37, ...]

Two separate concerns, kept deliberately distinct:

  * AXIS-LABEL repair  — safe and unambiguous. The axis grid is regular;
    a blank/garbled label has exactly one correct value given the fit.

  * DATA-POINT interpolation — a MEDICAL nuance. A day with no marker means
    no measurement was taken. We DO offer interpolation (so the curve can be
    drawn continuously), but the interpolated points are returned in a
    SEPARATE `interpolated` mask and never merged into `present`. Downstream
    code must treat measured ≠ interpolated: presenting a synthesised BBT as
    if it were measured could mislead a conception/contraception decision.
"""

from __future__ import annotations

import numpy as np

from .axis_ticks import AxisMapping, TickLabel, _parse_text
from .constants import BBT_SCALES, LH_RANGE


# Plausible value windows used to reject OCR garbage before fitting.
#   right axis (BBT): covers both °C (~35-38) and °F (~94-101) conventions.
#   left axis  (LH ratio): the fixed 0.1-1.9 scale, with slack.
_PLAUSIBLE = {
    "right": (30.0, 105.0),
    "left": (0.0, 2.5),
}
# "Nice" grid steps we snap the inferred per-tick step onto.
_NICE_STEPS = (0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0)


def _robust_linfit(ys: np.ndarray, vs: np.ndarray) -> tuple[float, float, float]:
    """Least-squares value = a + b*y with one outlier-rejection pass.
    Returns (a, b, rmse)."""
    b, a = np.polyfit(ys, vs, 1)
    resid = vs - (a + b * ys)
    rms = float(np.sqrt(np.mean(resid * resid)))
    if rms > 1e-6 and ys.size >= 4:
        keep = np.abs(resid) <= 2.0 * rms
        if keep.sum() >= 2:
            b, a = np.polyfit(ys[keep], vs[keep], 1)
            resid = vs[keep] - (a + b * ys[keep])
            rms = float(np.sqrt(np.mean(resid * resid)))
    return float(a), float(b), rms


def _infer_step(b: float, ys: np.ndarray, side: str = "right") -> float:
    """Per-tick value step, snapped to a 'nice' increment."""
    if ys.size >= 2:
        dy = float(np.median(np.diff(np.sort(ys))))
        raw = abs(b) * dy
    else:
        raw = 0.5 if side == "right" else 0.2
    # snap raw to the nearest nice step
    best = min(_NICE_STEPS, key=lambda s: abs(s - raw))
    return best


def snap_axis_column(col) -> None:
    """Write the corrected on-grid value back onto every label of a FITTED
    axis column (duck-typed: needs `.a`, `.b`, `.boxes[*].cy/.value`).

    `axis_columns._fit_column` already produces a robust (a, b) by RANSACing
    through the reliable ticks and DISCARDING half-tick misreads — so the
    MAPPING is correct. But the discarded labels keep their garbled OCR value
    (the ".5" that read as "5", or a blank). This rewrites each label's
    `.value` to `round((a + b·y)/step)·step`, so the label set reads the true
    grid — exactly the [36, 5, 37] → [36, 36.5, 37] repair the user asked for.
    Idempotent; safe to call once per fitted column.
    """
    if not getattr(col, "boxes", None) or abs(getattr(col, "b", 0.0)) < 1e-12:
        return
    ys = np.array([bx.cy for bx in col.boxes], dtype=np.float64)
    step = _infer_step(col.b, ys)
    for bx in col.boxes:
        pred = col.a + col.b * bx.cy
        bx.value = float(round(round(pred / step) * step, 3))


def repair_axis_labels(
    labels: list[TickLabel], side: str,
) -> tuple[list[TickLabel], float]:
    """Snap every tick label onto the axis's inferred arithmetic grid.

    Mutates and returns the labels with corrected `.value`, plus the fit's
    RMSE (small = trustworthy). Labels keep their original `.text` so the
    overlay can still show what OCR actually read.
    """
    lo, hi = _PLAUSIBLE.get(side, (-1e9, 1e9))
    confident: list[tuple[float, float]] = []
    for lbl in labels:
        v, is_frag = _parse_text(lbl.text)
        if v is not None and not is_frag and lo <= v <= hi:
            confident.append((lbl.y, v))
    if len(confident) < 2:
        return labels, float("inf")

    ys = np.array([c[0] for c in confident], dtype=np.float64)
    vs = np.array([c[1] for c in confident], dtype=np.float64)
    a, b, _ = _robust_linfit(ys, vs)
    step = _infer_step(b, np.array([l.y for l in labels], dtype=np.float64), side)

    # Snap each label (confident OR garbled) to the grid the fit predicts.
    for lbl in labels:
        pred = a + b * lbl.y
        snapped = round(pred / step) * step
        # clean float noise (e.g. 36.50000001)
        lbl.value = float(round(snapped, 3))

    # Re-fit on the now-complete, snapped set for the final mapping accuracy.
    all_ys = np.array([l.y for l in labels], dtype=np.float64)
    all_vs = np.array([l.value for l in labels], dtype=np.float64)
    a2, b2, rms2 = _robust_linfit(all_ys, all_vs)
    return labels, rms2


def mapping_from_labels(labels: list[TickLabel], side: str) -> AxisMapping | None:
    """Least-squares AxisMapping through ALL repaired labels (more robust
    than a two-endpoint fit, which a single mis-snapped extreme would skew)."""
    pts = [(l.y, l.value) for l in labels if l.value is not None]
    if len(pts) < 2:
        return None
    ys = np.array([p[0] for p in pts], dtype=np.float64)
    vs = np.array([p[1] for p in pts], dtype=np.float64)
    a, b, rms = _robust_linfit(ys, vs)
    return AxisMapping(side=side, a=a, b=b, anchors=labels, rmse=rms,
                       source="repaired")


def classify_scale_from_labels(right_labels: list[TickLabel]) -> int | None:
    """°C vs °F purely from the repaired BBT-axis label VALUES.

    Far more reliable than glyph/template heuristics: after repair the labels
    carry real numbers, and their median lands unambiguously in one band.
    Returns an index into BBT_SCALES, or None if undecidable.
    """
    vals = [l.value for l in right_labels if l.value is not None]
    if len(vals) < 2:
        return None
    med = float(np.median(vals))
    # celsius axis values cluster ~35-37.5; fahrenheit ~95-99.5.
    best_idx, best_d = None, float("inf")
    for idx, (_name, (lo, hi)) in enumerate(BBT_SCALES):
        centre = 0.5 * (lo + hi)
        d = abs(med - centre)
        if d < best_d:
            best_d, best_idx = d, idx
    # require the median to actually be near a known band (within 5 units),
    # else the labels are garbage and we abstain.
    return best_idx if best_d <= 5.0 else None


def interpolate_series(
    value: np.ndarray, present: np.ndarray, max_gap: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Linear-interpolate short gaps between MEASURED days.

    Returns (value_filled, interpolated_mask). `interpolated_mask` is 1.0 on
    days that were synthesised and 0.0 on measured days, so the caller can
    keep the two classes visually + semantically distinct. `present` is NOT
    modified — interpolated days remain absent from the measured set.

    Only gaps of length ≤ `max_gap` are filled; longer gaps stay empty (we
    don't invent a week of missing temperatures). No extrapolation past the
    first / last measured day.
    """
    n = len(value)
    out_v = value.astype(np.float32).copy()
    interp = np.zeros(n, dtype=np.float32)
    present_idx = [i for i in range(n) if present[i] > 0.5]
    for j in range(len(present_idx) - 1):
        a, b = present_idx[j], present_idx[j + 1]
        gap = b - a - 1
        if 1 <= gap <= max_gap:
            va, vb = float(value[a]), float(value[b])
            for k in range(a + 1, b):
                t = (k - a) / (b - a)
                out_v[k] = va * (1.0 - t) + vb * t
                interp[k] = 1.0
    return out_v, interp
