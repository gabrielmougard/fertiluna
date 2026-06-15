"""Pipeline-level confidence + status — the production failure signal.

The pipeline always returns *something*; without a quality signal the
consumer can't tell "digitised a clean chart" from "found nothing in a photo
of a cat". This module folds the observable per-stage signals into a single
`confidence ∈ [0,1]` and a coarse `status` the app can branch on:

    "extracted"      → trust it (optionally one-tap confirm)
    "low_confidence" → show the result but prompt the user to verify / correct
    "not_a_chart"    → no chart found; ask for a different image / manual entry

The score is a transparent weighted blend of:
    axis    — did we fit a BBT axis (enough ticks, low residual)?
    scale   — °C/°F classifier margin
    markers — how many data points were recovered
    grid    — how the day grid was found (date-row > gridlines > autocorr > uniform)
    plot    — how the plot region was found (tick/axis-anchored > ink+grid > fallback)

`reasons` lists the human-readable deductions so the UI can explain itself
and so eval failures are debuggable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class QualityReport:
    confidence: float                      # [0,1]
    status: str                            # extracted | low_confidence | not_a_chart
    components: dict = field(default_factory=dict)   # per-signal sub-scores
    reasons: list[str] = field(default_factory=list)


# Blend weights (sum = 1).
_W = {"axis": 0.25, "scale": 0.15, "markers": 0.30, "grid": 0.20, "plot": 0.10}
# Below this the result is flagged for manual verification.
_LOW_CONF = 0.45


def _grid_score(source: str | None) -> float:
    return {
        "date-row": 1.0,
        "gridlines": 0.95,
        "autocorr": 0.80,
        "marker-spacing": 0.60,
        "uniform": 0.30,
    }.get(source or "", 0.4)


def _plot_score(method: str | None) -> float:
    m = method or ""
    if "axiscol" in m or "ticks" in m:
        return 1.0
    if "ink+grid" in m:
        return 0.8
    if "ink" in m:
        return 0.6
    return 0.3  # fallback ratios


def assess_quality(
    *,
    present: np.ndarray,           # (N_SERIES, N_DAYS)
    scale_confidence: float,
    axes,                          # ResolvedAxes | None
    grid_source: str | None,
    plot_method: str | None,
    visible_days: int,
    truncated: bool,
) -> QualityReport:
    reasons: list[str] = []

    # ── axis ──────────────────────────────────────────────────────────────
    bbt = getattr(axes, "bbt", None) if axes is not None else None
    if bbt is not None and getattr(bbt, "n_fit", 0) >= 2:
        n_fit = int(bbt.n_fit)
        rmse = float(getattr(bbt, "rmse", 0.0))
        # n_fit saturates at ~8 ticks; rmse penalty (a clean axis fits to <0.1).
        axis_score = min(1.0, n_fit / 8.0) * float(np.exp(-min(rmse, 3.0)))
        if n_fit < 5:
            reasons.append(f"BBT axis fit on only {n_fit} ticks")
        if rmse > 0.5:
            reasons.append(f"BBT axis fit residual high (rmse={rmse:.2f})")
    else:
        axis_score = 0.0
        reasons.append("no BBT temperature axis detected")

    # ── scale ─────────────────────────────────────────────────────────────
    scale_score = float(np.clip(scale_confidence, 0.0, 1.0))
    if scale_score < 0.5:
        reasons.append("°C/°F scale uncertain")

    # ── markers ───────────────────────────────────────────────────────────
    bbt_n = int(present[0].sum())
    lh_n = int(present[1].sum())
    best_n = max(bbt_n, lh_n)
    # ≥10 recovered points = fully confident on coverage.
    marker_score = float(np.clip(best_n / 10.0, 0.0, 1.0))
    if best_n == 0:
        reasons.append("no data points recovered")
    elif best_n < 5:
        reasons.append(f"few data points recovered ({best_n})")

    # ── grid / plot ─────────────────────────────────────────────────────────
    grid_score = _grid_score(grid_source)
    plot_score = _plot_score(plot_method)
    if grid_source in (None, "uniform"):
        reasons.append("day grid fell back to uniform spacing")

    # A readable temperature axis is a GATE, not just a weighted signal: with
    # no axis we can't calibrate pixel→°C/°F, and spurious markers (random
    # white blobs that happen to sit near coloured pixels — e.g. a noisy
    # photo) would otherwise inflate the score. So markers only count at full
    # weight when an axis was actually fit; without one they're discounted.
    axis_present = axis_score > 0.0
    eff_marker = marker_score * (1.0 if axis_present else 0.25)

    components = {
        "axis": round(axis_score, 3),
        "scale": round(scale_score, 3),
        "markers": round(eff_marker, 3),
        "markers_raw": round(marker_score, 3),
        "grid": round(grid_score, 3),
        "plot": round(plot_score, 3),
        "bbt_points": bbt_n,
        "lh_points": lh_n,
    }
    confidence = (
        _W["axis"] * axis_score
        + _W["scale"] * scale_score
        + _W["markers"] * eff_marker
        + _W["grid"] * grid_score
        + _W["plot"] * plot_score
    )
    confidence = float(np.clip(confidence, 0.0, 1.0))

    # ── status ──────────────────────────────────────────────────────────────
    if not axis_present and best_n < 2:
        # No axis and essentially no data points → not a cycle chart.
        status = "not_a_chart"
        confidence = min(confidence, 0.15)
        reasons.append("no axis and no data points — likely not a cycle chart")
    elif not axis_present:
        # Markers but no calibratable axis: can't trust the values. Cap at
        # low_confidence so the UI prompts manual verification rather than
        # silently presenting uncalibrated numbers.
        status = "low_confidence"
        confidence = min(confidence, _LOW_CONF - 0.01)
        reasons.append("no readable temperature axis — values not calibrated")
    elif confidence < _LOW_CONF:
        status = "low_confidence"
    else:
        status = "extracted"

    if truncated:
        reasons.append(
            f"chart shows ~{visible_days} days; only the first 35 were kept"
        )

    return QualityReport(
        confidence=round(confidence, 3),
        status=status,
        components=components,
        reasons=reasons,
    )
