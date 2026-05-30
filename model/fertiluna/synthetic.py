"""Physiologically-grounded synthetic cycle generator.

Generates BBT (basal body temperature) and optional LH curves for one menstrual
cycle, with realistic noise, missing data, and outliers. Each curve is labeled
according to the SENSIPLAN / NICE sympto-thermal rules so the downstream model
learns those rules with noise robustness.

Archetypes covered (sampled by the `archetype` parameter):
    "normal"            — clear ovulation, normal luteal length
    "doubtful"          — borderline thermal rise or noisy plateau
    "short_luteal"      — confirmed ovulation but luteal phase 7-9 days
    "anovulation"       — no thermal shift, flat or chaotic curve
    "insufficient"      — heavy missing data (>30% NaN) or <15 measured days

Output for one cycle:
    temps         np.ndarray shape (CYCLE_MAX_DAYS,) — °C or NaN
    lh            np.ndarray shape (CYCLE_MAX_DAYS,) — relative units (0-3) or NaN
    label_idx     int — index into LABELS
    truth         dict — ground-truth annotations for analysis
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .constants import CYCLE_MAX_DAYS, LABEL_TO_INDEX

ARCHETYPES = ("normal", "doubtful", "short_luteal", "anovulation", "insufficient")
ARCHETYPE_WEIGHTS = (0.45, 0.15, 0.15, 0.15, 0.10)


@dataclass
class CycleSample:
    temps: np.ndarray
    lh: np.ndarray
    label_idx: int
    truth: dict


def _gauss(rng: np.random.Generator, mu: float, sigma: float) -> float:
    return float(rng.normal(mu, sigma))


def _make_temperature_curve(
    rng: np.random.Generator,
    follicular_len: int,
    luteal_len: int,
    baseline_temp: float,
    rise_amplitude: float,
    rise_duration: int,
    follicular_noise: float,
    luteal_noise: float,
) -> np.ndarray:
    """Build a CYCLE_MAX_DAYS-vector of temperatures with NaN past cycle end."""
    n = follicular_len + luteal_len
    temps = np.full(CYCLE_MAX_DAYS, np.nan, dtype=np.float32)

    # follicular: baseline + small downward drift just before ovulation
    for d in range(follicular_len):
        # subtle nadir 1-2 days before ovulation (real physiology)
        dip = -0.05 if d in (follicular_len - 1, follicular_len - 2) else 0.0
        temps[d] = baseline_temp + dip + _gauss(rng, 0.0, follicular_noise)

    # ovulation rise spread over 1-3 days
    for k in range(rise_duration):
        idx = follicular_len + k
        if idx < CYCLE_MAX_DAYS:
            progress = (k + 1) / rise_duration
            temps[idx] = (
                baseline_temp
                + rise_amplitude * progress
                + _gauss(rng, 0.0, follicular_noise)
            )

    # luteal plateau with gentle drift down in last 2-3 days
    plateau_start = follicular_len + rise_duration
    for k in range(luteal_len - rise_duration):
        idx = plateau_start + k
        if idx >= CYCLE_MAX_DAYS or idx >= n:
            break
        end_drift = (
            -0.05 * ((k - (luteal_len - rise_duration - 3)) / 3)
            if k >= (luteal_len - rise_duration - 3)
            else 0.0
        )
        temps[idx] = (
            baseline_temp
            + rise_amplitude
            + end_drift
            + _gauss(rng, 0.0, luteal_noise)
        )

    return temps


def _make_lh_curve(
    rng: np.random.Generator,
    ovulation_day: int,
    peak_value: float,
    has_peak: bool,
    extra_peaks: int,
) -> np.ndarray:
    """LH series, baseline ~0.8, peak 1.5-3.0 occurring 12-36h before ovulation."""
    lh = np.full(CYCLE_MAX_DAYS, np.nan, dtype=np.float32)
    for d in range(CYCLE_MAX_DAYS):
        lh[d] = max(0.0, _gauss(rng, 0.8, 0.12))

    if has_peak:
        # SENSIPLAN says LH surge precedes thermal rise by 12-36h, so peak day
        # is 1-2 days before ovulation
        peak_day = max(0, ovulation_day - int(rng.choice([1, 2])))
        if 0 <= peak_day < CYCLE_MAX_DAYS:
            lh[peak_day] = peak_value
        # shoulder days
        for offset, factor in ((-1, 0.55), (1, 0.70)):
            d = peak_day + offset
            if 0 <= d < CYCLE_MAX_DAYS:
                lh[d] = max(lh[d], peak_value * factor)

    # additional spurious peaks (PCOS pattern)
    for _ in range(extra_peaks):
        d = int(rng.integers(0, CYCLE_MAX_DAYS))
        lh[d] = max(lh[d], _gauss(rng, 1.8, 0.2))

    return lh


def _apply_missing_data(
    rng: np.random.Generator, arr: np.ndarray, missing_rate: float
) -> np.ndarray:
    mask = rng.uniform(size=arr.shape) < missing_rate
    out = arr.copy()
    out[mask] = np.nan
    return out


def _apply_outliers(rng: np.random.Generator, temps: np.ndarray) -> np.ndarray:
    """Inject realistic temperature artifacts: fever, late measurement, short night."""
    out = temps.copy()
    # 10% of cycles get a 1-2 day fever artifact (+0.3 to +0.6°C)
    if rng.uniform() < 0.10:
        start = int(rng.integers(0, max(1, CYCLE_MAX_DAYS - 2)))
        n_days = int(rng.choice([1, 2]))
        bump = float(rng.uniform(0.3, 0.6))
        for d in range(start, min(start + n_days, CYCLE_MAX_DAYS)):
            if not np.isnan(out[d]):
                out[d] += bump
    # 15% get isolated late-measurement spikes (+0.15 to +0.25°C on random days)
    if rng.uniform() < 0.15:
        n_spikes = int(rng.integers(1, 4))
        idxs = rng.choice(CYCLE_MAX_DAYS, size=n_spikes, replace=False)
        for d in idxs:
            if not np.isnan(out[d]):
                out[d] += float(rng.uniform(0.10, 0.25))
    return out


def generate_cycle(
    rng: np.random.Generator,
    archetype: Optional[str] = None,
) -> CycleSample:
    """Generate one labeled synthetic cycle.

    Args:
        rng: numpy Generator (use np.random.default_rng for reproducibility)
        archetype: one of ARCHETYPES, or None to sample by weights

    Returns:
        CycleSample with temps, lh, label_idx, and ground-truth annotations.
    """
    if archetype is None:
        archetype = str(rng.choice(ARCHETYPES, p=ARCHETYPE_WEIGHTS))

    # --- Sample physiological parameters per archetype ---
    if archetype == "normal":
        follicular_len = int(rng.integers(11, 18))   # 11-17
        luteal_len = int(rng.integers(11, 16))       # 11-15
        rise_amplitude = float(rng.uniform(0.25, 0.45))
        rise_duration = int(rng.choice([1, 2, 2, 3], p=[0.2, 0.4, 0.3, 0.1]))
        has_lh_peak = rng.uniform() < 0.95
        extra_lh_peaks = 0
        missing_temp = float(rng.uniform(0.03, 0.18))
        missing_lh = float(rng.uniform(0.40, 0.85))  # LH is optional
        label = "ovulation_confirmee"

    elif archetype == "doubtful":
        follicular_len = int(rng.integers(11, 22))
        luteal_len = int(rng.integers(10, 15))
        rise_amplitude = float(rng.uniform(0.08, 0.22))
        rise_duration = int(rng.choice([2, 3, 4]))
        has_lh_peak = rng.uniform() < 0.7
        extra_lh_peaks = int(rng.choice([0, 0, 1]))
        missing_temp = float(rng.uniform(0.10, 0.25))
        missing_lh = float(rng.uniform(0.40, 0.90))
        label = "ovulation_douteuse"

    elif archetype == "short_luteal":
        follicular_len = int(rng.integers(11, 18))
        luteal_len = int(rng.integers(7, 10))        # SHORT
        rise_amplitude = float(rng.uniform(0.22, 0.40))
        rise_duration = int(rng.choice([1, 2]))
        has_lh_peak = rng.uniform() < 0.9
        extra_lh_peaks = 0
        missing_temp = float(rng.uniform(0.05, 0.20))
        missing_lh = float(rng.uniform(0.40, 0.85))
        label = "phase_luteale_courte"

    elif archetype == "anovulation":
        # no consistent rise — model as a noisy flat or chaotic curve
        follicular_len = int(rng.integers(20, 30))
        luteal_len = int(rng.integers(0, 6))
        rise_amplitude = float(rng.uniform(0.0, 0.10))
        rise_duration = int(rng.choice([1, 2, 3]))
        has_lh_peak = rng.uniform() < 0.20    # mostly no peak, sometimes spurious
        extra_lh_peaks = int(rng.choice([0, 1, 2, 3]))
        missing_temp = float(rng.uniform(0.05, 0.20))
        missing_lh = float(rng.uniform(0.40, 0.85))
        label = "anovulation"

    elif archetype == "insufficient":
        # could be any underlying pattern but with crippling missing data
        follicular_len = int(rng.integers(11, 22))
        luteal_len = int(rng.integers(8, 16))
        rise_amplitude = float(rng.uniform(0.0, 0.40))
        rise_duration = int(rng.choice([1, 2, 3]))
        has_lh_peak = rng.uniform() < 0.5
        extra_lh_peaks = 0
        missing_temp = float(rng.uniform(0.45, 0.85))   # WAY too much missing
        missing_lh = float(rng.uniform(0.70, 1.0))
        label = "donnees_insuffisantes"

    else:
        raise ValueError(f"unknown archetype: {archetype}")

    baseline_temp = float(rng.uniform(36.05, 36.55))
    follicular_noise = float(rng.uniform(0.04, 0.08))
    luteal_noise = float(rng.uniform(0.05, 0.10))

    temps = _make_temperature_curve(
        rng,
        follicular_len=follicular_len,
        luteal_len=luteal_len,
        baseline_temp=baseline_temp,
        rise_amplitude=rise_amplitude,
        rise_duration=rise_duration,
        follicular_noise=follicular_noise,
        luteal_noise=luteal_noise,
    )
    temps = _apply_outliers(rng, temps)
    temps = _apply_missing_data(rng, temps, missing_temp)

    lh_peak_value = float(rng.uniform(1.6, 2.8))
    lh = _make_lh_curve(
        rng,
        ovulation_day=follicular_len,
        peak_value=lh_peak_value,
        has_peak=has_lh_peak,
        extra_peaks=extra_lh_peaks,
    )
    lh = _apply_missing_data(rng, lh, missing_lh)

    truth = {
        "archetype": archetype,
        "follicular_len": follicular_len,
        "luteal_len": luteal_len,
        "ovulation_day": follicular_len,        # 0-indexed
        "rise_amplitude": rise_amplitude,
        "rise_duration": rise_duration,
        "baseline_temp": baseline_temp,
        "has_lh_peak": has_lh_peak,
    }

    return CycleSample(
        temps=temps.astype(np.float32),
        lh=lh.astype(np.float32),
        label_idx=LABEL_TO_INDEX[label],
        truth=truth,
    )


def generate_dataset(
    n: int,
    seed: int = 42,
    archetype_weights: Optional[dict] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Generate `n` labeled cycles.

    Returns:
        temps:  (n, CYCLE_MAX_DAYS) float32
        lh:     (n, CYCLE_MAX_DAYS) float32
        labels: (n,) int64
        truths: list of n ground-truth dicts
    """
    rng = np.random.default_rng(seed)
    temps_out = np.empty((n, CYCLE_MAX_DAYS), dtype=np.float32)
    lh_out = np.empty((n, CYCLE_MAX_DAYS), dtype=np.float32)
    labels_out = np.empty(n, dtype=np.int64)
    truths_out: list[dict] = []

    if archetype_weights is not None:
        archetypes = list(archetype_weights.keys())
        weights = np.array([archetype_weights[k] for k in archetypes], dtype=np.float64)
        weights = weights / weights.sum()
    else:
        archetypes = list(ARCHETYPES)
        weights = np.array(ARCHETYPE_WEIGHTS, dtype=np.float64)

    drawn = rng.choice(archetypes, size=n, p=weights)
    for i in range(n):
        sample = generate_cycle(rng, archetype=str(drawn[i]))
        temps_out[i] = sample.temps
        lh_out[i] = sample.lh
        labels_out[i] = sample.label_idx
        truths_out.append(sample.truth)

    return temps_out, lh_out, labels_out, truths_out
