"""Feature extraction from raw (temps, lh) curves.

This module is the canonical implementation. The browser-side TypeScript port
(src/lib/features.ts) MUST produce identical outputs for the same input — there
is a parity test in tests/test_feature_parity.py that compares against fixtures.

Design notes:
    - NaN-safe everywhere. A missing day is missing, not zero.
    - All features are scalars (no variable-length outputs) so the model input
      is a fixed (N_FEATURES,) vector.
    - The "estimated ovulation day" feature uses the SENSIPLAN 3-over-6 rule
      (3 consecutive temps strictly above the max of the prior 6), which is the
      same rule the model is implicitly learning. This gives the model a strong
      handcrafted feature alongside the raw aggregates.
"""

from __future__ import annotations

import numpy as np

from .constants import CYCLE_MAX_DAYS, FEATURE_NAMES, N_FEATURES


def _nan_count(a: np.ndarray) -> int:
    return int(np.sum(~np.isnan(a)))


def _safe_mean(a: np.ndarray) -> float:
    if _nan_count(a) == 0:
        return 0.0
    return float(np.nanmean(a))


def _safe_std(a: np.ndarray) -> float:
    if _nan_count(a) < 2:
        return 0.0
    return float(np.nanstd(a, ddof=0))


def _detect_ovulation_day_sensiplan(temps: np.ndarray) -> int:
    """SENSIPLAN 3-over-6 rule.

    Find the first day d such that temps[d:d+3] are all strictly > max(temps[d-6:d]),
    and the third of those temps is at least 0.2 above that 6-day max.
    Returns the index of the LAST follicular day (one before the rise), or -1 if
    no rise is detected.
    """
    n = len(temps)
    for d in range(6, n - 2):
        prior = temps[d - 6 : d]
        if _nan_count(prior) < 4:
            continue
        rising = temps[d : d + 3]
        if _nan_count(rising) < 3:
            continue
        prior_max = float(np.nanmax(prior))
        if not np.all(rising > prior_max):
            continue
        if rising[2] - prior_max < 0.20:
            continue
        return d - 1   # last follicular day (0-indexed)
    return -1


def _longest_run(mask: np.ndarray) -> int:
    """Longest consecutive run of True in a 1D bool mask."""
    longest = 0
    current = 0
    for v in mask:
        if v:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _find_local_maxima(arr: np.ndarray, threshold: float) -> list[int]:
    """Indices where arr[i] > threshold and arr[i] >= both neighbors (NaN-safe)."""
    out = []
    n = len(arr)
    for i in range(n):
        v = arr[i]
        if np.isnan(v) or v <= threshold:
            continue
        left = arr[i - 1] if i > 0 else -np.inf
        right = arr[i + 1] if i < n - 1 else -np.inf
        left = -np.inf if np.isnan(left) else left
        right = -np.inf if np.isnan(right) else right
        if v >= left and v >= right:
            out.append(i)
    return out


def extract_features(temps: np.ndarray, lh: np.ndarray) -> np.ndarray:
    """Return a (N_FEATURES,) float32 vector.

    Args:
        temps: (CYCLE_MAX_DAYS,) array of temperatures (°C) with NaN for missing
        lh:    (CYCLE_MAX_DAYS,) array of LH values with NaN for missing

    Indices into the output vector are given by constants.FEATURE_NAMES.
    """
    assert temps.shape == (CYCLE_MAX_DAYS,)
    assert lh.shape == (CYCLE_MAX_DAYS,)

    f = np.zeros(N_FEATURES, dtype=np.float32)

    n_temp = _nan_count(temps)
    n_lh = _nan_count(lh)

    f[0] = n_temp
    f[1] = n_lh
    f[2] = 1.0 - n_temp / CYCLE_MAX_DAYS
    f[3] = 1.0 - n_lh / CYCLE_MAX_DAYS

    f[4] = _safe_mean(temps)
    f[5] = _safe_std(temps)
    f[6] = float(np.nanmin(temps)) if n_temp > 0 else 0.0
    f[7] = float(np.nanmax(temps)) if n_temp > 0 else 0.0
    f[8] = f[7] - f[6]

    # Detected ovulation (last follicular day, 0-indexed). +1 for human-friendly day.
    ov = _detect_ovulation_day_sensiplan(temps)
    f[9] = float(ov + 1) if ov >= 0 else 0.0

    if ov >= 0:
        follicular = temps[: ov + 1]
        luteal = temps[ov + 1 :]
        f[10] = _safe_mean(follicular)
        f[11] = _safe_std(follicular)
        f[12] = _safe_mean(luteal)
        f[13] = _safe_std(luteal)
        f[14] = max(0.0, f[12] - f[10])
        # rise steepness: max single-day jump in [ov-1, ov+3]
        steepness = 0.0
        lo = max(0, ov - 1)
        hi = min(CYCLE_MAX_DAYS - 1, ov + 3)
        for d in range(lo, hi):
            if not (np.isnan(temps[d]) or np.isnan(temps[d + 1])):
                steepness = max(steepness, float(temps[d + 1] - temps[d]))
        f[15] = steepness
        # plateau days: count of post-ovulation days above follicular_mean + 0.15
        plateau_threshold = f[10] + 0.15
        post = luteal[~np.isnan(luteal)]
        f[16] = float(np.sum(post > plateau_threshold))
        # longest consecutive run above plateau threshold
        post_mask_full = ~np.isnan(luteal) & (luteal > plateau_threshold)
        f[17] = _longest_run(post_mask_full)
        # post-rise dips (luteal days dropping below follicular max)
        foll_max = float(np.nanmax(follicular)) if _nan_count(follicular) > 0 else 0.0
        dips = post < foll_max
        f[18] = float(np.sum(dips))
        f[23] = float(ov + 1)
        f[24] = float(np.sum(~np.isnan(luteal)))
    else:
        # No detected rise → leave most fields at 0; report descriptive aggregates only.
        f[10] = _safe_mean(temps)
        f[11] = _safe_std(temps)
        f[12] = 0.0
        f[13] = 0.0
        f[14] = 0.0
        f[15] = 0.0
        f[16] = 0.0
        f[17] = 0.0
        f[18] = 0.0
        f[23] = 0.0
        f[24] = 0.0

    # LH features
    if n_lh > 0:
        lh_baseline = float(np.nanmedian(lh))
        peaks = _find_local_maxima(lh, threshold=max(1.3, lh_baseline * 1.5))
        if peaks:
            # Highest peak
            best = max(peaks, key=lambda i: lh[i])
            f[19] = float(best + 1)        # 1-indexed day
            f[20] = float(lh[best])
            f[21] = float(len(peaks))
            if ov >= 0:
                f[22] = float((ov + 1) - (best + 1))
            else:
                f[22] = 0.0
        else:
            f[19] = 0.0
            f[20] = float(np.nanmax(lh))
            f[21] = 0.0
            f[22] = 0.0
    else:
        f[19] = 0.0
        f[20] = 0.0
        f[21] = 0.0
        f[22] = 0.0

    # Slopes around (estimated) ovulation
    if ov >= 0:
        pre_window = temps[max(0, ov - 4) : ov + 1]
        post_window = temps[ov + 1 : min(CYCLE_MAX_DAYS, ov + 5)]
        f[25] = _slope(pre_window)
        f[26] = _slope(post_window)
    else:
        f[25] = _slope(temps[:10])
        f[26] = _slope(temps[10:20])

    if n_temp > 0:
        overall_mean = f[4]
        valid = ~np.isnan(temps)
        below = (temps < overall_mean) & valid
        above = (temps >= overall_mean) & valid
        f[27] = float(np.sum(below)) / max(1, n_temp)
        f[28] = float(_longest_run(above))
        f[29] = float(_longest_run(below))
    else:
        f[27] = 0.0
        f[28] = 0.0
        f[29] = 0.0

    assert f.shape == (N_FEATURES,)
    return f


def _slope(arr: np.ndarray) -> float:
    """Least-squares slope over NaN-filtered (x, y) where x is day index."""
    valid = ~np.isnan(arr)
    if int(np.sum(valid)) < 2:
        return 0.0
    x = np.arange(len(arr), dtype=np.float64)[valid]
    y = arr[valid].astype(np.float64)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = float(np.sum((x - x_mean) ** 2))
    if denom == 0.0:
        return 0.0
    return float(np.sum((x - x_mean) * (y - y_mean)) / denom)


def batch_extract(temps: np.ndarray, lh: np.ndarray) -> np.ndarray:
    """Apply extract_features to (n, CYCLE_MAX_DAYS) inputs. Returns (n, N_FEATURES)."""
    assert temps.ndim == 2 and lh.ndim == 2
    n = temps.shape[0]
    out = np.empty((n, N_FEATURES), dtype=np.float32)
    for i in range(n):
        out[i] = extract_features(temps[i], lh[i])
    return out


# Re-export names so callers don't have to import constants separately.
__all__ = ["extract_features", "batch_extract", "FEATURE_NAMES", "N_FEATURES"]
