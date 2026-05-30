"""Shared constants. Keep in sync with src/lib/constants.ts on the TS side."""

CYCLE_MAX_DAYS = 35

LABELS = [
    "ovulation_confirmee",
    "ovulation_douteuse",
    "anovulation",
    "phase_luteale_courte",
    "donnees_insuffisantes",
]
LABEL_TO_INDEX = {name: i for i, name in enumerate(LABELS)}

CONFIDENCE_THRESHOLD = 0.60

FEATURE_NAMES = [
    "n_temp_observed",
    "n_lh_observed",
    "missing_rate_temp",
    "missing_rate_lh",
    "temp_mean_overall",
    "temp_std_overall",
    "temp_min",
    "temp_max",
    "temp_range",
    "estimated_ovulation_day",
    "follicular_mean",
    "follicular_std",
    "luteal_mean",
    "luteal_std",
    "thermal_rise_amplitude",
    "rise_steepness_max",
    "plateau_days_above_baseline",
    "n_consecutive_high_days",
    "post_rise_dip_count",
    "lh_peak_day",
    "lh_peak_value",
    "lh_peak_count",
    "days_lh_peak_to_thermal_rise",
    "follicular_length",
    "luteal_length",
    "spline_slope_pre_ovulation",
    "spline_slope_post_ovulation",
    "fraction_days_below_mean",
    "longest_run_above_mean",
    "longest_run_below_mean",
]
N_FEATURES = len(FEATURE_NAMES)
