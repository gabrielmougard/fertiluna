import json
from pathlib import Path

import numpy as np

from fertiluna.constants import CYCLE_MAX_DAYS, N_FEATURES
from fertiluna.features import extract_features
from fertiluna.synthetic import generate_cycle


def test_feature_vector_shape():
    rng = np.random.default_rng(0)
    s = generate_cycle(rng, archetype="normal")
    f = extract_features(s.temps, s.lh)
    assert f.shape == (N_FEATURES,)
    assert f.dtype == np.float32


def test_features_finite():
    rng = np.random.default_rng(0)
    for arch in ("normal", "doubtful", "anovulation", "short_luteal", "insufficient"):
        s = generate_cycle(rng, archetype=arch)
        f = extract_features(s.temps, s.lh)
        assert np.all(np.isfinite(f)), f"non-finite feature in archetype {arch}: {f}"


def test_features_nan_safe_empty_input():
    temps = np.full(CYCLE_MAX_DAYS, np.nan, dtype=np.float32)
    lh = np.full(CYCLE_MAX_DAYS, np.nan, dtype=np.float32)
    f = extract_features(temps, lh)
    assert np.all(np.isfinite(f))
    # no observations -> these specific aggregates should be exactly zero
    assert f[0] == 0  # n_temp_observed
    assert f[1] == 0  # n_lh_observed


def test_export_fixtures_for_ts_parity(tmp_path):
    """Emit fixture data for the TypeScript parity test."""
    rng = np.random.default_rng(42)
    cases = []
    for arch in ("normal", "doubtful", "anovulation", "short_luteal", "insufficient"):
        for k in range(3):
            s = generate_cycle(rng, archetype=arch)
            cases.append(
                {
                    "archetype": arch,
                    "k": k,
                    "temps": [None if np.isnan(v) else float(v) for v in s.temps],
                    "lh": [None if np.isnan(v) else float(v) for v in s.lh],
                    "expected_features": extract_features(s.temps, s.lh).tolist(),
                }
            )
    fixture_path = (
        Path(__file__).resolve().parent.parent
        / "artifacts"
        / "feature-fixtures.json"
    )
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    with open(fixture_path, "w") as f:
        json.dump(cases, f, indent=2)
    assert fixture_path.exists()
