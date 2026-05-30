import numpy as np

from fertiluna.constants import CYCLE_MAX_DAYS, LABELS
from fertiluna.synthetic import generate_cycle, generate_dataset


def test_generate_cycle_shapes():
    rng = np.random.default_rng(0)
    sample = generate_cycle(rng, archetype="normal")
    assert sample.temps.shape == (CYCLE_MAX_DAYS,)
    assert sample.lh.shape == (CYCLE_MAX_DAYS,)
    assert 0 <= sample.label_idx < len(LABELS)


def test_generate_dataset_balanced():
    temps, lh, y, truths = generate_dataset(n=2000, seed=1)
    assert temps.shape == (2000, CYCLE_MAX_DAYS)
    assert lh.shape == (2000, CYCLE_MAX_DAYS)
    assert y.shape == (2000,)
    assert len(truths) == 2000
    # every class represented
    assert set(np.unique(y).tolist()) == set(range(len(LABELS)))


def test_archetype_distribution_realistic():
    """Normal/doubtful should be the most common."""
    temps, lh, y, _ = generate_dataset(n=5000, seed=2)
    counts = np.bincount(y, minlength=len(LABELS))
    # normal (idx 0) should be the plurality
    assert counts[0] == counts.max()


def test_anovulation_has_no_clear_rise():
    rng = np.random.default_rng(7)
    sample = generate_cycle(rng, archetype="anovulation")
    valid = ~np.isnan(sample.temps)
    if int(np.sum(valid)) > 10:
        rng_temps = sample.temps[valid]
        assert (rng_temps.max() - rng_temps.min()) < 1.5  # no fever-sized swing


def test_insufficient_has_lots_of_nans():
    rng = np.random.default_rng(11)
    sample = generate_cycle(rng, archetype="insufficient")
    nan_rate = float(np.mean(np.isnan(sample.temps)))
    assert nan_rate >= 0.30
