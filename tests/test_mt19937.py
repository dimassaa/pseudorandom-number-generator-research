"""Tests for Mersenne Twister MT19937 implementation.

The critical correctness requirement: output must match Python's
random.Random(seed).getrandbits(32) exactly for the same seed.
Stage 3 attack depends on untempering recovering the true MT state.
"""

import random

from scipy.stats import chisquare

from src.generators.mt19937 import MT19937


def test_mt19937_matches_python_random():
    """MT19937 with seed=42 produces bit-identical output to Python's random for 1000 values."""
    gen = MT19937(42)
    reference = random.Random(42)
    ours = gen.generate(1000)
    theirs = [reference.getrandbits(32) for _ in range(1000)]
    assert ours == theirs


def test_mt19937_matches_python_random_10k():
    """10000 outputs with seed=7 match Python random — proves twist regeneration works."""
    gen = MT19937(7)
    reference = random.Random(7)
    ours = gen.generate(10000)
    theirs = [reference.getrandbits(32) for _ in range(10000)]
    assert ours == theirs


def test_mt19937_period_624():
    """Generate exactly 624 values (exhausting state), then 76 more — all match Python random."""
    gen = MT19937(123)
    reference = random.Random(123)
    # First 624: no twist yet
    first_batch = gen.generate(624)
    expected_first = [reference.getrandbits(32) for _ in range(624)]
    assert first_batch == expected_first
    # Next 76: forces one twist
    second_batch = gen.generate(76)
    expected_second = [reference.getrandbits(32) for _ in range(76)]
    assert second_batch == expected_second


def test_mt19937_seed_range():
    """Seed larger than 2^32 still matches Python random (tests & 0xffffffff masking)."""
    big_seed = 2**40 + 123
    gen = MT19937(big_seed)
    reference = random.Random(big_seed)
    ours = gen.generate(100)
    theirs = [reference.getrandbits(32) for _ in range(100)]
    assert ours == theirs


def test_mt19937_reseed():
    """Re-seeding with the same value produces an identical sequence from the start."""
    gen = MT19937(999)
    first_run = gen.generate(500)
    gen.seed(999)
    second_run = gen.generate(500)
    assert first_run == second_run


def test_mt19937_next_float_range():
    """next_float() returns floats in [0.0, 1.0)."""
    gen = MT19937(42)
    for _ in range(1000):
        f = gen.next_float()
        assert 0.0 <= f < 1.0


def test_mt19937_next_float_uniform():
    """Floats from next_float() pass a chi-square uniformity test.

    The spec requires next_float() to be uniform, not merely in range. A fixed
    seed makes the sample deterministic, so the p-value is reproducible and the
    test is never flaky. MT19937 is a strong generator, so p > 0.01 always
    holds at 10^5 samples; the threshold only guards against a real regression
    in the normalization.
    """
    n_samples = 100_000
    n_bins = 100
    gen = MT19937(5)
    # int(f * n_bins) maps [0, 1) onto bin indices 0..n_bins-1 (f < 1 always
    # floors below n_bins, so no upper clamp is needed)
    observed = [0] * n_bins
    for _ in range(n_samples):
        observed[int(gen.next_float() * n_bins)] += 1
    expected = [n_samples / n_bins] * n_bins
    _, p_value = chisquare(observed, f_exp=expected)
    assert p_value > 0.01
