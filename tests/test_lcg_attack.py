"""Tests for LCG parameter recovery and prediction attacks."""

import pytest

from src.generators.lcg import (
    AnsiCLCG,
    GlibcLCG,
    NumericalRecipesLCG,
    BadLCG,
)
from src.attacks.lcg_attack import (
    LCGPredictor,
    recover_modulus_and_parameters,
    recover_parameters,
    attack_lcg,
)


# --- recover_parameters (known modulus) ---


def test_recover_parameters_known_modulus():
    """Recover (a, c) for AnsiCLCG from 3 observations."""
    gen = AnsiCLCG(seed_value=42)
    obs = gen.generate(4)
    a, c = recover_parameters(obs, gen.m)
    assert a == gen.a
    assert c == gen.c


def test_recover_parameters_different_seed():
    """Recovery works across multiple seeds and LCG variants."""
    for seed in [0, 1, 99, 2**30]:
        gen = AnsiCLCG(seed_value=seed)
        obs = gen.generate(4)
        a, c = recover_parameters(obs, gen.m)
        assert a == gen.a
        assert c == gen.c


def test_recover_parameters_insufficient_data():
    """Fewer than 3 observations raises ValueError."""
    with pytest.raises(ValueError, match="at least 3 observations"):
        recover_parameters([1, 2], 2**31)


def test_recover_parameters_non_invertible():
    """Equal consecutive values (X1 == X0 mod m) raises ValueError."""
    # Construct observations where x1 - x0 == 0 mod m (e.g. repeated values).
    # An LCG with seed where a*seed+c == seed (fixed point) would produce [X, X, ...].
    # Simpler: just call with [7, 7, 7, 7] — x1-x0=0, gcd(0, m) != 1.
    with pytest.raises(ValueError, match="not invertible"):
        recover_parameters([7, 7, 7], 101)


# --- LCGPredictor ---


def test_lcg_predictor_1000_outputs():
    """Predict 1000 outputs from AnsiCLCG and confirm all match."""
    gen = AnsiCLCG(seed_value=123)
    obs = gen.generate(10)
    a, c = recover_parameters(obs, gen.m)
    predictor = LCGPredictor(a, c, gen.m, last_state=obs[-1])
    predicted = predictor.predict(1000)
    actual = gen.generate(1000)
    assert predicted == actual


# --- recover_modulus_and_parameters (unknown modulus) ---


def test_recover_modulus_unknown():
    """Recover (m, a, c) for BadLCG from observations alone."""
    gen = BadLCG(seed_value=7)
    obs = gen.generate(10)
    m, a, c = recover_modulus_and_parameters(obs)
    assert m == gen.m
    assert a == gen.a
    assert c == gen.c


def test_recover_large_modulus():
    """Recover (m, a, c) for NumericalRecipesLCG (m=2^32) without knowing m."""
    gen = NumericalRecipesLCG(seed_value=42)
    obs = gen.generate(10)
    m, a, c = recover_modulus_and_parameters(obs)
    assert m == gen.m
    assert a == gen.a
    assert c == gen.c


def test_recover_modulus_insufficient_data():
    """Fewer than 5 observations raises ValueError."""
    with pytest.raises(ValueError, match="at least 5 observations"):
        recover_modulus_and_parameters([1, 2, 3, 4])


# --- attack_lcg end-to-end ---


def test_attack_lcg_known_modulus_full():
    """End-to-end known-modulus attack achieves 100% accuracy."""
    for cls, seed in [(AnsiCLCG, 0), (NumericalRecipesLCG, 42)]:
        gen = cls(seed_value=seed)
        result = attack_lcg(gen, num_observations=10, num_predictions=1000, modulus_known=True)
        assert result["accuracy"] == 1.0
        assert result["match_count"] == 1000


def test_attack_lcg_unknown_modulus_full():
    """End-to-end unknown-modulus attack achieves 100% accuracy."""
    for cls, seed in [(BadLCG, 7), (NumericalRecipesLCG, 42)]:
        gen = cls(seed_value=seed)
        result = attack_lcg(gen, num_observations=10, num_predictions=1000, modulus_known=False)
        assert result["accuracy"] == 1.0
        assert result["match_count"] == 1000


def test_attack_glibc_truncated_raises():
    """attack_lcg on GlibcLCG raises ValueError about truncated output."""
    gen = GlibcLCG(seed_value=42)
    with pytest.raises(ValueError, match="truncat"):
        attack_lcg(gen)
