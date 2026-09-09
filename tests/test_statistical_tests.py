"""Unit tests for the Stage 2 statistical test functions.

Tests use deterministic data from random.Random and numpy only — no new
dependencies. Each test exercises one spec §4 case with real behavior.
"""

import random

import numpy as np

from src.tests.statistical_tests import (
    TestResult,
    chi_square_test,
    autocorrelation_test,
    spectral_test_2d,
    runs_test,
    histogram_data,
)


def _uniform_ints(n: int, seed: int = 42) -> list[int]:
    """Deterministic 32-bit uniform integer sample for tests."""
    rng = random.Random(seed)
    return [rng.getrandbits(32) for _ in range(n)]


def test_chi_square_uniform_input():
    """Uniform random integers pass the chi-square uniformity test (p > 0.01)."""
    values = _uniform_ints(10_000)
    result = chi_square_test(values, generator_name="Test")
    assert result.passed
    assert result.p_value > 0.01


def test_chi_square_nonuniform_input():
    """A constant (all-7) sequence fails uniformity (p < 0.01), as expected."""
    values = [7] * 10_000
    result = chi_square_test(values, generator_name="Test")
    assert not result.passed
    assert result.p_value < 0.01


def test_chi_square_adjusts_bins_small_input():
    """Too few values for the requested bins triggers bin reduction."""
    values = _uniform_ints(50)
    result = chi_square_test(values, num_bins=1000, generator_name="Test")
    assert result.details["num_bins"] == 5  # 50 // 10
    assert "reduced num_bins" in result.comment


def test_autocorrelation_zero_lag1():
    """Independent uniform ints have near-zero lag-1 correlation."""
    values = _uniform_ints(10_000)
    result = autocorrelation_test(values, generator_name="Test")
    assert abs(result.details["correlations"][1]) < 2 / np.sqrt(len(values))
    assert result.passed


def test_autocorrelation_detects_pattern():
    """Alternating values show strong negative lag-1 correlation -> failure."""
    values = [0 if i % 2 == 0 else 1 for i in range(10_000)]
    result = autocorrelation_test(values, lags=[1], generator_name="Test")
    assert abs(result.details["correlations"][1]) > 0.9
    assert not result.passed


def test_spectral_returns_pairs():
    """Spectral test returns sample_size normalized pairs in [0, 1)."""
    values = _uniform_ints(1001)
    sample_size = 1000
    result = spectral_test_2d(values, sample_size=sample_size, generator_name="Test")
    assert len(result.details["pairs_x"]) == sample_size
    assert len(result.details["pairs_y"]) == sample_size
    assert all(0.0 <= px < 1.0 for px in result.details["pairs_x"])
    assert all(0.0 <= py < 1.0 for py in result.details["pairs_y"])


def test_runs_test_fair_coin():
    """Random bits produce |Z| < 1.96 -> independence not rejected."""
    values = _uniform_ints(10_000)
    result = runs_test(values, generator_name="Test")
    assert abs(result.details["z"]) < 1.96
    assert result.passed


def test_runs_test_alternating():
    """Perfectly alternating bits maximize run count -> large |Z|, failure."""
    values = [i % 2 for i in range(10_000)]
    result = runs_test(values, generator_name="Test")
    assert abs(result.details["z"]) > 1.96
    assert not result.passed


def test_histogram_counts_match_input():
    """Histogram bin counts sum to the full input length."""
    values = _uniform_ints(2000)
    result = histogram_data(values, num_bins=200, generator_name="Test")
    assert sum(result.details["bin_counts"]) == len(values)


def test_test_result_dataclass():
    """Every TestResult field is populated with the correct type."""
    values = _uniform_ints(10_000)
    result = chi_square_test(values, generator_name="Test")
    assert isinstance(result, TestResult)
    assert isinstance(result.test_name, str)
    assert isinstance(result.generator_name, str)
    assert isinstance(result.statistic, float)
    assert isinstance(result.p_value, float)
    assert isinstance(result.passed, bool)
    assert isinstance(result.details, dict)
    assert isinstance(result.comment, str)
