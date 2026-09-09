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
    run_all_tests,
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


# --- Regression tests for degenerate paths (Issue 8) ---


def test_chi_square_too_few_values_raises():
    """Chi-square on 1 value raises ValueError; on 10 values (1 bin) too."""
    import pytest

    with pytest.raises(ValueError, match="at least 2 values"):
        chi_square_test([42], generator_name="Test")

    with pytest.raises(ValueError, match="insufficient samples"):
        chi_square_test([1] * 10, num_bins=1000, generator_name="Test")


def test_autocorrelation_constant_input_no_nan():
    """Constant list -> passed=False, no exceptions, no RuntimeWarning."""
    values = [7] * 1000
    result = autocorrelation_test(values, generator_name="Test")
    assert result.passed is False
    assert np.isnan(result.p_value)
    assert np.isinf(result.statistic)


def test_autocorrelation_lag_too_large_raises():
    """Lag >= n raises ValueError."""
    import pytest

    with pytest.raises(ValueError, match="lag.*>= number of samples"):
        autocorrelation_test([1, 2, 3], lags=[3], generator_name="Test")


def test_modulus_explicit_badlcg():
    """BadLCG(42).generate(1000) with modulus=101 — proves modulus fix.

    Without explicit modulus, bit-length inference gives modulus=128 →
    quantization error → p=0.0. With modulus=101, p > 0.0 (values truly
    uniform over their 101-domain). We use few bins so the expected count
    per bin isn't pathologically small.
    """
    from src.generators import BadLCG

    gen = BadLCG()
    gen.seed(42)
    values = gen.generate(1000)
    result = chi_square_test(values, modulus=101, num_bins=10, generator_name="BadLCG")
    # The key assertion: modulus fix prevents the false p=0.0 from
    # bit-length misinference (128 instead of 101).
    assert result.p_value > 0.0, (
        f"modulus=101 should yield p>0.0, got p={result.p_value}"
    )


def test_run_all_tests_ansi_lcg_normalizes_by_output_domain():
    """AnsiCLCG (m=2^31) must pass chi-square; dividing by 2^32 would
    cluster all values in the lower half-range and falsify the result."""
    from src.generators import AnsiCLCG

    results = run_all_tests(AnsiCLCG(), n=200_000, n_spectral=10_000, seed=0)
    chi = results["chi_square"]
    assert 0.01 <= chi.p_value <= 0.99, (
        f"AnsiCLCG should pass uniformity with correct modulus, got p={chi.p_value}"
    )


def test_run_all_tests_glibc_normalizes_by_output_domain():
    """GlibcLCG outputs 15 bits; the chi-square statistic must stay in a
    meaningful range, not saturate from division by 2^32."""
    from src.generators import GlibcLCG

    results = run_all_tests(GlibcLCG(), n=200_000, n_spectral=10_000, seed=0)
    chi = results["chi_square"]
    assert chi.statistic < 1_000_000, (
        f"GlibcLCG statistic {chi.statistic} indicates normalization artifact"
    )


def test_run_all_tests_badlcg_normalizes_by_output_domain():
    """BadLCG (m=101) must fail uniformity on the merits of its tiny domain,
    with values spread over its 101-value domain rather than collapsed into a
    single bin by an oversized normalization modulus."""
    from src.generators import BadLCG

    results = run_all_tests(BadLCG(), n=200_000, n_spectral=10_000, seed=0)
    chi = results["chi_square"]
    occupied_bins = sum(1 for c in chi.details["bin_counts"] if c > 0)
    assert occupied_bins >= 50, (
        f"BadLCG values collapsed into {occupied_bins} bins; "
        "normalization modulus is too large"
    )
    assert not chi.passed, "BadLCG should genuinely fail uniformity"


def test_run_all_tests_integration():
    """run_all_tests returns correct structure and chi_square passes."""
    from src.generators import PCG32

    gen = PCG32()
    results = run_all_tests(gen, n=200_000, n_spectral=10_000, seed=0)

    assert set(results.keys()) == {
        "chi_square", "autocorrelation", "spectral", "runs", "histogram"
    }
    for name, result in results.items():
        assert isinstance(result, TestResult)
        assert result.test_name == name or (
            name == "spectral" and result.test_name == "spectral"
        )
        assert result.generator_name == "PCG32"
        assert isinstance(result.passed, bool)

    assert results["chi_square"].passed
