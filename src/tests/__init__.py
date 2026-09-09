"""Statistical test and visualization package for PRNG output (Stage 2)."""

from .statistical_tests import (
    TestResult,
    chi_square_test,
    autocorrelation_test,
    spectral_test_2d,
    runs_test,
    histogram_data,
    run_all_tests,
)

__all__ = [
    "TestResult",
    "chi_square_test",
    "autocorrelation_test",
    "spectral_test_2d",
    "runs_test",
    "histogram_data",
    "run_all_tests",
]
