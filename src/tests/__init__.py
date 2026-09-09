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
from .visualization import (
    plot_chi_square_comparison,
    plot_autocorrelation,
    plot_spectral,
    plot_runs_z_scores,
    plot_histograms,
    generate_all_plots,
)

__all__ = [
    "TestResult",
    "chi_square_test",
    "autocorrelation_test",
    "spectral_test_2d",
    "runs_test",
    "histogram_data",
    "run_all_tests",
    "plot_chi_square_comparison",
    "plot_autocorrelation",
    "plot_spectral",
    "plot_runs_z_scores",
    "plot_histograms",
    "generate_all_plots",
]
