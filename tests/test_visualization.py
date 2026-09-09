"""Smoke tests for Stage 2 visualization functions (spec §4).

Each test builds real TestResult objects via the actual test functions
on a small MT19937(42) sample, then verifies the plot function produces
a non-empty PNG in a temporary directory.  No visual correctness is
checked — only that the file is created and has positive size.
"""

import os

import numpy as np
import pytest

from src.generators import MT19937
from src.tests.statistical_tests import (
    chi_square_test,
    autocorrelation_test,
    spectral_test_2d,
    runs_test,
    histogram_data,
    run_all_tests,
)
from src.tests.visualization import (
    plot_chi_square_comparison,
    plot_autocorrelation,
    plot_spectral,
    plot_runs_z_scores,
    plot_histograms,
    generate_all_plots,
)


@pytest.fixture()
def small_sample():
    """5 000 values from MT19937(42) — fast and large enough for all tests."""
    gen = MT19937()
    gen.seed(42)
    return gen.generate(5000)


@pytest.fixture()
def chi_result(small_sample):
    return chi_square_test(small_sample, num_bins=100, generator_name="MT19937")


@pytest.fixture()
def autocorr_result(small_sample):
    return autocorrelation_test(small_sample, generator_name="MT19937")


@pytest.fixture()
def spectral_result(small_sample):
    return spectral_test_2d(small_sample, sample_size=2000, generator_name="MT19937")


@pytest.fixture()
def runs_result(small_sample):
    return runs_test(small_sample, generator_name="MT19937")


@pytest.fixture()
def hist_result(small_sample):
    return histogram_data(small_sample, num_bins=50, generator_name="MT19937")


# PNG magic bytes: \x89PNG\r\n\x1a\n.
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _assert_valid_png(path):
    """Assert *path* is an existing, non-empty file whose first 8 bytes are the PNG magic."""
    assert os.path.isfile(path)
    assert os.path.getsize(path) > 0
    with open(path, "rb") as fh:
        assert fh.read(len(_PNG_MAGIC)) == _PNG_MAGIC


def test_plot_chi_square_comparison_creates_file(chi_result, tmp_path):
    path = plot_chi_square_comparison({"MT19937": chi_result}, output_dir=str(tmp_path))
    _assert_valid_png(path)


def test_plot_autocorrelation_creates_file(autocorr_result, tmp_path):
    path = plot_autocorrelation({"MT19937": autocorr_result}, output_dir=str(tmp_path))
    _assert_valid_png(path)


def test_plot_spectral_creates_file(spectral_result, tmp_path):
    path = plot_spectral({"MT19937": spectral_result}, output_dir=str(tmp_path))
    _assert_valid_png(path)


def test_plot_runs_creates_file(runs_result, tmp_path):
    path = plot_runs_z_scores({"MT19937": runs_result}, output_dir=str(tmp_path))
    _assert_valid_png(path)


def test_plot_histograms_creates_file(hist_result, tmp_path):
    path = plot_histograms({"MT19937": hist_result}, output_dir=str(tmp_path))
    _assert_valid_png(path)


def test_generate_all_plots_creates_multiple_files(tmp_path):
    """Exercise generate_all_plots with 2 generators and reduced sample size."""
    from src.generators import AnsiCLCG

    # Build a 2-generator result dict with small n.
    gen_mt = MT19937()
    gen_lcg = AnsiCLCG()
    results: dict[str, dict] = {}
    for gen in [gen_mt, gen_lcg]:
        name = type(gen).__name__
        results[name] = run_all_tests(
            gen, n=20_000, n_spectral=5_000, num_bins=200, seed=42
        )

    paths = generate_all_plots(results, output_dir=str(tmp_path))
    # 5 plot types × 1 PNG each = at least 5 PNG files.
    assert len(paths) >= 5
    for p in paths:
        _assert_valid_png(p)
