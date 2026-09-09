"""Plot generation functions for Stage 2 statistical test results.

Produces publication-quality PNG plots: chi-square p-value comparison,
autocorrelation line plots, spectral 2D scatter (2x3 grid), runs Z-scores,
and histogram overlays. All functions write to a configurable output directory
and return the absolute path to the saved file.

Style: seaborn-v0_8-whitegrid with tab10 colour mapping by sorted generator
name for cross-plot consistency.
"""

import math
import os
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .statistical_tests import TestResult

# Canonical figure ordering mandated by spec §3.2: LCG variants first, then
# MT19937, xorshift variants, V8, and PCG.  Any generator not listed here is
# appended after the known families (never silently dropped).
BASE = [
    "AnsiCLCG",
    "NumericalRecipesLCG",
    "GlibcLCG",
    "BadLCG",
    "MT19937",
    "XorShift32",
    "XorShift64",
    "XorShift128Plus",
    "V8Random",
    "PCG32",
]


def _sort_key(name: str) -> tuple[int, str]:
    """Rank *name* by spec ordering, then alphabetically as a tiebreak.

    Unknown names get a group index larger than every known family so they
    sort last but are still included.
    """
    ordinal = BASE.index(name) if name in BASE else len(BASE) + 1
    return (0 if ordinal <= len(BASE) else 1, name)


def _ordered_names(results: dict[str, TestResult]) -> list[str]:
    """Return generator names in canonical spec order (unknown names last)."""
    return sorted(results.keys(), key=_sort_key)

# Consistent colour assignment: sorted generator names → tab10 slots.
# Built lazily per call but deterministic across calls when keys match.


def _build_color_map(generator_names: list[str]) -> dict[str, str]:
    """Map each generator name to a tab10 colour by sorted-name index."""
    cmap = plt.cm.tab10
    sorted_names = sorted(set(generator_names))
    return {
        name: cmap(i % cmap.N)
        for i, name in enumerate(sorted_names)
    }


def _get_style_context():
    """Return a style context manager; falls back to default if style missing."""
    try:
        return plt.style.context("seaborn-v0_8-whitegrid")
    except OSError:
        return plt.style.context("default")


def _save_and_close(fig: plt.Figure, path: str, dpi: int = 150) -> str:
    """Save figure to *path* at *dpi*, close it, and return the absolute path."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(path)


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------


def plot_chi_square_comparison(
    results: dict[str, TestResult],
    output_dir: str = "results/figures",
    dpi: int = 150,
    figsize: tuple[int, int] = (10, 8),
) -> str:
    """Bar chart comparing chi-square p-values across generators.

    Horizontal dashed lines mark the 0.01 and 0.99 pass-band thresholds.
    Generators outside that band fail the uniformity test.

    Returns the absolute path to the saved PNG.
    """
    os.makedirs(output_dir, exist_ok=True)
    color_map = _build_color_map(list(results.keys()))

    with _get_style_context():
        fig, ax = plt.subplots(figsize=figsize)
        names = _ordered_names(results)
        p_vals = [results[n].p_value for n in names]
        colours = [color_map[n] for n in names]

        ax.bar(names, p_vals, color=colours, edgecolor="black", linewidth=0.5)
        ax.set_ylabel("P-value")
        ax.set_title("Chi-Square P-values — Uniformity Test")
        ax.axhline(0.01, color="red", linestyle="--", linewidth=0.8, label="0.01")
        ax.axhline(0.99, color="red", linestyle="--", linewidth=0.8, label="0.99")
        ax.legend()
        ax.tick_params(axis="x", rotation=45)

    return _save_and_close(fig, os.path.join(output_dir, "chi_square_comparison.png"), dpi)


def plot_autocorrelation(
    results: dict[str, TestResult],
    output_dir: str = "results/figures",
    dpi: int = 150,
    figsize: tuple[int, int] = (10, 8),
) -> str:
    """Line plot of autocorrelation r vs lag for each generator.

    A horizontal dashed band at ±2/sqrt(N) indicates the 95 % confidence
    region for uncorrelated noise.  Lines that escape the band signal
    dependence at those lags.

    Returns the absolute path to the saved PNG.
    """
    os.makedirs(output_dir, exist_ok=True)
    color_map = _build_color_map(list(results.keys()))

    with _get_style_context():
        fig, ax = plt.subplots(figsize=figsize)
        # Collect all lags seen across results so the x-axis covers everything.
        all_lags: set[int] = set()
        for tr in results.values():
            all_lags.update(tr.details.get("correlations", {}).keys())

        # Draw the ±2/sqrt(N) band using the first result's N (all generators
        # are tested on the same sample size within a single run_all_tests call,
        # but when called standalone the N may differ — use the max N).
        max_n = max(
            (tr.details.get("n", 0) for tr in results.values()),
            default=1,
        )
        band = 2.0 / np.sqrt(max_n)
        ax.axhline(band, color="grey", linestyle="--", linewidth=0.8, label=f"±2/√N (N={max_n})")
        ax.axhline(-band, color="grey", linestyle="--", linewidth=0.8)

        for name in _ordered_names(results):
            tr = results[name]
            corrs = tr.details.get("correlations", {})
            if not corrs:
                continue
            lags = sorted(corrs.keys())
            r_vals = [corrs[l] for l in lags]
            ax.plot(lags, r_vals, marker="o", markersize=4, label=name, color=color_map[name])

        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation r")
        ax.set_title("Autocorrelation by Lag")
        ax.legend(fontsize="small")

    return _save_and_close(fig, os.path.join(output_dir, "autocorrelation.png"), dpi)


def plot_spectral(
    spectral_results: dict[str, TestResult],
    output_dir: str = "results/figures",
    dpi: int = 150,
    figsize: tuple[int, int] = (10, 8),
) -> str:
    """Scatter grid (2x3 by default) of (X_n, X_{n+1}) pairs for each generator.

    Generators whose name contains 'LCG' are labelled "Structured"; all
    others are labelled "Random".  Small markers and low alpha keep the
    100k-point scatter readable.  The grid grows to fit more than six
    generators instead of silently hiding them.

    Returns the absolute path to the saved PNG.
    """
    if not spectral_results:
        raise ValueError("spectral_results is empty")

    os.makedirs(output_dir, exist_ok=True)

    names = _ordered_names(spectral_results)
    n_plots = len(names)
    n_cols = min(max(n_plots, 1), 3)
    n_rows = math.ceil(n_plots / n_cols) if n_plots else 1

    with _get_style_context():
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        flat_axes = np.asarray(axes).flatten()

        for idx, ax in enumerate(flat_axes):
            if idx >= n_plots:
                ax.set_visible(False)
                continue
            name = names[idx]
            tr = spectral_results[name]
            pairs_x = np.asarray(tr.details["pairs_x"])
            pairs_y = np.asarray(tr.details["pairs_y"])

            # "LCG" anywhere in the generator name → Structured
            tag = "Structured" if "LCG" in name.upper() else "Random"
            ax.scatter(pairs_x, pairs_y, s=0.1, alpha=0.3, edgecolors="none")
            ax.set_aspect("equal")
            ax.set_xlabel("X$_{n}$")
            ax.set_ylabel("X$_{n+1}$")
            ax.set_title(f"{name} ({tag})")

    fig.suptitle("Spectral Test — 2D Consecutive Pairs", y=1.01)
    return _save_and_close(fig, os.path.join(output_dir, "spectral.png"), dpi)


def plot_runs_z_scores(
    results: dict[str, TestResult],
    output_dir: str = "results/figures",
    dpi: int = 150,
    figsize: tuple[int, int] = (10, 8),
) -> str:
    """Bar chart of |Z| from the runs test for each generator.

    A horizontal dashed line at +1.96 marks the 95 % confidence boundary
    (bars are |Z|, so only the positive threshold is reachable).

    Returns the absolute path to the saved PNG.
    """
    os.makedirs(output_dir, exist_ok=True)
    color_map = _build_color_map(list(results.keys()))

    with _get_style_context():
        fig, ax = plt.subplots(figsize=figsize)
        names = _ordered_names(results)
        z_scores = [abs(results[n].details.get("z", 0.0)) for n in names]
        colours = [color_map[n] for n in names]

        ax.bar(names, z_scores, color=colours, edgecolor="black", linewidth=0.5)
        ax.set_ylabel("|Z|")
        ax.set_title("Runs Test — |Z| Scores")
        ax.axhline(1.96, color="red", linestyle="--", linewidth=0.8, label="1.96 (95% threshold)")
        ax.legend()
        ax.tick_params(axis="x", rotation=45)

    return _save_and_close(fig, os.path.join(output_dir, "runs_z_scores.png"), dpi)


def plot_histograms(
    hist_results: dict[str, TestResult],
    output_dir: str = "results/figures",
    dpi: int = 150,
    figsize: tuple[int, int] = (10, 8),
) -> str:
    """Overlay histograms of normalised value distributions.

    All generators share one axes for direct comparison.  Semilog y-scale
    helps reveal tail behaviour.

    Returns the absolute path to the saved PNG.
    """
    os.makedirs(output_dir, exist_ok=True)
    color_map = _build_color_map(list(hist_results.keys()))

    with _get_style_context():
        fig, ax = plt.subplots(figsize=figsize)

        for name in _ordered_names(hist_results):
            tr = hist_results[name]
            bin_edges = np.asarray(tr.details["bin_edges"])
            bin_counts = np.asarray(tr.details["bin_counts"])
            # centres for bar-width = bin width
            centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            ax.bar(
                centres,
                bin_counts,
                width=bin_edges[1] - bin_edges[0],
                alpha=0.5,
                label=name,
                color=color_map[name],
            )

        ax.set_yscale("log")
        ax.set_xlabel("Normalised value")
        ax.set_ylabel("Count")
        ax.set_title("Histogram Overlay — Value Distribution")
        ax.legend(fontsize="small")

    return _save_and_close(fig, os.path.join(output_dir, "histograms.png"), dpi)


# ---------------------------------------------------------------------------
# Aggregate plot generator
# ---------------------------------------------------------------------------


def generate_all_plots(
    generator_results: dict[str, dict[str, TestResult]],
    output_dir: str = "results/figures",
) -> list[str]:
    """Generate all five plot types from per-generator test results.

    Parameters
    ----------
    generator_results:
        ``{generator_name: {test_name: TestResult}}`` — typically produced by
        calling ``run_all_tests`` for each generator and merging the dicts.
    output_dir:
        Directory where PNGs are written.  Created if it does not exist.

    Returns
    -------
    list[str]
        Absolute paths to every PNG file written.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths: list[str] = []

    # Collect per-test-name dicts from the nested structure.
    chi_square_map: dict[str, TestResult] = {}
    autocorr_map: dict[str, TestResult] = {}
    spectral_map: dict[str, TestResult] = {}
    runs_map: dict[str, TestResult] = {}
    hist_map: dict[str, TestResult] = {}

    for gen_name, test_dict in generator_results.items():
        if "chi_square" in test_dict:
            chi_square_map[gen_name] = test_dict["chi_square"]
        if "autocorrelation" in test_dict:
            autocorr_map[gen_name] = test_dict["autocorrelation"]
        if "spectral" in test_dict:
            spectral_map[gen_name] = test_dict["spectral"]
        if "runs" in test_dict:
            runs_map[gen_name] = test_dict["runs"]
        if "histogram" in test_dict:
            hist_map[gen_name] = test_dict["histogram"]

    if chi_square_map:
        paths.append(plot_chi_square_comparison(chi_square_map, output_dir))
    if autocorr_map:
        paths.append(plot_autocorrelation(autocorr_map, output_dir))
    if spectral_map:
        paths.append(plot_spectral(spectral_map, output_dir))
    if runs_map:
        paths.append(plot_runs_z_scores(runs_map, output_dir))
    if hist_map:
        paths.append(plot_histograms(hist_map, output_dir))

    return paths
