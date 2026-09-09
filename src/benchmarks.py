"""Benchmark runner and visualization for PRNG throughput comparison.

Measures outputs/second for all 10 project generators plus Python's built-in
``random`` (C-implemented) as a baseline.  Results are saved as JSON with full
environment metadata and visualized as a log-scale bar chart.

Design rationale — pure-Python loops (no numba):
    Only the Stage 4 brute-force attack uses numba for JIT acceleration.
    This benchmark measures *standard* generation throughput in a plain Python
    loop so that the timings reflect real-world per-call overhead including the
    Python interpreter.  Wrapping the loop in numba would mask the very
    interpreter cost we want to measure.

Design rationale — ``mean_mbps`` key name:
    The spec mandates the key ``mean_mbps`` but its value is *outputs per
    second*, not megabits per second.  The name is kept for spec compliance
    but documented in every docstring to prevent misinterpretation.
"""

import json
import os
import platform
import random
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.generators import (
    AnsiCLCG,
    BadLCG,
    GlibcLCG,
    MT19937,
    NumericalRecipesLCG,
    PCG32,
    V8Random,
    XorShift128Plus,
    XorShift32,
    XorShift64,
)

# ---------------------------------------------------------------------------
# Generator registry (spec §1.2) — 10 entries mapping name → zero-arg factory
# ---------------------------------------------------------------------------

GENERATORS = {
    "lcg_ansi_c": lambda: AnsiCLCG(),
    "lcg_numerical_recipes": lambda: NumericalRecipesLCG(),
    "lcg_glibc": lambda: GlibcLCG(),
    "lcg_bad": lambda: BadLCG(),
    "mt19937": lambda: MT19937(),
    "xorshift32": lambda: XorShift32(),
    "xorshift64": lambda: XorShift64(),
    "xorshift128+": lambda: XorShift128Plus(),
    "v8_math_random": lambda: V8Random(),
    "pcg32": lambda: PCG32(),
}


def time_generator(
    gen_factory,
    n: int = 10_000_000,
    repeats: int = 3,
    seed: int = 42,
) -> dict:
    """Time a single generator's throughput over multiple runs.

    Each repeat creates a fresh generator via ``gen_factory()``, warms it up
    with 1 000 calls (to trigger lazy init such as MT19937's twist), then
    measures ``n`` next_int() calls in a pure-Python loop.  Values are
    discarded to avoid ~80 MB list allocation that would bias timing.

    Args:
        gen_factory: Callable returning a fresh generator instance (zero-arg).
        n: Number of outputs to generate per run.
        repeats: Number of independent timing repetitions.
        seed: Seed applied to each fresh generator via ``.seed(seed)``.

    Returns:
        Dict with keys:
            mean_s          — mean elapsed seconds across repeats.
            std_s           — sample std-dev of elapsed seconds.
            mean_ns_per_output — mean_s * 1e9 / n  (nanoseconds per output).
            mean_mbps       — n / mean_s  (**outputs per second**, despite the
                              spec-mandated key name "mbps"; do NOT interpret
                              as megabits per second).

    Raises:
        ValueError: If ``n < 1`` or ``repeats < 1``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")

    timings = []
    for _ in range(repeats):
        gen = gen_factory()
        gen.seed(seed)
        # Warm-up: 1 000 calls discarded.  Triggers MT19937 twist and any
        # other lazy-init that would otherwise pollute the first timed run.
        for _ in range(1000):
            gen.next_int()

        start = time.perf_counter()
        for _ in range(n):
            gen.next_int()
        end = time.perf_counter()
        timings.append(end - start)

    mean_s = float(np.mean(timings))
    std_s = float(np.std(timings, ddof=1)) if repeats > 1 else 0.0
    mean_ns_per_output = mean_s * 1e9 / n
    # Despite the name, this is outputs per second (spec-mandated key name).
    mean_mbps = n / mean_s

    return {
        "mean_s": mean_s,
        "std_s": std_s,
        "mean_ns_per_output": mean_ns_per_output,
        "mean_mbps": mean_mbps,
    }


def time_builtin_random(
    n: int = 10_000_000,
    repeats: int = 3,
    seed: int = 42,
) -> dict:
    """Time ``random.Random(seed).getrandbits(32)`` as the C-implemented baseline.

    Uses identical methodology to ``time_generator``: fresh ``Random`` instance
    per repeat, warm-up discarded, pure-Python loop.

    Args:
        n: Number of outputs per run.
        repeats: Number of independent timing repetitions.
        seed: Seed for the random.Random instance.

    Returns:
        Dict with the same keys as ``time_generator`` (see its docstring).
        ``mean_mbps`` is outputs/second despite the key name (spec mandate).

    Raises:
        ValueError: If ``n < 1`` or ``repeats < 1``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")

    timings = []
    for _ in range(repeats):
        rng = random.Random(seed)
        # Warm-up
        for _ in range(1000):
            rng.getrandbits(32)

        start = time.perf_counter()
        for _ in range(n):
            rng.getrandbits(32)
        end = time.perf_counter()
        timings.append(end - start)

    mean_s = float(np.mean(timings))
    std_s = float(np.std(timings, ddof=1)) if repeats > 1 else 0.0
    mean_ns_per_output = mean_s * 1e9 / n
    # Despite the name, this is outputs per second (spec-mandated key name).
    mean_mbps = n / mean_s

    return {
        "mean_s": mean_s,
        "std_s": std_s,
        "mean_ns_per_output": mean_ns_per_output,
        "mean_mbps": mean_mbps,
    }


def run_benchmarks(
    n: int = 10_000_000,
    repeats: int = 3,
    seed: int = 42,
    output_file: str = "results/benchmark.json",
) -> dict:
    """Run all benchmarks (10 project generators + built-in random), save JSON.

    The built-in ``random`` result is stored inside ``results["generator"]``
    under the key ``"builtin_random"`` alongside the 10 project generators,
    so every entry shares the same data structure.  An additional top-level key
    ``results["builtin_random_name"]`` = ``"builtin_random"`` allows the plot
    function (and any consumer) to distinguish the C baseline.

    Results also include ``results["n"]``, ``results["repeats"]``, and
    ``results["seed"]`` so that the plot title and JSON are self-documenting.

    Args:
        n: Number of outputs per generator per run.
        repeats: Number of independent timing repetitions.
        seed: Seed for all generators and the built-in baseline.
        output_file: Path to write the JSON results file.  Parent directories
            are created if they don't exist.

    Returns:
        The full results dict (also written to ``output_file``).
    """
    results: dict = {}
    results["n"] = n
    results["repeats"] = repeats
    results["seed"] = seed

    environment = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    results["environment"] = environment
    results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    generator_results: dict = {}
    for name, factory in GENERATORS.items():
        generator_results[name] = time_generator(factory, n=n, repeats=repeats, seed=seed)

    # Built-in random stored inside the same generator dict for uniform access.
    builtin_name = "builtin_random"
    generator_results[builtin_name] = time_builtin_random(
        n=n, repeats=repeats, seed=seed
    )

    results["generator"] = generator_results
    results["builtin_random_name"] = builtin_name

    # Persist to disk — create parent directories if needed.
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# Visualization (spec §2)
# ---------------------------------------------------------------------------


def _throughput_error(std_s: float, mean_s: float, n: int) -> float:
    """Propagate time std-dev to throughput std-dev via first-order Taylor expansion.

    Throughput ``R = n / T``.  By standard error propagation for a function
    ``f(T) = n / T``:

        σ_R ≈ |df/dT| · σ_T = (n / T²) · σ_T

    This is a first-order (linear) approximation that is accurate when
    σ_T << mean_T, which holds for stable CI timings where std_s is
    typically < 5 % of mean_s.  Clamped to >= 0 to avoid negative values
    from floating-point noise.

    Args:
        std_s: Standard deviation of elapsed time (seconds).
        mean_s: Mean elapsed time (seconds).
        n: Number of outputs generated.

    Returns:
        Estimated standard deviation of throughput (outputs/second).
    """
    if mean_s <= 0:
        return 0.0
    rate_std = (n / (mean_s ** 2)) * std_s
    return max(rate_std, 0.0)


def plot_benchmark_results(
    results: dict,
    output_dir: str = "results/figures",
) -> str:
    """Bar chart of generator throughput with logarithmic y-axis.

    Follows the project's matplotlib conventions (mirrors
    ``src/tests/visualization.py``): Agg backend, seaborn-v0_8-whitegrid
    with fallback, tab10 color cycle, dpi=150, bbox_inches="tight".

    The built-in ``random`` bar is coloured distinctly and labelled
    "C baseline (built-in random)" in the legend.

    Args:
        results: Full results dict as returned by ``run_benchmarks()``.
            Must contain ``results["generator"]`` with per-generator dicts
            keyed by name, plus ``results["builtin_random_name"]`` and
            ``results["n"]`` / ``results["repeats"]`` for the title.
        output_dir: Directory for the saved PNG.  Created if absent.

    Returns:
        Absolute path to the saved PNG.
    """
    os.makedirs(output_dir, exist_ok=True)

    gen_results = results["generator"]
    builtin_name = results.get("builtin_random_name", "builtin_random")

    # Preserve GENERATORS insertion order, then append builtin at the end.
    ordered_names = list(GENERATORS.keys()) + [builtin_name]

    names = [n for n in ordered_names if n in gen_results]
    throughputs = [gen_results[n]["mean_mbps"] for n in names]
    std_times = [gen_results[n]["std_s"] for n in names]
    mean_times = [gen_results[n]["mean_s"] for n in names]
    n_val = results.get("n", 10_000_000)

    errors = [
        _throughput_error(s, m, n_val)
        for s, m in zip(std_times, mean_times)
    ]

    # Colours: tab10 for project generators, distinct colour for built-in.
    cmap = plt.cm.tab10
    colours = []
    builtin_colour = "#e67e22"  # distinct orange, outside tab10's default cycle
    for i, name in enumerate(names):
        if name == builtin_name:
            colours.append(builtin_colour)
        else:
            colours.append(cmap(i % cmap.N))

    n_display = results.get("n", 10_000_000)
    repeats_display = results.get("repeats", 3)
    title = f"PRNG Throughput Comparison (N={n_display:,}, {repeats_display} repeats)"

    try:
        style_ctx = plt.style.context("seaborn-v0_8-whitegrid")
    except OSError:
        style_ctx = plt.style.context("default")

    with style_ctx:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.bar(names, throughputs, color=colours, edgecolor="black",
               linewidth=0.5, yerr=errors, capsize=3)
        ax.set_yscale("log")
        ax.set_ylabel("Outputs / second (log scale)")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)

        # Legend: label every bar; the built-in baseline is called out
        # explicitly so the C-implemented comparison point is unambiguous.
        from matplotlib.patches import Patch
        patches = []
        for i, name in enumerate(names):
            if name == builtin_name:
                patches.append(Patch(facecolor=builtin_colour, edgecolor="black",
                                     label="C baseline (built-in random)"))
            else:
                patches.append(Patch(facecolor=colours[i], edgecolor="black",
                                     label=name))
        ax.legend(handles=patches, fontsize="small", loc="best")

    return _save_and_close(fig, os.path.join(output_dir, "benchmark.png"))


def _save_and_close(fig: plt.Figure, path: str, dpi: int = 150) -> str:
    """Save figure to *path* at *dpi*, close it, return absolute path.

    Mirrors ``src/tests/visualization.py``'s ``_save_and_close``.
    """
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(path)
