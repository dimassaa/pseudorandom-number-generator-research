"""Tests for src/benchmarks.py — benchmark runner and visualization.

All tests use small n/repeats to stay CI-safe (< 10 s total).
No test invokes run_benchmarks() with the default n (10^7 × 11 × 3).
"""

import json
import os

from src.benchmarks import (
    GENERATORS,
    plot_benchmark_results,
    run_benchmarks,
    time_builtin_random,
    time_generator,
)


# ---------------------------------------------------------------------------
# time_generator tests
# ---------------------------------------------------------------------------


def test_time_generator_returns_fields():
    """Result dict contains all four required keys with plausible values."""
    gen_factory = lambda: __import__("src.generators", fromlist=["AnsiCLCG"]).AnsiCLCG()
    result = time_generator(gen_factory, n=2000, repeats=1)
    for key in ("mean_s", "std_s", "mean_ns_per_output", "mean_mbps"):
        assert key in result, f"Missing key: {key}"
    assert result["mean_s"] > 0
    assert result["mean_ns_per_output"] > 0
    assert result["mean_mbps"] > 0


def test_time_generator_mean_positive():
    """mean_s must be strictly positive for any non-zero workload."""
    gen_factory = lambda: __import__("src.generators", fromlist=["AnsiCLCG"]).AnsiCLCG()
    result = time_generator(gen_factory, n=2000, repeats=1)
    assert result["mean_s"] > 0, "mean_s should be > 0 for a real timing run"


def test_time_generator_repeat_count():
    """Verify the loop runs exactly ``repeats`` times using a counting factory."""

    class CountingGen:
        """Minimal generator that counts how many fresh instances were created."""

        instance_count = 0

        def __init__(self):
            CountingGen.instance_count += 1

        def seed(self, v):
            pass

        def next_int(self):
            return 0

    # Reset class-level counter before the test
    CountingGen.instance_count = 0

    def counting_factory():
        return CountingGen()

    time_generator(counting_factory, n=10, repeats=5)
    assert CountingGen.instance_count == 5, (
        f"Expected 5 fresh instances (one per repeat), got {CountingGen.instance_count}"
    )


# ---------------------------------------------------------------------------
# run_benchmarks tests (small n for CI)
# ---------------------------------------------------------------------------


def test_run_benchmarks_environment():
    """Results contain an 'environment' dict with 'python_version' key."""
    result = run_benchmarks(n=1000, repeats=1, output_file="/dev/null")
    assert "environment" in result
    assert "python_version" in result["environment"]


def test_run_benchmarks_generators_present():
    """All 10 GENERATORS names plus 'builtin_random' appear in results['generator']."""
    result = run_benchmarks(n=1000, repeats=1, output_file="/dev/null")
    gen_names = set(result["generator"].keys())
    for name in GENERATORS:
        assert name in gen_names, f"Missing generator: {name}"
    assert "builtin_random" in gen_names, "Missing builtin_random entry"


def test_run_benchmarks_saves_json(tmp_path):
    """Output file exists and contains valid JSON."""
    out = tmp_path / "bench.json"
    run_benchmarks(n=1000, repeats=1, output_file=str(out))
    assert out.exists()
    with open(out) as f:
        data = json.load(f)
    assert "generator" in data


# ---------------------------------------------------------------------------
# Builtin random comparison
# ---------------------------------------------------------------------------


def test_builtin_random_faster_than_python():
    """C-implemented random.getrandbits(32) must be faster than pure-Python xorshift32.

    Uses n=200_000 with 3 repeats for timing stability on CI.  The C-vs-Python
    gap is typically > 10× on modern hardware, so we assert with a 1.1× safety
    margin to tolerate slow CI runners without masking real regressions.
    """
    from src.generators import XorShift32

    n = 200_000
    repeats = 3
    builtin = time_builtin_random(n=n, repeats=repeats)
    xor32 = time_generator(lambda: XorShift32(), n=n, repeats=repeats)
    assert builtin["mean_mbps"] > xor32["mean_mbps"] * 1.1, (
        f"Expected built-in random ({builtin['mean_mbps']:.0f} ops/s) to be "
        f"> 1.1× xorshift32 ({xor32['mean_mbps']:.0f} ops/s)"
    )


# ---------------------------------------------------------------------------
# Plot tests
# ---------------------------------------------------------------------------


def test_plot_benchmark_creates_file(tmp_path):
    """Plotting a tiny fake results dict produces a non-empty PNG."""
    fake_results = {
        "n": 100,
        "repeats": 1,
        "generator": {
            "gen_a": {"mean_s": 0.001, "std_s": 0.0, "mean_ns_per_output": 10000.0, "mean_mbps": 100000.0},
            "gen_b": {"mean_s": 0.005, "std_s": 0.001, "mean_ns_per_output": 50000.0, "mean_mbps": 20000.0},
            "gen_c": {"mean_s": 0.010, "std_s": 0.002, "mean_ns_per_output": 100000.0, "mean_mbps": 10000.0},
            "builtin_random": {"mean_s": 0.0002, "std_s": 0.00005, "mean_ns_per_output": 2000.0, "mean_mbps": 500000.0},
        },
        "builtin_random_name": "builtin_random",
    }
    path = plot_benchmark_results(fake_results, output_dir=str(tmp_path))
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0
    # Verify it's a valid PNG (magic bytes)
    with open(path, "rb") as f:
        assert f.read(8)[:4] == b"\x89PNG"


# ---------------------------------------------------------------------------
# Smoke test — full pipeline with tiny n
# ---------------------------------------------------------------------------


def test_benchmark_small_n_smoke(tmp_path):
    """run_benchmarks + plot_benchmark_results complete without error for n=1000."""
    out_json = tmp_path / "smoke.json"
    results = run_benchmarks(n=1000, repeats=1, output_file=str(out_json))
    assert out_json.exists()

    fig_dir = tmp_path / "figures"
    png_path = plot_benchmark_results(results, output_dir=str(fig_dir))
    assert os.path.exists(png_path)
    assert os.path.getsize(png_path) > 0
