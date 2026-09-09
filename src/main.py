"""CLI orchestration for the PRNG analysis pipeline.

Wires together all prior stages (generate, test, attack, benchmark, report)
into one reproducible command-line pipeline.  Each stage is a separate
``run_*`` helper that takes a parsed argparse Namespace and resolves its output
paths from ``args.results_dir`` / ``args.figures_dir``.  Keeping the paths
argument-driven lets tests point the pipeline at throwaway tmp directories
instead of writing into the real repo tree.

The project's fail=record-and-continue design decision applies throughout:
if one generator or one attack fails, we record the error into the stage's
JSON output and move on rather than aborting the whole stage.
"""

import json
import os
import pathlib
import platform
import sys
import time

import numpy as np

from src.attacks import (
    analyze_xorshift_family,
    attack_lcg,
    attack_mt19937,
)
from src.benchmarks import plot_benchmark_results, run_benchmarks
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
from src.tests.statistical_tests import run_all_tests
from src.tests.visualization import generate_all_plots
from src.report import generate_report

# The ten project generators in their canonical ordering.  Each is a zero-arg
# constructor whose instances expose .seed(int), .generate(n) and
# .generate_floats(n).  Class names are the natural JSON keys.
GENERATORS = [
    AnsiCLCG,
    NumericalRecipesLCG,
    GlibcLCG,
    BadLCG,
    MT19937,
    XorShift32,
    XorShift64,
    XorShift128Plus,
    V8Random,
    PCG32,
]

# Matches the benchmarks module default; 3 repeats is the project's benchmark
# standard for stable mean/std timing estimates.
BENCHMARK_REPEATS = 3


def environment_block() -> dict:
    """Return the environment metadata dict shared across stage JSON outputs.

    Reuses exactly the same field set as ``src.benchmarks.run_benchmarks`` so
    every result file records an identical environment signature.
    """
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def _json_safe(value):
    """Recursively convert numpy scalars and NaN/inf to JSON-safe values.

    TestResult objects may carry numpy types and NaN/inf floats (e.g. spectral
    tests report NaN statistics).  The standard json module rejects these, so
    every serialized stage must be sanitized first.  NaN/inf become None.
    """
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        # NaN/inf cannot be encoded by the stdlib json module; map to null.
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    return value


def _ensure_dirs(args) -> None:
    """Create results/, figures/ and the docs/ parents if missing."""
    pathlib.Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.figures_dir).mkdir(parents=True, exist_ok=True)


def build_parser() -> "argparse.ArgumentParser":
    """Build the CLI argument parser for the PRNG analysis pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="prng-analysis",
        description="PRNG analysis, attacks, and benchmarking pipeline.",
    )
    parser.add_argument(
        "--stage",
        choices=["generate", "test", "attack", "benchmark", "report", "all"],
        default="all",
        help="Which stage(s) to run (default: all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1_000_000,
        help="Number of values for statistical tests (default: 1000000).",
    )
    parser.add_argument(
        "--n-spectral",
        type=int,
        default=100_000,
        help="Number of pairs for spectral test (default: 100000).",
    )
    parser.add_argument(
        "--n-bench",
        type=int,
        default=10_000_000,
        help="Number of values for benchmark (default: 10000000).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory for outputs (default: results/).",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default="results/figures",
        help="Directory for figures (default: results/figures/).",
    )
    return parser


def run_generate(args) -> str:
    """Generate sequences from every generator and persist them.

    Sequences are saved as a numpy .npz archive (one array per generator)
    because 1M integers per generator as JSON would be both large and slow to
    parse; numpy is already a project dependency.  The .npz records the seed
    used and each generator's class name as an ``.npy``-equivalent array key.

    Returns the path to the written archive.
    """
    _ensure_dirs(args)
    sequences = {}
    for cls in GENERATORS:
        gen = cls()
        gen.seed(args.seed)
        # Use a modest sample for the stored artifact regardless of --n so
        # the generated_sequences file stays small and reproducible; the
        # statistical stage draws its own larger sample directly.
        sequences[cls.__name__] = np.asarray(gen.generate(args.n))
    out = os.path.join(args.results_dir, "generated_sequences.npz")
    np.savez_compressed(out, seed=args.seed, **sequences)
    return out


def run_tests(args) -> str:
    """Run all statistical tests + plots; write metrics.json and PNGs.

    Runs ``run_all_tests`` on every generator, merges the per-generator
    TestResult dicts into the nested structure generate_all_plots expects,
    then produces the five canonical PNGs.  A sanitized subset of each
    TestResult is written to metrics.json.

    Returns the path to the written metrics.json.
    """
    _ensure_dirs(args)
    generator_results = {}
    for cls in GENERATORS:
        gen = cls()
        try:
            generator_results[cls.__name__] = run_all_tests(
                gen,
                n=args.n,
                n_spectral=args.n_spectral,
                seed=args.seed,
            )
        except Exception as exc:
            # Record the failure per-generator so a single broken generator
            # does not abort the whole test stage.
            print(
                f"warning: statistical tests failed for {cls.__name__}: {exc}",
                file=sys.stderr,
            )
            generator_results[cls.__name__] = {"error": str(exc)}

    # Generate the figures from every generator that has real results.
    fig_paths = generate_all_plots(generator_results, output_dir=args.figures_dir)

    metrics = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": environment_block(),
        "generators": _serialize_test_results(generator_results),
        "figures": [os.path.basename(p) for p in fig_paths],
    }

    out = os.path.join(args.results_dir, "metrics.json")
    with open(out, "w") as f:
        json.dump(_json_safe(metrics), f, indent=2)
    return out


def _serialize_test_results(generator_results: dict) -> dict:
    """Reduce per-generator TestResult dicts to the small JSON schema.

    Extracts only scalar fields plus the small details the report needs
    (per-lag correlations, the runs z-score).  Large details arrays such as
    histogram bin counts and spectral pair coordinates are deliberately omitted
    — dumping millions of floats into metrics.json would bloat the artifact
    without feeding anything the report consumes.
    """
    out = {}
    for gen_name, test_dict in generator_results.items():
        if "error" in test_dict:
            out[gen_name] = {"error": test_dict["error"]}
            continue
        tests = {}
        for test_name, tr in test_dict.items():
            if test_name == "histogram":
                # histogram TestResult is folded into the generator block as
                # its own key (per the spec's schema) rather than a "test".
                continue
            entry = {
                "statistic": _json_safe(tr.statistic),
                "p_value": _json_safe(tr.p_value),
                "passed": bool(tr.passed),
                "comment": tr.comment,
            }
            details = {}
            if "correlations" in tr.details:
                details["correlations"] = {
                    str(k): _json_safe(v)
                    for k, v in tr.details["correlations"].items()
                }
            if "z" in tr.details:
                details["z"] = _json_safe(tr.details["z"])
            if details:
                entry["details"] = details
            tests[test_name] = entry

        # Fold the histogram result (visualization data) alongside the tests.
        hist = test_dict.get("histogram")
        histogram_entry = None
        if hist is not None:
            histogram_entry = {
                "statistic": _json_safe(hist.statistic),
                "p_value": None,
                "passed": bool(hist.passed),
                "comment": hist.comment,
            }
        out[gen_name] = {"tests": tests, "histogram": histogram_entry}
    return out


def run_attacks(args) -> str:
    """Run the LCG, MT19937 and xorshift-family attacks; write attacks.json.

    Each attack is wrapped in try/except so a failure is recorded as
    {"error": ...} for that key and the remaining attacks still run.

    Returns the path to the written attacks.json.
    """
    _ensure_dirs(args)
    attacks = {}

    # LCG: ANSI C with a known modulus, BadLCG with an unknown modulus.  Both
    # are seeded from args.seed; pass seeded instances as attack_lcg requires.
    try:
        ansi = AnsiCLCG()
        ansi.seed(args.seed)
        lcg_known = attack_lcg(ansi, modulus_known=True)
        # recovered_parameters is a tuple; convert to a list for JSON.
        lcg_known["recovered_parameters"] = list(
            lcg_known["recovered_parameters"]
        )
        attacks["lcg_known"] = lcg_known
    except Exception as exc:
        print(f"warning: LCG (known modulus) attack failed: {exc}", file=sys.stderr)
        attacks["lcg_known"] = {"error": str(exc)}

    try:
        bad = BadLCG()
        bad.seed(args.seed)
        lcg_unknown = attack_lcg(bad, modulus_known=False)
        lcg_unknown["recovered_parameters"] = list(
            lcg_unknown["recovered_parameters"]
        )
        attacks["lcg_unknown"] = lcg_unknown
    except Exception as exc:
        print(
            f"warning: LCG (unknown modulus) attack failed: {exc}", file=sys.stderr
        )
        attacks["lcg_unknown"] = {"error": str(exc)}

    try:
        attacks["mt19937"] = attack_mt19937(seed=args.seed)
    except Exception as exc:
        print(f"warning: MT19937 attack failed: {exc}", file=sys.stderr)
        attacks["mt19937"] = {"error": str(exc)}

    # analyze_xorshift_family never raises; run it once and split the result
    # into the attacks.json keys "xorshift32", "xorshift64" and "v8".
    family = analyze_xorshift_family()
    attacks["xorshift32"] = family.get("xorshift32")
    attacks["xorshift64"] = family.get("xorshift64")
    attacks["v8"] = family.get("v8_xorshift128plus")

    out = os.path.join(args.results_dir, "attacks.json")
    payload = {"attacks": attacks}
    with open(out, "w") as f:
        json.dump(_json_safe(payload), f, indent=2)
    return out


def run_benchmark(args) -> str:
    """Run benchmarks and produce benchmark.png; write benchmark.json.

    Forwards args.seed and args.n_bench into run_benchmarks (which writes its
    own JSON at args.results_dir/benchmark.json) and plots the resulting chart
    into args.figures_dir.

    Returns the path to the written benchmark.json.
    """
    _ensure_dirs(args)
    out = os.path.join(args.results_dir, "benchmark.json")
    results = run_benchmarks(
        n=args.n_bench,
        repeats=BENCHMARK_REPEATS,
        seed=args.seed,
        output_file=out,
    )
    plot_benchmark_results(results, output_dir=args.figures_dir)
    return out


def run_report(args) -> str:
    """Generate docs/report.md from saved metrics/attacks/benchmark results.

    Reads inputs from args.results_dir / args.figures_dir so tests can point
    them at fixture directories.  The output path comes from the optional
    ``args.report_file`` attribute (set by tests to avoid writing into the real
    repo ``docs/``); when absent it defaults to the spec-mandated
    ``docs/report.md`` relative to the repo root, with parents created as
    needed.  Note: ``report_file`` is deliberately NOT a CLI argument — the
    parser stays spec-exact, this is a test-only wiring hook on the namespace.

    Returns the path to the written report.
    """
    output_file = getattr(args, "report_file", "docs/report.md")
    return generate_report(
        metrics_file=os.path.join(args.results_dir, "metrics.json"),
        attacks_file=os.path.join(args.results_dir, "attacks.json"),
        benchmark_file=os.path.join(args.results_dir, "benchmark.json"),
        figures_dir=args.figures_dir,
        output_file=output_file,
    )


def _dispatch(args) -> None:
    """Run the stages ``main()`` selected, in the spec's execution order.

    Kept separate from ``main()`` so the integration test can exercise the
    exact dispatch logic (not a re-inlined copy of it) without depending on
    real process argv — the Namespace still carries the output paths.
    """
    if args.stage in ("all", "generate"):
        run_generate(args)
    if args.stage in ("all", "test"):
        run_tests(args)
    if args.stage in ("all", "attack"):
        run_attacks(args)
    if args.stage in ("all", "benchmark"):
        run_benchmark(args)
    if args.stage in ("all", "report"):
        run_report(args)


def main() -> None:
    """Entry point: parse args and dispatch to the requested stage(s).

    Stages run in dependency order (generate -> test -> attack -> benchmark ->
    report) when --stage all is selected.
    """
    args = build_parser().parse_args()
    _dispatch(args)


if __name__ == "__main__":
    main()
