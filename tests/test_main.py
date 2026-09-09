"""CLI smoke tests for src/main.py.

Tests drive the pipeline hermetically: they parse args pointing results-dir /
figures-dir at pytest's tmp_path, then invoke the run_* helper (or main()) with
that Namespace.  Nothing writes into the real repo tree.  All sizes are tiny to
keep CI fast.
"""

import json
import os

import pytest

from src.main import (
    build_parser,
    run_generate,
    run_tests,
    run_attacks,
    run_benchmark,
    run_report,
)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def test_build_parser_defaults():
    """Default stage/seed/n are 'all'/42/1000000; bench & figure defaults per spec."""
    args = build_parser().parse_args([])
    assert args.stage == "all"
    assert args.seed == 42
    assert args.n == 1_000_000
    assert args.n_spectral == 100_000
    assert args.n_bench == 10_000_000
    assert args.results_dir == "results"
    assert args.figures_dir == "results/figures"


def test_build_parser_stage_options():
    """Only the documented stage choices are accepted; others raise SystemExit."""
    for choice in ("generate", "test", "attack", "benchmark", "report", "all"):
        args = build_parser().parse_args(["--stage", choice])
        assert args.stage == choice
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--stage", "nope"])


# ---------------------------------------------------------------------------
# Stage helpers (hermetic — point at tmp dirs)
# ---------------------------------------------------------------------------


def test_main_generate_stage_smoke(tmp_path):
    """generate stage creates the results dir and the sequences artifact."""
    results_dir = tmp_path / "res"
    figures_dir = results_dir / "figures"
    args = build_parser().parse_args(
        ["--stage", "generate", "--n", "1000",
         "--results-dir", str(results_dir), "--figures-dir", str(figures_dir)]
    )
    out = run_generate(args)
    assert os.path.isdir(results_dir)
    assert os.path.isfile(out)
    assert os.path.basename(out) == "generated_sequences.npz"


def test_main_test_stage_smoke(tmp_path):
    """test stage writes metrics.json and the five figure PNGs."""
    results_dir = tmp_path / "res"
    figures_dir = results_dir / "figures"
    args = build_parser().parse_args(
        ["--stage", "test", "--n", "1000", "--n-spectral", "100",
         "--results-dir", str(results_dir), "--figures-dir", str(figures_dir)]
    )
    out = run_tests(args)
    assert os.path.basename(out) == "metrics.json"
    with open(out) as f:
        data = json.load(f)
    assert "generators" in data
    # Verify the five canonical PNGs actually got produced.
    expected_pngs = {
        "chi_square_comparison.png",
        "autocorrelation.png",
        "spectral.png",
        "runs_z_scores.png",
        "histograms.png",
    }
    produced = {name for name in os.listdir(figures_dir) if name.endswith(".png")}
    assert expected_pngs <= produced


def test_main_attack_stage_smoke(tmp_path):
    """attack stage writes attacks.json with all expected keys."""
    results_dir = tmp_path / "res"
    figures_dir = results_dir / "figures"
    args = build_parser().parse_args(
        ["--stage", "attack",
         "--results-dir", str(results_dir), "--figures-dir", str(figures_dir)]
    )
    out = run_attacks(args)
    assert os.path.basename(out) == "attacks.json"
    with open(out) as f:
        data = json.load(f)
    assert "attacks" in data
    for key in ("lcg_known", "lcg_unknown", "mt19937", "xorshift32", "xorshift64", "v8"):
        assert key in data["attacks"]


def test_main_benchmark_stage_smoke(tmp_path):
    """benchmark stage writes benchmark.json and benchmark.png."""
    results_dir = tmp_path / "res"
    figures_dir = results_dir / "figures"
    args = build_parser().parse_args(
        ["--stage", "benchmark", "--n-bench", "1000",
         "--seed", "42",
         "--results-dir", str(results_dir), "--figures-dir", str(figures_dir)]
    )
    run_benchmark(args)
    assert os.path.isfile(os.path.join(results_dir, "benchmark.json"))
    assert os.path.isfile(os.path.join(figures_dir, "benchmark.png"))


# ---------------------------------------------------------------------------
# Report stage with fixture input files
# ---------------------------------------------------------------------------


def _write_fixtures(tmp_path):
    """Write minimal JSON fixtures for the report stage into tmp_path."""
    results_dir = tmp_path / "res"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = results_dir / "figures"

    metrics = {
        "generators": {
            "AnsiCLCG": {
                "tests": {
                    "chi_square": {"statistic": 5.0, "p_value": 0.5, "passed": True, "comment": ""},
                    "autocorrelation": {"statistic": 0.01, "p_value": None, "passed": True, "comment": "", "details": {"correlations": {"1": 0.001}, "z": None}},
                    "runs": {"statistic": 1.2, "p_value": 0.2, "passed": True, "comment": "", "details": {"z": 1.2}},
                    "spectral": {"statistic": None, "p_value": None, "passed": True, "comment": "visual test"},
                },
                "histogram": {"statistic": 1.0, "p_value": None, "passed": True, "comment": "visualization data"},
            },
            "BadLCG": {
                "tests": {
                    "chi_square": {"statistic": 99.0, "p_value": 0.0001, "passed": False, "comment": ""},
                    "autocorrelation": {"statistic": 0.5, "p_value": None, "passed": False, "comment": "", "details": {"correlations": {"1": 0.5}, "z": None}},
                    "runs": {"statistic": 2.5, "p_value": 0.01, "passed": False, "comment": "", "details": {"z": 2.5}},
                    "spectral": {"statistic": None, "p_value": None, "passed": True, "comment": "visual test"},
                },
                "histogram": {"statistic": 1.0, "p_value": None, "passed": True, "comment": "visualization data"},
            },
        }
    }
    (results_dir / "metrics.json").write_text(json.dumps(metrics))

    attacks = {
        "attacks": {
            "lcg_known": {"recovered_parameters": [1103515245, 12345], "actual_next": [1, 2], "predicted_next": [1, 2], "match_count": 1000, "accuracy": 1.0},
            "lcg_unknown": {"recovered_parameters": [101, 3, 7], "actual_next": [1], "predicted_next": [1], "match_count": 1000, "accuracy": 1.0},
            "mt19937": {"state_recovered": True, "actual_next": [1], "predicted_next": [1], "match_count": 1000, "accuracy": 1.0},
            "xorshift32": {"recoverable": True, "accuracy": 1.0, "match_count": 20},
            "xorshift64": {"recoverable": True, "accuracy": 1.0, "match_count": 20},
            "v8": {"recoverable": True, "note": "state recovered via z3"},
        }
    }
    (results_dir / "attacks.json").write_text(json.dumps(attacks))

    benchmark = {
        "n": 1000,
        "repeats": 1,
        "seed": 42,
        "generator": {
            "lcg_ansi_c": {"mean_s": 0.001, "std_s": 0.0, "mean_ns_per_output": 10000.0, "mean_mbps": 100000.0},
            "xorshift32": {"mean_s": 0.002, "std_s": 0.0, "mean_ns_per_output": 20000.0, "mean_mbps": 50000.0},
            "builtin_random": {"mean_s": 0.0001, "std_s": 0.0, "mean_ns_per_output": 1000.0, "mean_mbps": 1000000.0},
        },
        "builtin_random_name": "builtin_random",
    }
    (results_dir / "benchmark.json").write_text(json.dumps(benchmark))

    return results_dir, figures_dir


def test_main_report_stage(tmp_path):
    """report stage produces report.md from fixture result files."""
    results_dir, _ = _write_fixtures(tmp_path)
    args = build_parser().parse_args(
        ["--stage", "report",
         "--results-dir", str(results_dir), "--figures-dir", str(results_dir / "figures")]
    )
    # Point the report output into tmp (not the real repo docs/) so the test
    # stays hermetic; run_report falls back to "docs/report.md" when absent.
    args.report_file = str(tmp_path / "docs" / "report.md")
    out = run_report(args)
    assert out == args.report_file
    assert os.path.isfile(out)
    with open(out) as f:
        content = f.read()
    assert "PRNG Analysis Report" in content


# ---------------------------------------------------------------------------
# Full 'all' integration
# ---------------------------------------------------------------------------


def test_main_all_stage(tmp_path):
    """Full pipeline with tiny n produces every expected artifact."""
    results_dir = tmp_path / "res"
    figures_dir = results_dir / "figures"
    args = build_parser().parse_args(
        ["--stage", "all", "--n", "500", "--n-spectral", "100", "--n-bench", "1000",
         "--seed", "7",
         "--results-dir", str(results_dir), "--figures-dir", str(figures_dir)]
    )
    # Route the report output (and its copied figures/) into tmp so the full
    # pipeline integration never touches the real repo docs/.
    args.report_file = str(tmp_path / "docs" / "report.md")
    report_dir = os.path.dirname(args.report_file)

    # Drive the same run_* sequence `main()` dispatches, with our Namespace so
    # we stay hermetic and do not depend on real process argv.
    run_generate(args)
    run_tests(args)
    run_attacks(args)
    run_benchmark(args)
    run_report(args)

    assert os.path.isfile(os.path.join(results_dir, "generated_sequences.npz"))
    assert os.path.isfile(os.path.join(results_dir, "metrics.json"))
    assert os.path.isfile(os.path.join(results_dir, "attacks.json"))
    assert os.path.isfile(os.path.join(results_dir, "benchmark.json"))
    assert os.path.isfile(args.report_file)
    # The report is self-contained: its copied figures live beside it, not in
    # the repo's docs/figures/.
    assert os.path.isdir(os.path.join(report_dir, "figures"))
