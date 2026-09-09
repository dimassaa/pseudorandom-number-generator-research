"""Tests for src/report.py report generation.

Tests create minimal JSON fixtures in tmp_path (fast, isolated) rather than
running the real analysis.  The report generator is exercised against these
fixtures to verify structure, embeds, conclusions and graceful degradation.
"""

import json
import os

from src.report import generate_report


def _write_fixture_files(tmp_path, figures=True):
    """Write realistic minimal JSON fixtures into tmp_path.

    Returns a dict of the file/dir paths to pass into generate_report.  When
    ``figures`` is False no figures directory is created (to exercise the
    missing-figures degradation path).
    """
    results = tmp_path / "res"
    results.mkdir(parents=True, exist_ok=True)
    figures_dir = results / "figures"
    if figures:
        figures_dir.mkdir(parents=True, exist_ok=True)
        # A stub PNG with valid magic bytes so the copy path is exercised.
        (figures_dir / "chi_square_comparison.png").write_bytes(b"\x89PNG\r\n\x1a\nstub")
        (figures_dir / "spectral.png").write_bytes(b"\x89PNG\r\n\x1a\nstub")

    metrics = {
        "generators": {
            "AnsiCLCG": {
                "tests": {
                    "chi_square": {"statistic": 5.0, "p_value": 0.5, "passed": True, "comment": ""},
                    "autocorrelation": {"statistic": 0.01, "p_value": None, "passed": True, "comment": "", "details": {"correlations": {"1": 0.001}}},
                    "runs": {"statistic": 1.2, "p_value": 0.2, "passed": True, "comment": "", "details": {"z": 1.2}},
                    "spectral": {"statistic": None, "p_value": None, "passed": True, "comment": "visual test"},
                },
                "histogram": {"statistic": 1.0, "p_value": None, "passed": True, "comment": "visualization data"},
            },
            "BadLCG": {
                "tests": {
                    "chi_square": {"statistic": 99.0, "p_value": 0.0001, "passed": False, "comment": ""},
                    "autocorrelation": {"statistic": 0.5, "p_value": None, "passed": False, "comment": "", "details": {"correlations": {"1": 0.5}}},
                    "runs": {"statistic": 2.5, "p_value": 0.01, "passed": False, "comment": "", "details": {"z": 2.5}},
                    "spectral": {"statistic": None, "p_value": None, "passed": True, "comment": "visual test"},
                },
                "histogram": {"statistic": 1.0, "p_value": None, "passed": True, "comment": "visualization data"},
            },
        }
    }
    (results / "metrics.json").write_text(json.dumps(metrics))

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
    (results / "attacks.json").write_text(json.dumps(attacks))

    benchmark = {
        "generator": {
            "lcg_ansi_c": {"mean_s": 0.001, "std_s": 0.0, "mean_ns_per_output": 10000.0, "mean_mbps": 100000.0},
            "xorshift32": {"mean_s": 0.002, "std_s": 0.0, "mean_ns_per_output": 20000.0, "mean_mbps": 50000.0},
            "builtin_random": {"mean_s": 0.0001, "std_s": 0.0, "mean_ns_per_output": 1000.0, "mean_mbps": 1000000.0},
        },
        "builtin_random_name": "builtin_random",
    }
    (results / "benchmark.json").write_text(json.dumps(benchmark))

    return {
        "metrics": str(results / "metrics.json"),
        "attacks": str(results / "attacks.json"),
        "benchmark": str(results / "benchmark.json"),
        "figures_dir": str(figures_dir),
        "results_dir": str(results),
    }


def _run_report(tmp_path, paths=None, output_name="report.md"):
    """Convenience wrapper running generate_report against fixtures."""
    if paths is None:
        paths = _write_fixture_files(tmp_path)
    output = str(tmp_path / output_name)
    return generate_report(
        metrics_file=paths["metrics"],
        attacks_file=paths["attacks"],
        benchmark_file=paths["benchmark"],
        figures_dir=paths["figures_dir"],
        output_file=output,
    ), output


def test_generate_report_creates_file(tmp_path):
    """Report file is created and non-empty from fixture result dicts."""
    path, output = _run_report(tmp_path)
    assert path == output
    assert os.path.isfile(output)
    assert os.path.getsize(output) > 0


def test_report_contains_tables(tmp_path):
    """Report has markdown tables for generators and benchmarks."""
    _, output = _run_report(tmp_path)
    with open(output) as f:
        content = f.read()
    # Generator table header.
    assert "| Generator | Chi2 p-val | Autocorr (lag1) | Runs Z | Spectral |" in content
    # Benchmark table header.
    assert "| Generator | Mean (s) | Std (s) | Outputs/s |" in content
    # At least one generator row references BadLCG as a failed generator.
    assert "BadLCG" in content


def test_report_embeds_figures(tmp_path):
    """Report references figures/*.png for each expected figure."""
    _, output = _run_report(tmp_path)
    with open(output) as f:
        content = f.read()
    assert "![chi_square_comparison.png](figures/chi_square_comparison.png)" in content
    assert "![spectral.png](figures/spectral.png)" in content


def test_report_contains_conclusions(tmp_path):
    """Report includes a conclusions section with attack/statistical findings."""
    _, output = _run_report(tmp_path)
    with open(output) as f:
        content = f.read()
    assert "## 7. Conclusions" in content
    assert "not cryptographically secure" in content
    assert "BadLCG" in content  # flagged as statistically weak


def test_generate_report_missing_files(tmp_path):
    """Missing inputs produce a valid report with warnings, no crash."""
    missing = {
        "metrics": str(tmp_path / "nope_metrics.json"),
        "attacks": str(tmp_path / "nope_attacks.json"),
        "benchmark": str(tmp_path / "nope_benchmark.json"),
        "figures_dir": str(tmp_path / "nope_figures"),
    }
    output = str(tmp_path / "report.md")
    generate_report(
        metrics_file=missing["metrics"],
        attacks_file=missing["attacks"],
        benchmark_file=missing["benchmark"],
        figures_dir=missing["figures_dir"],
        output_file=output,
    )
    assert os.path.isfile(output)
    with open(output) as f:
        content = f.read()
    # Every missing-file note surfaced as a warning, and sections still render.
    assert "> [!WARNING]" in content
    assert "no statistical data" in content
    assert "no attack data" in content
    assert "no benchmark data" in content
    assert "## 7. Conclusions" in content


def test_report_copies_figures(tmp_path):
    """Figures are copied into <parent_of_output>/figures/."""
    _, output = _run_report(tmp_path)
    parent = os.path.dirname(output)
    assert os.path.isfile(os.path.join(parent, "figures", "chi_square_comparison.png"))
    assert os.path.isfile(os.path.join(parent, "figures", "spectral.png"))


def test_report_figures_same_dir_as_output(tmp_path):
    """No SameFileError when the source figures dir equals <parent>/figures.

    If the user places the report output next to the figures dir such that
    dest resolves to the source itself, the copy step must skip rather than
    raise SameFileError.
    """
    paths = _write_fixture_files(tmp_path)
    # Place figures directly under <parent>/figures (identical to the copy
    # destination) so the copy step would collide with its own source.
    same_dir = tmp_path / "figures"
    same_dir.mkdir(parents=True, exist_ok=True)
    (same_dir / "chi_square_comparison.png").write_bytes(b"\x89PNG\r\n\x1a\nstub")
    output = str(tmp_path / "report.md")
    generate_report(
        metrics_file=paths["metrics"],
        attacks_file=paths["attacks"],
        benchmark_file=paths["benchmark"],
        figures_dir=str(same_dir),
        output_file=output,
    )
    assert os.path.isfile(output)
    with open(output) as f:
        assert "![chi_square_comparison.png]" in f.read()
