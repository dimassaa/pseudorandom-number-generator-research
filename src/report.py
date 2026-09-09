"""Auto-generate docs/report.md from saved metrics, attacks and benchmark results.

Consumes the three JSON artifacts produced by the CLI pipeline
(metrics.json, attacks.json, benchmark.json) plus the PNG figures directory and
renders a self-contained markdown report.  The report embeds figures by copying
them into ``<parent_of_output_file>/figures/`` so it works standalone.

Graceful degradation is mandatory: any missing input file yields a visible
warning line in the report and partial data, never a crash.
"""

import json
import os
import platform
import shutil
import sys
import time


def _load_json(path: str):
    """Load a JSON file, returning None if missing or unparseable.

    A missing or corrupt file is a report-quality concern, not a fatal one; the
    caller emits a warning and continues with partial data.
    """
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"warning: could not parse {path}: {exc}", file=sys.stderr)
        return None


def _fmt_pvalue(value) -> str:
    """Format a p-value for the report table, tolerating None/NaN."""
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_stat(value) -> str:
    """Format a statistic for the report table, tolerating None/NaN."""
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_accuracy(value) -> str:
    """Format an accuracy fraction as a percentage."""
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _copy_figures(figures_dir: str, output_file: str) -> list[str]:
    """Copy every PNG in figures_dir into <parent>/figures/.

    Returns the list of copied basenames.  A missing or empty figures_dir
    yields an empty list and the caller writes a note instead of failing.
    """
    parent = os.path.dirname(os.path.abspath(output_file))
    dest_dir = os.path.join(parent, "figures")
    os.makedirs(dest_dir, exist_ok=True)

    copied = []
    if not os.path.isdir(figures_dir):
        return copied
    for name in sorted(os.listdir(figures_dir)):
        if not name.endswith(".png"):
            continue
        src = os.path.join(figures_dir, name)
        dst = os.path.join(dest_dir, name)
        if os.path.isfile(src):
            # Guard against src and dest resolving to the same file (e.g. when
            # <parent>/figures IS the source figures_dir); copying onto itself
            # raises SameFileError.  The figure is already in place either way.
            if os.path.abspath(src) != os.path.abspath(dst):
                shutil.copy2(src, dst)
            copied.append(name)
    return copied


def _report_environment() -> str:
    """Render the environment section."""
    return (
        f"## 1. Environment\n"
        f"- Python: {sys.version.split()[0]}\n"
        f"- Platform: {platform.platform()}\n"
        f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    )


def _report_generators(metrics) -> str:
    """Render the list of generators tested, from metrics if available."""
    lines = ["## 2. Generators Tested\n"]
    if metrics and "generators" in metrics:
        for name in metrics["generators"]:
            lines.append(f"- {name}")
    else:
        lines.append(
            "- LCG (ANSI C, Numerical Recipes, glibc, bad)\n"
            "- MT19937 (Python random)\n"
            "- Xorshift (32, 64, 128+)\n"
            "- V8 Math.random (xorshift128+)\n"
            "- PCG32"
        )
    return "\n".join(lines) + "\n"


def _report_statistics(metrics) -> str:
    """Render the statistical test results table from metrics.json."""
    lines = ["## 3. Statistical Test Results\n"]
    if not metrics or "generators" not in metrics:
        lines.append("> [!WARNING] missing metrics.json — no statistical data.")
        return "\n".join(lines) + "\n"

    lines.append(
        "| Generator | Chi2 p-val | Autocorr (lag1) | Runs Z | Spectral |"
    )
    lines.append("|---|---|---|---|---|")
    for name, gen_block in metrics["generators"].items():
        tests = gen_block.get("tests", {})
        chi = tests.get("chi_square", {})
        auto = tests.get("autocorrelation", {})
        runs = tests.get("runs", {})
        spectral = tests.get("spectral", {})

        # Extract lag-1 correlation if present; otherwise n/a.
        corr = auto.get("details", {}).get("correlations", {}).get("1")
        corr_str = _fmt_stat(corr)
        z = runs.get("details", {}).get("z")
        z_str = _fmt_stat(z)
        spectral_str = "visual" if spectral.get("passed") is not None else "n/a"

        lines.append(
            f"| {name} | {_fmt_pvalue(chi.get('p_value'))} "
            f"| {corr_str} | {z_str} | {spectral_str} |"
        )
    return "\n".join(lines) + "\n"


def _report_visualization(figures: list[str]) -> str:
    """Render the visualization section referencing every copied PNG."""
    lines = ["## 4. Visualization\n"]
    if figures:
        for name in figures:
            # Relative path keeps the report self-contained alongside figures/.
            lines.append(f"![{name}](figures/{name})")
    else:
        lines.append("> [!WARNING] no figures found — visualization section empty.")
    return "\n".join(lines) + "\n"


def _report_attacks(attacks) -> str:
    """Render the attack results section from attacks.json."""
    lines = ["## 5. Attack Results\n"]
    if not attacks or "attacks" not in attacks:
        lines.append("> [!WARNING] missing attacks.json — no attack data.")
        return "\n".join(lines) + "\n"

    data = attacks["attacks"]

    lcg_known = data.get("lcg_known", {})
    lcg_unknown = data.get("lcg_unknown", {})
    if "error" in lcg_known:
        lines.append("### LCG\n")
        lines.append(f"- Known-modulus attack failed: {lcg_known['error']}")
    else:
        params = lcg_known.get("recovered_parameters")
        lines.append("### LCG\n")
        if params and len(params) >= 2:
            lines.append(f"- Parameters recovered: {{a: {params[0]}, c: {params[1]}}}")
        lines.append(f"- Prediction accuracy: {_fmt_accuracy(lcg_known.get('accuracy'))}")
    if "error" in lcg_unknown:
        lines.append(f"- Unknown-modulus attack failed: {lcg_unknown['error']}")
    else:
        params = lcg_unknown.get("recovered_parameters")
        if params and len(params) >= 3:
            lines.append(
                f"- Unknown-modulus recovered: {{m: {params[0]}, a: {params[1]}, c: {params[2]}}}"
            )

    mt = data.get("mt19937", {})
    if "error" in mt:
        lines.append(f"### MT19937\n- State recovered: failed ({mt['error']})")
    else:
        state_recovered = mt.get("state_recovered")
        recovered_str = "yes" if state_recovered else "no"
        lines.append(
            f"### MT19937\n- State recovered: {recovered_str}\n"
            f"- Prediction accuracy: {_fmt_accuracy(mt.get('accuracy'))}"
        )

    x32 = data.get("xorshift32", {})
    x64 = data.get("xorshift64", {})
    v8 = data.get("v8", {})
    lines.append("### Xorshift")
    if "error" in x32:
        lines.append(f"- xorshift32: failed ({x32['error']})")
    else:
        lines.append(
            f"- xorshift32: state recovered via brute force, "
            f"{_fmt_accuracy(x32.get('accuracy'))}"
        )
    if "error" in x64:
        lines.append(f"- xorshift64: failed ({x64['error']})")
    else:
        note = "state recovered" if x64.get("recoverable") else "not recoverable"
        lines.append(
            f"- xorshift64: {note}, {_fmt_accuracy(x64.get('accuracy'))}"
        )
    if "error" in v8:
        lines.append(f"- V8 xorshift128+: failed ({v8['error']})")
    elif v8.get("recoverable"):
        lines.append(f"- V8 xorshift128+: {v8.get('note', 'recovered')}")
    else:
        lines.append(f"- V8 xorshift128+: not recovered ({v8.get('note', 'n/a')})")

    return "\n".join(lines) + "\n"


def _report_benchmark(benchmark) -> str:
    """Render the benchmark results table from benchmark.json."""
    lines = ["## 6. Benchmark Results\n"]
    if not benchmark or "generator" not in benchmark:
        lines.append("> [!WARNING] missing benchmark.json — no benchmark data.")
        return "\n".join(lines) + "\n"

    lines.append("| Generator | Mean (s) | Std (s) | Outputs/s |")
    lines.append("|---|---|---|---|")
    for name, gen in benchmark["generator"].items():
        mean_s = gen.get("mean_s")
        std_s = gen.get("std_s")
        mean_mbps = gen.get("mean_mbps")
        lines.append(
            f"| {name} | {_fmt_stat(mean_s)} | {_fmt_stat(std_s)} "
            f"| {_fmt_throughput(mean_mbps)} |"
        )
    return "\n".join(lines) + "\n"


def _fmt_throughput(value) -> str:
    """Format outputs/second (the 'mean_mbps' key) readably."""
    if value is None:
        return "n/a"
    try:
        return f"{float(value):,.0f}"
    except (TypeError, ValueError):
        return "n/a"


def _report_conclusions(metrics, attacks, benchmark) -> str:
    """Auto-generate the conclusions section from the data that exists."""
    lines = ["## 7. Conclusions\n"]

    # Statistical strength: list passing vs failing generators.
    if metrics and "generators" in metrics:
        passing = []
        failing = []
        for name, gen_block in metrics["generators"].items():
            tests = gen_block.get("tests", {})
            if not tests:
                continue
            if all(t.get("passed") for t in tests.values()):
                passing.append(name)
            else:
                failing.append(name)
        if passing:
            lines.append(f"- Statistical strength: passed all tests — {', '.join(passing)}")
        if failing:
            lines.append(
                f"- Statistical weakness: failed at least one test — {', '.join(failing)}"
            )
    else:
        lines.append("> [!WARNING] no statistical data for conclusions.")

    # Attack outcomes: full recoverability means not cryptographically secure.
    if attacks and "attacks" in attacks:
        data = attacks["attacks"]
        broken = []
        for key in ("lcg_known", "lcg_unknown", "mt19937"):
            entry = data.get(key, {})
            if not isinstance(entry, dict):
                continue
            acc = entry.get("accuracy")
            if acc is not None and float(acc) >= 1.0:
                broken.append(key)
        for key, label in (
            ("xorshift32", "xorshift32"),
            ("xorshift64", "xorshift64"),
            ("v8", "V8 xorshift128+"),
        ):
            entry = data.get(key, {})
            if isinstance(entry, dict) and entry.get("recoverable"):
                broken.append(label)
        if broken:
            lines.append(
                "- Attack outcomes: fully recoverable / not cryptographically "
                f"secure — {', '.join(broken)}"
            )
        else:
            lines.append("- Attack outcomes: no generator fully recovered.")
    else:
        lines.append("> [!WARNING] no attack data for conclusions.")

    # Performance: fastest / slowest and the C-vs-Python gap.
    if benchmark and "generator" in benchmark:
        gens = benchmark["generator"]
        if gens:
            fastest = max(gens.items(), key=lambda kv: kv[1].get("mean_mbps") or 0.0)
            slowest = min(gens.items(), key=lambda kv: kv[1].get("mean_mbps") or 0.0)
            lines.append(f"- Performance: fastest {fastest[0]}, slowest {slowest[0]}")
            builtin = benchmark.get("builtin_random_name")
            if builtin and builtin in gens:
                pure_python = {
                    k: v["mean_mbps"]
                    for k, v in gens.items()
                    if k != builtin and (v.get("mean_mbps") or 0.0) > 0
                }
                if pure_python:
                    gap = gens[builtin]["mean_mbps"] / max(pure_python.values())
                    lines.append(
                        f"- C-vs-Python gap: built-in random ~{gap:.1f}× faster "
                        f"than the fastest pure-Python generator."
                    )
    else:
        lines.append("> [!WARNING] no benchmark data for conclusions.")

    return "\n".join(lines) + "\n"


def generate_report(
    metrics_file: str = "results/metrics.json",
    attacks_file: str = "results/attacks.json",
    benchmark_file: str = "results/benchmark.json",
    figures_dir: str = "results/figures",
    output_file: str = "docs/report.md",
) -> str:
    """Auto-generate docs/report.md from saved results and figures.

    Args:
        metrics_file: Path to results/metrics.json.
        attacks_file: Path to results/attacks.json.
        benchmark_file: Path to results/benchmark.json.
        figures_dir: Directory holding the PNG figures to embed.
        output_file: Path where the markdown report is written.

    Returns:
        Path to the written report (``output_file``, as given).

    Raises:
        OSError: If the output directory cannot be created or the file written.
    """
    metrics = _load_json(metrics_file)
    attacks = _load_json(attacks_file)
    benchmark = _load_json(benchmark_file)

    figures = _copy_figures(figures_dir, output_file)

    sections = [
        "# PRNG Analysis Report\n",
        _report_environment(),
        _report_generators(metrics),
        _report_statistics(metrics),
        _report_visualization(figures),
        _report_attacks(attacks),
        _report_benchmark(benchmark),
        _report_conclusions(metrics, attacks, benchmark),
    ]
    content = "\n".join(sections)

    out_parent = os.path.dirname(os.path.abspath(output_file))
    os.makedirs(out_parent, exist_ok=True)
    with open(output_file, "w") as f:
        f.write(content)
    return output_file


def main() -> None:
    """CLI entry point for the report generator (python -m src.report)."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="prng-report",
        description="Generate docs/report.md from saved PRNG analysis results.",
    )
    parser.add_argument(
        "--metrics",
        default="results/metrics.json",
        help="Path to metrics.json (default: results/metrics.json).",
    )
    parser.add_argument(
        "--attacks",
        default="results/attacks.json",
        help="Path to attacks.json (default: results/attacks.json).",
    )
    parser.add_argument(
        "--benchmark",
        default="results/benchmark.json",
        help="Path to benchmark.json (default: results/benchmark.json).",
    )
    parser.add_argument(
        "--figures-dir",
        default="results/figures",
        help="Directory of PNG figures to embed (default: results/figures).",
    )
    parser.add_argument(
        "--output",
        default="docs/report.md",
        help="Output markdown path (default: docs/report.md).",
    )
    args = parser.parse_args()

    generated = generate_report(
        metrics_file=args.metrics,
        attacks_file=args.attacks,
        benchmark_file=args.benchmark,
        figures_dir=args.figures_dir,
        output_file=args.output,
    )
    print(f"Report written to {generated}")


if __name__ == "__main__":
    main()
