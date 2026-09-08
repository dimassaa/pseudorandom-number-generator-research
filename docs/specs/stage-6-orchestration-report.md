# Stage 6 Spec: Orchestration (main.py), Auto-Generated Report, README

## Goal

Wire together all prior stages into one reproducible pipeline with a CLI (`main.py`), auto-generate `docs/report.md` from collected metrics/figures, and write a comprehensive `README.md` with reproduction instructions.

## Scope

CLI orchestration, report generator, README, requirements.txt, final integration tests. No new analysis logic.

## Dependencies

- All modules from Stages 1–5
- `argparse` (stdlib), `json`, `os`, `pathlib`

---

## Files to Create

| File | Responsibility |
|------|---------------|
| `src/main.py` | CLI orchestration |
| `src/report.py` | Auto-generates `docs/report.md` from results |
| `requirements.txt` | Minimum-version dependencies |
| `README.md` | Project documentation + reproduction guide |
| `tests/test_main.py` | CLI smoke tests |
| `tests/test_report.py` | Report generation tests |

---

## Specifications

### 1. CLI Orchestration (`src/main.py`)

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prng-analysis",
        description="PRNG analysis, attacks, and benchmarking pipeline."
    )
    parser.add_argument(
        "--stage",
        choices=["generate", "test", "attack", "benchmark", "report", "all"],
        default="all",
        help="Which stage(s) to run (default: all)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Global seed for reproducibility (default: 42)."
    )
    parser.add_argument(
        "--n", type=int, default=1_000_000,
        help="Number of values for statistical tests (default: 1000000)."
    )
    parser.add_argument(
        "--n-spectral", type=int, default=100_000,
        help="Number of pairs for spectral test (default: 100000)."
    )
    parser.add_argument(
        "--n-bench", type=int, default=10_000_000,
        help="Number of values for benchmark (default: 10000000)."
    )
    parser.add_argument(
        "--results-dir", type=str, default="results",
        help="Directory for outputs (default: results/)."
    )
    parser.add_argument(
        "--figures-dir", type=str, default="results/figures",
        help="Directory for figures (default: results/figures/)."
    )
    return parser
```

**Stage behaviors:**

| Stage | Actions |
|-------|---------|
| `generate` | Instantiate all generators, generate sequences, save `results/generated_sequences.json` (or numpy files) |
| `test` | Run all statistical tests + plots; save `results/metrics.json`, figures to `results/figures/` |
| `attack` | Run LCG, MT19937, xorshift attacks; save `results/attacks.json` |
| `benchmark` | Run benchmarks; save `results/benchmark.json`, `results/figures/benchmark.png` |
| `report` | Generate `docs/report.md` from all saved results |
| `all` | Run generate → test → attack → benchmark → report in order |

**Execution model:**
```python
def main():
    args = build_parser().parse_args()
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
```

**Directory setup:** Ensure `results/`, `results/figures/`, `docs/` exist before writing (using `pathlib.Path.mkdir(parents=True, exist_ok=True)`).

---

### 2. Report Generation (`src/report.py`)

```python
def generate_report(
    metrics_file: str = "results/metrics.json",
    attacks_file: str = "results/attacks.json",
    benchmark_file: str = "results/benchmark.json",
    figures_dir: str = "results/figures",
    output_file: str = "docs/report.md",
) -> str:
    """Auto-generate docs/report.md from saved results and figures.

    Returns path to written report.
    """
```

**Auto-generated report structure:**

```markdown
# PRNG Analysis Report

## 1. Environment
- Python: {version}
- Platform: {platform}
- Generated: {timestamp}

## 2. Generators Tested
- LCG (ANSI C, Numerical Recipes, glibc, bad)
- MT19937 (Python random)
- Xorshift (32, 64, 128+)
- V8 Math.random (xorshift128+)
- PCG32

## 3. Statistical Test Results
{for each generator, table of test results: name, statistic, p-value, passed}

| Generator | Chi2 p-val | Autocorr (lag1) | Runs Z | Spectral |
|-----------|-----------|----------------|--------|----------|
...

## 4. Visualization
![Chi-square Comparison](figures/chi_square_comparison.png)
![Autocorrelation](figures/autocorrelation.png)
![Spectral Test](figures/spectral_test.png)
![Runs Test](figures/runs_test.png)
![Histograms](figures/histograms.png)
![Benchmark](figures/benchmark.png)

## 5. Attack Results
### LCG
- Parameters recovered: {a, c}
- Prediction accuracy: {100.0%}

### MT19937
- State recovered: {yes}
- Prediction accuracy: {100.0%}

### Xorshift
- xorshift32: state recovered via brute force, {100.0%}
- xorshift64: {recovered / documented infeasible}
- V8 xorshift128+: {recovered via z3 / z3 unavailable, described}

## 6. Benchmark Results
| Generator | Mean (s) | Std (s) | Outputs/s |
|-----------|----------|---------|-----------|
...

## 7. Conclusions
- {auto-generated summary: which generators are statistically strong, which are weak}
- {attack outcomes: which generators are cryptographically broken}
```

**Embedding figures:** Report uses **relative paths** `figures/...` so it renders correctly when `docs/report.md` references `docs/figures/` or the figures are copied there. **Decision:** Copy the generated PNGs into `docs/figures/` so the report is self-contained. Report figures reference `figures/filename.png`.

**Auto-conclusions:**
- Statistical: list generators that passed all tests vs those that failed (e.g., `lcg_bad`).
- Attack: list generators where prediction accuracy == 100% → "fully recoverable / not cryptographically secure".
- Performance: identify fastest/slowest, note C-vs-Python gap.

---

### 3. README.md

Comprehensive, targets "developers, researchers, cryptography and statistics enthusiasts."

**Sections:**
1. **Title + one-line description**
2. **Features** (bulleted: generators, statistical tests, attacks, benchmark)
3. **Installation** (Python 3.10+, `pip install -r requirements.txt`)
4. **Quick Start** (run `python -m src.main --stage all`)
5. **Usage / CLI reference** (all `--stage` options documented)
6. **Project Structure** (reproduce the directory tree)
7. **Methodology** (concise explanation of each statistical test and attack)
8. **Results Summary** (headline findings + link to `docs/report.md`)
9. **Reproducibility** (fixed seeds, environment capture)
10. **Limitations** (z3 optional, xorshift64 partial, float-only V8 ambiguity)
11. **Dependencies** table
12. **License** (if applicable — leave section header)

**Style:** Code-friendly, tables for results, no marketing fluff. Match the technical tone of Stark Industries.

---

### 4. requirements.txt

```txt
numpy>=1.24
matplotlib>=3.7
scipy>=1.10
numba>=0.57
z3-solver>=4.12
```

**Note:** z3-solver marked optional in docs but included in requirements (since it's harmless as a dependency and the plan lists it as optional). Final decision: **include z3 in requirements.txt** but code must gracefully degrade if `import z3` fails (defensive). Document in README that it's optional and the attack degrades without it.

---

### 5. `main.py` As Module Entry Point

Support `python -m src.main` and `python -m src.report`:

```python
if __name__ == "__main__":
    main()
```

---

### 6. Unit Tests

#### `tests/test_main.py`

Tests to write:
- `test_build_parser_defaults` — default stage == "all", seed == 42
- `test_build_parser_stage_options` — valid choices are generate/test/attack/benchmark/report/all
- `test_main_generate_stage_smoke` — run with `--stage generate --n 1000`, verify results dir created, sequences file exists. Use a temp dir via `tmp_path`.
- `test_main_test_stage_smoke` — run with `--stage test --n 1000 --n-spectral 100`, verify metrics.json and figures created.
- `test_main_attack_stage_smoke` — run with `--stage attack`, verify attacks.json.
- `test_main_benchmark_stage_smoke` — run with `--stage benchmark --n-bench 1000`, verify benchmark.json.
- `test_main_report_stage` — given fixture result files, run `--stage report`, verify report.md created.
- `test_main_all_stage` — integration test with tiny n; verify all result files exist. (Use tiny n to keep CI fast.)

#### `tests/test_report.py`

Tests to write:
- `test_generate_report_creates_file` — from fixture result dicts, verify report.md created and non-empty.
- `test_report_contains_tables` — report contains markdown table syntax (`|`) for generators and benchmarks.
- `test_report_embeds_figures` — report references `figures/*.png` for each expected figure.
- `test_report_contains_conclusions` — report includes conclusions section.
- `test_generate_report_missing_files` — gracefully handles missing metrics/attacks/benchmark files (writes report with partial data + warning).
- `test_report_copies_figures` — figures copied into `docs/figures/`.

**Fixture approach:** Tests should create minimal JSON fixtures in `tmp_path` rather than running full analysis (fast, isolated).

---

## Exit Criteria

- [ ] `main.py` CLI supports all 6 stages individually and `all`
- [ ] Each stage produces expected artifacts (JSON, PNG)
- [ ] `docs/report.md` auto-generated from results with tables and embedded figures
- [ ] Figures copied to `docs/figures/` for self-contained report
- [ ] `README.md` complete with installation, usage, structure, methodology, limitations
- [ ] `requirements.txt` with minimum versions
- [ ] `python -m src.main` and `python -m src.report` work as entry points
- [ ] All unit tests pass (smoke tests use tiny n for CI speed)
- [ ] Running `--stage all` with small n produces a complete, valid report end-to-end

---

## Cross-Stage Integration Requirements

For `--stage all` to work from scratch:

1. Sequences generated in Stage 1 must be importable by Stage 2 tests.
2. Stage 2 metrics must be savable to `results/metrics.json` with a consistent schema.
3. Stages 3–4 attacks must write `results/attacks.json`.
4. Stage 5 benchmarks write `results/benchmark.json`.
5. Stage 6 report reads all three JSON files + figures.

**JSON schema consistency (critical for report):**

```json
{
    "generated_at": "ISO timestamp",
    "environment": {...},
    "generators": {
        "gen_name": {
            "tests": {"chi_square": {...}, "autocorrelation": {...}, ...},
            "histogram": {...}
        }
    },
    "attacks": {
        "lcg_known": {...},
        "lcg_unknown": {...},
        "mt19937": {...},
        "xorshift32": {...},
        "xorshift64": {...},
        "v8": {...}
    },
    "benchmarks": {
        "generators": {...},
        "builtin_random": {...}
    }
}
```
