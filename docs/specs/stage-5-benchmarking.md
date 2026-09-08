# Stage 5 Spec: Performance Benchmarking

## Goal

Compare generation speed of all generators under identical conditions, measure throughput (numbers/second) with proper statistical rigor (multiple repeats, mean + stddev), and visualize as a log-scale bar chart.

## Scope

A benchmark script, visualization of results, and reproducibility metadata (Python version, CPU, numpy version). No attacks, no statistical tests.

## Dependencies

- All generators from Stage 1
- `time` (stdlib), `numba` (for JIT-accelerated brute force only, optional in bench)
- `matplotlib` for the bar chart
- `platform`, `sys` for environment metadata

---

## Files to Create

| File | Responsibility |
|------|---------------|
| `src/benchmarks.py` | Benchmark runner — measures and saves results |
| `tests/test_benchmarks.py` | Unit tests for benchmark output correctness |

---

## Specifications

### 1. Benchmark Runner (`src/benchmarks.py`)

```python
import time
import platform
import sys
import numpy as np
from src.generators import LCG, MT19937, XorShift32, XorShift64, XorShift128Plus, V8Random, PCG32
from src.generators.lcg import ansi_c, numerical_recipes, glibc, bad  # or factory API
```

#### 1.1 Generator Timing Function

```python
def time_generator(
    gen_factory,
    n: int = 10_000_000,
    repeats: int = 3,
    seed: int = 42,
) -> dict:
    """Time generator throughput.

    Args:
        gen_factory: Callable returning a fresh generator instance.
        n: Number of values to generate per run.
        repeats: Number of timing repetitions.
        seed: Seed for reproducibility.

    Returns:
        Dict: {mean_s, std_s, mean_ns_per_output, mean_mbps}
    """
```

**Methodology:**
1. Fresh generator per run (avoids state carryover) — call `gen_factory()`.
2. Seed with `seed`.
3. Run numpy? No — this is a *pure Python loop* benchmark. The plan explicitly says "generate ... in a Python loop."
4. Warm-up: generate a small batch (e.g., 1000) to trigger any lazy init/JIT (important for MT twist, numba).
5. `start = time.perf_counter()`.
6. Loop `for _ in range(n): gen.next_int()` — discard values to avoid list memory overhead.
7. `end = time.perf_counter()`.
8. Compute elapsed.
9. Repeat `repeats` times; compute mean and stddev via `numpy`.
10. Derive `ns_per_output = (mean_s * 1e9) / n` and `mbps = n / mean_s` (outputs per second).

**Important — pure Python loop, not numba:** Only the *brute force* attack uses numba (Stage 4). The benchmark measures *standard* generation throughput, so no numba here except where a generator's native impl uses it — but our generators are pure Python. Document this.

**Note on memory:** Do NOT preallocate a list of n values for n=10^7 — that's ~80MB and biases timing toward memory allocation. The loop discards values.

#### 1.2 Generator Registry

```python
GENERATORS = {
    "lcg_ansi_c": lambda: ansi_c_mod(...),   # exact factory wiring
    "lcg_numerical_recipes": ...,
    "lcg_glibc": ...,
    "lcg_bad": ...,
    "mt19937": lambda: MT19937(),
    "xorshift32": lambda: XorShift32(),
    "xorshift64": lambda: XorShift64(),
    "xorshift128+": lambda: XorShift128Plus(),
    "v8_math_random": lambda: V8Random(),
    "pcg32": lambda: PCG32(),
}
```

**Built-in `random` comparison:** The plan requires comparing against Python's built-in `random` (`random.getrandbits(32)`). Add it as a special case:

```python
def time_builtin_random(n: int = 10_000_000, repeats: int = 3, seed: int = 42) -> dict:
    """Time random.getrandbits(32) as the C-implemented baseline."""
    rng = random.Random(seed)
    start = time.perf_counter()
    for _ in range(n):
        rng.getrandbits(32)
    end = time.perf_counter()
    ...
```

**Rationale:** Python's `random` is C-implemented; our generators are pure Python. This comparison quantifies the Python-vs-C speed gap.

#### 1.3 Full Benchmark Orchestration

```python
def run_benchmarks(
    n: int = 10_000_000,
    repeats: int = 3,
    seed: int = 42,
    output_file: str = "results/benchmark.json",
) -> dict:
    """Run all benchmarks, save results, return data.

    Results dict includes:
        - environment: python_version, platform, cpu_count, numpy_version
        - generator: {name: {mean_s, std_s, ns_per_output, mbps}}
        - timestamp
    """
```

**Environment capture:**
```python
environment = {
    "python_version": sys.version,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "processor": platform.processor(),
    "cpu_count": os.cpu_count(),
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
}
```

---

### 2. Benchmark Visualization

```python
def plot_benchmark_results(
    results: dict,
    output_dir: str = "results/figures",
) -> str:
    """Bar chart of throughput with logarithmic scale.

    Returns path to saved PNG.
    """
```

**Design:**
- X-axis: generator names (rotated 45° if needed).
- Y-axis: outputs/second (log scale, since C vs Python gap can be 10-100x).
- **Log scale is mandatory** — plan explicitly requires it.
- Error bars from stddev.
- Color bar for built-in `random` differently (e.g., highlighted) to distinguish C baseline.
- Title: "PRNG Throughput Comparison (N=10^7, 3 repeats)".
- Save to `results/figures/benchmark.png`.

---

### 3. Expected Results (documented in spec, not hard assertions)

These are *expectations*, not exit criteria (system-dependent):
- `random.getrandbits(32)` (C) will be fastest — typically 10-100M outputs/sec.
- Pure Python generators: ~1-10M outputs/sec, MT19937 and xorshift128+ (64-bit mult) slower than simple LCG.
- `lcg_bad` (m=101) is trivial (`a*state+c mod 101`) but the small modulus means overhead is Python-loop dominated, similar to others.
- The gap C-vs-Python is the headline finding.

---

### 4. Unit Tests

#### `tests/test_benchmarks.py`

Tests to write:
- `test_time_generator_returns_fields` — result dict has mean_s, std_s, ns_per_output, mbps
- `test_time_generator_mean_positive` — mean_s > 0
- `test_time_generator_repeat_count` — uses exactly `repeats` iterations (can check via monkeypatched time or run count)
- `test_run_benchmarks_environment` — results has environment dict with python_version
- `test_run_benchmarks_generators_present` — all registered generators present in results
- `test_run_benchmarks_saves_json` — output file exists and is valid JSON
- `test_builtin_random_faster_than_python` — `random.getrandbits(32)` mbps > pure-Python xorshift32 mbps (this is reliable enough on CI as the gap is 10x+)
- `test_plot_benchmark_creates_file` — PNG created and non-empty
- `test_benchmark_small_n_smoke` — run with n=1000, repeats=1 to verify no crash (CI-safe)

**Test runtime concern:** Actual benchmarks are heavy (10^7 × repeats × 10 generators). Unit tests must NOT run full benchmarks — use tiny `n` (e.g., 1000) and `repeats=1`, or monkeypatch `n`. The plan requires a special "smoke" path. A `--quick` parameter or separate light-weight test path is expected.

---

## Exit Criteria

- [ ] Benchmarks run for all 10 generators + built-in random with proper timing
- [ ] Results include mean, stddev, outputs/sec, ns/output
- [ ] Environment metadata captured in results
- [ ] Log-scale bar chart generated with error bars
- [ ] Built-in `random` clearly identified as C baseline in chart
- [ ] Results saved to `results/benchmark.json`
- [ ] Unit tests pass (with small-n smoke tests, full benchmarks excluded from CI)
- [ ] No new dependencies beyond Stage 1 + matplotlib
