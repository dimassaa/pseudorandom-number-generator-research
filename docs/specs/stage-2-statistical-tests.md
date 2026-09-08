# Stage 2 Spec: Statistical Tests and Visualization

## Goal

Implement a battery of statistical tests and visualization functions that analyze PRNG output sequences, producing both quantitative metrics (JSON) and publication-quality PNG plots.

## Scope

Statistical test functions (chi-square, autocorrelation, spectral, runs) + visualization module. No attacks, no benchmarking.

## Dependencies from Stage 1

All 6 generator classes from `src/generators/` must be implemented and passing tests.

---

## Files to Create

| File | Responsibility |
|------|---------------|
| `src/tests/statistical_tests.py` | Statistical test functions |
| `src/tests/visualization.py` | Plot generation functions |
| `src/tests/__init__.py` | Package exports |
| `tests/test_statistical_tests.py` | Unit tests for statistical functions |
| `tests/test_visualization.py` | Unit tests for visualization (smoke tests) |

---

## Specifications

### 1. Statistical Tests (`src/tests/statistical_tests.py`)

Each test function takes an array of integers (or floats) and returns a structured result dictionary.

**Common result format:**

```python
@dataclass
class TestResult:
    test_name: str
    generator_name: str
    statistic: float        # Test statistic value
    p_value: float          # P-value (where applicable)
    passed: bool            # Whether test passed threshold
    details: dict           # Additional info (bins, lags, etc.)
    comment: str            # Human-readable interpretation
```

#### 1.1 Chi-Square Test for Uniformity

```python
def chi_square_test(
    values: list[int],
    num_bins: int = 1000,
    generator_name: str = "unknown"
) -> TestResult:
    """Chi-square goodness-of-fit test for uniformity.

    Divides the range of values into `num_bins` equal bins,
    counts observed frequencies, compares to expected (uniform).
    """
```

**Algorithm:**
1. Normalize values to `[0, 1)` range (divide by max possible value).
2. Create `num_bins` equal-width bins.
3. Count observations in each bin.
4. Expected count per bin = `len(values) / num_bins`.
5. Compute chi-square statistic: `Σ (observed - expected)^2 / expected`
6. Compute p-value using `scipy.stats.chi2.sf(chi2_stat, df=num_bins - 1)`.
7. Pass criterion: `0.01 <= p_value <= 0.99`.

**Edge cases:**
- If any bin has 0 observations, chi-square is inflated — still compute it (this indicates poor uniformity).
- If `len(values) < num_bins`, reduce `num_bins` to `len(values) // 10` and warn.

#### 1.2 Autocorrelation Test

```python
def autocorrelation_test(
    values: list[int],
    lags: list[int] = [1, 2, 5, 10],
    generator_name: str = "unknown"
) -> TestResult:
    """Pearson autocorrelation at specified lags.

    For ideal PRNG, correlation should be near 0 (± 1/√N).
    """
```

**Algorithm:**
1. Convert to float array.
2. For each lag `t` in `lags`:
   - Compute Pearson r between `values[:-t]` and `values[t:]`
   - Using `numpy.corrcoef` or manual formula
3. Test criterion: `|r| < 2 / sqrt(N)` for all lags (95% confidence)
4. Return all lag correlations in `details["correlations"]`.

**Note:** Use `numpy` for speed. The correlation formula:
```
r = Σ((x_i - mean_x) * (y_i - mean_y)) / (std_x * std_y * N)
```

#### 1.3 Spectral Test (2D Pairs)

```python
def spectral_test_2d(
    values: list[int],
    sample_size: int = 100_000,
    generator_name: str = "unknown"
) -> TestResult:
    """Plot 2D pairs (X_n, X_{n+1}) to reveal lattice structure.

    Uses sample_size pairs from the first sample_size+1 values.
    LCGs show parallel lines; good generators fill the space.
    """
```

**Algorithm:**
1. Take first `sample_size + 1` values.
2. Create pairs: `(values[0], values[1]), (values[1], values[2]), ...`
3. Normalize both axes to `[0, 1)`.
4. Return pairs in `details["pairs_x"]` and `details["pairs_y"]` for visualization.
5. **No p-value** — this is a visual test. Set `passed = True` always; interpretation is visual.

#### 1.4 Runs Test

```python
def runs_test(
    values: list[int],
    generator_name: str = "unknown"
) -> TestResult:
    """Wald-Wolfowitz runs test for independence.

    Converts values to binary (bit 0), counts runs of consecutive
    identical bits, compares to expected under randomness.
    """
```

**Algorithm:**
1. Extract bit 0 from each value: `bits = [v & 1 for v in values]`
2. Count runs: consecutive identical bits form a run.
3. Expected runs: `E = (2 * n_0 * n_1) / n + 1` where `n_0` = count of 0s, `n_1` = count of 1s, `n` = total.
4. Variance: `V = (2 * n_0 * n_1 * (2 * n_0 * n_1 - n)) / (n^2 * (n - 1))`
5. Z-statistic: `Z = (observed_runs - E) / sqrt(V)`
6. Pass criterion: `|Z| < 1.96` (95% confidence, two-tailed).

#### 1.5 Histogram Data (for visualization)

```python
def histogram_data(
    values: list[int],
    num_bins: int = 200,
    generator_name: str = "unknown"
) -> TestResult:
    """Compute histogram bin counts for value distribution.

    Returns bin edges and counts in details for plotting.
    """
```

**Algorithm:**
1. Normalize to `[0, 1)`.
2. Use `numpy.histogram` with `num_bins` bins.
3. Return `details["bin_edges"]` and `details["bin_counts"]`.

---

### 2. Test Runner

```python
def run_all_tests(
    generator: PRNG,
    n: int = 1_000_000,
    n_spectral: int = 100_000,
    num_bins: int = 1000,
    seed: int = 42
) -> dict[str, TestResult]:
    """Run all statistical tests on a generator.

    Returns dict mapping test names to TestResult objects.
    """
```

**Behavior:**
1. Seed the generator.
2. Generate `n` values.
3. Run each test, collect results.
4. For spectral test, use `n_spectral` pairs.
5. Return all results as dict.

---

### 3. Visualization (`src/tests/visualization.py`)

All functions save PNG files to a specified output directory.

**Common parameters:**
- `output_dir: str = "results/figures"` — directory for saved PNGs
- `dpi: int = 150` — image resolution
- `figsize: tuple = (10, 8)` — figure size in inches

#### 3.1 Plot Functions

```python
def plot_chi_square_comparison(
    results: dict[str, TestResult],
    output_dir: str = "results/figures"
) -> str:
    """Bar chart comparing chi-square p-values across generators.

    Returns path to saved PNG.
    """
```

```python
def plot_autocorrelation(
    results: dict[str, TestResult],
    output_dir: str = "results/figures"
) -> str:
    """Line plot of autocorrelation by lag for each generator.

    Returns path to saved PNG.
    """
```

```python
def plot_spectral(
    spectral_results: dict[str, TestResult],
    output_dir: str = "results/figures"
) -> str:
    """2D scatter plot of (X_n, X_{n+1}) pairs for each generator.

    Subplots: one per generator. LCG shows lines, others fill space.
    Returns path to saved PNG.
    """
```

```python
def plot_runs_z_scores(
    results: dict[str, TestResult],
    output_dir: str = "results/figures"
) -> str:
    """Bar chart of Z-scores from runs test.

    Horizontal lines at ±1.96 (95% confidence).
    Returns path to saved PNG.
    """
```

```python
def plot_histograms(
    hist_results: dict[str, TestResult],
    output_dir: str = "results/figures"
) -> str:
    """Histogram overlay of value distributions.

    All generators on one plot for comparison.
    Returns path to saved PNG.
    """
```

#### 3.2 Report Generator

```python
def generate_all_plots(
    generator_results: dict[str, dict[str, TestResult]],
    output_dir: str = "results/figures"
) -> list[str]:
    """Generate all plots from test results.

    Args:
        generator_results: {generator_name: {test_name: TestResult}}

    Returns:
        List of paths to saved PNG files.
    """
```

**Plot style:**
- Use `matplotlib.pyplot.style.context('seaborn-v0_8-whitegrid')` or similar clean style.
- Consistent color palette across plots (e.g., `tab10` colormap).
- Generators ordered: LCG variants first, then MT19937, xorshift variants, V8, PCG.
- Titles include generator name and test name.
- Axis labels with units where applicable.

**Spectral test subplot layout:**
- 2x3 grid (6 generators + variants).
- Each subplot: scatter plot of pairs, equal aspect ratio, axis labels "X_n" and "X_{n+1}".
- LCG subplot title includes "Structured" marker; others "Random" marker.

---

### 4. Unit Tests

#### `tests/test_statistical_tests.py`

Tests to write:
- `test_chi_square_uniform_input` — uniform random numbers pass chi-square (p > 0.01)
- `test_chi_square_nonuniform_input` — constant values fail chi-square (p < 0.01)
- `test_chi_square_adjusts_bins_small_input` — fewer values than bins triggers bin reduction
- `test_autocorrelation_zero_lag1` — independent values have |r| < 2/sqrt(N)
- `test_autocorrelation_detects_pattern` — alternating bits have high correlation at lag 1
- `test_spectral_returns_pairs` — output contains correct number of pairs
- `test_runs_test_fair_coin` — random bits have |Z| < 1.96
- `test_runs_test_alternating` — alternating bits have high |Z|
- `test_histogram_counts_match_input` — bin counts sum to input length
- `test_test_result_dataclass` — TestResult fields are populated correctly

#### `tests/test_visualization.py`

Smoke tests (verify plots generate without error, not visual correctness):
- `test_plot_chi_square_comparison_creates_file`
- `test_plot_autocorrelation_creates_file`
- `test_plot_spectral_creates_file`
- `test_plot_runs_creates_file`
- `test_plot_histograms_creates_file`
- `test_generate_all_plots_creates_multiple_files`

Each test: create minimal TestResult objects, call plot function, assert PNG file exists and is non-empty.

---

## Exit Criteria

- [ ] All 5 statistical test functions return correct `TestResult`
- [ ] Chi-square test correctly identifies uniform vs non-uniform distributions
- [ ] Autocorrelation detects known patterns (alternating bits)
- [ ] Spectral test produces valid 2D pairs
- [ ] Runs test correctly classifies random vs structured sequences
- [ ] All 5 plot functions generate valid PNG files
- [ ] `generate_all_plots` produces one PNG per test type
- [ ] All unit tests pass
- [ ] No new dependencies beyond `numpy`, `matplotlib`, `scipy`
