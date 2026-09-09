"""Statistical tests for PRNG output sequences.

Stage 2 (spec: docs/specs/stage-2-statistical-tests.md). Each test function
takes an array of integers (or floats) produced by a generator and returns a
structured TestResult. No attack logic, no benchmarking — only the five test
type(s) the spec calls for.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class TestResult:
    """Structured outcome of a single statistical test.

    Fields mirror the spec's common result format exactly. ``passed`` is a
    boolean threshold decision; ``details`` carries extra data (bin counts,
    per-lag correlations, spectral pairs, etc.) for reporting and plotting.
    """

    test_name: str
    generator_name: str
    statistic: float
    p_value: float
    passed: bool
    details: dict = field(default_factory=dict)
    comment: str = ""

    # Prevent pytest from collecting this dataclass as a test class just
    # because its name starts with "Test".
    __test__ = False


def _normalize(values: list) -> np.ndarray:
    """Normalize values to floats in [0, 1).

    Accepts either integers or already-normalized floats. For integer input,
    generators differ in bit-width (32-bit standard, 64-bit V8), so the modulus
    is inferred from the largest output's bit length rather than plumbed through
    the spec-fixed signatures. Float input (from generate_floats) is already in
    [0, 1) and used as-is.
    """
    if isinstance(values[0], float):
        return np.asarray(values, dtype=np.float64)
    modulus = 1 << max(v.bit_length() for v in values)
    return np.asarray(values, dtype=np.float64) / modulus


def chi_square_test(
    values: list[int],
    num_bins: int = 1000,
    generator_name: str = "unknown",
) -> TestResult:
    """Chi-square goodness-of-fit test for uniformity.

    Divides the normalized [0, 1) range into ``num_bins`` equal bins, counts
    observed frequencies, and compares them to the uniform expectation.

    Edge case: with fewer values than bins the test is meaningless (many empty
    bins inflate chi-square), so bin count is reduced to ``len(values) // 10``
    and the reduction is noted in the comment.
    """
    n = len(values)
    comment = ""

    if n < num_bins:
        num_bins = max(n // 10, 1)
        comment = (
            f"input too small ({n} values for {num_bins} bins); "
            f"reduced num_bins to {num_bins}"
        )

    normalized = _normalize(values)
    counts, _ = np.histogram(normalized, bins=num_bins, range=(0.0, 1.0))
    expected = n / num_bins

    # Chi-square statistic: sum of (observed - expected)^2 / expected. Empty
    # bins simply contribute their expected value to the sum, which is exactly
    # the inflation the spec wants surfaced for poor uniformity.
    statistic = float(np.sum((counts - expected) ** 2 / expected))
    df = num_bins - 1
    p_value = float(stats.chi2.sf(statistic, df=df))

    # Wide acceptance band (1%..99%); only gross departures from uniformity
    # fail, matching the spec's goal of catching structural defects.
    passed = 0.01 <= p_value <= 0.99

    return TestResult(
        test_name="chi_square",
        generator_name=generator_name,
        statistic=statistic,
        p_value=p_value,
        passed=passed,
        details={
            "num_bins": num_bins,
            "bin_counts": counts.tolist(),
            "expected": expected,
            "n": n,
        },
        comment=comment,
    )


def autocorrelation_test(
    values: list[int],
    lags: list[int] = [1, 2, 5, 10],
    generator_name: str = "unknown",
) -> TestResult:
    """Pearson autocorrelation at the given lags.

    For an ideal PRNG, successive values are uncorrelated, so Pearson r at any
    lag should be near 0. Works on the raw ints directly — the correlation is
    scale-invariant, so normalization is unnecessary.
    """
    a = np.asarray(values, dtype=np.float64)
    n = len(a)

    correlations: dict[int, float] = {}
    for t in lags:
        # Compare each value against the value t positions ahead. numpy's
        # corrcoef returns the 2x2 correlation matrix; [0, 1] is r(x, y).
        x = a[:-t]
        y = a[t:]
        r = float(np.corrcoef(x, y)[0, 1])
        correlations[t] = r

    # 95% confidence band is roughly 2/sqrt(N); require every lag to fall
    # inside it. statistic is the worst (largest magnitude) correlation.
    threshold = 2.0 / np.sqrt(n)
    statistic = max(abs(r) for r in correlations.values())
    passed = all(abs(r) < threshold for r in correlations.values())

    return TestResult(
        test_name="autocorrelation",
        generator_name=generator_name,
        statistic=statistic,
        p_value=float("nan"),  # not a hypothesis test; per-lag band is reported
        passed=passed,
        details={
            "correlations": correlations,
            "threshold": float(threshold),
            "n": n,
        },
        comment="expected |r| < 2/sqrt(N) for all lags",
    )


def spectral_test_2d(
    values: list[int],
    sample_size: int = 100_000,
    generator_name: str = "unknown",
) -> TestResult:
    """Extract normalized 2D pairs (X_n, X_{n+1}) for lattice inspection.

    Reveals structure: a bad LCG shows parallel lines, good generators fill
    the space. This is a visual test only — pass is always True and the caller
    judges layout from the returned pairs.
    """
    # Need sample_size+1 values to form sample_size consecutive pairs.
    sample = values[: sample_size + 1]

    x = sample[:-1]
    y = sample[1:]

    # Normalize both axes with the bit-length modulus inferred from the whole
    # sample; see _normalize for the bit-width rationale.
    modulus = 1 << max(max(v.bit_length() for v in x), max(v.bit_length() for v in y))
    pairs_x = [float(v) / modulus for v in x]
    pairs_y = [float(v) / modulus for v in y]

    return TestResult(
        test_name="spectral",
        generator_name=generator_name,
        statistic=float("nan"),
        p_value=float("nan"),
        passed=True,
        details={
            "pairs_x": pairs_x,
            "pairs_y": pairs_y,
            "sample_size": len(pairs_x),
        },
        comment="visual test — inspect lattice structure",
    )


def runs_test(
    values: list[int],
    generator_name: str = "unknown",
) -> TestResult:
    """Wald-Wolfowitz runs test for independence.

    Converts each value to a single bit (bit 0), counts runs of consecutive
    identical bits, and compares the observed run count to the expectation
    under randomness via the Z statistic. Alternating or clustered bits yield
    a large |Z|.
    """
    bits = [v & 1 for v in values]
    n = len(bits)
    n0 = bits.count(0)
    n1 = bits.count(1)

    # Degenerate input: if either symbol never appears there is exactly one
    # run and variance is zero, so Z is undefined. Report the failure loudly.
    if n0 == 0 or n1 == 0:
        return TestResult(
            test_name="runs",
            generator_name=generator_name,
            statistic=1.0,
            p_value=float("nan"),
            passed=False,
            details={"n0": n0, "n1": n1, "n": n},
            comment="degenerate input (only one bit value present)",
        )

    observed_runs = 1 + sum(1 for i in range(1, n) if bits[i] != bits[i - 1])

    expectation = (2 * n0 * n1) / n + 1
    variance = (2 * n0 * n1 * (2 * n0 * n1 - n)) / (n * n * (n - 1))
    std = np.sqrt(variance)
    z = (observed_runs - expectation) / std

    # Two-tailed 95% cutoff; |Z| >= 1.96 signals non-random structure.
    passed = bool(abs(z) < 1.96)

    return TestResult(
        test_name="runs",
        generator_name=generator_name,
        statistic=float(observed_runs),
        p_value=float("nan"),  # Z-based decision; report Z in details
        passed=passed,
        details={
            "z": float(z),
            "observed_runs": observed_runs,
            "expected_runs": float(expectation),
            "variance": float(variance),
            "n0": n0,
            "n1": n1,
            "n": n,
        },
        comment="expected |Z| < 1.96 for random bits",
    )


def histogram_data(
    values: list[int],
    num_bins: int = 200,
    generator_name: str = "unknown",
) -> TestResult:
    """Compute histogram bin counts for the value distribution.

    Returns bin edges and counts in details for plotting. ``statistic`` is a
    rough spread metric: max count minus the uniform expectation (positive when
    values cluster, near zero for uniform).
    """
    normalized = _normalize(values)
    counts, bin_edges = np.histogram(normalized, bins=num_bins, range=(0.0, 1.0))

    expected = len(values) / num_bins
    statistic = float(np.max(counts) - expected)

    return TestResult(
        test_name="histogram",
        generator_name=generator_name,
        statistic=statistic,
        p_value=float("nan"),
        passed=True,
        details={
            "bin_edges": bin_edges.tolist(),
            "bin_counts": counts.tolist(),
            "num_bins": num_bins,
            "n": len(values),
        },
        comment="visualization data",
    )


def run_all_tests(
    generator,
    n: int = 1_000_000,
    n_spectral: int = 100_000,
    num_bins: int = 1000,
    seed: int = 42,
) -> dict[str, TestResult]:
    """Run all statistical tests on a generator and collect the results.

    Seeds the generator, draws ``n`` values once, and runs every test over that
    sample so the tests share a single deterministic stream. Spectral uses
    ``n_spectral`` pairs from the head of the sample.
    """
    generator.seed(seed)
    values = generator.generate(n)
    floats = generator.generate_floats(n)
    generator_name = type(generator).__name__

    # The histogram characterizes the float distribution; floats are already
    # in [0, 1) so pass them straight through the normalization.
    return {
        "chi_square": chi_square_test(values, num_bins=num_bins, generator_name=generator_name),
        "autocorrelation": autocorrelation_test(values, generator_name=generator_name),
        "spectral": spectral_test_2d(values, sample_size=n_spectral, generator_name=generator_name),
        "runs": runs_test(values, generator_name=generator_name),
        "histogram": histogram_data(floats, generator_name=generator_name),
    }
