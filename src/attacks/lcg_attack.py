"""LCG parameter recovery and prediction attacks."""

from __future__ import annotations

import math
from functools import reduce
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.generators.lcg import LCG


def recover_parameters(observations: list[int], m: int) -> tuple[int, int]:
    """Recover LCG multiplier a and increment c given modulus m.

    Uses two consecutive observations to solve the linear system:
        X1 = (a * X0 + c) mod m
        X2 = (a * X1 + c) mod m

    Requires X1 - X0 to be invertible mod m (gcd == 1).

    Args:
        observations: At least 3 consecutive LCG outputs [X_0, X_1, X_2].
        m: Known modulus.

    Returns:
        (a, c) recovered LCG parameters.

    Raises:
        ValueError: If fewer than 3 observations given.
        ValueError: If X1 - X0 is not invertible mod m.
    """
    if len(observations) < 3:
        raise ValueError("at least 3 observations required")

    x0, x1, x2 = observations[0], observations[1], observations[2]

    # Compute modular inverse of (x1 - x0) mod m
    # pow raises ValueError if the base is not invertible mod m (gcd != 1)
    try:
        inv = pow(x1 - x0, -1, m)
    except ValueError:
        raise ValueError("cannot recover: (X1 - X0) not invertible mod m")

    # Solve for a: a = (x2 - x1) * inv(x1 - x0) mod m
    a = (x2 - x1) * inv % m

    # Solve for c: c = (x1 - a * x0) mod m
    c = (x1 - a * x0) % m

    # Verify against 4th observation if available
    if len(observations) >= 4:
        x3 = observations[3]
        predicted = (a * x2 + c) % m
        if predicted != x3:
            raise ValueError(
                f"verification failed: predicted {predicted} for obs[3], got {x3}"
            )

    return a, c


class LCGPredictor:
    """Predict future LCG outputs given recovered parameters.

    Usage:
        obs = generator.generate(5)
        a, c = recover_parameters(obs[:3], m)
        predictor = LCGPredictor(a, c, m, last_state=obs[-1])
        predicted = predictor.predict(1000)
    """

    def __init__(self, a: int, c: int, m: int, last_state: int):
        self.a = a
        self.c = c
        self.m = m
        self.state = last_state

    def predict(self, n: int) -> list[int]:
        """Return next n LCG outputs (100% accurate if parameters correct)."""
        result = []
        for _ in range(n):
            self.state = (self.a * self.state + self.c) % self.m
            result.append(self.state)
        return result


def _divisors(n: int) -> list[int]:
    """Return all positive divisors of n in ascending order (n > 0)."""
    if n <= 0:
        return []
    small: list[int] = []
    large: list[int] = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            small.append(i)
            if i != n // i:
                large.append(n // i)
    return small + large[::-1]


def _trial_divide(n: int) -> list[int]:
    """Return prime factors of n (with repetition) via trial division."""
    factors: list[int] = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def _reduce_gcd(g: int) -> list[int]:
    """Generate candidate divisors of g by trial-dividing out small prime factors.

    Returns divisors in ascending order. If g has a large prime factor (> 1000)
    that makes full factoring expensive, we still include g itself as a candidate
    (caller tests it).
    """
    candidates: list[int] = []

    # Start by trying g itself
    candidates.append(g)

    # Try dividing out small prime factors repeatedly
    remaining = g
    for factor in _trial_divide(g):
        if factor > 1000:
            # Large prime factor — stop dividing; g itself is the best candidate
            break
        while remaining % factor == 0:
            remaining //= factor
            candidates.append(remaining)

    return sorted(set(candidates))


def recover_modulus_and_parameters(
    observations: list[int],
) -> tuple[int, int, int]:
    """Recover LCG modulus m, multiplier a, and increment c from outputs alone.

    Uses the GCD-of-differences method: for consecutive differences T_i = X_{i+1} - X_i,
    m divides T_i * T_{i+2} - T_{i+1}^2 for all overlapping triples.

    Args:
        observations: At least 5 consecutive LCG outputs.

    Returns:
        (m, a, c) recovered parameters.

    Raises:
        ValueError: If fewer than 5 observations given.
        ValueError: If modulus cannot be recovered.
    """
    if len(observations) < 5:
        raise ValueError("at least 5 observations required")

    # Step 1: compute consecutive differences
    diffs = [observations[i + 1] - observations[i] for i in range(len(observations) - 1)]

    # Step 2: compute GCD of T_i * T_{i+2} - T_{i+1}^2 for overlapping triples
    cross_products = []
    for i in range(len(diffs) - 2):
        val = diffs[i] * diffs[i + 2] - diffs[i + 1] * diffs[i + 1]
        if val != 0:
            cross_products.append(abs(val))

    if not cross_products:
        raise ValueError("cannot recover modulus: all cross-products are zero")

    # GCD over all cross-products
    g = cross_products[0]
    for v in cross_products[1:]:
        g = math.gcd(g, v)

    if g == 0:
        raise ValueError("cannot recover modulus: GCD is zero")

    # Step 3: try candidate divisors of g, smallest first
    # For power-of-2 moduli, g will be a power-of-2 multiple; trial dividing by 2
    # finds the true modulus. For small m (like 101), g may be a multiple.
    candidates = _reduce_gcd(g)

    for candidate_m in candidates:
        if candidate_m <= 1:
            continue
        # Restrict: reject absurdly large moduli where factoring is too expensive
        if candidate_m > 2**40:
            raise ValueError("modulus too large to recover")
        try:
            a, c = recover_parameters(observations, candidate_m)
        except ValueError:
            continue
        # Verify full sequence from obs[0] with (a, c, m)
        state = observations[0]
        ok = True
        for obs in observations[1:]:
            state = (a * state + c) % candidate_m
            if state != obs:
                ok = False
                break
        if ok:
            return candidate_m, a, c

    raise ValueError("cannot recover modulus: no valid divisor found")


def attack_lcg(
    generator: LCG,
    num_observations: int = 10,
    num_predictions: int = 1000,
    modulus_known: bool = True,
) -> dict:
    """Complete LCG attack: recover parameters, predict future outputs, verify.

    Args:
        generator: The LCG instance to attack.
        num_observations: How many consecutive outputs to observe.
        num_predictions: How many subsequent outputs to predict.
        modulus_known: Whether the modulus is known.

    Returns:
        Dict with:
            - recovered_parameters: (a, c) or (m, a, c)
            - actual_next: first min(10, num_predictions) actual outputs
            - predicted_next: first min(10, num_predictions) predicted outputs
            - match_count: how many predictions matched actual
            - accuracy: match_count / num_predictions
    """
    # GlibcLCG truncates output to 15 bits; observed values are not raw states,
    # so recovery of (a, c) from them is impossible — the mapping is many-to-one.
    from src.generators.lcg import GlibcLCG

    if isinstance(generator, GlibcLCG):
        raise ValueError(
            "GlibcLCG truncates output to 15 bits; parameter recovery "
            "requires full state outputs"
        )

    observations = generator.generate(num_observations)

    if modulus_known:
        m = generator.m
        a, c = recover_parameters(observations, m)
        recovered = (a, c)
    else:
        m, a, c = recover_modulus_and_parameters(observations)
        recovered = (m, a, c)

    # Predict next outputs from the last observed state
    predictor = LCGPredictor(a, c, m, last_state=observations[-1])
    predicted = predictor.predict(num_predictions)

    # Collect actual outputs the generator produces next
    actual = generator.generate(num_predictions)

    match_count = sum(p == a for p, a in zip(predicted, actual))

    return {
        "recovered_parameters": recovered,
        "actual_next": actual[:10],
        "predicted_next": predicted[:10],
        "match_count": match_count,
        "accuracy": match_count / num_predictions,
    }
