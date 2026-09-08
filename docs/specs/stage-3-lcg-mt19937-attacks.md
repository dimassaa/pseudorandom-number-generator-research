# Stage 3 Spec: LCG and MT19937 Attacks

## Goal

Implement state/parameter recovery attacks on LCG (known and unknown modulus) and full state recovery + prediction for MT19937, verified to 100% bit-identical accuracy on 1000 predicted outputs.

## Scope

LCG attack module (known modulus, unknown modulus), MT19937 attack module (inverse tempering, state recovery, prediction), unit tests with cleanup defined below. No xorshift attacks (Stage 4), no benchmark.

## Dependencies

- `src/generators/lcg.py` — LCG class
- `src/generators/mt19937.py` — MT19937 class
- Python built-in `random` module (source of outputs for MT attack)

---

## Files to Create

| File | Responsibility |
|------|---------------|
| `src/attacks/lcg_attack.py` | LCG parameter recovery + prediction |
| `src/attacks/mt19937_attack.py` | Inverse tempering, state recovery, prediction |
| `src/attacks/__init__.py` | Package exports |
| `tests/test_lcg_attack.py` | LCG attack unit tests |
| `tests/test_mt19937_attack.py` | MT19937 attack unit tests |

---

## Specifications

### 1. LCG Attack (`src/attacks/lcg_attack.py`)

#### 1.1 Known Modulus Attack

```python
def recover_parameters(observations: list[int], m: int) -> Tuple[int, int]:
    """Recover LCG multiplier a and increment c given modulus m.

    Args:
        observations: At least 3 consecutive LCG outputs [X_0, X_1, X_2]
        m: Known modulus

    Returns:
        (a, c) LCG parameters.

    Raises:
        ValueError: If fewer than 3 observations given.
        ValueError: If X_1 == X_0 (mod m) — cannot recover from repeated value.
    """
```

**Algorithm:**
1. `a = (X_2 - X_1) * inverse(X_1 - X_0, m) mod m`
   - Using `pow(X_1 - X_0, -1, m)` (Python 3.8+ modular inverse).
2. `c = (X_1 - a * X_0) mod m`
3. Return `(a, c)`.
4. If `X_1 - X_0` is not invertible mod m (gcd != 1), raise `ValueError`.
5. **Verify:** check `(a * X_2 + c) % m == X_3` if a 4th observation is provided. If not, warn in return tuple.

```python
class LCGPredictor:
    """Predict future LCG outputs given a recovered parameter set.

    Usage:
        obs = generator.generate(5)          # observed outputs
        a, c = recover_parameters(obs[:3], m)
        predictor = LCGPredictor(a, c, m, last_state=obs[2])
        predicted = predictor.predict(1000)  # next 1000 outputs
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
```

**Verification requirement:** Predicted outputs must match actual generator output bit-for-bit for at least 1000 subsequent values.

#### 1.2 Unknown Modulus Attack

```python
def recover_modulus_and_parameters(observations: list[int]) -> Tuple[int, int, int]:
    """Recover LCG modulus m, multiplier a, and increment c.

    Uses the difference method: m divides the GCD of
    T_n * T_{n+2} - T_{n+1}^2 for several n, where T_n = X_{n+1} - X_n.

    Args:
        observations: At least 5 consecutive LCG outputs.

    Returns:
        (m, a, c) recovered parameters.

    Raises:
        ValueError: If fewer than 5 observations given.
        ValueError: If modulus cannot be determined (all differences are 0).
    """
```

**Algorithm:**
1. Differences: `T = [X_{i+1} - X_i for i in range(len(observations) - 1)]`
2. Compute `candidate = gcd(T[0]*T[2] - T[1]^2,  T[1]*T[3] - T[2]^2, ...)` for all overlapping triples.
3. Try to factor the GCD into a plausible LCG modulus:
   - If `m` is a power of 2 / small factor, reduce.
   - If GCD is small (e.g., < 1000), use directly.
   - If GCD is large, may be multiple of actual m — test each divisor (or candidates via factorization).
4. With recovered `m`, call `recover_parameters(observations, m)` to get `(a, c)`.
5. Verify: predict 4th value from first 3, compare to actual.
6. If verification fails, raise `ValueError` with diagnostic message.

**Caveat:** For moduli that are large primes and with limited observations, the GCD method may return a multiple of the true m. In that case, restrict search to divisors of the GCD that also satisfy the LCG recurrence. Limit divisor testing to `m <= 2^40` (beyond that, factorization is expensive).

#### 1.3 High-Level Attack Function

```python
def attack_lcg(
    generator: LCG,
    num_observations: int = 10,
    num_predictions: int = 1000,
    modulus_known: bool = True,
) -> dict:
    """Complete LCG attack: recover params, predict, verify.

    Args:
        generator: The LCG instance to attack.
        num_observations: How many consecutive outputs to observe.
        num_predictions: How many subsequent outputs to predict.
        modulus_known: Whether modulus is known.

    Returns:
        Dict with keys:
            - recovered_parameters: (a, c) or (m, a, c)
            - actual_next: first 10 actual outputs
            - predicted_next: first 10 predicted outputs
            - match_count: how many predictions matched actual (should be 1000)
            - accuracy: match_count / num_predictions
    """
```

---

### 2. MT19937 Attack (`src/attacks/mt19937_attack.py`)

#### 2.1 Inverse Tempering Functions

```python
def inverse_right_shift_xor(y: int, shift: int) -> int:
    """Invert operation y ^= y >> shift (bit by bit).

    Original: y = y ^ (y >> shift)
    Inverse: reconstruct original from tempered output.
    """
```

**Algorithm (bit-by-bit from MSB):**
```
For a 32-bit value:
    result = 0
    for i in range(31, -1, -1):  # from MSB to LSB
        bit = (y >> i) & 1
        # XOR with previously set higher bit that shifted into position i
        if i + shift < 32 and ((result >> (i + shift)) & 1):
            bit ^= 1
        result |= bit << i
    return result
```

**Verification:** For all random 32-bit inputs, `inverse_right_shift_xor(x ^ (x >> shift), shift) == x`.

```python
def inverse_left_shift_xor_mask(y: int, shift: int, mask: int) -> int:
    """Invert operation y ^= (y << shift) & mask (bit by bit).

    Original: y = y ^ ((y << shift) & mask)
    Inverse: reconstruct original from tempered output.
    """
```

**Algorithm (bit-by-bit from LSB):**
```
For a 32-bit value:
    result = 0
    for i in range(0, 32):  # from LSB to MSB
        bit = (y >> i) & 1
        # XOR with previously set lower bit that shifted into position i
        if i - shift >= 0 and mask bit i is set:
            if ((result >> (i - shift)) & 1):
                bit ^= 1
        result |= bit << i
    return result
```

**Verification:** For all random 32-bit inputs, `inverse_left_shift_xor_mask(x ^ ((x << shift) & mask), shift, mask) == x`.

#### 2.2 Full Tempering Inverse

```python
def untemper(y: int) -> int:
    """Apply inverse tempering to recover original MT state value.

    Applies the four inverse operations in reverse order.
    """
```

**Algorithm (reverse of `_temper`):**
1. `y = inverse_right_shift_xor(y, 18)`
2. `y = inverse_left_shift_xor_mask(y, 15, 0xefc60000)`
3. `y = inverse_left_shift_xor_mask(y, 7, 0x9d2c5680)`
4. `y = inverse_right_shift_xor(y, 11)`
5. Return y.

**Verification:** `untemper(generator._temper(x)) == x` for random x.

#### 2.3 State Recovery Attack

```python
def recover_mt19937_state(observations: list[int]) -> MT19937:
    """Recover full MT19937 state from 624 consecutive outputs.

    Args:
        observations: Exactly 624 consecutive 32-bit outputs.

    Returns:
        A new MT19937 instance with recovered state, ready to predict.

    Raises:
        ValueError: If len(observations) < 624.
    """
```

**Algorithm:**
1. Untemper all 624 outputs.
2. Create `MT19937()` instance.
3. Set `instance.mt = [untemper(x) for x in observations]`.
4. Set `instance.index = 624` (triggers twist on next call).
5. Return.

**Critical correctness note:** This works because MT19937 with the same `mt` array and `index=624` generates identical subsequent outputs to the original generator (at the observed point). The twist is deterministic given the full mt array.

#### 2.4 High-Level Attack Function

```python
def attack_mt19937(
    num_observations: int = 624,
    num_predictions: int = 1000,
    seed: int = 2024,
) -> dict:
    """Complete MT19937 attack using Python's random module as target.

    Args:
        num_observations: Must be >= 624.
        num_predictions: How many future outputs to predict.
        seed: Seed for Python's random module (defines target sequence).

    Returns:
        Dict with:
            - state_recovered: bool
            - actual_next: first 10 actual future outputs
            - predicted_next: first 10 predicted outputs
            - match_count: number of exact matches (expect num_predictions)
            - accuracy: match_count / num_predictions
    """
```

**Algorithm:**
1. Seed `random.Random(seed)`.
2. Collect 624 consecutive `random.getrandbits(32)` values.
3. Recover state with `recover_mt19937_state`.
4. Generate 1000 predictions from recovered instance.
5. Generate 1000 actual values from `random`.
6. Compare element-wise; count matches.
7. Return result dict.

**Important:** The recovered MT19937 outputs are 32-bit unsigned integers. Python's `random.getrandbits(32)` returns the same. There is no sign/byte-order mismatch — both match directly. This is why output comparison (not state comparison) is the verification method we chose.

---

### 3. Unit Tests

#### `tests/test_lcg_attack.py`

Tests to write:
- `test_recover_parameters_known_modulus` — generate LCG, collect 3 outputs, recover a,c, verify against actual params
- `test_recover_parameters_different_seed` — test with multiple seeds
- `test_recover_parameters_insufficient_data` — fewer than 3 observations raises ValueError
- `test_recover_parameters_non_invertible` — X_1 == X_0 (mod m) raises ValueError
- `test_lcg_predictor_1000_outputs` — predict 1000, compare to actual, all match
- `test_recover_modulus_unknown` — generate LCG with known m, hide it, recover m via GCD, verify
- `test_recover_large_modulus` — test with m = 2^32 (Numerical Recipes LCG)
- `test_recover_modulus_insufficient_data` — fewer than 5 observations raises ValueError
- `test_attack_lcg_known_modulus_full` — end-to-end, accuracy == 1.0
- `test_attack_lcg_unknown_modulus_full` — end-to-end, accuracy == 1.0

#### `tests/test_mt19937_attack.py`

Tests to write:
- `test_inverse_right_shift_xor_roundtrip` — random inputs, verify roundtrip
- `test_inverse_right_shift_xor_shift_variants` — test with shift 11 and 18
- `test_inverse_left_shift_xor_mask_roundtrip` — random inputs, verify roundtrip
- `test_inverse_left_shift_xor_mask_masks` — test with masks 0x9d2c5680 and 0xefc60000
- `test_untemper_roundtrip` — random x, `untemper(_temper(x)) == x`
- `test_recover_mt19937_state_matches` — recover from 624 outputs, generate, compare to original
- `test_recover_mt19937_insufficient_data` — < 624 observations raises ValueError
- `test_attack_mt19937_single_seed` — attack with seed=2024, accuracy == 1.0
- `test_attack_mt19937_multiple_seeds` — attack with seeds [0, 1, 42, 999], all accuracy == 1.0
- `test_attack_mt19937_1000_predictions` — after recovery, all 1000 predictions match exactly

---

## Exit Criteria

- [ ] LCG known-modulus recovery works for all 4 LCG variants
- [ ] LCG unknown-modulus recovery works for small and large moduli
- [ ] LCG prediction accuracy is 100% (bit-identical) over 1000 outputs
- [ ] MT19937 inverse tempering functions roundtrip correctly
- [ ] MT19937 state recovery works from 624 consecutive outputs
- [ ] MT19937 prediction accuracy is 100% (bit-identical) over 1000 outputs
- [ ] All unit tests pass
- [ ] No new dependencies beyond Stage 1 + Stage 2
