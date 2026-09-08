# Stage 1 Spec: Generator Implementations

## Goal

Implement all PRNG generators as standalone, testable Python modules with a unified interface, ready for statistical testing and attack demonstrations.

## Scope

6 generator modules + base class + unit tests. No statistical analysis, no attacks, no visualization.

## Decisions Applied

- Python 3.10+
- Hardcoded V8 entropy seed
- Full standalone MT19937
- PCG32-EXS (XSH-RR)
- Top-level `tests/` directory
- PNG figures (not relevant here, but noted for consistency)

---

## Files to Create

| File | Responsibility |
|------|---------------|
| `src/generators/base.py` | Abstract base class for all generators |
| `src/generators/lcg.py` | LCG with 4 parameter sets |
| `src/generators/mt19937.py` | Full MT19937 implementation |
| `src/generators/xorshift.py` | xorshift32, xorshift64, xorshift128+ |
| `src/generators/v8_random.py` | V8 Math.random emulation |
| `src/generators/pcg.py` | PCG32-EXS (XSH-RR) |
| `src/generators/__init__.py` | Package exports |
| `tests/test_lcg.py` | LCG unit tests |
| `tests/test_mt19937.py` | MT19937 unit tests |
| `tests/test_xorshift.py` | Xorshift unit tests |
| `tests/test_v8_random.py` | V8 emulation unit tests |
| `tests/test_pcg.py` | PCG32 unit tests |

---

## Specifications

### 1. Base Class (`src/generators/base.py`)

Abstract base class defining the generator interface.

```python
from abc import ABC, abstractmethod

class PRNG(ABC):
    """Base class for all pseudorandom number generators."""

    @abstractmethod
    def seed(self, value: int) -> None:
        """Set the generator state from an integer seed."""
        ...

    @abstractmethod
    def next_int(self) -> int:
        """Return the next 32-bit unsigned integer in the sequence."""
        ...

    def next_float(self) -> float:
        """Return the next float in [0.0, 1.0) by normalizing next_int()."""
        return self.next_int() / 0x100000000  # divide by 2^32

    def generate(self, n: int) -> list[int]:
        """Generate n consecutive 32-bit integers."""
        return [self.next_int() for _ in range(n)]

    def generate_floats(self, n: int) -> list[float]:
        """Generate n floats in [0.0, 1.0)."""
        return [self.next_float() for _ in range(n)]
```

**Design rationale:** `next_float` uses `/ 0x100000000` (2^32) for uniform [0,1) distribution. The `generate` method returns a list for test compatibility, not a generator, to allow indexed access.

---

### 2. LCG (`src/generators/lcg.py`)

Four LCG variants as classes inheriting from `PRNG`.

**Class structure:**

```python
class LCG(PRNG):
    """Linear Congruential Generator: X_{n+1} = (a * X_n + c) mod m."""

    def __init__(self, a: int, c: int, m: int, seed_value: int = 0):
        self.a = a
        self.c = c
        self.m = m
        self.state = seed_value % m

    def seed(self, value: int) -> None:
        self.state = value % self.m

    def next_int(self) -> int:
        self.state = (self.a * self.state + self.c) % self.m
        return self.state

    def next_float(self) -> float:
        return self.next_int() / self.m
```

**Note on `next_float`:** LCG override uses `/ self.m` instead of `/ 0x100000000` because modulus varies (not always 2^32).

**Preset variants (factory functions or class methods):**

| Variant | a | c | m | Notes |
|---------|---|---|---|-------|
| `ansi_c` | 1103515245 | 12345 | 2^31 | C `rand()` |
| `numerical_recipes` | 1664525 | 1013904223 | 2^32 | Numerical Recipes |
| `glibc` | 1103515245 | 12345 | **2^32** | glibc `rand_r`, upper 15 bits |
| `bad` | 3 | 7 | 101 | Intentionally weak |

**glibc output method:** The glibc variant uses `m = 2^32` internally and overrides `next_int()` to return the **upper 15 bits**: `(state >> 16) & 0x7fff`. This matches classic glibc `rand_r` behavior (take high-order bits) and genuinely differs from the ANSI C variant (which returns the full 31-bit state modulo 2^31). Note: at `m = 2^31`, a lower-bits mask would be a no-op (state is always `< 2^31`), so `m = 2^32` is required to make the high-bit extraction meaningful.

**glibc `next_float`:** Because glibc's `next_int()` returns a 15-bit quantity in `[0, 2^15)`, it must ALSO override `next_float()` to normalize by `2^15` (not `m = 2^32`): `return self.next_int() / 2**15`. Dividing by `2^32` would cluster all floats near zero (`[0, 2^-17)`), not uniform `[0,1)`.

**Verification against reference:** The ANSI C LCG with seed=0 must produce the first outputs `12345, 1406932606, 654583775, 1449466924, ...`, computed from the recurrence `state = (1103515245 * state + 12345) mod 2^31` starting at state 0.

---

### 3. MT19937 (`src/generators/mt19937.py`)

Full Mersenne Twister implementation matching Python's `random` module behavior.

**Constants:**
- `N = 624` (degree of recurrence)
- `M = 397` (middle word offset)
- `MATRIX_A = 0x9908b0df` (constant vector a)
- `UPPER_MASK = 0x80000000` (most significant w-r bits)
- `LOWER_MASK = 0x7fffffff` (least significant r bits)

**State:**
- `mt: list[int]` — array of 624 32-bit integers
- `index: int` — current position in mt (0..623, with 624 triggering twist)

**Key methods:**

```python
class MT19937(PRNG):
    N = 624
    M = 397
    MATRIX_A = 0x9908b0df
    UPPER_MASK = 0x80000000
    LOWER_MASK = 0x7fffffff

    def seed(self, value: int) -> None:
        """Initialize the 624-word state array from an integer seed.

        Matches Python's random.seed(value) for integer seeds.
        CRITICAL: CPython seeds integers via init_by_array(), NOT the direct
        init_genrand() formula. init_by_array first runs init_genrand(19650218)
        (a fixed constant), then folds the integer seed's 32-bit little-endian
        words through two mixing passes, and finally forces mt[0] = 0x80000000.
        Replicate CPython's _randommodule.c exactly so the untempered state
        recovers identically (required for the Stage 3 attack).
        """
        # Split abs(value) into little-endian 32-bit words (init_key).
        n = abs(value)
        init_key = [0] if n == 0 else []
        while n:
            init_key.append(n & 0xffffffff)
            n >>= 32
        # _init_by_array: init_genrand(19650218) then two mixing passes,
        # CPIython multiplier constants 1664525 and 1566083941, force mt[0] MSB.
        ...

    def _twist(self) -> None:
        """Generate next N words from the algorithm."""
        for i in range(624):
            y = (self.mt[i] & UPPER_MASK) | (self.mt[(i+1) % 624] & LOWER_MASK)
            self.mt[i] = self.mt[(i + M) % 624] ^ (y >> 1)
            if y & 1:
                self.mt[i] ^= MATRIX_A

    def _temper(self, y: int) -> int:
        """Apply tempering transformations to output. Mask to 32 bits at the end."""
        y ^= (y >> 11)
        y ^= (y << 7) & 0x9d2c5680
        y ^= (y << 15) & 0xefc60000
        y ^= (y >> 18)
        return y & 0xffffffff

    def next_int(self) -> int:
        if self.index >= 624:
            self._twist()
        y = self.mt[self.index]
        self.index += 1
        return self._temper(y)
```

**Critical alert: do NOT use the naive `init_genrand` seeding (`mt[0]=seed; mt[i]=1812433253*(...)`).** It does NOT match CPython and will fail the acid test (`MT19937().seed(42).generate(10000) == random.Random(42).getrandbits(32)`). The binding requirement is byte-identical output to CPython; the seeding algorithm must follow CPython's `init_by_array`. Also: CPython `random()` uses 53-bit resolution (two words) — our inherited `next_float` (div by 2^32) is intentionally NOT bit-identical to `random.random()`, which is correct per spec.

**Critical verification:** After seeding with `seed(42)`, the first 5 outputs from our implementation must match Python's `random.Random(42).getrandbits(32)` exactly. This is the primary correctness test.

**Additional verification:** `random.getstate()` returns `(3, tuple_of_625_ints, None)`. After seeding our MT with the same seed, `self.mt` must match `random.getstate()[1][:624]` (element-wise, accounting for Python's internal tuple format).

---

### 4. Xorshift (`src/generators/xorshift.py`)

Three xorshift variants in one module.

#### 4.1 xorshift32

```python
class XorShift32(PRNG):
    """32-bit xorshift with period 2^32 - 1."""

    def __init__(self, seed_value: int = 0):
        self.state = seed_value & 0xffffffff
        if self.state == 0:
            self.state = 1  # xorshift requires non-zero state

    def seed(self, value: int) -> None:
        self.state = value & 0xffffffff
        if self.state == 0:
            self.state = 1

    def next_int(self) -> int:
        self.state ^= (self.state << 13) & 0xffffffff
        self.state ^= (self.state >> 17)
        self.state ^= (self.state << 5) & 0xffffffff
        return self.state
```

**Algorithm:** Standard Marsaglia xorshift32 with shifts (13, 17, 5). State must never be 0 (degenerate).

#### 4.2 xorshift64

```python
class XorShift64(PRNG):
    """64-bit xorshift with period 2^64 - 1."""

    def __init__(self, seed_value: int = 0):
        self.state = seed_value & 0xffffffffffffffff
        if self.state == 0:
            self.state = 1

    def seed(self, value: int) -> None:
        self.state = value & 0xffffffffffffffff
        if self.state == 0:
            self.state = 1

    def next_int(self) -> int:
        self.state ^= (self.state << 13) & 0xffffffffffffffff
        self.state ^= (self.state >> 7)
        self.state ^= (self.state << 17) & 0xffffffffffffffff
        return self.state & 0xffffffff  # Return lower 32 bits
```

**Note:** Internal state is 64-bit, but output is 32-bit (lower half) for consistency with other generators.

#### 4.3 xorshift128+

```python
class XorShift128Plus(PRNG):
    """xorshift128+ with 128-bit state, period 2^128 - 1."""

    def __init__(self, s0: int = 0, s1: int = 0):
        self.s0 = s0 & 0xffffffffffffffff
        self.s1 = s1 & 0xffffffffffffffff

    def seed(self, value: int) -> None:
        """Derive two 64-bit states from single seed using splitmix64."""
        self.s0 = self._splitmix64(value)
        self.s1 = self._splitmix64(self.s0)

    @staticmethod
    def _splitmix64(x: int) -> int:
        """SplitMix64 generator for seeding xorshift128+."""
        x = (x + 0x9e3779b97f4a7c15) & 0xffffffffffffffff
        z = x
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9 & 0xffffffffffffffff
        z = (z ^ (z >> 27)) * 0x94d049bb133111eb & 0xffffffffffffffff
        return z ^ (z >> 31)

    def next_int(self) -> int:
        x = self.s0
        y = self.s1
        self.s0 = y
        x ^= (x << 23) & 0xffffffffffffffff
        self.s1 = x ^ y ^ (x >> 17) ^ (y >> 26)
        result = (self.s1 + y) & 0xffffffffffffffff
        return result & 0xffffffff  # Lower 32 bits
```

**Critical:** The `_splitmix64` seeding method must produce the same state as V8's actual initialization when given the hardcoded seed. The constants `0x9e3779b97f4a7c15`, `0xbf58476d1ce4e5b9`, `0x94d049bb133111eb` are from V8 source.

---

### 5. V8 Random (`src/generators/v8_random.py`)

V8 `Math.random()` emulation using xorshift128+ with 53-bit float conversion.

```python
class V8Random(PRNG):
    """Emulation of V8 engine's Math.random() (xorshift128+)."""

    # Hardcoded V8 entropy seed (from V8 source: src/base-random-number-generator.h)
    V8_SEED_S0 = 0x012de6b2
    V8_SEED_S1 = 0x09501088

    def __init__(self):
        self.impl = XorShift128Plus(self.V8_SEED_S0, self.V8_SEED_S1)

    def seed(self, value: int) -> None:
        """Re-seed the internal xorshift128+ state."""
        self.impl.seed(value)

    def next_int(self) -> int:
        """Return raw 64-bit xorshift128+ result (for attack compatibility)."""
        x = self.impl.s0
        y = self.impl.s1
        self.impl.s0 = y
        x ^= (x << 23) & 0xffffffffffffffff
        self.impl.s1 = x ^ y ^ (x >> 17) ^ (y >> 26)
        return (self.impl.s1 + y) & 0xffffffffffffffff

    def next_float(self) -> float:
        """Return 53-bit float as JavaScript's Math.random() does.

        V8 uses: (xorshift128plus_result >> 11) / 2^53
        """
        raw = self.next_int()
        return (raw >> 11) / (1 << 53)
```

**V8 float conversion detail:** JavaScript's `Number` is IEEE 754 double (53-bit mantissa). V8's `Math.random()` shifts right by 11 bits to get 53 significant bits, then divides by `2^53`. This produces values in `[0, 1)`.

**Verification:** With the hardcoded seed, the first 5 outputs must match Node.js `Math.random()` sequence exactly (when seeded identically). Since V8 doesn't expose seeding, we verify by confirming our implementation produces the expected internal state transition.

---

### 6. PCG32 (`src/generators/pcg.py`)

PCG32-EXS (XSH-RR) variant — 64-bit state, 32-bit output.

```python
class PCG32(PRNG):
    """PCG32-EXS (XSH-RR) pseudorandom number generator.

    State: 64-bit LCG + 64-bit increment (stream selector).
    Output: 32-bit via XSH-RR (xorshift high, random rotation).
    """

    def __init__(self, init_state: int = 0, init_seq: int = 0):
        self.state = 0
        self.inc = (init_seq << 1) | 1  # Ensure odd increment
        self._advance()  # Warm up
        self.state = self.state + init_state
        self._advance()

    def seed(self, value: int) -> None:
        """Reinitialize with new seed (stream defaults to 0)."""
        self.__init__(value, 0)

    def _advance(self) -> None:
        """Advance LCG state: state = state * 6364136223846793005 + inc."""
        oldstate = self.state
        self.state = oldstate * 6364136223846793005 + self.inc

    def next_int(self) -> int:
        """Generate next 32-bit output via XSH-RR."""
        oldstate = self.state
        self._advance()
        # XSH-RR output function
        xorshifted = ((oldstate >> 18) ^ oldstate) >> 27
        rot = oldstate >> 59
        return self._rotr32(xorshifted, rot)

    @staticmethod
    def _rotr32(value: int, rot: int) -> int:
        """32-bit right rotation."""
        return ((value >> rot) | (value << (32 - rot))) & 0xffffffff
```

**LCG multiplier:** `6364136223846793005` is the official PCG multiplier (from L'Ecuyer's tables). The increment is `(init_seq << 1) | 1` — standard PCG convention ensuring full period.

**Verification:** Seed with `PCG32(42, 54)` — first output must match reference PCG32 implementation. The reference values are documented in PCG paper (Figure 4.2).

---

### 7. Package Exports (`src/generators/__init__.py`)

```python
from .lcg import LCG
from .mt19937 import MT19937
from .xorshift import XorShift32, XorShift64, XorShift128Plus
from .v8_random import V8Random
from .pcg import PCG32
```

---

### 8. Unit Tests

#### `tests/test_lcg.py`

Tests to write:
- `test_lcg_ansi_c_sequence` — seed(0), verify first 10 outputs match C `rand()` reference
- `test_lcg_numerical_recipes_sequence` — seed(0), verify first 10 outputs
- `test_lcg_glibc_upper_bits` — seed(0), verify glibc (m=2^32, upper 15 bits) returns values in [0, 2^15) that differ from ANSI C for the same seed
- `test_lcg_glibc_next_float_uniform` — verify glibc next_float() spans [0,1) (max value close to 1.0, not clustered near zero) — guards against the divide-by-2^32 bug
- `test_lcg_bad_lattice` — generate 1000 pairs `(X_n, X_{n+1})`, assert they lie on at most `m` distinct planes (verify structural weakness)
- `test_lcg_next_float_range` — all outputs in `[0.0, 1.0)`
- `test_lcg_reseed` — seed, generate, re-seed same value, verify identical sequence

#### `tests/test_mt19937.py`

Tests to write:
- `test_mt19937_matches_python_random` — seed(42), compare first 1000 outputs to `random.Random(42).getrandbits(32)`
- `test_mt19937_matches_python_random_10k` — same for 10000 outputs (ensures twist is triggered)
- `test_mt19937_period_624` — verify twist triggers exactly at index 624
- `test_mt19937_next_float_uniform` — chi-square on 10^5 floats, p > 0.01
- `test_mt19937_reseed` — re-seed produces identical sequence

#### `tests/test_xorshift.py`

Tests to write:
- `test_xorshift32_nonzero_state` — seed(0) must not produce all-zero state
- `test_xorshift32_period` — verify period (state repeats after exactly 2^32 - 1 outputs) — test with small seed, check repeat count
- `test_xorshift32_output_range` — all outputs in `[1, 2^32 - 1]`
- `test_xorshift64_output_range` — outputs are 32-bit values
- `test_xorshift128plus_deterministic` — same seed → same sequence
- `test_xorshift128plus_splitmix64` — verify splitmix64 produces expected output for known input

#### `tests/test_v8_random.py`

Tests to write:
- `test_v8_default_seed_deterministic` — default construction produces identical sequence
- `test_v8_next_float_range` — all outputs in `[0.0, 1.0)`
- `test_v8_next_float_53bit` — verify no output has more than 53 significant bits
- `test_v8_raw_int_64bit` — `next_int()` returns values in `[0, 2^64)`
- `test_v8_seeded_reproducible` — re-seed produces identical sequence

#### `tests/test_pcg.py`

Tests to write:
- `test_pcg32_deterministic` — seed(42, 54), verify first 100 outputs match PCG reference
- `test_pcg32_output_range` — all outputs in `[0, 2^32)`
- `test_pcg32_reseed` — re-seed produces identical sequence
- `test_pcg32_increment_odd` — verify internal increment is always odd

---

## Dependencies

```
numpy>=1.24
matplotlib>=3.7
scipy>=1.10
numba>=0.57
```

(`z3-solver>=4.12` — optional, not needed in Stage 1)

## Exit Criteria

- [ ] All 6 generator classes implement `PRNG` interface
- [ ] All unit tests pass
- [ ] MT19937 output matches Python `random` module exactly for same seed
- [ ] V8 emulation is deterministic with hardcoded seed
- [ ] No external dependencies beyond `numpy`, `matplotlib`, `scipy`, `numba`
