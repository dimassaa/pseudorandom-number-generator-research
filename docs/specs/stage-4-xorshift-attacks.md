# Stage 4 Spec: Xorshift Attacks and V8 Predictability Analysis

## Goal

Demonstrate the non-cryptographic-security of xorshift-family generators: brute-force full state recovery for xorshift32, linear algebra state recovery for xorshift64, and a (best-effort) z3/SMT-based attack on V8's xorshift128+ — with graceful degradation when z3 is unavailable.

## Scope

xorshift attack module, V8 predictability analysis with optional z3. Success criteria: xorshift32 and xorshift64 proven recoverable+predicted with 100% bit-identical accuracy; V8 attack documented and attempted if z3 available.

## Dependencies

- `src/generators/xorshift.py` — XorShift32, XorShift64, XorShift128Plus
- `src/generators/v8_random.py` — V8Random emulation
- `numba` (required) — JIT acceleration for brute force
- `z3-solver` (optional) — SMT solving for xorshift128+ attack

---

## Files to Create

| File | Responsibility |
|------|---------------|
| `src/attacks/xorshift_attack.py` | xorshift32 brute force, xorshift64 linear algebra, xorshift128+ z3 attack |
| `tests/test_xorshift_attack.py` | Unit tests for all xorshift attacks |

---

## Specifications

### 1. xorshift32 Brute Force (`src/attacks/xorshift_attack.py`)

#### 1.1 Numba-Accelerated Brute Force

```python
def brute_force_xorshift32(observations: list[int], num_predictions: int = 1000) -> dict:
    """Find the xorshift32 state that produces the observed outputs.

    Brute-forces all 2^32 states via numba JIT and predicts future outputs.

    Args:
        observations: At least 3 consecutive 32-bit xorshift32 outputs.
        num_predictions: How many future outputs to predict.

    Returns:
        Dict with state, predicted outputs, accuracy.

    Raises:
        RuntimeError: If numba is unavailable (should not happen — required dep).
    """
```

**Numba JIT core (essential for speed):**

```python
import numpy as np
from numba import njit

@njit
def _xorshift32_state_matches(start_state: int, obs: np.ndarray) -> bool:
    """Return True if running xorshift32 from start_state reproduces obs."""
    state = start_state
    for target in obs:
        state ^= (state << 13) & 0xffffffff
        state ^= state >> 17
        state ^= (state << 5) & 0xffffffff
        if state != target:
            return False
    return True

@njit
def _brute_force_x32(obs: np.ndarray):
    """Brute force over all 32-bit states (excluding 0)."""
    for state in range(1, 0x100000000):
        if _xorshift32_state_matches(state, obs):
            return state
    return -1  # not found
```

**Performance estimate:** With ~3 observations and numba @njit (nopython mode), each state check is ~10-20ns. Full 2^32 sweep ≈ 40-80 seconds. Acceptable.

**Optimization options (documented, not mandatory):**
- Reverse-lookup using the algebraic structure (nls/mdva or reverse xorshift) — faster but complex.
- Reduce observations to 2 to speed up each check (fewer state transitions).
- Parallelize with `njit(parallel=True)` and `prange`.

**Prediction:** Once state is found, set `XorShift32(state)` and predict next `num_predictions`.

**Verification:** Predicted outputs must match actual generator 100% over `num_predictions`.

---

### 2. xorshift64 Linear Algebra (`src/attacks/xorshift_attack.py`)

#### 2.1 Linear System Over GF(2)

```python
def recover_xorshift64_state(observations: list[int]) -> int:
    """Recover xorshift64 internal state via GF(2) linear algebra.

    xorshift is linear over GF(2) bits. Given the transformation matrix
    T (64x64 bits), and outputs y = T^k * s, we can invert to find s.

    Args:
        observations: At least 64 consecutive 32-bit outputs
                      (each output = lower 32 bits of 64-bit state).

    Returns:
        The recovered 64-bit internal state.

    Raises:
        ValueError: If fewer than 64 observations given.
    """
```

**Approach (two viable methods — implement the inversion method):**

**Method A — Direct matrix inversion:**
1. Build a 64-bit LFSR-style transformation. The xorshift64 with shifts (13, 7, 17) is:
   ```
   x = state << 13
   x = x ^ state
   x = x >> 7        (note: these are non-commuting ops)
   ...
   ```
2. Represent each bit position's evolution as GF(2) equations.
3. Given 2 consecutive full 64-bit states (obtained from 2 consecutive 32-bit outputs), solve the linear system `s1 = T * s0` for both.

**Method B — State reconstruction by reverse xorshift (simpler, recommended):**
Because xorshift64's update is fully invertible, we can reconstruct the state going **backward** if we know consecutive outputs:

```python
@njit
def _reverse_xorshift64(state: int) -> int:
    """Reverse one xorshift64 step."""
    # Reverse x = x ^ (x << 17) & mask  first (last operation)
    # ... reverse in reverse order of operations
```

However, this only works if we know the *full* 64-bit state at some point. Since outputs only give lower 32 bits, we must recover the upper 32 bits too.

**Recommended implementation:** Use the linear-system-of-equations approach over GF(2) with `numpy` (treat bits as 0/1 and solve with `numpy.linalg` or a custom GF(2) Gaussian elimination):

```python
def _gf2_gaussian_elimination(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve A x = b over GF(2) using binary Gaussian elimination.

    Matrix is 64x64 (or smaller for subset), values 0/1.
    Returns solution vector of 0/1 (length = columns), or raises if singular.
    """
```

**Important caveat (must be handled):** Observing only the lower 32 bits gives partial information. The observation equation relating hidden upper bits to lower outputs requires building the full 64→32 mapping. If the system is underdetermined for a single output, collect more outputs and form an overdetermined system, solved in least-squares-over-GF(2) sense.

**Given complexity, the spec allows:**
- If linear system is singular/underdetermined with available observations, document this clearly and fall back to demonstrating on a *reduced* xorshift (e.g., 32-bit xorshift, or a truncated shift chosen so lower bits uniquely determine state). The **key success criterion** below still requires a working demonstration.

**Success criterion (must hold):** For xorshift64, at minimum demonstrate that *given two consecutive full 64-bit states* (e.g., instrumented generator), the state can be recovered and future outputs predicted 100%. If full state is recoverable from outputs alone, even better.

---

### 3. V8 xorshift128+ Attack (`src/attacks/xorshift_attack.py`)

#### 3.1 Z3-Based Attack (optional, graceful skip)

```python
def attack_v8_xorshift128(
    observations: list[int],      # 64-bit xorshift128+ raw outputs (from V8Random.next_int)
    num_predictions: int = 10,
) -> dict:
    """Attempt xorshift128+ state recovery via z3 SMT solver.

    Builds a system of non-linear bitwise equations from consecutive outputs
    and solves for (s0, s1). Returns recovered state + predictions.

    Note: Raw 64-bit outputs are needed. If only 53-bit floats are available,
    the top 11 bits are unknown, adding complexity (documented but not
    necessarily solved in this stage).

    Returns dict with:
        - success: bool
        - recovered_state: (s0, s1) or None
        - predicted_next: list or None
        - note: explanation of outcome
    """
```

**Algorithm (when z3 is available):**
1. Import `z3`. If `ImportError`, return `{"success": False, "note": "z3 not installed"}`.
2. Declare `BitVec('s0', 64)`, `BitVec('s1', 64)`.
3. For each observed output, add a solver constraint:
   ```
   x = s0; y = s1
   s0' = y
   x = x ^ (x << 23)
   s1' = x ^ y ^ (x >> 17) ^ (y >> 26)
   output = s1' + y   (mod 2^64)
   ```
   Add `output == observation`.
4. Update symbolic s0, s1 for next iteration.
5. Solve. If sat, extract `s0`, `s1` values.
6. If unsat, report failure (wrong observation format, overflow, etc.).

**Important V8 detail that affects feasibility:**
- V8's raw result is `(s1 + y) & (2^64 - 1)` — **not** zero-extended cleanly; it's the full 64-bit sum.
- V8 then does `(result >> 11)` to produce the float. If we only feed floats, we lose the low 11 bits → ambiguous. **Therefore:**
  - If we feed full 64-bit `next_int()` outputs (raw sum), the system is exact and solvable.
  - If we feed float outputs, we document the ambiguity and do not guarantee recovery.

**Decision:** The attack assumes access to raw 64-bit outputs (`V8Random.next_int()`), which is a valid demonstration that *if* an attacker can observe the raw internal outputs, state recovery works. This is consistent with the plan's framing ("predictability analysis").

**Runtime concern:** SMT solving of XOR-only systems is generally fast (milliseconds-seconds). Set z3 timeout to, e.g., 30 seconds to avoid hangs: `solver.set("timeout", 30000)`.

---

### 4. High-Level Predictability Analysis

```python
def analyze_xorshift_family() -> dict:
    """Run all xorshift-family attacks and report predictability.

    Covers: xorshift32 brute force, xorshift64 (best-effort),
    V8 xorshift128+ (z3 if available).

    Returns dict with per-generator results and a summary flag
    'all_recoverable'.
    """
```

---

### 5. Unit Tests

#### `tests/test_xorshift_attack.py`

Tests to write:
- `test_brute_force_xorshift32_finds_state` — generate 3 outputs, brute force, verify state matches
- `test_brute_force_xorshift32_small_state` — test with reduced state (e.g., 16-bit xorshift) for fast CI
- `test_brute_force_prediction_1000` — predict 1000, all match actual
- `test_brute_force_handles_zero_state` — seed(0) auto-sets to 1, verify no crash
- `test_recover_xorshift64_given_states` — with 2 full states, recover and predict 1000, all match
- `test_recover_xorshift64_from_outputs` — best-effort; if implemented, verify 100% match; if not, verify it raises documented exception
- `test_attack_v8_without_z3` — if z3 missing, returns success=False gracefully (monkeypatch import to simulate)
- `test_attack_v8_with_z3` — if z3 present, recover state + predict, verify match
- `test_analyze_xorshift_family_summary` — returns dict with all sub-results
- `test_gf2_gaussian_elimination_known_system` — verify correct solve of a small hand-constructed GF(2) system

**Note on z3-dependent tests:** Tests must `pytest.importorskip("z3")` for z3-specific tests so the suite passes when z3 is absent.

---

## Exit Criteria

- [ ] xorshift32 brute force recovers state within acceptable time (numba; document runtime)
- [ ] xorshift32 prediction accuracy 100% over 1000 outputs
- [ ] xorshift64 state recovery demonstrated (full-state or from outputs as per feasibility)
- [ ] xorshift64 prediction 100% where recovery succeeds
- [ ] V8 xorshift128+ attack: z3-based recovery works when z3 installed; graceful skip otherwise
- [ ] All unit tests pass (including when z3 absent)
- [ ] numba is used for brute force acceleration
- [ ] Documentation of the low-11-bits ambiguity for float-only V8 observations included in docstrings
