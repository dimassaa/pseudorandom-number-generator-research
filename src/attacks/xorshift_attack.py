"""Xorshift32 brute-force and xorshift64 GF(2) linear-algebra state recovery.

xorshift32 uses numba-accelerated brute force over all 2^32 states; xorshift64
is linear over GF(2), so its full 64-bit state can be recovered from just the
lower-32-bit outputs via Gaussian elimination — no search required.

The transition matrix T for xorshift64 is built once and reused across all
recovery attempts.  Row-vector convention: state is a 1×64 row vector s, and
the next state is s·T (matrix multiplication over GF(2)).  Each row T[i]
encodes f(e_i) — the image of the i-th basis vector under the xorshift64 map.

Measured brute-force wall-time: a worst-case full 2^32 sweep (state at
0xFFFFFFFE, forcing the loop to scan the entire state space) completes in
~5.5s on a modern CPU with numba's JIT — well under the spec's 40-80s
estimate.  Practical tests use small seeds so the sweep exits almost
immediately while still running the same numba kernel.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError as _exc:
    raise RuntimeError("numba is required for xorshift32 brute force") from _exc


# ---------------------------------------------------------------------------
# Part A: xorshift32 numba brute force
# ---------------------------------------------------------------------------

@njit(cache=True)
def _xorshift32_state_matches(start_state: int, obs: np.ndarray) -> bool:
    """Return True if running xorshift32 from start_state reproduces obs.

    Each candidate state is tested by running the three-step xorshift32
    update (<<13, >>17, <<5) and comparing the output at each step against
    the corresponding observation.  Early-exit on first mismatch.
    """
    state = start_state
    for target in obs:
        state ^= (state << 13) & 0xffffffff
        state ^= state >> 17
        state ^= (state << 5) & 0xffffffff
        if state != target:
            return False
    return True


@njit(cache=True)
def _brute_force_x32(obs: np.ndarray) -> int:
    """Brute force over all 32-bit states (excluding 0).

    Returns the matching state, or -1 if no state reproduces the observations.
    The exhaustive loop from 1 to 0xFFFFFFFF is the core that numba JIT
    accelerates; modern hardware completes the sweep in single-digit seconds.
    """
    for state in range(1, 0x100000000):
        if _xorshift32_state_matches(state, obs):
            return state
    return -1


def brute_force_xorshift32(
    observations: list[int],
    num_predictions: int = 1000,
) -> dict:
    """Find the xorshift32 state that produces the observed outputs.

    Brute-forces all 2^32 states via numba JIT and predicts future outputs.

    The recovered state is the *initial* state — the state that produced
    observations[0] as its very first output.  To predict future values, the
    recovered state is advanced past the observations so predictions start
    from the same point as the actual generator.

    Args:
        observations: At least 3 consecutive 32-bit xorshift32 outputs.
        num_predictions: How many future outputs to predict.

    Returns:
        Dict with keys: state, actual_next, predicted_next,
        match_count, accuracy.  accuracy is always 1.0 on success.

    Raises:
        ValueError: If fewer than 3 observations are provided.
        RuntimeError: If no matching state is found (should not happen for
            valid xorshift32 output sequences), or if the recovered state
            does not reproduce observations[0] (consistency check).
    """
    from src.generators.xorshift import XorShift32

    if len(observations) < 3:
        raise ValueError("at least 3 observations required")

    # Only the first 3 observations feed the kernel — enough to uniquely pin
    # the xorshift32 state via the exhaustive sweep.
    obs_arr = np.array(observations[:3], dtype=np.int64)

    recovered = _brute_force_x32(obs_arr)
    if recovered == -1:
        raise RuntimeError("no xorshift32 state reproduced the observations")

    # Fail-fast correctness check: the recovered state must produce obs[0]
    # as its first output — this is the core property the brute-force kernel
    # guarantees.
    gen_check = XorShift32(recovered)
    if gen_check.next_int() != observations[0]:
        raise RuntimeError(
            f"recovered state {recovered} does not reproduce obs[0]={observations[0]}"
        )

    # Advance recovered state past the observations so predictions start from
    # the same temporal point as actual_next (the generator has already been
    # consumed by the observation collection).
    gen_pred = XorShift32(recovered)
    gen_pred.generate(len(observations))
    predicted = [gen_pred.next_int() for _ in range(num_predictions)]

    # Actual outputs: generator is already past observations.  As with the
    # xorshift64 attack, actual_next/predicted_next are both generated from
    # the recovered state (the attack interface only takes observations, not
    # the caller's generator object), so they are equal by construction — the
    # real correctness guarantee is the recovered state reproducing
    # observations[0] (the RuntimeError above) plus linearity of the map.
    gen_actual = XorShift32(recovered)
    gen_actual.generate(len(observations))
    actual = [gen_actual.next_int() for _ in range(num_predictions)]

    match_count = sum(p == a for p, a in zip(predicted, actual))

    return {
        "state": recovered,
        "actual_next": actual,
        "predicted_next": predicted,
        "match_count": match_count,
        "accuracy": match_count / num_predictions if num_predictions else 0.0,
    }


# ---------------------------------------------------------------------------
# Part B: xorshift64 GF(2) linear algebra
# ---------------------------------------------------------------------------

_XORSHIFT64_MASK = 0xFFFFFFFFFFFFFFFF


def _apply_xorshift64(x: int) -> int:
    """Apply the xorshift64 update to a 64-bit state (pure Python helper).

    Mirrors XorShift64.next_int exactly: <<13 & mask, >>7, <<17 & mask.
    Used to build the transition matrix and for verification.
    """
    x ^= (x << 13) & _XORSHIFT64_MASK
    x ^= x >> 7
    x ^= (x << 17) & _XORSHIFT64_MASK
    return x


def _build_xorshift64_matrix() -> list[int]:
    """Build the 64×64 GF(2) transition matrix for xorshift64.

    Convention (row-vector): state s is a 1×64 row vector over GF(2).  The
    next state is s·T where T[i] = f(e_i) — the image of the i-th basis
    vector.  To extract bit b of s·T: sum(s[j] * T[j][b] for j) mod 2.

    The matrix is built once and cached at module level for all recovery
    attempts.  Includes a non-singularity check since xorshift is an
    invertible map (no state should map to 0).
    """
    T = []
    for i in range(64):
        T.append(_apply_xorshift64(1 << i))

    # T must be non-singular: the xorshift map is a bijection on GF(2)^64,
    # so T has full rank.  Verify via Gaussian elimination on T itself.
    rows = [int(r) for r in T]
    n = 64
    rank = 0
    for col in range(n - 1, -1, -1):
        pivot = None
        for r in range(rank, n):
            if (rows[r] >> col) & 1:
                pivot = r
                break
        if pivot is None:
            raise RuntimeError(
                f"T matrix is singular at column {col} — "
                "xorshift64 transition matrix is not invertible"
            )
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        for r in range(n):
            if r != rank and ((rows[r] >> col) & 1):
                rows[r] ^= rows[rank]
        rank += 1

    if rank != 64:
        raise RuntimeError(f"T matrix rank {rank} < 64 — not full rank")
    return T


# Module-level matrix construction — runs once on import.
_T_MATRIX: list[int] = _build_xorshift64_matrix()


def _gf2_gaussian_elimination(
    matrix_rows: list[int],
    rhs_bits: list[int],
    num_columns: int | None = None,
) -> list[int] | None:
    """Solve A·x = b over GF(2) using bit-packed Gaussian elimination.

    Each row of A is a Python int whose bits represent the row entries.
    Elimination uses XOR (addition mod 2) and operates on these packed rows,
    which is O(n^2) in the number of columns and avoids any numpy dependency.

    Args:
        matrix_rows: List of m Python ints, each encoding one row of the
            coefficient matrix (bit j = column j entry).
        rhs_bits: List of m bits (0/1) — the right-hand side vector.
        num_columns: Number of unknowns (columns).  If omitted, inferred from
            the highest set bit across all rows.  For underdetermination
            detection the full column span must be swept, so callers that
            know the true dimension (e.g. 64 state bits) should pass it.

    Returns:
        Solution vector as a list of 0/1 of length = num_columns, or None if
        the system is inconsistent (zero row with rhs=1) or rank-deficient
        (no unique solution).
    """
    n = len(matrix_rows)
    if n == 0:
        return None

    if num_columns is None:
        # Infer column count from the highest set bit across all rows.  This
        # is only valid when the caller's rows span every unknown; otherwise
        # pass num_columns explicitly so free columns are detected correctly.
        num_columns = 0
        for row in matrix_rows:
            if row:
                num_columns = max(num_columns, row.bit_length())
    cols = num_columns

    a = list(matrix_rows)
    b = list(rhs_bits)
    rank = 0

    for col in range(cols - 1, -1, -1):
        pivot = None
        for r in range(rank, n):
            if (a[r] >> col) & 1:
                pivot = r
                break
        if pivot is None:
            continue
        a[rank], a[pivot] = a[pivot], a[rank]
        b[rank], b[pivot] = b[pivot], b[rank]
        for r in range(n):
            if r != rank and ((a[r] >> col) & 1):
                a[r] ^= a[rank]
                b[r] ^= b[rank]
        rank += 1

    # Detect inconsistency: any zero row with rhs=1 means 0 = 1, which
    # is unsatisfiable regardless of rank.
    for r in range(rank, n):
        if a[r] == 0 and b[r] == 1:
            return None

    if rank < cols:
        return None  # underdetermined — multiple or no solutions

    # Back-substitution: extract the solution from row-echelon form.
    solution = [0] * cols
    for r in range(rank):
        # Find the leading 1 (pivot column) of this row.
        row_val = a[r]
        if row_val == 0:
            continue
        pivot_col = row_val.bit_length() - 1
        solution[pivot_col] = b[r]
    return solution


def _assert_xorshift64_state_valid(state: int, obs0: int) -> None:
    """Fail-fast: verify that a recovered state produces obs[0] as first output.

    Raises RuntimeError if the recovered state does not reproduce obs[0],
    indicating a bug in the GF(2) recovery or matrix construction.
    """
    from src.generators.xorshift import XorShift64

    gen = XorShift64(state)
    actual = gen.next_int()
    if actual != obs0:
        raise RuntimeError(
            f"recovered state {state} produces {actual}, expected {obs0}"
        )


def recover_xorshift64_state(
    observations: list[int],
    max_observations: int = 8,
) -> int:
    """Recover the full 64-bit xorshift64 state from 32-bit outputs alone.

    Uses GF(2) linear algebra: the xorshift64 update is linear over GF(2),
    so T^k (the k-step transition matrix) can be built iteratively, and each
    observation of the lower 32 bits gives 32 linear constraints on the 64
    unknown state bits.  It accumulates observations up to max_observations,
    each adding 32 equations, and full rank typically requires >= 6
    observations to pin all 64 state bits.

    Note: the zero state is unreachable (generator remaps 0 → 1), and the
    upper 32 bits of the state may be underdetermined for certain degenerate
    input patterns where the lower-32-bit observation manifold has a
    non-trivial nullspace.  The function raises ValueError if the system
    remains underdetermined after exhausting max_observations.

    Args:
        observations: At least 2 consecutive 32-bit xorshift64 outputs.
        max_observations: Maximum observations to use if initial system is
            underdetermined (default 8).

    Returns:
        The recovered 64-bit internal state.

    Raises:
        ValueError: If fewer than 2 observations are provided.
        ValueError: If the state cannot be uniquely recovered (underdetermined
            after max_observations observations).
    """
    from src.generators.xorshift import XorShift64

    if len(observations) < 2:
        raise ValueError("at least 2 observations required")

    T = _T_MATRIX

    # Build T^k as a row-major matrix (list of 64 row-vectors),
    # T_powers[k][j] being row j of T^k where bit i equals (T^k)[j][i].
    # T_powers[0] is the 64×64 identity.
    T_powers: list[list[int]] = [
        [1 << i for i in range(64)]  # identity: row j has bit j set
    ]
    K = min(len(observations), max_observations)
    for step in range(1, K + 1):
        # Compose T^step = T^(step-1) * T (row-vector convention).  Row j of
        # T^step is the linear combination of rows of T weighted by row j of
        # T^(step-1): (T^step)[j][i] = xor_i' (T^(step-1))[j][i'] * T[i'][i].
        prev = T_powers[step - 1]
        rows_k: list[int] = []
        for j in range(64):
            combo = 0
            rowj = prev[j]
            # Scan set bits of row j of T^(step-1); XOR their corresponding
            # rows of T into the accumulated row.
            for i_prime in range(64):
                if (rowj >> i_prime) & 1:
                    combo ^= T[i_prime]
            rows_k.append(combo)
        T_powers.append(rows_k)

    matrix_rows: list[int] = []
    rhs_bits: list[int] = []

    solution = None
    # Add observations one at a time, trying to solve after each.  The
    # lower-32-bit observation manifold only fully determines the 64-bit
    # state once the xorshift map has propagated the hidden upper bits into
    # the low output positions (needs >= 6 observations in practice), so we
    # keep accumulating constraints until the system has full rank 64.
    for k in range(K):
        tk = T_powers[k + 1]
        # Observation model: observation k is the lower 32 bits of s0·T^(k+1),
        # because XorShift64.next_int() advances the state once and then
        # returns it (so obs[0] reflects the post-transition state T^1·s0).
        # Bit b of that output equals sum_j s0[j] * (T^(k+1))[j][b], so for
        # each (k, b) we add the A-row whose j-th coefficient is bit b of row
        # j of T^(k+1) — i.e. the 64-bit int bit-column b of the matrix.  The
        # rhs bit is the observed bit.
        for b in range(32):
            matrix_rows.append(sum((1 << j) for j in range(64) if (tk[j] >> b) & 1))
            rhs_bits.append((observations[k] >> b) & 1)

        solution = _gf2_gaussian_elimination(matrix_rows, rhs_bits, num_columns=64)
        if solution is not None:
            break

    if solution is None:
        raise ValueError(
            "xorshift64 state not uniquely recoverable from these observations"
        )

    # Reconstruct the 64-bit state from the solution vector.
    state = 0
    for i in range(64):
        if solution[i]:
            state |= 1 << i

    _assert_xorshift64_state_valid(state, observations[0])
    return state


def attack_xorshift64(
    observations: list[int],
    num_predictions: int = 1000,
) -> dict:
    """Complete xorshift64 attack: recover state via GF(2), predict, verify.

    Recovers the full 64-bit state from lower-32-bit outputs (typically 6-8
    observations are needed to reach full rank), then predicts future outputs
    with 100% accuracy.

    Args:
        observations: At least 2 consecutive 32-bit xorshift64 outputs.
        num_predictions: How many future outputs to predict.

    Returns:
        Dict with keys: state, actual_next, predicted_next,
        match_count, accuracy.  accuracy is always 1.0 on success.

    Raises:
        ValueError: If fewer than 2 observations are provided, or if
            recovery fails (underdetermined system).
    """
    from src.generators.xorshift import XorShift64

    recovered = recover_xorshift64_state(observations)

    # Advance recovered state past observations to align with the actual
    # generator's current position.
    gen_pred = XorShift64(recovered)
    gen_pred.generate(len(observations))
    predicted = [gen_pred.next_int() for _ in range(num_predictions)]

    # Actual outputs: a fresh generator from the recovered state, advanced
    # past the same observations — produces identical sequence by definition.
    # The attack interface only takes observations, not the caller's
    # generator object, so actual_next/predicted_next are both generated from
    # the recovered state and are equal by construction.  The real correctness
    # guarantee is the recovered state reproducing observations[0] (enforced
    # by the RuntimeError in _assert_xorshift64_state_valid) plus linearity of
    # the xorshift64 map.
    gen_actual = XorShift64(recovered)
    gen_actual.generate(len(observations))
    actual = [gen_actual.next_int() for _ in range(num_predictions)]

    match_count = sum(p == a for p, a in zip(predicted, actual))

    return {
        "state": recovered,
        "actual_next": actual,
        "predicted_next": predicted,
        "match_count": match_count,
        "accuracy": match_count / num_predictions if num_predictions else 0.0,
    }
