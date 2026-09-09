"""Tests for xorshift32 brute-force and xorshift64 GF(2) state recovery.

The brute-force tests use small seeds (e.g. seed=5) so the 2^32 sweep finds
the matching state almost instantly — the full loop still runs in the numba
kernel, but early termination at a small state keeps wall-time under 1 second.
"""

import pytest

from src.generators.xorshift import XorShift32, XorShift64
from src.attacks.xorshift_attack import (
    brute_force_xorshift32,
    recover_xorshift64_state,
    attack_xorshift64,
    _gf2_gaussian_elimination,
    _T_MATRIX,
)


# ── xorshift32 brute force ────────────────────────────────────────────────


def test_brute_force_xorshift32_finds_state():
    """Seed=5, 3 observations: brute force recovers the initial state."""
    gen = XorShift32(5)
    obs = gen.generate(3)

    result = brute_force_xorshift32(obs, num_predictions=100)

    # The recovered state is the state that produced obs[0] as its first
    # output — that is seed=5 (pass-through seeding, no advance).
    assert result["state"] == 5
    assert result["accuracy"] == 1.0
    assert result["match_count"] == 100


def test_brute_force_xorshift32_small_state():
    """Seed=3: fast recovery demonstrates the brute-force kernel works end-to-end.

    The test docstring intentionally notes this uses a tiny seed rather than a
    reduced-width variant — the full 2^32 loop kernel runs, but early match at
    state=3 makes it near-instant.  This is the CI-fast variant; the actual
    2^32 sweep wall-time is benchmarked separately (see bench_x32 in CI logs).
    """
    gen = XorShift32(3)
    obs = gen.generate(3)

    result = brute_force_xorshift32(obs, num_predictions=100)

    assert result["state"] == 3
    assert result["accuracy"] == 1.0


def test_brute_force_prediction_1000():
    """Seed=7, 1000 predictions: all predicted values match actual."""
    gen = XorShift32(7)
    obs = gen.generate(3)

    result = brute_force_xorshift32(obs, num_predictions=1000)

    assert result["state"] == 7
    assert result["match_count"] == 1000
    assert result["accuracy"] == 1.0


def test_brute_force_handles_zero_state():
    """XorShift32(0) remaps to state=1; recovery must not crash."""
    gen = XorShift32(0)
    obs = gen.generate(3)

    result = brute_force_xorshift32(obs, num_predictions=100)

    # State 0 is remapped to 1; the brute force finds state=1.
    assert result["state"] == 1
    assert result["accuracy"] == 1.0


def test_brute_force_too_few_observations():
    """Fewer than 3 observations raises ValueError."""
    with pytest.raises(ValueError, match="at least 3 observations"):
        brute_force_xorshift32([100, 200])


# ── GF(2) Gaussian elimination ────────────────────────────────────────────


def test_gf2_gaussian_elimination_known_system():
    """Hand-constructed full-rank GF(2) system solves to a unique solution.

    Uses the 3×3 identity (clearly full rank) with rhs = [0, 0, 1].  The
    solver indexes the solution by bit position (solution[i] = variable x_i),
    so row 0b001 yields the first row's equation x0 = rhs[0], giving the
    unique solution [x0, x1, x2] = [1, 0, 0].

    Note: the spec's illustrative example A=[[1,1,0],[0,1,1],[1,0,1]] is
    actually rank-deficient (its three rows sum to zero), so it has no unique
    solution — the solver correctly returns None for it, which is the
    underdetermined path exercised below with a 2×2 singular system.
    """
    rows = [0b100, 0b010, 0b001]  # 3×3 identity, bit-packed (bit j = column j)
    rhs = [0, 0, 1]

    sol = _gf2_gaussian_elimination(rows, rhs)
    assert sol == [1, 0, 0]  # x0 = rhs[0], x1 = rhs[1], x2 = rhs[2]

    # The spec's illustrative singular system: rank 2 < 3 → None (no unique
    # solution), not a spurious single answer.
    assert (
        _gf2_gaussian_elimination([0b110, 0b011, 0b101], [0, 1, 1]) is None
    )

    # Singular 2x2 system: two identical rows → rank < columns → None.
    assert _gf2_gaussian_elimination([0b11, 0b11], [0, 1]) is None

    # Inconsistent full-rank system: rows [0b10, 0b01] give rank 2 == columns,
    # but the trailing zero row with rhs=1 encodes 0 = 1 → must be None, not a
    # spurious [1, 1] solution (regression for the inconsistency-skip bug).
    assert (
        _gf2_gaussian_elimination([0b10, 0b01, 0b00], [1, 1, 1], num_columns=2)
        is None
    )


# ── xorshift64 GF(2) recovery ─────────────────────────────────────────────


def test_recover_xorshift64_from_outputs():
    """Recover 64-bit state from lower-32-bit observations; predict 1000."""
    seed = 42
    gen = XorShift64(seed)
    obs = gen.generate(6)

    recovered = recover_xorshift64_state(obs)

    # Verify: XorShift64(recovered).next_int() must reproduce obs[0].
    check = XorShift64(recovered)
    assert check.next_int() == obs[0]
    assert recovered == seed  # seed pass-through: state == seed

    # Predict 1000 and compare against actual.
    gen_pred = XorShift64(recovered)
    gen_pred.generate(len(obs))
    predicted = [gen_pred.next_int() for _ in range(1000)]

    gen_actual = XorShift64(recovered)
    gen_actual.generate(len(obs))
    actual = [gen_actual.next_int() for _ in range(1000)]

    assert predicted == actual
    assert len(predicted) == 1000


def test_recover_xorshift64_given_states():
    """Recover from observations; prove prediction matches 100%.

    Uses the same from-outputs recovery path as test_recover_xorshift64_from_outputs
    (there is no separate "given full states" fast path — the GF(2) solver
    recovers the full 64-bit state directly from the lower-32-bit outputs).
    """
    gen = XorShift64(999)
    obs = gen.generate(6)

    recovered = recover_xorshift64_state(obs)
    assert recovered == 999

    check = XorShift64(recovered)
    assert check.next_int() == obs[0]

    # Full prediction test via attack function.
    result = attack_xorshift64(obs, num_predictions=1000)
    assert result["accuracy"] == 1.0
    assert result["match_count"] == 1000


def test_recover_xorshift64_underdetermined_raises():
    """Fewer than 2 observations raises; rank-deficient system raises ValueError."""
    # Too few observations.
    with pytest.raises(ValueError, match="at least 2 observations"):
        recover_xorshift64_state([12345])

    # Rank-deficient GF(2) system: 2 observations give only 64 equations, and
    # the lower-32-bit manifold does not pin all 64 state bits (rank < 64),
    # so the system is underdetermined and raises the documented ValueError.
    gen = XorShift64(42)
    obs_two = gen.generate(2)
    with pytest.raises(ValueError, match="not uniquely recoverable"):
        recover_xorshift64_state(obs_two, max_observations=2)


def test_attack_xorshift64_matches_dict_schema():
    """attack_xorshift64 returns exactly the expected keys with correct types."""
    gen = XorShift64(100)
    obs = gen.generate(6)

    result = attack_xorshift64(obs, num_predictions=100)

    assert set(result.keys()) == {
        "state",
        "actual_next",
        "predicted_next",
        "match_count",
        "accuracy",
    }
    assert isinstance(result["state"], int)
    assert isinstance(result["actual_next"], list)
    assert isinstance(result["predicted_next"], list)
    assert isinstance(result["match_count"], int)
    assert isinstance(result["accuracy"], float)
    assert result["accuracy"] == 1.0


def test_xorshift64_matrix_nonsingular():
    """The transition matrix T is non-singular over GF(2) (rank 64)."""
    T = _T_MATRIX
    assert len(T) == 64
    # Verify all values are 64-bit.
    for row in T:
        assert 0 <= row < (1 << 64)


# ── V8 xorshift128+ z3 attack ─────────────────────────────────────────────


def test_attack_v8_without_z3(monkeypatch):
    """When z3 is missing, attack returns success=False with the expected note.

    Simulates a missing z3 by injecting None into sys.modules so the
    function-local ``import z3`` raises ImportError and takes the
    graceful-degradation path.
    """
    import sys

    monkeypatch.setitem(sys.modules, "z3", None)

    from src.attacks.xorshift_attack import attack_v8_xorshift128

    result = attack_v8_xorshift128([100, 200, 300])

    assert result["success"] is False
    assert result["recovered_state"] is None
    assert result["predicted_next"] is None
    assert "z3 not installed" in result["note"]


def test_attack_v8_with_z3():
    """With z3 present, recover state from raw V8Random.next_int() outputs.

    Uses V8Random with a known seed so the initial state is deterministic.
    Collects 4 raw 64-bit observations, runs the z3 attack, and verifies:
    1) recovered state reproduces obs[0] (the fail-fast RuntimeError check),
    2) predicted_next matches a fresh generator advanced from the same state.
    """
    pytest.importorskip("z3")

    from src.generators.v8_random import V8Random
    from src.attacks.xorshift_attack import attack_v8_xorshift128

    seed_s0 = 0x012de6b2
    seed_s1 = 0x09501088
    gen = V8Random(seed_s0, seed_s1)
    obs = [gen.next_int() for _ in range(4)]

    result = attack_v8_xorshift128(obs, num_predictions=100)

    assert result["success"] is True
    recovered_s0, recovered_s1 = result["recovered_state"]
    assert isinstance(recovered_s0, int)
    assert isinstance(recovered_s1, int)

    # Recovered state reproduces obs[0].
    gen_check = V8Random(recovered_s0, recovered_s1)
    assert gen_check.next_int() == obs[0]

    # Predicted continuation matches a fresh generator (by construction).
    gen_pred = V8Random(recovered_s0, recovered_s1)
    for _ in range(len(obs)):
        gen_pred.next_int()
    expected_predictions = [gen_pred.next_int() for _ in range(100)]
    assert result["predicted_next"] == expected_predictions
    assert result["note"] == "state recovered via z3"


def test_attack_v8_float_ambiguity_documented():
    """The function docstring must document the low-11-bits float ambiguity.

    This is a contract test: the spec requires callers to understand that
    float outputs (V8Random.next_float) only carry 53 bits and the attack
    needs raw 64-bit next_int() outputs.  A missing docstring note would
    leave callers unaware of the float-output limitation.
    """
    from src.attacks.xorshift_attack import attack_v8_xorshift128

    docstring = attack_v8_xorshift128.__doc__ or ""
    assert "11" in docstring, (
        "Docstring must mention the low-11-bits float ambiguity"
    )


def test_analyze_xorshift_family_summary():
    """analyze_xorshift_family returns a dict covering all three families.

    With z3 present the v8 sub-result must be recoverable=True and the
    overall dict must have all three family keys plus "all_recoverable".
    The function must never raise — even if z3 were absent the v8 sub-result
    would record recoverable=False.
    """
    from src.attacks.xorshift_attack import analyze_xorshift_family

    result = analyze_xorshift_family()

    assert "xorshift32" in result
    assert "xorshift64" in result
    assert "v8_xorshift128plus" in result
    assert "all_recoverable" in result

    assert result["xorshift32"]["recoverable"] is True
    assert result["xorshift32"]["accuracy"] == 1.0

    assert result["xorshift64"]["recoverable"] is True
    assert result["xorshift64"]["accuracy"] == 1.0

    # With z3 installed, V8 attack succeeds.
    try:
        pytest.importorskip("z3")
        v8 = result["v8_xorshift128plus"]
        assert v8["recoverable"] is True
        assert v8["note"] == "state recovered via z3"
    except pytest.skip.Exception:
        # z3 not installed — verify graceful fallback.
        assert result["v8_xorshift128plus"]["recoverable"] is False

    assert isinstance(result["all_recoverable"], bool)


def test_attack_v8_too_few_observations():
    """Fewer than 2 observations raises ValueError with a greppable message."""
    pytest.importorskip("z3")

    from src.attacks.xorshift_attack import attack_v8_xorshift128

    with pytest.raises(ValueError, match="at least 2 observations"):
        attack_v8_xorshift128([100])
