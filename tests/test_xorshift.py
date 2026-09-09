"""Tests for XorShift family generators (32, 64, 128+)."""

from src.generators.xorshift import XorShift32, XorShift64, XorShift128Plus


# ---------------------------------------------------------------------------
# XorShift32
# ---------------------------------------------------------------------------

def test_xorshift32_known_values():
    """XorShift32 with seed=42 matches independently computed reference."""
    gen = XorShift32(seed_value=42)
    expected = [
        11355432, 2836018348, 476557059, 3648046016, 3759983556,
        1441438134, 3713466840, 2431644334, 3120216979, 1067267639,
    ]
    assert gen.generate(10) == expected


def test_xorshift32_zero_seed_not_zero_output():
    """seed(0) must never produce a zero output (state is remapped from 0)."""
    gen = XorShift32(seed_value=0)
    for _ in range(200):
        assert gen.next_int() != 0


def test_xorshift32_output_range():
    """All outputs are 32-bit unsigned integers [0, 2^32)."""
    gen = XorShift32(seed_value=1)
    for v in gen.generate(500):
        assert 0 <= v < 2**32


# ---------------------------------------------------------------------------
# XorShift64
# ---------------------------------------------------------------------------

def test_xorshift64_known_values():
    """XorShift64 with seed=42 matches independently computed reference."""
    gen = XorShift64(seed_value=42)
    expected = [
        2505132714, 4160881343, 3520153978, 1608512872, 2335445122,
        862984227, 2779943123, 3961758002, 2832591600, 1273691429,
    ]
    assert gen.generate(10) == expected


def test_xorshift64_output_range():
    """Outputs are 32-bit (lower 32 bits of 64-bit state)."""
    gen = XorShift64(seed_value=1)
    for v in gen.generate(500):
        assert 0 <= v < 2**32


def test_xorshift64_nonzero_state():
    """seed(0) maps state to 1 — never produces all-zero output."""
    gen = XorShift64(seed_value=0)
    values = gen.generate(100)
    assert all(v != 0 for v in values)


# ---------------------------------------------------------------------------
# XorShift128Plus
# ---------------------------------------------------------------------------

def test_xorshift128plus_known_values():
    """XorShift128+ with seed=42 matches independently computed reference."""
    gen = XorShift128Plus()
    gen.seed(42)
    expected = [
        14512833, 2845005652, 2150214822, 2670627382, 2468376569,
        2109118363, 3774817149, 1523778099, 2002366048, 1032209310,
    ]
    assert gen.generate(10) == expected


def test_xorshift128plus_deterministic():
    """Same seed always produces the same sequence."""
    a = XorShift128Plus()
    a.seed(999)
    seq_a = a.generate(50)
    b = XorShift128Plus()
    b.seed(999)
    seq_b = b.generate(50)
    assert seq_a == seq_b


def test_xorshift128plus_splitmix64():
    """splitmix64 is deterministic and returns valid 64-bit values."""
    vals = [XorShift128Plus._splitmix64(i) for i in range(100)]
    # Deterministic: same input → same output
    assert all(XorShift128Plus._splitmix64(i) == vals[i] for i in range(100))
    # All values fit in 64 bits
    assert all(0 <= v < 2**64 for v in vals)
    # Not all identical (basic sanity)
    assert len(set(vals)) > 1


def test_xorshift128plus_output_range():
    """Outputs are 32-bit unsigned integers."""
    gen = XorShift128Plus()
    gen.seed(1)
    for v in gen.generate(500):
        assert 0 <= v < 2**32


def test_xorshift128plus_default_state_not_degenerate():
    """Default s0=s1=0 must not output zeros forever (zero-guard → s0=1)."""
    gen = XorShift128Plus()
    for _ in range(100):
        assert gen.next_int() != 0


# ---------------------------------------------------------------------------
# Cross-generator
# ---------------------------------------------------------------------------

def test_xorshift_all_reseed():
    """Reseeding any generator reproduces the original sequence."""
    for cls, kwargs in [
        (XorShift32, {"seed_value": 42}),
        (XorShift64, {"seed_value": 42}),
        (XorShift128Plus, {}),
    ]:
        gen = cls(**kwargs)
        gen.seed(42)
        first = gen.generate(20)
        gen.seed(42)
        second = gen.generate(20)
        assert first == second, f"{cls.__name__} reseed mismatch"
