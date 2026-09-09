"""Tests for V8Random, the emulation of V8's Math.random() (xorshift128+).

The reference values in test_v8_reference_values were computed independently
with a standalone xorshift128+ step (not via V8Random) so the tests verify
behavior against ground truth rather than against the implementation under
test.
"""

from src.generators.v8_random import V8Random

# Hardcoded V8 entropy seed constants (mirror the class).
V8_SEED_S0 = 0x012de6b2
V8_SEED_S1 = 0x09501088

# Independently computed first four float outputs for the default seed.
REFERENCE_FLOATS = [
    8.997306694902285e-06,
    7.378592042839305e-05,
    0.47482048598660853,
    0.49447271548625016,
]


def test_v8_default_seed_deterministic():
    """Two default-constructed generators produce identical sequences."""
    a = V8Random()
    b = V8Random()
    for _ in range(1000):
        assert a.next_float() == b.next_float()


def test_v8_next_float_range():
    """next_float() outputs stay in [0.0, 1.0)."""
    gen = V8Random()
    for _ in range(1000):
        val = gen.next_float()
        assert 0.0 <= val < 1.0


def test_v8_next_float_53bit():
    """Every float can be written as k / 2^53 (53-bit resolution invariant)."""
    gen = V8Random()
    for _ in range(1000):
        val = gen.next_float()
        # Exactly representable as k/2^53: the numerator must be an integer.
        assert (val * (1 << 53)).is_integer()


def test_v8_raw_int_64bit():
    """next_int() returns full 64-bit values in [0, 2^64)."""
    gen = V8Random()
    seen = gen.generate(500)
    assert all(0 <= v < 2**64 for v in seen)
    # Sanity: some outputs should exceed 32 bits, proving full-width returns.
    assert any(v >= 2**32 for v in seen)


def test_v8_seeded_reproducible():
    """seed(x) then re-seed(x) reproduces the identical sequence."""
    gen = V8Random()
    gen.seed(12345)
    first = gen.generate_floats(50)
    gen.seed(12345)
    second = gen.generate_floats(50)
    assert first == second


def test_v8_default_seed_matches_hardcoded():
    """Default construction uses V8's hardcoded entropy seed."""
    gen = V8Random()
    assert gen.impl.s0 == V8_SEED_S0
    assert gen.impl.s1 == V8_SEED_S1


def test_v8_reference_values():
    """Default seed matches independently computed reference floats."""
    gen = V8Random()
    got = [gen.next_float() for _ in range(len(REFERENCE_FLOATS))]
    assert got == REFERENCE_FLOATS
