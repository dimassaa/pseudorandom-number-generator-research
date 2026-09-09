"""Tests for PCG32 (PCG-XSH-RR), a permuted congruential generator.

The reference values in test_pcg32_reference_values are the canonical PCG
test vector for (init_state=42, init_seq=54), taken from the official
pcg-c test harness (test-high/expected/check-pcg32.out) and the
pcg-random.org "Using the Full C Implementation" page:

    0xa15c02b7 0x7b47f409 0xba1d3330 0x83d2f293 0xbfa4784b 0xcbed606e

They were cross-checked against the Rosetta Code decimal block
(2707161783 2068313097 3122475824 2211639955 3215226955) and reproduced
independently here from the paper's step/permutation formulas.
"""

from src.generators.pcg import PCG32

# Official PCG-XSH-RR test vector for (init_state=42, init_seq=54).
REFERENCE_VALUES = [
    0xA15C02B7,
    0x7B47F409,
    0xBA1D3330,
    0x83D2F293,
    0xBFA4784B,
    0xCBED606E,
]


def test_pcg32_reference_values():
    """Canonical (42, 54) seeding reproduces the official PCG test vector."""
    gen = PCG32(init_state=42, init_seq=54)
    assert gen.generate(len(REFERENCE_VALUES)) == REFERENCE_VALUES


def test_pcg32_deterministic():
    """Same (init_state, init_seq) yields the identical sequence."""
    a = PCG32(init_state=1234, init_seq=56)
    b = PCG32(init_state=1234, init_seq=56)
    assert a.generate(1000) == b.generate(1000)


def test_pcg32_output_range():
    """next_int() outputs are 32-bit unsigned values in [0, 2^32)."""
    gen = PCG32(init_state=1, init_seq=2)
    seen = gen.generate(1000)
    assert all(0 <= v < 2**32 for v in seen)


def test_pcg32_max_seed():
    """Huge seeds are masked into the 64-bit state without error."""
    gen = PCG32(init_state=2**100, init_seq=7)
    seen = gen.generate(100)
    assert all(0 <= v < 2**32 for v in seen)
    assert all(isinstance(v, int) for v in seen)


def test_pcg32_max_reseed():
    """seed(x) is re-entrant: re-seeding the same value replays the stream."""
    gen = PCG32()
    gen.seed(987654321)
    first = gen.generate(50)
    gen.seed(987654321)
    second = gen.generate(50)
    assert first == second


def test_pcg32_different_streams():
    """Different init_seq values select different random streams."""
    a = PCG32(init_state=100, init_seq=1)
    b = PCG32(init_state=100, init_seq=2)
    assert a.generate(20) != b.generate(20)


def test_pcg32_increment_odd():
    """Stream selector always yields an odd increment (full-period condition)."""
    for seq in range(64):
        assert PCG32(init_state=0, init_seq=seq).inc % 2 == 1
