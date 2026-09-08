import pytest
from src.generators.base import PRNG


class FakeGen(PRNG):
    """Minimal concrete subclass for testing the base interface."""

    def seed(self, value: int) -> None:
        self.state = value
        self._count = 0

    def next_int(self) -> int:
        self._count += 1
        return self.state + self._count


def test_prng_is_abstract():
    """PRNG cannot be instantiated directly — missing seed and next_int."""
    with pytest.raises(TypeError, match="abstract method"):
        PRNG()


def test_concrete_subclass_instantiation():
    """A subclass implementing seed and next_int can be instantiated."""
    gen = FakeGen()
    gen.seed(42)
    assert gen.state == 42


def test_next_float_normalization():
    """next_float returns next_int() / 2^32, in [0.0, 1.0)."""
    gen = FakeGen()
    gen.seed(0)
    result = gen.next_float()
    expected = (0 + 1) / 0x100000000
    assert result == pytest.approx(expected)
    assert 0.0 <= result < 1.0


def test_generate_returns_n_ints():
    """generate(n) returns a list of n 32-bit integers."""
    gen = FakeGen()
    gen.seed(10)
    result = gen.generate(5)
    assert len(result) == 5
    assert all(isinstance(v, int) for v in result)
    assert result == [11, 12, 13, 14, 15]


def test_generate_floats_returns_n_floats():
    """generate_floats(n) returns n floats in [0.0, 1.0)."""
    gen = FakeGen()
    gen.seed(10)
    result = gen.generate_floats(4)
    assert len(result) == 4
    assert all(isinstance(v, float) for v in result)
    assert all(0.0 <= v < 1.0 for v in result)
    expected = [(10 + i + 1) / 0x100000000 for i in range(4)]
    assert result == pytest.approx(expected)
