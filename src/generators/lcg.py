"""Linear Congruential Generator: X_{n+1} = (a * X_n + c) mod m."""

from .base import PRNG


class LCG(PRNG):
    """Base LCG with configurable a, c, m parameters.

    The modulus m may vary across variants, so next_float divides by self.m
    instead of the fixed 2^32 used by the base class.
    """

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
        # Division by self.m (not 2^32) because modulus varies per variant
        return self.next_int() / self.m


class AnsiCLCG(LCG):
    """ANSI C rand() LCG: a=1103515245, c=12345, m=2^31."""

    def __init__(self, seed_value: int = 0):
        super().__init__(a=1103515245, c=12345, m=2**31, seed_value=seed_value)


class NumericalRecipesLCG(LCG):
    """Numerical Recipes LCG: a=1664525, c=1013904223, m=2^32."""

    def __init__(self, seed_value: int = 0):
        super().__init__(a=1664525, c=1013904223, m=2**32, seed_value=seed_value)


class GlibcLCG(LCG):
    """glibc rand() LCG: same a, c as ANSI C but uses m=2^32 and upper bits.

    Uses the upper 15 bits (>> 16 & 0x7fff) instead of the full state.
    Upper bits of an LCG have better statistical independence than lower bits
    because the modular multiplication mixes high-order bits more thoroughly —
    this is the core reason glibc's design extracts upper bits.
    """

    def __init__(self, seed_value: int = 0):
        super().__init__(a=1103515245, c=12345, m=2**32, seed_value=seed_value)

    def next_int(self) -> int:
        # Upper 15 bits: statistically superior to low bits for LCG outputs.
        # glibc uses this extraction because the low bits of an LCG have short
        # cycles and visible correlation, while upper bits pass more tests.
        self.state = (self.a * self.state + self.c) % self.m
        return (self.state >> 16) & 0x7fff

    def next_float(self) -> float:
        # Normalize by the upper-15-bit output range 2**15, NOT by m=2**32
        # (which would cluster glibc floats near zero and break uniformity).
        return self.next_int() / (1 << 15)


class BadLCG(LCG):
    """Intentionally weak LCG: a=3, c=7, m=101.

    Produces short periods and visible lattice structure for analysis.
    """

    def __init__(self, seed_value: int = 0):
        super().__init__(a=3, c=7, m=101, seed_value=seed_value)
