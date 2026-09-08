"""XorShift family: fast, lightweight generators based on xor operations.

XorShift32 and XorShift64 use Marsaglia's original single-state design.
XorShift128Plus is the V8 Math.random algorithm — its internal update is
kept bit-exact so the next task (V8Random) can reuse it directly.
"""

from .base import PRNG


class XorShift32(PRNG):
    """32-bit xorshift with period 2^32 - 1.

    Marsaglia's xorshift32 with shifts (13, 17, 5).  State must never be
    zero — the xorshift map is degenerate at 0, so seed() and __init__
    remap 0 → 1.
    """

    SHIFT_LEFT_1 = 13
    SHIFT_RIGHT = 17
    SHIFT_LEFT_2 = 5

    def __init__(self, seed_value: int = 0):
        self.state = seed_value & 0xffffffff
        if self.state == 0:
            self.state = 1  # xorshift requires non-zero state

    def seed(self, value: int) -> None:
        self.state = value & 0xffffffff
        if self.state == 0:
            self.state = 1

    def next_int(self) -> int:
        # Left shifts are masked to 32 bits to prevent Python's arbitrary-
        # precision ints from growing the state beyond 32 bits.
        self.state ^= (self.state << self.SHIFT_LEFT_1) & 0xffffffff
        self.state ^= self.state >> self.SHIFT_RIGHT
        self.state ^= (self.state << self.SHIFT_LEFT_2) & 0xffffffff
        return self.state


class XorShift64(PRNG):
    """64-bit xorshift with period 2^64 - 1, returning lower 32 bits.

    Internal state is 64-bit with shifts (13, 7, 17).  Output is the
    lower 32 bits for interface consistency with other generators.
    """

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
        self.state ^= self.state >> 7
        self.state ^= (self.state << 17) & 0xffffffffffffffff
        # Return lower 32 bits for interface consistency
        return self.state & 0xffffffff


class XorShift128Plus(PRNG):
    """xorshift128+ with 128-bit state (2x 64-bit), period 2^128 - 1.

    Uses splitmix64 to derive two 64-bit states from a single integer seed.
    The next_int step matches V8's xorshift128+ exactly so that V8Random
    (next task) can call the internal 64-bit path without reimplementation.
    """

    def __init__(self, s0: int = 0, s1: int = 0):
        self.s0 = s0 & 0xffffffffffffffff
        self.s1 = s1 & 0xffffffffffffffff

    def seed(self, value: int) -> None:
        """Derive two 64-bit states from a single seed via splitmix64."""
        self.s0 = self._splitmix64(value)
        self.s1 = self._splitmix64(self.s0)

    @staticmethod
    def _splitmix64(x: int) -> int:
        """SplitMix64 — deterministic 64-bit hash used for seeding."""
        x = (x + 0x9e3779b97f4a7c15) & 0xffffffffffffffff
        z = x
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9 & 0xffffffffffffffff
        z = (z ^ (z >> 27)) * 0x94d049bb133111eb & 0xffffffffffffffff
        return (z ^ (z >> 31)) & 0xffffffffffffffff

    def next_int(self) -> int:
        # V8's xorshift128+ update — kept bit-exact for V8Random reuse.
        x = self.s0
        y = self.s1
        self.s0 = y
        x ^= (x << 23) & 0xffffffffffffffff
        self.s1 = x ^ y ^ (x >> 17) ^ (y >> 26)
        result = (self.s1 + y) & 0xffffffffffffffff
        # Return lower 32 bits for interface consistency with other generators
        return result & 0xffffffff
