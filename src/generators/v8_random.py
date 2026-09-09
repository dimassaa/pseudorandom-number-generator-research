"""V8Random: emulation of V8 engine's Math.random().

V8 (Chrome 49+) generates random doubles using xorshift128+ with two 64-bit
state words, then converts the raw 64-bit result to a float by keeping the
top 53 bits: (result >> 11) / 2^53.  This class wraps XorShift128Plus for
state management and replicates the raw 64-bit step so callers can obtain the
full-width sum needed for the Stage 4 state-recovery attack.
"""

from .base import PRNG
from .xorshift import XorShift128Plus


class V8Random(PRNG):
    """Emulation of V8 engine's Math.random() (xorshift128+).

    Default construction uses V8's hardcoded seed constants so output is
    reproducible across runs.
    """

    # Entropy seed constants from V8 source (src/base/random-number-generator.h).
    V8_SEED_S0 = 0x012de6b2
    V8_SEED_S1 = 0x09501088

    def __init__(self, s0: int = V8_SEED_S0, s1: int = V8_SEED_S1):
        self.impl = XorShift128Plus(s0, s1)

    def seed(self, value: int) -> None:
        """Re-seed the internal xorshift128+ state from an integer."""
        self.impl.seed(value)

    def next_int(self) -> int:
        """Return the raw 64-bit xorshift128+ result (V8 repeats this step).

        NOTE: this intentionally deviates from the base-class contract which
        returns a 32-bit value.  The Stage 4 attack needs the full 64-bit sum,
        so this method replicates the xorshift128+ update inline (reading
        impl.s0/impl.s1 directly) instead of calling impl.next_int(), which
        only exposes the lower 32 bits.  Returns the full 64-bit sum.
        """
        x = self.impl.s0
        y = self.impl.s1
        self.impl.s0 = y
        x ^= (x << 23) & 0xffffffffffffffff
        self.impl.s1 = x ^ y ^ (x >> 17) ^ (y >> 26)
        return (self.impl.s1 + y) & 0xffffffffffffffff

    def next_float(self) -> float:
        """Return a double in [0, 1) exactly as JavaScript's Math.random().

        V8 keeps the top 53 bits of the raw result and divides by 2^53.
        """
        raw = self.next_int()
        return (raw >> 11) / (1 << 53)
