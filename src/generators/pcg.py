"""PCG32-EXS (XSH-RR): a permuted congruential generator.

A 64-bit LCG core (full period 2^64 because the increment is odd) whose
output is passed through the XSH-RR permutation (xorshift-high, then a
rotation by the top 5 bits of the state).  See M.E. O'Neill's PCG paper.
"""

from .base import PRNG

MASK64 = 0xFFFFFFFFFFFFFFFF
MASK32 = 0xFFFFFFFF


def _rotr32(value: int, rot: int) -> int:
    """Rotate a 32-bit value right by rot bits (defensive mod 32)."""
    return ((value >> (rot % 32)) | (value << ((32 - rot) % 32))) & MASK32


class PCG32(PRNG):
    """PCG32-EXS (XSH-RR): 64-bit LCG state + 32-bit permuted output.

    Uses the XSH-RR output function (xorshift-high then random rotation).
    Full-period 64-bit LCG; the increment selects a stream (must be odd).
    """

    MULTIPLIER = 6364136223846793005

    def __init__(self, init_state: int = 0, init_seq: int = 0):
        # PCG seeding: start from zero, fold the stream selector into an odd
        # increment, warm up once, then inject the explicit state and step
        # again so a zero init_state still produces a mixed stream.
        self.state = 0
        self.inc = (init_seq << 1) | 1
        self._advance()
        self.state = (self.state + init_state) & MASK64
        self._advance()

    def seed(self, value: int) -> None:
        """Reinitialize with a new seed (stream defaults to 0)."""
        self.__init__(value, 0)

    def _advance(self) -> None:
        """Advance the LCG core: state = state * MULTIPLIER + inc (masked)."""
        self.state = (self.state * self.MULTIPLIER + self.inc) & MASK64

    def next_int(self) -> int:
        """Generate the next 32-bit output via the XSH-RR output function."""
        oldstate = self.state
        self._advance()
        # XSH-RR: take high 32 bits of the middle, then rotate by top 5 bits.
        # Masking xorshifted to 32 bits mirrors the C uint32_t truncation and
        # is required to reproduce the canonical PCG test vector.
        xorshifted = (((oldstate >> 18) ^ oldstate) >> 27) & MASK32
        rot = oldstate >> 59
        return _rotr32(xorshifted, rot)
