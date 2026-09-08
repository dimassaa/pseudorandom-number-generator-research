"""Mersenne Twister MT19937: 32-bit PRNG matching Python's random module.

Output is bit-identical to random.Random(seed).getrandbits(32) for the same
integer seed — critical for Stage 3 untempering attack correctness.

NOTE: Python's random seeds integers via init_by_array(), NOT the direct
init_genrand() call documented in the original MT19937 reference. init_by_array
first seeds with the fixed constant 19650218 then folds the integer seed's
32-bit chunks through the state. We replicate CPython's _randommodule.c exactly
so the untempered state recovers identically.
"""

from .base import PRNG


class MT19937(PRNG):
    """Mersenne Twister MT19937, matching Python's random module output.

    Seed initialization and tempering replicate CPython's _randommodule.c so
    that for the same seed this produces byte-identical output to
    random.Random(seed).getrandbits(32).
    """

    N = 624
    M = 397
    MATRIX_A = 0x9908b0df
    UPPER_MASK = 0x80000000
    LOWER_MASK = 0x7fffffff

    def __init__(self, seed_value: int = 0):
        self.mt = [0] * self.N
        self.index = self.N
        self.seed(seed_value)

    def _init_genrand(self, s: int) -> None:
        """Fill the state with the standard MT19937 sequence from a 32-bit seed.

        Mirrors CPython's init_genrand(). The multiplication may overflow 32
        bits in Python's arbitrary-precision ints, so each word is masked.
        """
        self.mt[0] = s & 0xffffffff
        for i in range(1, self.N):
            # Knuth's multiplicative init (TAOCP Vol2 3rd Ed P.106).
            self.mt[i] = (1812433253 * (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)) + i) & 0xffffffff
        self.index = self.N

    def _init_by_array(self, init_key: list[int]) -> None:
        """Fold an array of 32-bit words into the state, matching CPython.

        This is how Python seeds integer seeds: init_genrand(19650218) then
        two mixing passes. The final mt[0] is forced to have its MSB set to
        guarantee a non-zero initial state.
        """
        key_length = len(init_key)
        self._init_genrand(19650218)
        i = 1
        j = 0
        k = max(self.N, key_length)
        while k:
            # Non-linear first mixing pass. Outer parens around the XOR are
            # required: in C '^' binds looser than '+', so the additions apply
            # to the XOR result, not to the multiplier term.
            self.mt[i] = ((self.mt[i] ^ ((self.mt[i - 1] ^ (self.mt[i - 1] >> 30)) * 1664525))
                          + init_key[j] + j) & 0xffffffff
            i += 1
            j += 1
            if i >= self.N:
                self.mt[0] = self.mt[self.N - 1]
                i = 1
            if j >= key_length:
                j = 0
            k -= 1
        k = self.N - 1
        while k:
            # Second mixing pass with a different multiplier. Outer parens
            # around the XOR are required ('^' binds looser than '-' in Python).
            self.mt[i] = ((self.mt[i] ^ ((self.mt[i - 1] ^ (self.mt[i - 1] >> 30)) * 1566083941))
                          - i) & 0xffffffff
            i += 1
            if i >= self.N:
                self.mt[0] = self.mt[self.N - 1]
                i = 1
            k -= 1
        # MSB is 1; assures non-zero initial array.
        self.mt[0] = 0x80000000

    def seed(self, value: int) -> None:
        """Initialize the 624-word state array from an integer seed.

        Matches Python's random.seed() for integer seeds: the absolute value
        is split into little-endian 32-bit words and folded via init_by_array().
        """
        n = abs(value)
        if n == 0:
            init_key = [0]
        else:
            init_key = []
            while n:
                init_key.append(n & 0xffffffff)
                n >>= 32
        self._init_by_array(init_key)

    def _twist(self) -> None:
        """Generate the next N words of the state array."""
        for i in range(self.N):
            y = (self.mt[i] & self.UPPER_MASK) | (self.mt[(i + 1) % self.N] & self.LOWER_MASK)
            self.mt[i] = self.mt[(i + self.M) % self.N] ^ (y >> 1)
            # Apply the matrix multiplication if lowest bit of y is set.
            if y & 1:
                self.mt[i] ^= self.MATRIX_A
        self.index = 0

    def _temper(self, y: int) -> int:
        """Apply the four tempering transformations (all invertible).

        Final & 0xffffffff masks Python's arbitrary-precision ints back to 32 bits
        — left shifts can carry bits beyond 32 without this guard.
        """
        y ^= y >> 11
        y ^= (y << 7) & 0x9d2c5680
        y ^= (y << 15) & 0xefc60000
        y ^= y >> 18
        return y & 0xffffffff

    def next_int(self) -> int:
        """Return the next 32-bit tempered output, twisting the state when exhausted."""
        if self.index >= self.N:
            self._twist()
        y = self.mt[self.index]
        self.index += 1
        return self._temper(y)
