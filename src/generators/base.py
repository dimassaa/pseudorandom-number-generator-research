from abc import ABC, abstractmethod


class PRNG(ABC):
    """Base class for all pseudorandom number generators."""

    @abstractmethod
    def seed(self, value: int) -> None:
        """Set the generator state from an integer seed."""
        ...

    @abstractmethod
    def next_int(self) -> int:
        """Return the next 32-bit unsigned integer in the sequence."""
        ...

    def next_float(self) -> float:
        """Return the next float in [0.0, 1.0) by normalizing next_int()."""
        return self.next_int() / 0x100000000

    def generate(self, n: int) -> list[int]:
        """Generate n consecutive 32-bit integers."""
        return [self.next_int() for _ in range(n)]

    def generate_floats(self, n: int) -> list[float]:
        """Generate n floats in [0.0, 1.0)."""
        return [self.next_float() for _ in range(n)]
