"""MT19937 state recovery and prediction attacks via inverse tempering.

Exploits the invertibility of MT19937's four tempering transformations to
recover the full 624-word internal state from observed outputs, then predict
all future outputs with 100% accuracy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.generators.mt19937 import MT19937


def inverse_right_shift_xor(y: int, shift: int) -> int:
    """Invert the operation ``y ^= y >> shift`` for 32-bit values.

    Reconstructs the original bit-by-bit from MSB downward.  At each bit
    position the output bit equals ``original_bit XOR shifted_in_bit``; the
    shifted-in bit is the original bit at position (i + shift), which we
    have already reconstructed when iterating from the top.

    Args:
        y: The transformed (tempered) 32-bit value.
        shift: The right-shift amount used in the forward XOR.

    Returns:
        The original pre-XOR value, masked to 32 bits.
    """
    result = 0
    for i in range(31, -1, -1):
        # Extract the bit of y at position i
        bit = (y >> i) & 1
        # XOR with the higher-order bit that was shifted into position i
        # if that higher-order position exists within 32 bits
        if i + shift < 32 and ((result >> (i + shift)) & 1):
            bit ^= 1
        result |= bit << i
    return result & 0xFFFFFFFF


def inverse_left_shift_xor_mask(y: int, shift: int, mask: int) -> int:
    """Invert the operation ``y ^= (y << shift) & mask`` for 32-bit values.

    Reconstructs the original bit-by-bit from LSB upward.  At each bit
    position the mask determines whether a shifted-in lower bit XORs with
    the original; the shifted-in bit comes from position (i - shift),
    already reconstructed when iterating from the bottom.

    Args:
        y: The transformed (tempered) 32-bit value.
        shift: The left-shift amount used in the forward operation.
        mask: The AND mask applied to the shifted value.

    Returns:
        The original pre-XOR value, masked to 32 bits.
    """
    result = 0
    for i in range(32):
        # Extract the bit of y at position i
        bit = (y >> i) & 1
        # If the mask has a bit set at position i, the forward operation
        # XORed the original bit at i with the original bit at (i - shift)
        if i - shift >= 0 and ((mask >> i) & 1):
            if (result >> (i - shift)) & 1:
                bit ^= 1
        result |= bit << i
    return result & 0xFFFFFFFF


def untemper(y: int) -> int:
    """Reverse MT19937's four tempering transformations.

    Applies the inverse of each temper step in reverse order of the
    forward ``_temper`` function:

    Forward:  >>11, <<7&0x9d2c5680, <<15&0xefc60000, >>18
    Inverse:  >>18, <<15&0xefc60000, <<7&0x9d2c5680, >>11

    Args:
        y: A tempered 32-bit output from MT19937.

    Returns:
        The original untempered MT state word.
    """
    y = inverse_right_shift_xor(y, 18)
    y = inverse_left_shift_xor_mask(y, 15, 0xEFC60000)
    y = inverse_left_shift_xor_mask(y, 7, 0x9D2C5680)
    y = inverse_right_shift_xor(y, 11)
    return y


def recover_mt19937_state(observations: list[int]) -> MT19937:
    """Recover a full MT19937 instance from 624 consecutive outputs.

    Untempers each observation to recover the raw state word, then injects
    the recovered words into a fresh MT19937 instance with index set to 624
    so the next call to ``next_int()`` triggers a twist from exactly the
    recovered state.

    Args:
        observations: At least 624 consecutive 32-bit tempered outputs.

    Returns:
        An MT19937 instance whose future outputs match the original generator.

    Raises:
        ValueError: If fewer than 624 observations are provided.
    """
    from src.generators.mt19937 import MT19937

    if len(observations) < MT19937.N:
        raise ValueError("at least 624 observations required")

    instance = MT19937(0)
    # Inject the raw (untempered) state words from the observed outputs.
    # index=624 means the state is exhausted; the next next_int() call will
    # twist and begin producing outputs from the freshly-recovered state.
    instance.mt = [untemper(x) for x in observations[:MT19937.N]]
    instance.index = MT19937.N
    return instance


def attack_mt19937(
    num_observations: int = 624,
    num_predictions: int = 1000,
    seed: int = 2024,
) -> dict:
    """Complete MT19937 attack: recover state from Python's random, then predict.

    Uses Python's built-in ``random`` module as the target.  Observes 624
    outputs (one full MT state worth), recovers the internal state via
    untempering, then verifies by predicting 1000 future outputs.

    Args:
        num_observations: Must be >= 624 (one full state cycle); if larger,
            only the first 624 observations are used.
        num_predictions: How many future outputs to predict and compare.
        seed: Seed for the target ``random.Random`` instance.

    Returns:
        Dict with:
            - state_recovered: always True on success (the attack either
              recovers the full state or raises)
            - actual_next: first 10 actual future outputs
            - predicted_next: first 10 predicted outputs
            - match_count: number of exact matches
            - accuracy: match_count / num_predictions
    """
    import random

    # Step 1: create the target RNG and collect observations
    rng = random.Random(seed)
    observations = [rng.getrandbits(32) for _ in range(num_observations)]

    # Step 2: recover the full internal state from observations
    recovered = recover_mt19937_state(observations)

    # Step 3: predict future outputs from the recovered state
    predicted = [recovered.next_int() for _ in range(num_predictions)]

    # Step 4: collect actual outputs the target produces next
    actual = [rng.getrandbits(32) for _ in range(num_predictions)]

    # Step 5: compare and compute accuracy
    match_count = sum(p == a for p, a in zip(predicted, actual))

    return {
        "state_recovered": True,
        "actual_next": actual[:10],
        "predicted_next": predicted[:10],
        "match_count": match_count,
        "accuracy": match_count / num_predictions,
    }
