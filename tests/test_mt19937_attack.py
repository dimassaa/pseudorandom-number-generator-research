"""Tests for MT19937 inverse tempering and state recovery attacks.

Uses the project's own MT19937 implementation for temper/untemper roundtrips
and Python's random module as the external target for state recovery tests.
"""

import random

import pytest

from src.generators.mt19937 import MT19937
from src.attacks.mt19937_attack import (
    inverse_right_shift_xor,
    inverse_left_shift_xor_mask,
    untemper,
    recover_mt19937_state,
    attack_mt19937,
)


# --- inverse_right_shift_xor ---


def test_inverse_right_shift_xor_roundtrip():
    """Roundtrip: inverse_right_shift_xor(x ^ (x>>shift), shift) == x for shifts 1..30."""
    import random as _rng

    _rng.seed(999)
    for _ in range(100):
        x = _rng.getrandbits(32)
        for shift in [1, 11, 18, 30]:
            tempered = x ^ (x >> shift)
            recovered = inverse_right_shift_xor(tempered, shift)
            assert recovered == x, f"roundtrip failed for shift={shift}, x={x:#010x}"


def test_inverse_right_shift_xor_shift_variants():
    """Verify inverse on crafted patterns for shifts 11 and 18 specifically."""
    # Shift 11: upper 11 bits pass through unchanged; lower 21 bits are XORed
    # with shifted upper bits.  Construct x where upper and lower bits interact.
    x_11 = 0xDEADBEEF
    tempered_11 = x_11 ^ (x_11 >> 11)
    assert inverse_right_shift_xor(tempered_11, 11) == x_11

    # Shift 18: upper 18 bits pass through; lower 14 bits interact
    x_18 = 0xCAFEBABE
    tempered_18 = x_18 ^ (x_18 >> 18)
    assert inverse_right_shift_xor(tempered_18, 18) == x_18

    # Boundary: shift 1 — every bit depends on the one above
    x_1 = 0x55555555
    tempered_1 = x_1 ^ (x_1 >> 1)
    assert inverse_right_shift_xor(tempered_1, 1) == x_1

    # All-ones and all-zeros are trivially correct; also test a wide shift
    x_30 = 0x00000003
    tempered_30 = x_30 ^ (x_30 >> 30)
    assert inverse_right_shift_xor(tempered_30, 30) == x_30


# --- inverse_left_shift_xor_mask ---


def test_inverse_left_shift_xor_mask_roundtrip():
    """Roundtrip: inverse_left_shift_xor_mask(x ^ ((x<<shift)&mask), shift, mask) == x."""
    import random as _rng

    _rng.seed(1234)
    pairs = [(15, 0xEFC60000), (7, 0x9D2C5680)]
    for _ in range(100):
        x = _rng.getrandbits(32)
        for shift, mask in pairs:
            tempered = x ^ ((x << shift) & mask)
            recovered = inverse_left_shift_xor_mask(tempered, shift, mask)
            assert recovered == x, (
                f"roundtrip failed for shift={shift}, mask={mask:#010x}, x={x:#010x}"
            )


def test_inverse_left_shift_xor_mask_masks():
    """Verify both MT19937 mask constants produce correct roundtrips."""
    # Mask 0x9D2C5680 (shift 7): bits 7,9,10,13,14,15,16,17,18,19,20,21,22,24,25,28,29,30
    x_a = 0x12345678
    tempered_a = x_a ^ ((x_a << 7) & 0x9D2C5680)
    assert inverse_left_shift_xor_mask(tempered_a, 7, 0x9D2C5680) == x_a

    # Mask 0xEFC60000 (shift 15): bits 17,18,19,20,21,22,23,24,25,26,27,28,29,31
    x_b = 0xFEDCBA98
    tempered_b = x_b ^ ((x_b << 15) & 0xEFC60000)
    assert inverse_left_shift_xor_mask(tempered_b, 15, 0xEFC60000) == x_b

    # Both masks applied sequentially (like untemper steps 2+3)
    combined = x_a ^ ((x_a << 7) & 0x9D2C5680)
    combined = combined ^ ((combined << 15) & 0xEFC60000)
    # Reverse step 3 then step 2
    step3 = inverse_left_shift_xor_mask(combined, 15, 0xEFC60000)
    step2 = inverse_left_shift_xor_mask(step3, 7, 0x9D2C5680)
    assert step2 == x_a


# --- untemper ---


def test_untemper_roundtrip():
    """untemper(gen._temper(x)) == x for random x, verifying all four steps."""
    gen = MT19937(42)
    import random as _rng

    _rng.seed(5678)
    for _ in range(200):
        x = _rng.getrandbits(32)
        tempered = gen._temper(x)
        recovered = untemper(tempered)
        assert recovered == x, f"untemper roundtrip failed for x={x:#010x}"


# --- recover_mt19937_state ---


def test_recover_mt19937_state_matches():
    """Recover state from 624 outputs; next 100 outputs must match the original."""
    seed = 42
    gen = MT19937(seed)
    observations = gen.generate(624)

    recovered = recover_mt19937_state(observations)

    # Generate 100 more from the original and from the recovered instance
    original_next = gen.generate(100)
    recovered_next = recovered.generate(100)

    assert original_next == recovered_next


def test_recover_mt19937_insufficient_data():
    """Fewer than 624 observations raises ValueError."""
    with pytest.raises(ValueError, match="at least 624 observations"):
        recover_mt19937_state([0] * 623)


# --- attack_mt19937 end-to-end ---


def test_attack_mt19937_single_seed():
    """Attack with seed=2024 achieves 100% accuracy over 1000 predictions."""
    result = attack_mt19937(num_observations=624, num_predictions=1000, seed=2024)
    assert result["state_recovered"] is True
    assert result["accuracy"] == 1.0


def test_attack_mt19937_multiple_seeds():
    """Attack achieves 100% accuracy across seeds 0, 1, 42, 999."""
    for seed in [0, 1, 42, 999]:
        result = attack_mt19937(num_observations=624, num_predictions=1000, seed=seed)
        assert result["state_recovered"] is True
        assert result["accuracy"] == 1.0, f"failed for seed={seed}"


def test_attack_mt19937_1000_predictions():
    """match_count is exactly 1000 after a full recovery."""
    result = attack_mt19937(num_observations=624, num_predictions=1000, seed=2024)
    assert result["match_count"] == 1000
