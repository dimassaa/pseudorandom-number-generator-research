#!/usr/bin/env python3
"""Independent verification of MT19937 attack - do not trust the module's own tests."""

import random
import sys
sys.path.insert(0, '/mnt/d/opencode/new14')

from src.generators.mt19937 import MT19937
from src.attacks.mt19937_attack import (
    inverse_right_shift_xor,
    inverse_left_shift_xor_mask,
    untemper,
    recover_mt19937_state,
    attack_mt19937,
)


def test_inverse_right_shift_xor():
    """Test inverse_right_shift_xor roundtrip for shifts 1,11,18,30."""
    import random as rng
    rng.seed(12345)
    for _ in range(1000):
        x = rng.getrandbits(32)
        for shift in [1, 11, 18, 30]:
            tempered = x ^ (x >> shift)
            recovered = inverse_right_shift_xor(tempered, shift)
            assert recovered == x, f"roundtrip failed for shift={shift}, x={x:#010x}"
    print("✓ inverse_right_shift_xor roundtrip: PASSED (1000 tests × 4 shifts)")


def test_inverse_left_shift_xor_mask():
    """Test inverse_left_shift_xor_mask roundtrip for (15, 0xefc60000) and (7, 0x9d2c5680)."""
    import random as rng
    rng.seed(54321)
    for _ in range(1000):
        x = rng.getrandbits(32)
        for shift, mask in [(15, 0xefc60000), (7, 0x9d2c5680)]:
            tempered = x ^ ((x << shift) & mask)
            recovered = inverse_left_shift_xor_mask(tempered, shift, mask)
            assert recovered == x, f"roundtrip failed for shift={shift}, mask={mask:#010x}, x={x:#010x}"
    print("✓ inverse_left_shift_xor_mask roundtrip: PASSED (1000 tests × 2 masks)")


def test_untemper_roundtrip():
    """Test untemper(temper(x)) == x for random x."""
    gen = MT19937(42)
    import random as rng
    rng.seed(9999)
    for _ in range(2000):
        x = rng.getrandbits(32)
        tempered = gen._temper(x)
        recovered = untemper(tempered)
        assert recovered == x, f"untemper roundtrip failed for x={x:#010x}"
    print("✓ untemper roundtrip: PASSED (2000 tests)")


def test_recover_from_random_module():
    """Recover from random.Random(12345), predict 100, assert 100% match."""
    seed = 12345
    rng = random.Random(seed)
    observations = [rng.getrandbits(32) for _ in range(624)]
    
    recovered = recover_mt19937_state(observations)
    
    # Predict 100
    predicted = [recovered.next_int() for _ in range(100)]
    actual = [rng.getrandbits(32) for _ in range(100)]
    
    match_count = sum(p == a for p, a in zip(predicted, actual))
    assert match_count == 100, f"match_count={match_count}, expected 100"
    print(f"✓ Recover from random.Random(12345): PASSED ({match_count}/100 match)")


def test_recover_from_random_module_different_seed():
    """Recover from random.Random(67890), predict 100, assert 100% match."""
    seed = 67890
    rng = random.Random(seed)
    observations = [rng.getrandbits(32) for _ in range(624)]
    
    recovered = recover_mt19937_state(observations)
    
    predicted = [recovered.next_int() for _ in range(100)]
    actual = [rng.getrandbits(32) for _ in range(100)]
    
    match_count = sum(p == a for p, a in zip(predicted, actual))
    assert match_count == 100, f"match_count={match_count}, expected 100"
    print(f"✓ Recover from random.Random(67890): PASSED ({match_count}/100 match)")


def test_recover_from_mt19937_generator():
    """Recover from src.generators.MT19937 instance, predict 100, compare to actual next_int()."""
    gen = MT19937(42)
    observations = [gen.next_int() for _ in range(624)]
    
    recovered = recover_mt19937_state(observations)
    
    predicted = [recovered.next_int() for _ in range(100)]
    actual = [gen.next_int() for _ in range(100)]
    
    match_count = sum(p == a for p, a in zip(predicted, actual))
    assert match_count == 100, f"match_count={match_count}, expected 100"
    print(f"✓ Recover from MT19937(42): PASSED ({match_count}/100 match)")


if __name__ == "__main__":
    test_inverse_right_shift_xor()
    test_inverse_left_shift_xor_mask()
    test_untemper_roundtrip()
    test_recover_from_random_module()
    test_recover_from_random_module_different_seed()
    test_recover_from_mt19937_generator()
    print("\n✅ ALL INDEPENDENT VERIFICATION TESTS PASSED")