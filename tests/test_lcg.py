"""Tests for Linear Congruential Generator (LCG) and its preset variants."""

from src.generators.lcg import (
    LCG,
    AnsiCLCG,
    NumericalRecipesLCG,
    GlibcLCG,
    BadLCG,
)


def test_lcg_ansi_c_sequence():
    """ANSI C LCG with seed=0 produces the known reference sequence."""
    gen = AnsiCLCG(seed_value=0)
    # Reference values computed independently: (a*s+c) % 2^31
    expected = [12345, 1406932606, 654583775, 1449466924, 229283573,
                1109335178, 1051550459, 1293799192, 794471793, 551188310]
    result = gen.generate(10)
    assert result == expected


def test_lcg_numerical_recipes_sequence():
    """Numerical Recipes LCG with seed=0 is deterministic and reproducible."""
    gen = NumericalRecipesLCG(seed_value=0)
    expected = [1013904223, 1196435762, 3519870697, 2868466484, 1649599747,
                2670642822, 1476291629, 2748932008, 2180890343, 2498801434]
    result = gen.generate(10)
    assert result == expected


def test_lcg_glibc_differs_from_ansi_c():
    """Glibc extracts upper 15 bits with m=2^32, so outputs differ from ANSI C."""
    ansi = AnsiCLCG(seed_value=42)
    glibc = GlibcLCG(seed_value=42)
    ansi_out = ansi.generate(10)
    glibc_out = glibc.generate(10)
    # glibc outputs the upper bits, ANSI C outputs the full state, so they differ
    assert ansi_out != glibc_out


def test_lcg_glibc_reference_sequence():
    """Glibc with seed=0 produces the known upper-15-bit reference sequence."""
    gen = GlibcLCG(seed_value=0)
    # Reference computed independently: ( (a*s+c) % 2^32 ) >> 16 & 0x7fff
    # First output: (0*1103515245+12345) % 2^32 = 12345, then 12345 >> 16 = 0
    expected = [0, 21468, 9988, 22117, 3498, 16927, 16045, 19741, 12122, 8410]
    assert gen.generate(10) == expected


def test_lcg_glibc_output_range():
    """Glibc outputs are always in [0, 2^15)."""
    gen = GlibcLCG(seed_value=0)
    for value in gen.generate(1000):
        assert 0 <= value < 2**15


def test_lcg_glibc_next_float_uniform():
    """Glibc floats span [0, 1), not clustered near zero."""
    gen = GlibcLCG(seed_value=0)
    floats = [gen.next_float() for _ in range(2000)]
    assert all(0.0 <= f < 1.0 for f in floats)
    # 15-bit outputs can reach (2^15 - 1)/2^15 ≈ 0.99997, so a uniform
    # generator must produce values near 1.0; values that only reach a small
    # fraction of [0,1) indicate the normalization range is wrong.
    assert max(floats) > 0.99


def test_lcg_bad_lattice():
    """BadLCG with m=101 produces at most 101 distinct values (structural weakness)."""
    gen = BadLCG(seed_value=0)
    values = gen.generate(2000)
    unique = set(values)
    # Period <= m = 101, so at most 101 distinct values ever
    assert len(unique) <= 101
    # All values in valid range [0, m)
    assert all(0 <= v < 101 for v in values)


def test_lcg_next_float_range():
    """next_float() for each variant returns floats in [0.0, 1.0)."""
    variants = [AnsiCLCG, NumericalRecipesLCG, GlibcLCG, BadLCG]
    for cls in variants:
        gen = cls(seed_value=123)
        for _ in range(100):
            f = gen.next_float()
            assert 0.0 <= f < 1.0, f"{cls.__name__} produced {f}"


def test_lcg_reseed():
    """Re-seeding with the same value produces an identical sequence."""
    gen = LCG(a=1103515245, c=12345, m=2**31, seed_value=99)
    first_run = gen.generate(10)
    gen.seed(99)
    second_run = gen.generate(10)
    assert first_run == second_run


def test_lcg_all_variants_instantiate():
    """All 4 preset classes and base LCG instantiate successfully."""
    lcg = LCG(a=1103515245, c=12345, m=2**31)
    ansi = AnsiCLCG()
    nr = NumericalRecipesLCG()
    glibc = GlibcLCG()
    bad = BadLCG()
    # All should be usable immediately
    for gen in [lcg, ansi, nr, glibc, bad]:
        gen.seed(0)
        assert isinstance(gen.next_int(), int)
