import random

import numpy as np
import pytest

from prng_attacks import (
    LCG, MT19937, JavaRandom, Xorshift128Plus, clone_mt19937, gf2_rank,
    lsb_stream, recover_java_random_state, recover_lcg_parameters,
    recover_modulus, recover_seed, recover_truncated_state,
    recover_xorshift128plus, rewind_twist, solve_gf2, temper, untemper,
)
from prng_attacks.xorshift_attacks import lsb_equations


def test_mt_reference_vector():
    g = MT19937(5489)
    assert [g.next_u32() for _ in range(3)] == [3499211612, 581869302, 3890346734]


def test_untemper_inverts_temper():
    rng = random.Random(1)
    for _ in range(500):
        x = rng.getrandbits(32)
        assert untemper(temper(x)) == x
        assert temper(untemper(x)) == x


def test_clone_predicts_future_outputs():
    victim = MT19937(987654321)
    seen = victim.outputs(624)
    clone = clone_mt19937(seen)
    assert clone.outputs(2000) == victim.outputs(2000)


def test_clone_matches_python_random():
    r = random.Random(42)
    seen = [r.getrandbits(32) for _ in range(624)]
    clone = clone_mt19937(seen)
    assert [clone.next_u32() for _ in range(100)] == [r.getrandbits(32) for _ in range(100)]


def test_rewind_recovers_previous_block():
    g = MT19937(2024)
    block1 = g.outputs(624)
    block2 = g.outputs(624)
    prev = rewind_twist([untemper(o) for o in block2])
    assert [temper(w) for w in prev[1:]] == block1[1:]
    assert prev[0] & 0x80000000 == untemper(block1[0]) & 0x80000000


def test_recover_seed():
    for seed in (0, 1, 5489, 123456789, 0xDEADBEEF):
        assert recover_seed(MT19937(seed).outputs(624)) == seed


def test_recover_seed_rejects_non_initial_block():
    g = MT19937(7)
    g.outputs(624)
    assert recover_seed(g.outputs(624)) is None


def test_gf2_solver_and_rank():
    rows = [0b011, 0b110, 0b101]
    assert gf2_rank(rows) == 2
    assert solve_gf2([0b011, 0b110], [1, 0], 3) is not None
    assert solve_gf2(rows, [1, 1, 1], 3) is None
    sol = solve_gf2([0b001, 0b010, 0b100], [1, 0, 1], 3)
    assert sol == [1, 0, 1]


def test_gf2_solution_satisfies_random_system():
    rng = random.Random(3)
    x = [rng.getrandbits(1) for _ in range(40)]
    rows = [rng.getrandbits(40) for _ in range(80)]
    rhs = [sum(((r >> j) & 1) & x[j] for j in range(40)) & 1 for r in rows]
    sol = solve_gf2(rows, rhs, 40)
    assert sol is not None
    for r, b in zip(rows, rhs):
        assert sum(((r >> j) & 1) & sol[j] for j in range(40)) & 1 == b


def test_xorshift_lsb_equations_match_generator():
    g = Xorshift128Plus(0x0123456789ABCDEF, 0xFEDCBA9876543210)
    s = (g.s[0], g.s[1])
    bits = [o & 1 for o in g.outputs(200)]
    x = [(s[0] >> j) & 1 for j in range(64)] + [(s[1] >> j) & 1 for j in range(64)]
    for mask, b in zip(lsb_equations(200), bits):
        assert sum(((mask >> j) & 1) & x[j] for j in range(128)) & 1 == b


def test_xorshift_state_recovery_predicts_full_words():
    victim = Xorshift128Plus(0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9)
    lsbs = [o & 1 for o in victim.outputs(160)]
    clone = recover_xorshift128plus(lsbs)
    assert clone is not None
    assert clone.outputs(50) == victim.outputs(50)


def test_xorshift_recovery_needs_enough_bits():
    with pytest.raises(ValueError):
        recover_xorshift128plus([0, 1] * 20)


def test_lcg_full_parameter_recovery():
    g = LCG(12345, 1103515245, 12345, 2 ** 31)
    xs = [g.step() for _ in range(60)]
    assert recover_modulus(xs) == 2 ** 31
    assert recover_lcg_parameters(xs) == (2 ** 31, 1103515245, 12345)


def test_lcg_prime_modulus_recovery():
    m = 2 ** 31 - 1
    g = LCG(99, 48271, 0, m)
    xs = [g.step() for _ in range(40)]
    assert recover_lcg_parameters(xs) == (m, 48271, 0)


def test_truncated_lcg_recovery():
    g = LCG(0xCAFEBABE, 1664525, 1013904223, 2 ** 32, shift=16)
    outs = g.outputs(6)
    states = recover_truncated_state(outs, 1664525, 1013904223, 2 ** 32, 16)
    assert len(states) == 1
    x = states[0]
    ref = LCG(0xCAFEBABE, 1664525, 1013904223, 2 ** 32, shift=16)
    ref.outputs(0)
    assert (x >> 16) == outs[0]
    for _ in range(5):
        x = (1664525 * x + 1013904223) % 2 ** 32
    assert x == g.x


def test_java_random_prediction():
    victim = JavaRandom(20240229)
    seen = victim.outputs(3)
    clone = recover_java_random_state(seen)
    assert clone is not None
    assert clone.outputs(20) == victim.outputs(20)


def test_lsb_stream_extraction():
    assert lsb_stream([1, 2, 3, 4]).tolist() == [1, 0, 1, 0]
    assert lsb_stream([1, 2, 3, 4], bit=1).tolist() == [0, 1, 1, 0]
