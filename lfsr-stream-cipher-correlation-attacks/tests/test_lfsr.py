import numpy as np
import pytest

from stream_cryptanalysis import LFSR, berlekamp_massey, lfsr_from_connection, recover_state
from stream_cryptanalysis.lfsr import PRIMITIVE_TAPS, sequence_from_taps
from stream_cryptanalysis.gf2 import rank, solve


@pytest.mark.parametrize("n", [3, 4, 5, 7, 8, 9, 10, 11, 12, 13])
def test_primitive_polynomials_have_maximal_period(n):
    assert LFSR.primitive(n).period(1) == 2**n - 1


def test_stream_matches_list_reference():
    r = LFSR.primitive(8)
    init = [1, 0, 1, 1, 0, 0, 1, 0]
    state = sum(b << i for i, b in enumerate(init))
    ref = sequence_from_taps(r.taps, 8, init, 100)
    assert list(r.stream(state, 100)) == ref


def test_masks_describe_linear_dependence():
    r = LFSR.primitive(11)
    st = 0x5A3
    s = r.stream(st, 200)
    for t, m in enumerate(r.masks(200)):
        assert s[t] == bin(m & st).count("1") % 2


def test_recover_state_from_n_bits():
    r = LFSR.primitive(13)
    st = 0x1ABC
    s = r.stream(st, 30)
    assert recover_state(r, list(s[:13])) == st
    assert recover_state(r, list(s[5:25]), offset=5) == st


def test_recover_state_inconsistent_returns_none():
    r = LFSR.primitive(4)
    s = list(r.stream(5, 30))
    s[20] ^= 1
    assert recover_state(r, s) is None


def test_berlekamp_massey_recovers_register():
    r = LFSR.primitive(12)
    s = r.stream(0x321, 60)
    L, c = berlekamp_massey(list(s))
    assert L == 12
    r2 = lfsr_from_connection(L, c)
    assert r2.taps == tuple(sorted(r.taps))
    assert np.array_equal(r2.stream(0x321, 200), r.stream(0x321, 200))


def test_berlekamp_massey_known_sequences():
    assert berlekamp_massey([0, 0, 0, 0])[0] == 0
    assert berlekamp_massey([0, 0, 0, 1])[0] == 4
    assert berlekamp_massey([1, 1, 1, 1, 1])[0] == 1


def test_gf2_solve_and_nullspace():
    rows = [0b011, 0b110]
    sol, null = solve(rows, [1, 1], 3)
    for r, b in zip(rows, [1, 1]):
        assert bin(r & sol).count("1") % 2 == b
    assert len(null) == 1 and null[0] == 0b111
    assert solve([0b01, 0b01], [0, 1], 2) is None
    assert rank([0b01, 0b10, 0b11], 2) == 2


def test_invalid_lfsr_rejected():
    with pytest.raises(ValueError):
        LFSR(4, (0, 4))
    with pytest.raises(KeyError):
        LFSR.primitive(6)
