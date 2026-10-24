import numpy as np
import pytest

from secure_aggregation.quantize import (clip_l2, decode, encode,
                                         max_safe_clients, ring_add, ring_sub)


def test_encode_decode_roundtrip_negative():
    x = np.array([-1.5, 0.0, 2.25, -0.0001])
    assert np.allclose(decode(encode(x, 2.0**16), 2.0**16), x, atol=2.0**-16)


def test_sum_in_ring_equals_real_sum():
    rng = np.random.default_rng(0)
    xs = [rng.normal(size=8) for _ in range(20)]
    tot = np.zeros(8, dtype=np.uint64)
    for x in xs:
        tot = ring_add(tot, encode(x, 2.0**12))
    assert np.allclose(decode(tot, 2.0**12), sum(xs), atol=20 * 2.0**-13)


def test_ring_sub_inverts_add():
    a = np.array([0, 5, 2**32 - 1], dtype=np.uint64)
    b = np.array([7, 2**32 - 3, 9], dtype=np.uint64)
    assert np.array_equal(ring_sub(ring_add(a, b), b), a)


def test_clip_l2_and_overflow():
    v = clip_l2(np.array([3.0, 4.0]), 1.0)
    assert np.isclose(np.linalg.norm(v), 1.0)
    assert np.array_equal(clip_l2(np.array([0.3, 0.4]), 1.0), [0.3, 0.4])
    with pytest.raises(OverflowError):
        encode(np.array([1e6]), 2.0**16)
    assert max_safe_clients(1.0, 2.0**16) > 30000
