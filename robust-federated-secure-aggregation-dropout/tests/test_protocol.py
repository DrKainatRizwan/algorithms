import numpy as np
import pytest

from secure_aggregation import AbortError, run_protocol
from secure_aggregation.crypto import prg_vector
from secure_aggregation.protocol import Client
from secure_aggregation.quantize import ring_add

N, D = 12, 16


def make_inputs(n=N, d=D, seed=0):
    rng = np.random.default_rng(seed)
    return {u: rng.integers(0, 2**20, size=d).astype(np.uint64) for u in range(1, n + 1)}


def expected(inputs, users):
    tot = np.zeros(D, dtype=np.uint64)
    for u in users:
        tot = ring_add(tot, inputs[u])
    return tot


def test_no_dropout_exact_sum():
    xs = make_inputs()
    res = run_protocol(xs, 6, seed=1)
    assert res.survivors == list(range(1, N + 1)) and res.dropped == []
    assert np.array_equal(res.total, expected(xs, res.survivors))


@pytest.mark.parametrize("stage,who", [("advertise", {2}), ("share", {4, 5}),
                                       ("masked", {1, 7, 9}), ("unmask", {3, 8})])
def test_dropout_at_each_stage(stage, who):
    xs = make_inputs()
    res = run_protocol(xs, 6, {stage: who}, seed=2)
    assert np.array_equal(res.total, expected(xs, res.survivors))
    if stage == "masked":
        assert set(res.dropped) == who and not who & set(res.survivors)
    if stage in ("advertise", "share"):
        assert not who & set(res.survivors)


def test_mixed_dropouts_exact():
    xs = make_inputs()
    drops = {"advertise": {1}, "share": {2}, "masked": {3, 4}, "unmask": {5, 6}}
    res = run_protocol(xs, 5, drops, seed=3)
    assert np.array_equal(res.total, expected(xs, res.survivors))
    assert len(res.survivors) == N - 4


def test_too_many_dropouts_abort():
    xs = make_inputs()
    with pytest.raises(AbortError):
        run_protocol(xs, 8, {"masked": set(range(1, 7))}, seed=4)
    with pytest.raises(AbortError):
        run_protocol(xs, 8, {"unmask": set(range(1, 7))}, seed=4)


def test_masked_vector_hides_input():
    xs = make_inputs(n=4)
    clients = {u: Client(u, x, 2, 5) for u, x in xs.items()}
    directory = {u: c.advertise() for u, c in clients.items()}
    out = {u: c.share_keys(directory) for u, c in clients.items()}
    y = clients[1].masked_input({u: out[u][1] for u in out if u != 1})
    assert not np.array_equal(y, xs[1])
    # Without the masks the ciphertext looks uniform: strongly decorrelated.
    assert abs(np.corrcoef(y.astype(float), xs[1].astype(float))[0, 1]) < 0.9


def test_pairwise_masks_cancel_without_self_mask():
    xs = make_inputs(n=5)
    clients = {u: Client(u, x, 3, 6) for u, x in xs.items()}
    directory = {u: c.advertise() for u, c in clients.items()}
    out = {u: c.share_keys(directory) for u, c in clients.items()}
    ys = {v: clients[v].masked_input({u: out[u][v] for u in out if u != v}) for v in clients}
    from secure_aggregation.crypto import kdf
    from secure_aggregation.quantize import ring_sub
    tot = np.zeros(D, dtype=np.uint64)
    for v, y in ys.items():
        tot = ring_add(tot, y)
        tot = ring_sub(tot, prg_vector(kdf(clients[v]._b, b"self"), D))
    assert np.array_equal(tot, expected(xs, xs))


def test_client_refuses_small_survivor_set_and_never_leaks_both():
    xs = make_inputs(n=6)
    clients = {u: Client(u, x, 4, 7) for u, x in xs.items()}
    directory = {u: c.advertise() for u, c in clients.items()}
    out = {u: c.share_keys(directory) for u, c in clients.items()}
    for v, c in clients.items():
        c.masked_input({u: out[u][v] for u in out if u != v})
    with pytest.raises(AbortError):
        clients[1].unmask({1, 2})        # fewer than t survivors
    s_out, b_out = clients[1].unmask({1, 2, 3, 4})
    assert set(s_out) == {5, 6} and set(b_out) == {1, 2, 3, 4}
    assert not set(s_out) & set(b_out)


def test_tampered_share_detected():
    from secure_aggregation.crypto import AuthenticationError
    xs = make_inputs(n=3)
    clients = {u: Client(u, x, 2, 8) for u, x in xs.items()}
    directory = {u: c.advertise() for u, c in clients.items()}
    out = {u: c.share_keys(directory) for u, c in clients.items()}
    blob = out[2][1]
    out[2][1] = bytes([blob[0] ^ 1]) + blob[1:]
    with pytest.raises(AuthenticationError):
        clients[1].masked_input({u: out[u][1] for u in out if u != 1})
