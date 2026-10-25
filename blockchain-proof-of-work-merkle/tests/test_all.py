import pytest

from pow_chain import (DUPLICATE, EXTENDED, INVALID, ORPHAN, REORG, SIDE_BRANCH, Block, Blockchain,
                       ConsensusParams, Mempool, MerkleTree, NetworkSimulation, SimConfig,
                       Transaction, bits_to_target, confirmations_needed, merkle_root, mine,
                       mine_block, nakamoto_double_spend, orphan_rate_theory, retarget,
                       roots_collide, rosenfeld_double_spend, selfish_mining_revenue_theory,
                       selfish_threshold, simulate_double_spend, simulate_selfish_mining,
                       target_to_bits, verify_proof, work_from_target)


def leaves(n):
    return [f"tx{i}".encode() for i in range(n)]


def new_chain(**kw):
    return Blockchain(ConsensusParams(genesis_alloc=(("alice", 1000), ("bob", 50)), **kw))


def grow(chain, n, miner="m", t0=None, spacing=600, txs=None):
    t = t0 if t0 is not None else chain.tip.timestamp
    out = []
    for i in range(n):
        t += spacing
        blk = mine_block(chain.block_template(miner, txs or [], t))[0]
        out.append(blk)
        chain.add_block(blk)
    return out


# ---------------------------------------------------------------- merkle
@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 13])
@pytest.mark.parametrize("dup", [False, True])
def test_every_proof_verifies(n, dup):
    t = MerkleTree(leaves(n), dup)
    for i, leaf in enumerate(leaves(n)):
        assert verify_proof(leaf, t.proof(i), t.root)


def test_proof_rejects_wrong_leaf_and_root():
    t = MerkleTree(leaves(7))
    p = t.proof(3)
    assert not verify_proof(b"forged", p, t.root)
    assert not verify_proof(leaves(7)[3], p, MerkleTree(leaves(6)).root)
    assert len(p) <= 3


def test_incremental_update_matches_rebuild():
    for dup in (False, True):
        t = MerkleTree(leaves(11), dup)
        t.update(6, b"changed")
        ref = leaves(11)
        ref[6] = b"changed"
        assert t.root == MerkleTree(ref, dup).root


def test_duplicate_odd_mutation_only_in_bitcoin_style():
    assert roots_collide(leaves(3))
    assert merkle_root(leaves(3)) != merkle_root(leaves(3) + [leaves(3)[-1]])


def test_domain_separation_blocks_second_preimage():
    t = MerkleTree(leaves(4))
    inner = t.levels[1][0]
    assert MerkleTree([inner]).root != t.levels[1][0]
    fake = MerkleTree([t.levels[1][0], t.levels[1][1]]).root
    assert fake != t.root


# ---------------------------------------------------------------- pow
def test_compact_bits_roundtrip():
    for t in (1 << 248, 1 << 252, 0x1234567 << 200):
        back = bits_to_target(target_to_bits(t))
        assert t * 0.99 < back <= t
    assert bits_to_target(target_to_bits(255)) == 255


def test_mining_meets_target_and_work_matches_bits():
    chain = new_chain()
    blk = chain.block_template("m", [], 600)
    solved, tries = mine_block(blk)
    assert solved.header.hash_int <= bits_to_target(blk.header.bits)
    assert tries >= 1
    assert work_from_target(1 << 248) == 255


def test_mine_returns_none_when_range_exhausted():
    blk = new_chain().block_template("m", [], 600)
    impossible = Block.assemble(blk.prev_hash, list(blk.txs), 600, target_to_bits(1))
    assert mine(impossible.header, max_tries=500) is None


def test_retarget_clamps_and_direction():
    lim = 1 << 252
    t = 1 << 240
    assert retarget(t, 100, 1000, lim) < t           # too fast -> harder
    assert retarget(t, 10000, 1000, lim) > t         # too slow -> easier
    assert retarget(t, 1, 1000, lim) == t // 4
    assert retarget(1 << 251, 10**9, 1000, lim) == lim


# ---------------------------------------------------------------- chain
def test_extends_and_pays_reward():
    c = new_chain()
    grow(c, 3, "m")
    assert c.height == 3 and c.balance("m") == 150


def test_transfer_confirms_and_spv_proof():
    c = new_chain()
    tx = Transaction("alice", "bob", 30, 2, 0)
    grow(c, 1, "m", txs=[tx])
    assert c.balance("bob") == 80 and c.balance("alice") == 968 and c.balance("m") == 52
    grow(c, 2)
    bh, proof = c.prove_tx(tx.txid)
    assert c.verify_tx_proof(tx.txid, bh, proof)
    assert not c.verify_tx_proof(Transaction("alice", "bob", 31, 2, 0).txid, bh, proof)
    assert c.confirmations(tx.txid) == 3


def test_rejects_double_spend_replay_and_overdraft():
    c = new_chain()
    for bad in (Transaction("alice", "bob", 30, 0, 1), Transaction("bob", "alice", 60, 0, 0),
                Transaction("alice", "bob", -5, 0, 0)):
        blk = mine_block(c.block_template("m", [bad], 600))[0]
        r = c.add_block(blk)
        assert r.status == INVALID
    assert c.height == 0


def test_rejects_tampered_blocks():
    c = new_chain()
    good = mine_block(c.block_template("m", [Transaction("alice", "bob", 1, 0, 0)], 600))[0]
    greedy = Block(good.header, (Transaction.coinbase("m", 999, 1), *good.txs[1:]))
    assert c.add_block(greedy).status == INVALID                      # merkle mismatch
    cb = Transaction.coinbase("m", 999, 1)
    blk = mine_block(Block.assemble(c.tip.hash, [cb], 600, good.header.bits))[0]
    assert c.add_block(blk).reason == "bad coinbase"
    wrong_bits = Block.assemble(c.tip.hash, list(good.txs), 600, target_to_bits(1 << 251))
    assert c.add_block(mine_block(wrong_bits)[0]).status == INVALID
    assert c.add_block(good).status == EXTENDED
    assert c.add_block(good).status == DUPLICATE


def test_timestamp_must_exceed_median_time_past():
    c = new_chain()
    grow(c, 5)
    blk = Block.assemble(c.tip.hash, [Transaction.coinbase("m", 50, 6)], 1, c.tip.block.header.bits)
    assert c.add_block(mine_block(blk)[0]).reason.startswith("timestamp")


def test_orphan_blocks_connect_when_parent_arrives():
    src = new_chain()
    blocks = grow(src, 3)
    dst = new_chain()
    assert dst.add_block(blocks[2]).status == ORPHAN
    assert dst.add_block(blocks[1]).status == ORPHAN
    assert dst.add_block(blocks[0]).status == EXTENDED
    assert dst.height == 3 and dst.tip.hash == src.tip.hash and not dst.orphans


def test_heaviest_chain_reorg_and_state_rollback():
    c = new_chain()
    pay = Transaction("alice", "bob", 100, 0, 0)
    grow(c, 1, "a", t0=0, txs=[pay])
    assert c.balance("bob") == 150
    fork = new_chain()
    fb = grow(fork, 2, "b", t0=0)
    assert c.add_block(fb[0]).status == SIDE_BRANCH
    res = c.add_block(fb[1])
    assert res.status == REORG and res.reorg_depth == 1
    assert c.balance("bob") == 50 and c.balance("b") == 100 and c.balance("a") == 0
    assert not c.on_main_chain(res.disconnected[0].hash)
    assert len(c.stale_blocks()) == 1


def test_first_seen_wins_on_equal_work():
    c = new_chain()
    a = grow(c, 1, "a", t0=0)[0]
    other = new_chain()
    b = grow(other, 1, "b", t0=0)[0]
    assert c.add_block(b).status == SIDE_BRANCH and c.tip.hash == a.hash


def test_difficulty_rises_when_blocks_come_fast():
    c = new_chain(retarget_interval=5, target_spacing=600)
    grow(c, 4, spacing=60)
    before = bits_to_target(c.tip.block.header.bits)
    grow(c, 1, spacing=60)
    after = bits_to_target(c.tip.block.header.bits)
    assert after < before and after >= before // 4 - 1


def test_mempool_orders_by_fee_respects_nonces_and_reorg_restore():
    c = new_chain()
    mp = Mempool()
    t1 = Transaction("alice", "bob", 5, 1, 0)
    t2 = Transaction("alice", "bob", 5, 9, 1)     # richer fee but needs t1 first
    t3 = Transaction("bob", "alice", 5, 4, 0)
    for t in (t2, t1, t3):
        assert mp.add(t)
    sel = mp.select(c)
    assert sel.index(t1) < sel.index(t2) and set(sel) == {t1, t2, t3}
    assert mp.select(c, limit=1) == [t3]
    res = c.add_block(mine_block(c.block_template("m", sel, 600))[0])
    mp.update_after(c, res)
    assert len(mp) == 0


# ---------------------------------------------------------------- network
def small_sim(seed=3, **kw):
    cfg = SimConfig([0.5, 0.3, 0.2], params=ConsensusParams(target_spacing=10, retarget_interval=20),
                    latency_mean=kw.pop("latency", 1.0), tx_rate=0.3, seed=seed, **kw)
    return NetworkSimulation(cfg)


def test_simulation_converges_and_is_deterministic():
    r1 = small_sim().run(1200)
    r2 = small_sim().run(1200)
    assert r1 == r2 and r1.converged and r1.height > 50 and r1.txs_confirmed > 0


def test_simulation_share_tracks_hashrate():
    r = small_sim(seed=5).run(6000)
    share0 = r.main_chain_blocks[0] / sum(r.main_chain_blocks)
    assert abs(share0 - 0.5) < 0.1


def test_stale_rate_grows_with_latency():
    low = small_sim(latency=0.2, seed=9).run(4000)
    high = small_sim(latency=4.0, seed=9).run(4000)
    assert high.stale_rate > low.stale_rate
    assert high.converged and low.converged
    assert abs(low.stale_rate - orphan_rate_theory(0.2, 10)) < 0.05


# ---------------------------------------------------------------- attacks
def test_nakamoto_whitepaper_value():
    assert nakamoto_double_spend(0.1, 6) == pytest.approx(0.0002428, abs=1e-6)
    assert nakamoto_double_spend(0.6, 3) == 1.0


def test_exact_formula_matches_race_simulation():
    for q, z in ((0.3, 3), (0.2, 4)):
        exact = rosenfeld_double_spend(q, z, overtake=True)
        mc = simulate_double_spend(q, z, trials=30000, seed=1)
        assert mc == pytest.approx(exact, abs=0.01)
    assert rosenfeld_double_spend(0.3, 3) > rosenfeld_double_spend(0.3, 3, overtake=True)


def test_confirmations_needed_monotone_in_hashpower():
    z10 = confirmations_needed(0.1, 0.001)
    z30 = confirmations_needed(0.3, 0.001)
    assert z10 < z30 and rosenfeld_double_spend(0.1, z10) < 0.001
    with pytest.raises(ValueError):
        confirmations_needed(0.5, 0.01)


def test_selfish_mining_matches_theory_and_threshold():
    assert simulate_selfish_mining(0.3, 0.5, 300000, 1) == pytest.approx(
        selfish_mining_revenue_theory(0.3, 0.5), abs=0.01)
    assert selfish_threshold(0.0) == pytest.approx(1 / 3)
    assert selfish_threshold(0.5) == pytest.approx(0.25)
    assert selfish_mining_revenue_theory(0.35, 0.5) > 0.35
    assert selfish_mining_revenue_theory(0.2, 0.0) < 0.2
