"""Walk-through: Merkle proofs, mining, forks and reorgs, network simulation, attack maths."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pow_chain import (Blockchain, ConsensusParams, MerkleTree, NetworkSimulation, SimConfig,
                       Transaction, bits_to_target, confirmations_needed, mine_block,
                       nakamoto_double_spend, orphan_rate_theory, roots_collide,
                       rosenfeld_double_spend, selfish_mining_revenue_theory, selfish_threshold,
                       simulate_double_spend, simulate_selfish_mining, verify_proof)


def section(title):
    print(f"\n=== {title} ===")


def main():
    section("Merkle tree")
    items = [f"tx{i}".encode() for i in range(1000)]
    tree = MerkleTree(items)
    proof = tree.proof(417)
    print(f"1000 leaves, height {tree.height}, proof length {len(proof)} hashes "
          f"({32 * len(proof)} bytes), verifies: {verify_proof(items[417], proof, tree.root)}")
    print(f"Bitcoin-style duplicate-odd mutation possible: {roots_collide(items[:3])}; "
          f"with promotion policy: {MerkleTree(items[:3]).root == MerkleTree(items[:3] + items[2:3]).root}")

    section("Mining and fork choice")
    params = ConsensusParams(genesis_alloc=(("alice", 1000), ("bob", 0)), retarget_interval=5)
    a = Blockchain(params)
    b = Blockchain(params)
    pay = Transaction("alice", "bob", 40, 2, 0)
    t0 = time.perf_counter()
    blk, tries = mine_block(a.block_template("minerA", [pay], 600))
    print(f"mined block 1 in {tries} hashes ({time.perf_counter() - t0:.3f}s), "
          f"target 2^{bits_to_target(blk.header.bits).bit_length()}")
    a.add_block(blk)
    print(f"A: height {a.height}, bob={a.balance('bob')}, confirmations={a.confirmations(pay.txid)}")
    for i in range(2):
        b.add_block(mine_block(b.block_template("minerB", [], 600 * (i + 1)))[0])
    print(f"B (competing, no payment): height {b.height}")
    for node in b.main_chain()[1:]:
        r = a.add_block(node.block)
        print(f"  A receives B's block {node.height}: {r.status}, reorg depth {r.reorg_depth}")
    print(f"A after reorg: bob={a.balance('bob')}, payment confirmations={a.confirmations(pay.txid)}, "
          f"stale blocks={len(a.stale_blocks())}")

    section("Network simulation (4 miners, 10 s blocks)")
    print(f"{'latency':>8} {'height':>7} {'stale%':>7} {'theory%':>8} {'reorgs':>7} {'maxdepth':>9} {'interval':>9}")
    for lat in (0.1, 1.0, 3.0):
        cfg = SimConfig([0.4, 0.3, 0.2, 0.1], ConsensusParams(target_spacing=10, retarget_interval=20),
                        latency_mean=lat, tx_rate=0.5, seed=7)
        r = NetworkSimulation(cfg).run(3000)
        print(f"{lat:>8} {r.height:>7} {100 * r.stale_rate:>7.1f} {100 * orphan_rate_theory(lat, 10):>8.1f} "
              f"{r.reorgs:>7} {r.max_reorg_depth:>9} {r.mean_interval:>9.2f}  converged={r.converged}")
    shares = [c / sum(r.main_chain_blocks) for c in r.main_chain_blocks]
    print("main-chain share by miner:", [round(s, 2) for s in shares], "(hashrate 0.4/0.3/0.2/0.1)")

    section("Double-spend probability")
    print(f"{'q':>5} {'z':>3} {'Nakamoto':>10} {'tie-win':>9} {'overtake':>9} {'MonteCarlo':>11}")
    for q, z in ((0.1, 6), (0.3, 6), (0.3, 12), (0.4, 10)):
        print(f"{q:>5} {z:>3} {nakamoto_double_spend(q, z):>10.5f} {rosenfeld_double_spend(q, z):>9.5f} "
              f"{rosenfeld_double_spend(q, z, True):>9.5f} {simulate_double_spend(q, z, 20000, 1):>11.5f}")
    print("confirmations for risk < 0.1%:", {q: confirmations_needed(q, 1e-3) for q in (0.05, 0.1, 0.2, 0.3)})

    section("Selfish mining (Eyal-Sirer)")
    print(f"threshold at gamma=0: {selfish_threshold(0):.3f}, gamma=0.5: {selfish_threshold(0.5):.3f}")
    print(f"{'alpha':>6} {'gamma':>6} {'theory':>8} {'simulated':>10}")
    for al, ga in ((0.2, 0.5), (0.3, 0.5), (0.4, 0.0), (0.4, 1.0)):
        print(f"{al:>6} {ga:>6} {selfish_mining_revenue_theory(al, ga):>8.4f} "
              f"{simulate_selfish_mining(al, ga, 200000, 1):>10.4f}")


if __name__ == "__main__":
    main()
