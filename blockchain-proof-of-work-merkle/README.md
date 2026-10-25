# Blockchain Proof-of-Work Simulator with Merkle Trees and Fork Resolution

## Overview

A from-scratch proof-of-work blockchain in pure Python (standard library only). It has real double-SHA-256 mining, Merkle commitments with inclusion proofs, full block validation, difficulty retargeting, heaviest-chain fork choice with reorganisations, and a discrete-event network simulator in which forks and orphans arise from the actual consensus rules. A final module compares double-spend and selfish-mining closed forms against Monte Carlo simulation.

Modules (`pow_chain/`):

| Module | Contents |
|--------|----------|
| `merkle.py` | Domain-separated Merkle tree, inclusion proofs, incremental update, mutation (CVE-2012-2459) check |
| `block.py` | Account-model transactions, 80-byte headers, blocks |
| `pow.py` | Compact `bits` encoding, work, nonce grinding, Bitcoin-style retargeting with 4x clamp |
| `chain.py` | Block tree, validation, orphan pool, reorgs with per-block state, mempool, SPV proofs |
| `network.py` | Event-driven multi-miner simulation with latency and a transaction workload |
| `attacks.py` | Nakamoto/Rosenfeld double-spend probabilities, Eyal-Sirer selfish mining, orphan-rate model |

## How it works

**Merkle trees.** Leaves are hashed as `H(0x00 || data)` and interior nodes as `H(0x01 || l || r)`, which prevents interior nodes from being replayed as leaves. An unpaired node is promoted (default) or duplicated (Bitcoin style, which admits two transaction lists with one root). A proof is the list of siblings on the path to the root, so a light client holding only headers can check inclusion with `log2 n` hashes.

**Mining.** A header is valid if `SHA256d(header) <= target`. The target is stored in a 4-byte compact form; the expected number of hashes is `2^256 / (target + 1)`. Every `retarget_interval` blocks the target is scaled by (observed timespan / expected timespan), clamped to a factor of four.

**Validation.** A block must carry the exact expected `bits`, meet its target, commit to its transactions via the Merkle root, have a timestamp above the median of the previous 11 blocks, and apply cleanly to the parent's account state (sequential nonces, sufficient balance, coinbase = reward + fees).

**Fork choice.** Every valid block is kept with its post-state. The tip is the block with the greatest cumulative work, first-seen on ties. When the tip moves to another branch the chain reports the disconnected and connected blocks, and the mempool re-queues transactions from the abandoned branch. Blocks that arrive before their parent wait in an orphan pool.

**Simulation.** Each miner draws an exponential delay with rate `hashrate / work(target)` (restarted whenever its tip changes, which is valid by memorylessness), then grinds a genuine nonce. Blocks travel to every peer with random latency and are fully validated on arrival.

**Attack analysis.** Nakamoto's Poisson approximation and Rosenfeld's exact negative-binomial formula give the probability that an attacker with hashrate share `q` rewrites a payment after `z` confirmations. The Eyal-Sirer strategy is simulated as a state machine and compared to the published revenue formula; the profitability threshold is `(1 - gamma) / (3 - 2 gamma)`.

## Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Merkle build | O(n) | O(n) |
| Proof / verify / leaf update | O(log n) | O(log n) |
| Mining | expected `work(target)` hashes | O(1) |
| Add block (validation) | O(t + a) for t txs and a accounts (state copy) | O(a) per block |
| Retarget ancestor lookup | O(retarget interval) | O(1) |
| Reorg diff | O(reorg depth) | O(depth) |
| Mempool selection | O(m^2) for m pending txs | O(m) |

## Usage

```python
from pow_chain import Blockchain, ConsensusParams, Transaction, mine_block

chain = Blockchain(ConsensusParams(genesis_alloc=(("alice", 1000), ("bob", 0))))
tx = Transaction("alice", "bob", 40, fee=2, nonce=0)
block, tries = mine_block(chain.block_template("miner", [tx], timestamp=600))
print(chain.add_block(block).status)              # extended
bh, proof = chain.prove_tx(tx.txid)
print(chain.verify_tx_proof(tx.txid, bh, proof))  # True

from pow_chain import NetworkSimulation, SimConfig
report = NetworkSimulation(SimConfig([0.5, 0.3, 0.2], ConsensusParams(target_spacing=10),
                                     latency_mean=1.0, seed=1)).run(3000)
print(report.stale_rate, report.converged)
```

Run `python examples/demo.py` for the full walk-through and `python -m pytest -q` for the 37 tests.

## Results

Output of `examples/demo.py` (4 miners with 40/30/20/10 % hashrate, 10 s target spacing, 3000 s simulated, seed 7):

| Mean latency | Height | Stale blocks | `1-exp(-d/T)` | Reorgs (all nodes) | Max depth |
|--------------|--------|--------------|---------------|--------------------|-----------|
| 0.1 s | 288 | 0.7 % | 1.0 % | 4 | 1 |
| 1.0 s | 297 | 4.2 % | 9.5 % | 21 | 1 |
| 3.0 s | 274 | 17.7 % | 25.9 % | 80 | 3 |

The theoretical value is an upper bound because a competing block only becomes stale if it loses the race, and in all runs all nodes converged on one tip.

Double-spend success probability (attacker must strictly overtake):

| q | z | Nakamoto | Exact (tie) | Exact (overtake) | Monte Carlo |
|---|---|----------|-------------|------------------|-------------|
| 0.1 | 6 | 0.00024 | 0.00059 | 0.00011 | 0.00015 |
| 0.3 | 6 | 0.13211 | 0.15645 | 0.08911 | 0.08910 |
| 0.4 | 10 | 0.35998 | 0.37218 | 0.29063 | 0.29320 |

Confirmations needed for risk below 0.1 %: 4, 6, 13, 32 for q = 0.05, 0.1, 0.2, 0.3.

Selfish mining revenue share: alpha = 0.3, gamma = 0.5 gives 0.3269 in theory and 0.3281 simulated (honest share would be 0.30); alpha = 0.4, gamma = 1 gives 0.5674 against 0.5637.

## References

- S. Nakamoto, *Bitcoin: A Peer-to-Peer Electronic Cash System*, 2008.
- M. Rosenfeld, *Analysis of Hashrate-Based Double Spending*, arXiv:1402.2009, 2014.
- I. Eyal and E. G. Sirer, *Majority is not Enough: Bitcoin Mining is Vulnerable*, Financial Cryptography 2014.
- R. C. Merkle, *A Digital Signature Based on a Conventional Encryption Function*, CRYPTO 1987.
- C. Decker and R. Wattenhofer, *Information Propagation in the Bitcoin Network*, IEEE P2P 2013.
- CVE-2012-2459, Bitcoin Merkle tree duplicate-leaf mutation.

---

**Author:** Dr. Kainat Rizwan
