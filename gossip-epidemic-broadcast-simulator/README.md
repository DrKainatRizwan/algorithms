# Distributed Gossip Protocol and Epidemic Broadcast Simulator

## Overview
A from-scratch simulator for epidemic-style information dissemination in a cluster. It covers synchronous push, pull and push-pull rumor spreading, rumor mongering with loss of interest, push-sum gossip averaging, and anti-entropy reconciliation of replicated last-writer-wins stores. It runs on complete, Erdős–Rényi, small-world, scale-free and ring-lattice overlays, with message loss and crashed nodes, and compares results to analytical predictions.

## How it works
- **Spread (`protocols.py`)**: each round, every alive node contacts `fanout` random neighbours. Push sends the rumor from informed nodes; pull lets uninformed nodes request it; push-pull does both. Contacts are dropped with probability `loss`; crashed nodes stay in neighbour lists so contacts to them are wasted.
- **Rumor mongering (`rumor.py`)**: infective nodes push every round and lose interest after hitting an already-informed node (feedback-coin with probability 1/k, feedback-counter after k such hits, or blind-coin). The residue (nodes never informed) follows s = exp(-(k+1)(1-s)).
- **Push-sum (`aggregation.py`)**: each node holds (s, w), keeps half and sends half to a random neighbour. s/w converges to the global mean while total s and w are conserved.
- **Anti-entropy (`antientropy.py`)**: replicas hold key → (timestamp, origin, value); the highest tuple wins. Pairs skip work when content digests match.
- **Theory (`theory.py`)**: push needs about log2 n + ln n rounds; mean-field recurrences give expected informed fractions per round on K_n.

## Complexity
- Spread: O(rounds · n · fanout) time, O(n) space. On K_n rounds are Θ(log n) for all three modes; push-pull has the smallest constant, and total messages are O(n log n) (O(n log log n) is possible with address-oblivious variants).
- Push-sum: O(n) per round; rounds scale with log n + log(1/ε) on well-mixing graphs.
- Anti-entropy: O(|keys|) per reconciliation (hashing), O(log n) rounds.

## Usage
```python
import numpy as np
from gossip_sim import complete_graph, spread, run_trials

g = complete_graph(1000)
res = spread(g, "push-pull", np.random.default_rng(0))
print(res.rounds_to_full, res.messages)
print(run_trials(g, "push", trials=20, seed=1, loss=0.2))
```
Run `python examples/demo.py` for the full demo and `python -m pytest -q` for the tests (22 tests).

## Results (from `examples/demo.py`)
Rounds to inform every node, complete graph, mean of 15 trials:

| n | push | pull | push-pull | log2 n + ln n |
|---|------|------|-----------|---------------|
| 128 | 13.40 | 10.13 | 6.93 | 11.85 |
| 512 | 17.20 | 12.47 | 8.53 | 15.24 |
| 1024 | 17.73 | 13.73 | 9.07 | 16.93 |

Push-pull, n = 300 (10 trials): complete 8.0 rounds, G(n,0.03) 8.9, scale-free 9.4, small-world 15.0, ring k=2 117.3.
Message loss, n = 1000 push-pull: 9.1 rounds (0% loss), 12.8 (30%), 21.6 (60%).
Rumor residue, n = 2000 (simulated / theory): k=1 0.2035 / 0.2032, k=2 0.0590 / 0.0595, k=3 0.0172 / 0.0198, k=4 0.0075 / 0.0070.
Push-sum, n = 200, tol 1e-6: 44 rounds on the complete graph, 188 on a small-world graph.
Anti-entropy, 64 replicas: push 14 rounds, pull 9, push-pull 5.

## References
- Demers et al., "Epidemic Algorithms for Replicated Database Maintenance", PODC 1987.
- Karp, Schindelhauer, Shenker, Vöcking, "Randomized Rumor Spreading", FOCS 2000.
- Kempe, Dobra, Gehrke, "Gossip-Based Computation of Aggregate Information", FOCS 2003.
- Pittel, "On Spreading a Rumor", SIAM J. Appl. Math. 1987.
- Frieze, Grimmett, "The Shortest-Path Problem for Graphs with Random Arc-Lengths", Discrete Appl. Math. 1985.

---

**Author:** Dr. Kainat Rizwan
