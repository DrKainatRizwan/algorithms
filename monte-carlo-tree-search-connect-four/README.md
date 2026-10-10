# Monte Carlo Tree Search Engine with UCT for Connect-Four

## Overview
A from-scratch Monte Carlo Tree Search (MCTS) engine using the UCT selection rule, applied to Connect-Four on a 7x6 board. The package contains a bitboard game implementation, the UCT searcher with optional tactical rollouts and a game-theoretic solver, a fixed-depth alpha-beta baseline, and tournament utilities.

## How it works
Each search iteration has four phases:
1. **Selection** – descend from the root choosing the child maximising UCB1: `w/n + c*sqrt(ln N / n)`.
2. **Expansion** – add one untried child to the tree.
3. **Simulation** – play to the end of the game. The `heuristic` rollout plays an immediate win, otherwise blocks an immediate loss, otherwise a random move; the `random` rollout is uniform.
4. **Backpropagation** – update visits and win totals from each node mover's perspective (draws count 0.5).

With the solver enabled, terminal wins/losses are propagated as proven values (MCTS-Solver): a node with a winning child for the opponent is a proven loss, and a node whose children are all proven losses for the opponent is a proven win. Proven nodes are chosen or avoided during selection, and the search stops when the root is solved. The final move is the most visited child.

Modules: `board.py` (bitboards, 4-in-a-row detection by shifts), `mcts.py` (searcher), `players.py` (random, MCTS, negamax alpha-beta), `arena.py` (games and matches).

## Complexity
- Per iteration: O(d + L) where d is tree depth and L the rollout length (at most 42 plies); each move/win check is O(1) on bitboards.
- Memory: O(N) nodes for N iterations.
- Alpha-beta baseline: O(b^(d/2)) best case, O(b^d) worst case with b ≤ 7.

## Usage
```python
from mcts_connect4 import Board, MCTS

board = Board.from_moves([3, 3, 2])
engine = MCTS(c=1.4, rollout="heuristic", seed=0)
move = engine.search(board, iterations=2000)
print(move, engine.move_stats())
```

## Results (from `examples/demo.py`, ~9 s)
| Match (colours alternate) | W/D/L |
|---|---|
| MCTS-50 vs random (6 games) | 6/0/0 |
| MCTS-500 vs random (6 games) | 6/0/0 |
| MCTS-200 vs alpha-beta depth 3 (4 games) | 4/0/0 |
| MCTS-800 vs alpha-beta depth 3 (4 games) | 4/0/0 |
| heuristic vs random rollouts, 200 iterations (4 games) | 2/0/2 |

On the empty board with 2000 iterations the engine prefers the centre column (499 visits, win rate 0.595) over the edge columns (134 and 143 visits), matching known theory. On `[1,6,2,6,3]` the solver proves a forced win for the first player.

Run the tests with `python -m pytest -q` (18 tests).

## References
- Kocsis, L. and Szepesvári, C. (2006). Bandit based Monte-Carlo Planning. ECML.
- Browne, C. et al. (2012). A Survey of Monte Carlo Tree Search Methods. IEEE TCIAIG.
- Winands, M., Björnsson, Y. and Saito, J.-T. (2008). Monte-Carlo Tree Search Solver. CG 2008.
- Auer, P., Cesa-Bianchi, N. and Fischer, P. (2002). Finite-time Analysis of the Multiarmed Bandit Problem.

---

**Author:** Dr. Kainat Rizwan
