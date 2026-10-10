"""Demo: MCTS strength vs. iterations, ablations and a sample game."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcts_connect4 import AlphaBetaPlayer, Board, MCTS, MCTSPlayer, RandomPlayer, match


def main() -> None:
    t0 = time.time()
    print("== MCTS vs random (6 games, colours alternate) ==")
    for its in (50, 200, 500):
        r = match(MCTSPlayer(its, seed=1), RandomPlayer(seed=1), games=6)
        print(f"mcts-{its:<4} W/D/L = {r['wins']}/{r['draws']}/{r['losses']}  rate={r['rate']:.2f}")

    print("\n== MCTS vs alpha-beta depth 3 (4 games) ==")
    for its in (200, 800):
        r = match(MCTSPlayer(its, seed=2), AlphaBetaPlayer(3), games=4)
        print(f"mcts-{its:<4} W/D/L = {r['wins']}/{r['draws']}/{r['losses']}  rate={r['rate']:.2f}")

    print("\n== Ablation: heuristic vs random rollouts (200 iters, 4 games) ==")
    r = match(MCTSPlayer(200, rollout="heuristic", seed=3),
              MCTSPlayer(200, rollout="random", seed=3), games=4)
    print(f"heuristic vs random rollout: W/D/L = {r['wins']}/{r['draws']}/{r['losses']}")

    print("\n== Root statistics, empty board, 2000 iterations ==")
    eng = MCTS(seed=0)
    best = eng.search(Board(), 2000)
    for col, (v, wr) in sorted(eng.move_stats().items()):
        print(f"col {col}: visits={v:5d} winrate={wr:.3f}")
    print("best:", best, "tree size:", eng.root.size(), "depth:", eng.root.depth())

    print("\n== Forced win detection ==")
    b = Board.from_moves([1, 6, 2, 6, 3])
    e = MCTS(seed=0)
    print("move:", e.search(b, 3000), "root proven:", e.root.proven)
    print(b)
    print(f"\nelapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
