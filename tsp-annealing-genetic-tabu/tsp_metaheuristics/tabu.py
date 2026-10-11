"""Tabu search over the 2-opt neighbourhood with aspiration."""

from __future__ import annotations

import numpy as np

from .annealing import Result
from .instances import Instance, tour_length
from .local_search import nearest_neighbour, two_opt_delta


def tabu_search(inst: Instance, iterations: int = 300, tenure: int | None = None, seed: int = 0,
                candidates: int | None = None, start: np.ndarray | None = None) -> Result:
    """Best-admissible-move tabu search.

    A move reverses ``tour[i..j]``; it removes edges (a,b) and (c,e) and adds (a,c), (b,e).
    The removed edges become tabu for ``tenure`` iterations (re-adding them is forbidden)
    unless the move yields a new global best (aspiration).  ``candidates`` samples a random
    subset of the neighbourhood each iteration for speed.
    """
    rng = np.random.default_rng(seed)
    n = inst.n
    cur = nearest_neighbour(inst) if start is None else np.array(start)
    cur_len = tour_length(inst, cur)
    best, best_len = cur.copy(), cur_len
    history = [best_len]
    if n < 4:
        return Result(best, best_len, history, 0)
    tenure = tenure if tenure is not None else max(5, int(np.sqrt(n)))
    tabu: dict[tuple[int, int], int] = {}
    pairs = np.array([(i, j) for i in range(1, n - 1) for j in range(i + 1, n)])
    evals = 0
    for it in range(iterations):
        pool = pairs if candidates is None or candidates >= len(pairs) else \
            pairs[rng.choice(len(pairs), size=candidates, replace=False)]
        move, move_delta = None, np.inf
        for i, j in pool:
            i, j = int(i), int(j)
            dl = two_opt_delta(inst, cur, i, j)
            evals += 1
            a, c = int(cur[i - 1]), int(cur[j])
            b, e = int(cur[i]), int(cur[(j + 1) % n])
            added = ((min(a, c), max(a, c)), (min(b, e), max(b, e)))
            is_tabu = any(tabu.get(x, -1) > it for x in added)
            if is_tabu and not (cur_len + dl < best_len - 1e-12):
                continue
            if dl < move_delta:
                move, move_delta = (i, j), dl
        if move is None:
            tabu.clear()
            continue
        i, j = move
        a, b = int(cur[i - 1]), int(cur[i])
        c, e = int(cur[j]), int(cur[(j + 1) % n])
        for x in ((min(a, b), max(a, b)), (min(c, e), max(c, e))):
            tabu[x] = it + tenure
        cur[i:j + 1] = cur[i:j + 1][::-1]
        cur_len += move_delta
        if cur_len < best_len - 1e-12:
            best, best_len = cur.copy(), cur_len
        history.append(best_len)
    return Result(best, tour_length(inst, best), history, evals)
