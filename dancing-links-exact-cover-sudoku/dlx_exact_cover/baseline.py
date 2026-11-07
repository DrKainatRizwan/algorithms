"""Reference Algorithm X using dictionaries of sets (no dancing links).

Selecting a row deep-copies nothing: removed columns are returned from
``_select`` and restored by ``_deselect``.  It is correct but allocates sets
at every step, which is what the DLX benchmark compares against.
"""
from __future__ import annotations

from typing import Iterator, Sequence


def _build(n_primary: int, n_cols: int, options: Sequence[Sequence[int]]):
    X: dict[int, set[int]] = {c: set() for c in range(n_cols)}
    Y = {i: tuple(o) for i, o in enumerate(options)}
    for i, cols in Y.items():
        for c in cols:
            X[c].add(i)
    return X, Y


def _select(X, Y, r):
    removed = []
    for j in Y[r]:
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].remove(i)
        removed.append(X.pop(j))
    return removed


def _deselect(X, Y, r, removed):
    for j in reversed(Y[r]):
        X[j] = removed.pop()
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].add(i)


def _search(X, Y, n_primary, sol) -> Iterator[list[int]]:
    prim = [c for c in X if c < n_primary]
    if not prim:
        yield sorted(sol)
        return
    c = min(prim, key=lambda k: len(X[k]))
    for r in list(X[c]):
        sol.append(r)
        removed = _select(X, Y, r)
        yield from _search(X, Y, n_primary, sol)
        _deselect(X, Y, r, removed)
        sol.pop()


def solve_dict(n_primary: int, n_secondary: int,
               options: Sequence[Sequence[int]]) -> Iterator[list[int]]:
    """Yield all exact covers (as sorted option-index lists)."""
    X, Y = _build(n_primary, n_primary + n_secondary, options)
    yield from _search(X, Y, n_primary, [])


def count_dict(n_primary: int, n_secondary: int,
               options: Sequence[Sequence[int]]) -> int:
    """Count exact covers with the dictionary implementation."""
    return sum(1 for _ in solve_dict(n_primary, n_secondary, options))
