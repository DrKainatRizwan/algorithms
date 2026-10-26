"""Experiment driver: attack x defence x poison-rate grids."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .attacks import (AttackResult, boundary_flip, gradient_poisoning, influence_label_flip,
                      random_label_flip)
from .data import Dataset
from .defenses import DEFENSES
from .model import LogisticRegression


def _attack(name: str, ds: Dataset, frac: float, seed: int, l2: float) -> AttackResult:
    rng = np.random.default_rng(seed)
    X, y = ds.X_train, ds.y_train
    if name == "random-flip":
        return random_label_flip(X, y, frac, rng)
    if name == "boundary-flip":
        return boundary_flip(X, y, frac, l2)
    if name == "influence-flip":
        return influence_label_flip(X, y, ds.X_val, ds.y_val, frac, l2)
    if name == "gradient":
        return gradient_poisoning(X, y, ds.X_val, ds.y_val, frac, rng, l2, steps=25)
    raise ValueError(f"unknown attack {name!r}")


ATTACKS = ("random-flip", "boundary-flip", "influence-flip", "gradient")


def evaluate_grid(ds: Dataset, attacks: tuple[str, ...] = ATTACKS,
                  defenses: tuple[str, ...] | None = None, fracs: tuple[float, ...] = (0.1, 0.2),
                  l2: float = 1.0, seed: int = 0) -> list[dict]:
    """Return one row per (attack, frac, defence) with test accuracy and filter quality."""
    defenses = defenses or tuple(DEFENSES)
    clean_acc = LogisticRegression(l2).fit(ds.X_train, ds.y_train).accuracy(ds.X_test, ds.y_test)
    rows: list[dict] = []
    for a in attacks:
        for f in fracs:
            res = _attack(a, ds, f, seed, l2)
            for dname in defenses:
                out = DEFENSES[dname](res.X, res.y, l2=l2, alpha=f, remove_frac=f, beta=f)
                dropped = ~out.kept
                tp = int((dropped & res.poisoned).sum())
                rows.append({
                    "attack": a, "frac": f, "defense": dname,
                    "test_acc": out.model.accuracy(ds.X_test, ds.y_test),
                    "clean_acc": clean_acc,
                    "poison_recall": tp / max(1, res.n_poisoned),
                    "dropped": int(dropped.sum()),
                })
    return rows


def format_table(rows: list[dict]) -> str:
    """Pivot grid rows into a text table: one line per (attack, frac), one column per defence."""
    defs = list(dict.fromkeys(r["defense"] for r in rows))
    head = f"{'attack':<15}{'frac':>6}" + "".join(f"{d:>14}" for d in defs)
    lines = [head, "-" * len(head)]
    keys = list(dict.fromkeys((r["attack"], r["frac"]) for r in rows))
    for a, f in keys:
        cells = {r["defense"]: r["test_acc"] for r in rows if r["attack"] == a and r["frac"] == f}
        lines.append(f"{a:<15}{f:>6.2f}" + "".join(f"{cells[d]:>14.3f}" for d in defs))
    return "\n".join(lines)
