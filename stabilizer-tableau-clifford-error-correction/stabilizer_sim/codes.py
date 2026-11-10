"""Stabilizer codes, syndrome decoding and a Monte-Carlo memory experiment."""

from __future__ import annotations

from itertools import combinations, product

import numpy as np

from .pauli import Pauli
from .tableau import Tableau


class StabilizerCode:
    """An [[n, k=1, d]] stabilizer code with a lookup-table decoder."""

    def __init__(self, name: str, generators: list[str], logical_x: str,
                 logical_z: str, max_weight: int = 2) -> None:
        self.name = name
        self.generators = [Pauli.from_label(g) for g in generators]
        self.logical_x = Pauli.from_label(logical_x)
        self.logical_z = Pauli.from_label(logical_z)
        self.n = self.logical_x.n
        self._check()
        self.table = self._build_table(max_weight)

    def _check(self) -> None:
        for a in self.generators:
            for b in self.generators:
                if not a.commutes(b):
                    raise ValueError("stabilizer generators must commute")
            for lg in (self.logical_x, self.logical_z):
                if not a.commutes(lg):
                    raise ValueError("logical operators must commute with stabilizers")
        if self.logical_x.commutes(self.logical_z):
            raise ValueError("logical X and Z must anticommute")

    def syndrome_of(self, error: Pauli) -> tuple[int, ...]:
        """Syndrome bits: 1 where the error anticommutes with a generator."""
        return tuple(0 if g.commutes(error) else 1 for g in self.generators)

    def _build_table(self, max_weight: int) -> dict[tuple[int, ...], Pauli]:
        table: dict[tuple[int, ...], Pauli] = {tuple([0] * len(self.generators)): Pauli.from_label("I" * self.n)}
        for w in range(1, max_weight + 1):
            for qs in combinations(range(self.n), w):
                for kinds in product("XYZ", repeat=w):
                    chars = ["I"] * self.n
                    for q, k in zip(qs, kinds):
                        chars[q] = k
                    e = Pauli.from_label("".join(chars))
                    table.setdefault(self.syndrome_of(e), e)
        return table

    def decode(self, syndrome: tuple[int, ...]) -> Pauli | None:
        """Minimum-weight correction for ``syndrome`` or None if unknown."""
        return self.table.get(syndrome)

    def distance_upper_bound(self) -> int:
        """Smallest weight among logical operators found by exhaustive search."""
        best = self.n + 1
        for w in range(1, self.n + 1):
            for qs in combinations(range(self.n), w):
                for kinds in product("XYZ", repeat=w):
                    chars = ["I"] * self.n
                    for q, k in zip(qs, kinds):
                        chars[q] = k
                    e = Pauli.from_label("".join(chars))
                    if all(g.commutes(e) for g in self.generators) and not (
                            e.commutes(self.logical_x) and e.commutes(self.logical_z)):
                        return w
        return best

    def encode_zero(self, rng: np.random.Generator | None = None) -> Tableau:
        """Logical |0>: project |0..0> onto all stabilizers and logical Z = +1."""
        t = Tableau(self.n)
        rng = rng or np.random.default_rng(0)
        for g in self.generators + [self.logical_z]:
            t.measure_pauli(g, rng, forced=0)
        return t

    def encode_plus(self, rng: np.random.Generator | None = None) -> Tableau:
        """Logical |+>: project onto all stabilizers and logical X = +1."""
        t = Tableau(self.n)
        rng = rng or np.random.default_rng(0)
        for g in self.generators + [self.logical_x]:
            t.measure_pauli(g, rng, forced=0)
        return t

    def measure_syndrome(self, t: Tableau) -> tuple[int, ...]:
        """Syndrome of a (Pauli-corrupted) code state via exact expectations."""
        return tuple(0 if t.expectation(g) == 1 else 1 for g in self.generators)

    def logical_ok(self, t: Tableau, basis: str = "z") -> bool:
        op = self.logical_z if basis == "z" else self.logical_x
        return t.expectation(op) == 1


def bit_flip_code() -> StabilizerCode:
    """3-qubit repetition code (corrects one X error only)."""
    return StabilizerCode("bit-flip [[3,1,1]]", ["ZZI", "IZZ"], "XXX", "ZII", max_weight=1)


def steane_code() -> StabilizerCode:
    """CSS [[7,1,3]] Steane code built from the Hamming code."""
    rows = ["IIIXXXX", "IXXIIXX", "XIXIXIX"]
    gens = rows + [r.replace("X", "Z") for r in rows]
    return StabilizerCode("Steane [[7,1,3]]", gens, "XXXXXXX", "ZZZZZZZ")


def five_qubit_code() -> StabilizerCode:
    """The perfect [[5,1,3]] code (cyclic shifts of XZZXI)."""
    gens = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]
    return StabilizerCode("Five-qubit [[5,1,3]]", gens, "XXXXX", "ZZZZZ")


def memory_experiment(code: StabilizerCode, p: float, trials: int,
                      rng: np.random.Generator) -> float:
    """Logical failure rate under i.i.d. depolarising noise of strength ``p``.

    Each trial encodes logical |0> or |+> (alternating), applies an independent
    X/Y/Z error with total probability ``p`` per qubit, decodes by table lookup
    and checks the logical observable.
    """
    fails = 0
    for trial in range(trials):
        basis = "z" if trial % 2 == 0 else "x"
        t = code.encode_zero(rng) if basis == "z" else code.encode_plus(rng)
        for q in range(code.n):
            if rng.random() < p:
                t.apply_pauli(Pauli.single(code.n, q, str(rng.choice(["X", "Y", "Z"]))))
        corr = code.decode(code.measure_syndrome(t))
        if corr is not None:
            t.apply_pauli(corr)
        if not code.logical_ok(t, basis):
            fails += 1
    return fails / trials
