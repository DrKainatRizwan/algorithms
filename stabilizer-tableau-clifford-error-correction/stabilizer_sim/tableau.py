"""Aaronson-Gottesman stabilizer tableau with Clifford gates and measurement."""

from __future__ import annotations

import numpy as np

from .pauli import Pauli, phase_exponent


class Tableau:
    """Binary tableau of ``n`` destabilizers, ``n`` stabilizers and a scratch row.

    Rows ``0..n-1`` are destabilizers, ``n..2n-1`` stabilizers, row ``2n`` is
    scratch. Memory is O(n^2) and every gate costs O(n).
    """

    def __init__(self, n: int) -> None:
        if n < 1:
            raise ValueError("need at least one qubit")
        self.n = n
        self.x = np.zeros((2 * n + 1, n), dtype=np.uint8)
        self.z = np.zeros((2 * n + 1, n), dtype=np.uint8)
        self.r = np.zeros(2 * n + 1, dtype=np.uint8)
        for i in range(n):
            self.x[i, i] = 1        # destabilizer X_i
            self.z[n + i, i] = 1    # stabilizer  Z_i  -> state |0...0>

    def copy(self) -> "Tableau":
        t = Tableau(self.n)
        t.x, t.z, t.r = self.x.copy(), self.z.copy(), self.r.copy()
        return t

    # ---- Clifford gates -------------------------------------------------
    def h(self, a: int) -> None:
        self.r ^= self.x[:, a] & self.z[:, a]
        self.x[:, a], self.z[:, a] = self.z[:, a].copy(), self.x[:, a].copy()

    def s(self, a: int) -> None:
        self.r ^= self.x[:, a] & self.z[:, a]
        self.z[:, a] ^= self.x[:, a]

    def sdg(self, a: int) -> None:
        self.s(a); self.s(a); self.s(a)

    def cx(self, a: int, b: int) -> None:
        if a == b:
            raise ValueError("control and target must differ")
        self.r ^= self.x[:, a] & self.z[:, b] & (self.x[:, b] ^ self.z[:, a] ^ 1)
        self.x[:, b] ^= self.x[:, a]
        self.z[:, a] ^= self.z[:, b]

    def z_gate(self, a: int) -> None:
        self.s(a); self.s(a)

    def x_gate(self, a: int) -> None:
        self.h(a); self.z_gate(a); self.h(a)

    def y_gate(self, a: int) -> None:
        self.z_gate(a); self.x_gate(a)

    def cz(self, a: int, b: int) -> None:
        self.h(b); self.cx(a, b); self.h(b)

    def swap(self, a: int, b: int) -> None:
        self.cx(a, b); self.cx(b, a); self.cx(a, b)

    def apply_pauli(self, p: Pauli) -> None:
        """Apply a Pauli operator (global sign ignored) as a unitary."""
        for q in range(self.n):
            xb, zb = int(p.x[q]), int(p.z[q])
            if xb and zb:
                self.y_gate(q)
            elif xb:
                self.x_gate(q)
            elif zb:
                self.z_gate(q)

    # ---- row algebra ----------------------------------------------------
    def _rowsum(self, h: int, i: int) -> None:
        """Replace row ``h`` by the product ``row_i * row_h`` (tracking phase)."""
        k = 2 * int(self.r[h]) + 2 * int(self.r[i]) + phase_exponent(
            self.x[i], self.z[i], self.x[h], self.z[h])
        self.r[h] = (k % 4) // 2
        self.x[h] ^= self.x[i]
        self.z[h] ^= self.z[i]

    def stabilizers(self) -> list[Pauli]:
        """Current stabilizer generators as signed Pauli strings."""
        n = self.n
        return [Pauli(self.x[n + i], self.z[n + i], int(self.r[n + i])) for i in range(n)]

    def destabilizers(self) -> list[Pauli]:
        n = self.n
        return [Pauli(self.x[i], self.z[i], int(self.r[i])) for i in range(n)]

    # ---- measurement ----------------------------------------------------
    def measure(self, a: int, rng: np.random.Generator | None = None,
                forced: int | None = None) -> int:
        """Measure qubit ``a`` in the Z basis; returns 0 or 1.

        If the outcome is random it is drawn from ``rng`` unless ``forced`` is
        given (post-selection).
        """
        n = self.n
        pivots = np.nonzero(self.x[n:2 * n, a])[0]
        if pivots.size:
            p = n + int(pivots[0])
            for i in range(2 * n):
                if i != p and self.x[i, a]:
                    self._rowsum(i, p)
            self.x[p - n], self.z[p - n], self.r[p - n] = self.x[p].copy(), self.z[p].copy(), self.r[p]
            self.x[p] = 0
            self.z[p] = 0
            self.z[p, a] = 1
            if forced is not None:
                outcome = int(forced)
            else:
                if rng is None:
                    raise ValueError("random measurement needs an rng or forced outcome")
                outcome = int(rng.integers(2))
            self.r[p] = outcome
            return outcome
        return self._deterministic_outcome(Pauli.single(n, a, "Z")) 

    def _deterministic_outcome(self, p: Pauli) -> int:
        """Outcome bit of a Pauli known to lie in (+-) the stabilizer group."""
        n = self.n
        s = 2 * n
        self.x[s] = 0; self.z[s] = 0; self.r[s] = 0
        for i in range(n):
            if (int(np.dot(self.x[i].astype(int), p.z.astype(int)))
                    + int(np.dot(self.z[i].astype(int), p.x.astype(int)))) % 2:
                self._rowsum(s, n + i)
        return int(self.r[s]) ^ p.sign

    def expectation(self, p: Pauli) -> int:
        """Expectation value of Pauli ``p``: +1, -1, or 0 (undetermined)."""
        n = self.n
        sx = self.x[n:2 * n].astype(int); sz = self.z[n:2 * n].astype(int)
        anti = (sx @ p.z.astype(int) + sz @ p.x.astype(int)) % 2
        if anti.any():
            return 0
        return -1 if self._deterministic_outcome(p) else 1

    def measure_pauli(self, p: Pauli, rng: np.random.Generator | None = None,
                      forced: int | None = None) -> int:
        """Projectively measure a multi-qubit Pauli observable (outcome 0 -> +1)."""
        support = [q for q in range(self.n) if p.x[q] or p.z[q]]
        if not support:
            return p.sign
        ops: list[tuple[str, int, int]] = []
        for q in support:
            if p.x[q] and p.z[q]:
                ops += [("sdg", q, -1), ("h", q, -1)]
            elif p.x[q]:
                ops.append(("h", q, -1))
        root = support[0]
        ops += [("cx", q, root) for q in support[1:]]
        for name, a, b in ops:
            self._apply(name, a, b)
        out = self.measure(root, rng, None if forced is None else forced ^ p.sign)
        for name, a, b in reversed(ops):
            self._apply({"sdg": "s"}.get(name, name), a, b)
        return out ^ p.sign

    def _apply(self, name: str, a: int, b: int) -> None:
        if name == "cx":
            self.cx(a, b)
        else:
            getattr(self, name)(a)

    # ---- analysis -------------------------------------------------------
    def entanglement_entropy(self, region: list[int]) -> int:
        """Von Neumann entropy (in bits) of the reduced state on ``region``.

        S_A = rank_GF2(stabilizer matrix restricted to A) - |A|.
        """
        n = self.n
        cols = list(region)
        m = np.concatenate([self.x[n:2 * n][:, cols], self.z[n:2 * n][:, cols]], axis=1)
        return gf2_rank(m) - len(cols)

    def is_valid(self) -> bool:
        """Check commutation relations of the destabilizer/stabilizer basis."""
        n = self.n
        st, de = self.stabilizers(), self.destabilizers()
        for i in range(n):
            for j in range(n):
                if not st[i].commutes(st[j]) or not de[i].commutes(de[j]):
                    return False
                if st[i].commutes(de[j]) != (i != j):
                    return False
        return True


def gf2_rank(m: np.ndarray) -> int:
    """Rank of a binary matrix over GF(2) by Gaussian elimination."""
    a = (m.copy() % 2).astype(np.uint8)
    rows, cols = a.shape
    rank = 0
    for c in range(cols):
        piv = None
        for r in range(rank, rows):
            if a[r, c]:
                piv = r
                break
        if piv is None:
            continue
        a[[rank, piv]] = a[[piv, rank]]
        for r in range(rows):
            if r != rank and a[r, c]:
                a[r] ^= a[rank]
        rank += 1
        if rank == rows:
            break
    return rank
