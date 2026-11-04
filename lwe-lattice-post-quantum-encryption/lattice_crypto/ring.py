"""Arithmetic in R_q = Z_q[x]/(x^n + 1) with schoolbook and NTT multiplication."""
from __future__ import annotations

import numpy as np


def _is_prime(p: int) -> bool:
    if p < 2:
        return False
    i = 2
    while i * i <= p:
        if p % i == 0:
            return False
        i += 1
    return True


def _primitive_root_of_unity(order: int, q: int) -> int:
    """Return an element of exact multiplicative order ``order`` modulo prime q."""
    if (q - 1) % order:
        raise ValueError("no root of unity of that order")
    for g in range(2, q):
        w = pow(g, (q - 1) // order, q)
        # exact order is `order` (power of two) iff w^(order/2) == -1
        if pow(w, order // 2, q) == q - 1:
            return w
    raise ValueError("root not found")


def _bit_reverse(n: int) -> np.ndarray:
    bits = n.bit_length() - 1
    return np.array([int(format(i, f"0{bits}b")[::-1], 2) if bits else 0 for i in range(n)])


class NegacyclicRing:
    """Polynomial ring Z_q[x]/(x^n+1); n a power of two, q prime, q = 1 mod 2n for NTT."""

    def __init__(self, n: int, q: int) -> None:
        if n & (n - 1) or n < 2:
            raise ValueError("n must be a power of two")
        if not _is_prime(q):
            raise ValueError("q must be prime")
        self.n, self.q = n, q
        self.has_ntt = (q - 1) % (2 * n) == 0
        if self.has_ntt:
            self.psi = _primitive_root_of_unity(2 * n, q)
            self.omega = self.psi * self.psi % q
            self.psi_pows = np.array([pow(self.psi, i, q) for i in range(n)], dtype=np.int64)
            ipsi = pow(self.psi, q - 2, q)
            n_inv = pow(n, q - 2, q)
            self.ipsi_pows = np.array([pow(ipsi, i, q) * n_inv % q for i in range(n)], dtype=np.int64)
            self._rev = _bit_reverse(n)

    # --- basic ops -------------------------------------------------------
    def reduce(self, a) -> np.ndarray:
        return np.asarray(a, dtype=np.int64) % self.q

    def add(self, a, b) -> np.ndarray:
        return (np.asarray(a) + np.asarray(b)) % self.q

    def sub(self, a, b) -> np.ndarray:
        return (np.asarray(a) - np.asarray(b)) % self.q

    def mul_schoolbook(self, a, b) -> np.ndarray:
        """O(n^2) negacyclic convolution: x^n = -1."""
        a, b = self.reduce(a), self.reduce(b)
        n = self.n
        full = np.zeros(2 * n, dtype=np.int64)
        for i in range(n):
            full[i:i + n] = (full[i:i + n] + a[i] * b) % self.q
        return (full[:n] - full[n:]) % self.q

    def _ntt_cyclic(self, a: np.ndarray, root: int) -> np.ndarray:
        """Iterative radix-2 cyclic NTT of length n using the given primitive n-th root."""
        q, n = self.q, self.n
        a = a[self._rev].copy()
        length = 2
        while length <= n:
            w_len = pow(root, n // length, q)
            half = length // 2
            tw = np.array([pow(w_len, k, q) for k in range(half)], dtype=np.int64)
            a = a.reshape(-1, length)
            u = a[:, :half].copy()
            v = (a[:, half:] * tw) % q
            a[:, :half] = (u + v) % q
            a[:, half:] = (u - v) % q
            a = a.reshape(-1)
            length *= 2
        return a

    def ntt(self, a) -> np.ndarray:
        """Forward negacyclic NTT (twist by psi^i, then cyclic NTT)."""
        if not self.has_ntt:
            raise ValueError("modulus does not support NTT for this n")
        return self._ntt_cyclic(self.reduce(a) * self.psi_pows % self.q, self.omega)

    def intt(self, a_hat) -> np.ndarray:
        inv_omega = pow(self.omega, self.q - 2, self.q)
        return self._ntt_cyclic(self.reduce(a_hat), inv_omega) * self.ipsi_pows % self.q

    def mul(self, a, b) -> np.ndarray:
        """Product in R_q, via the NTT when available."""
        if not self.has_ntt:
            return self.mul_schoolbook(a, b)
        return self.intt(self.ntt(a) * self.ntt(b) % self.q)
