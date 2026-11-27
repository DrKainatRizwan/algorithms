"""Private information retrieval schemes.

Information-theoretic (multi-server, non-colluding up to a threshold):

* :class:`TrivialPIR` - download everything.
* :class:`XORPIR` - Chor-Goldreich-Kushilevitz-Sudan ``k``-server scheme,
  private against any ``k-1`` colluding servers, ``O(n)`` upload.
* :class:`CubePIR` - ``2^d`` servers on a ``d``-dimensional cube,
  ``O(d * n^(1/d))`` upload.

Computational (single server):

* :class:`PaillierPIR` - Kushilevitz-Ostrovsky style with encrypted selection
  vectors; ``rows=1`` is the linear scheme, ``rows~sqrt(n)`` the square-root scheme.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from .database import Database
from .paillier import PrivateKey, keygen


@dataclass
class PIRResult:
    """Retrieved record and the communication / work it cost."""

    record: bytes
    upload_bits: int
    download_bits: int
    server_ops: int


def _mask_to_bool(mask: int, n: int) -> np.ndarray:
    return np.array([(mask >> j) & 1 for j in range(n)], dtype=bool)


def _xor_select(data: np.ndarray, sel: np.ndarray) -> np.ndarray:
    rows = data[sel]
    if rows.shape[0] == 0:
        return np.zeros(data.shape[1], dtype=np.uint8)
    return np.bitwise_xor.reduce(rows, axis=0)


class TrivialPIR:
    """Baseline: the server sends the whole database."""

    def retrieve(self, db: Database, i: int, rng: random.Random | None = None) -> PIRResult:
        rec = db.record(i)
        return PIRResult(rec, 0, 8 * db.n * db.record_len, 0)


class XORPIR:
    """``k``-server XOR PIR (``k >= 2``), private against ``k-1`` colluders."""

    def __init__(self, servers: int = 2) -> None:
        if servers < 2:
            raise ValueError("need at least two servers")
        self.k = servers

    def coin_ranges(self, n: int) -> list[int]:
        """Size of each random-coin variable (``k-1`` masks of ``n`` bits)."""
        return [1 << n] * (self.k - 1)

    def queries_from_coins(self, n: int, i: int, coins: list[int]) -> list[tuple]:
        """Deterministic query construction from explicit coins (used by audits)."""
        last = 1 << i
        for c in coins:
            last ^= c
        return [(c,) for c in coins] + [(last,)]

    def query(self, n: int, i: int, rng: random.Random) -> list[tuple]:
        """Random queries, one tuple per server."""
        return self.queries_from_coins(n, i, [rng.getrandbits(n) for _ in range(self.k - 1)])

    def answer(self, db: Database, q: tuple) -> np.ndarray:
        """Server side: XOR of the selected records."""
        return _xor_select(db.data, _mask_to_bool(q[0], db.n))

    def reconstruct(self, answers: list[np.ndarray]) -> bytes:
        return np.bitwise_xor.reduce(np.stack(answers), axis=0).tobytes()

    def retrieve(self, db: Database, i: int, rng: random.Random) -> PIRResult:
        qs = self.query(db.n, i, rng)
        ans = [self.answer(db, q) for q in qs]
        return PIRResult(self.reconstruct(ans), self.k * db.n,
                         self.k * 8 * db.record_len, self.k * db.n // 2)


class CubePIR:
    """``2^d``-server PIR over a ``d``-dimensional cube of side ``m``."""

    def __init__(self, dim: int = 2) -> None:
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self.d = dim

    @property
    def servers(self) -> int:
        return 2 ** self.d

    def side(self, n: int) -> int:
        """Smallest ``m`` with ``m^d >= n``."""
        m = max(1, round(n ** (1 / self.d)))
        while m ** self.d < n:
            m += 1
        while m > 1 and (m - 1) ** self.d >= n:
            m -= 1
        return m

    def coords(self, n: int, i: int) -> list[int]:
        """Base-``m`` digits of index ``i``."""
        m = self.side(n)
        out = []
        for _ in range(self.d):
            out.append(i % m)
            i //= m
        return out

    def coin_ranges(self, n: int) -> list[int]:
        return [1 << self.side(n)] * self.d

    def queries_from_coins(self, n: int, i: int, coins: list[int]) -> list[tuple]:
        c = self.coords(n, i)
        out = []
        for s in range(self.servers):
            out.append(tuple(coins[j] ^ ((1 << c[j]) if (s >> j) & 1 else 0) for j in range(self.d)))
        return out

    def query(self, n: int, i: int, rng: random.Random) -> list[tuple]:
        m = self.side(n)
        return self.queries_from_coins(n, i, [rng.getrandbits(m) for _ in range(self.d)])

    def answer(self, db: Database, q: tuple) -> np.ndarray:
        """XOR of all records whose coordinates lie in the product of the subsets."""
        m = self.side(db.n)
        cube = db.padded(m ** self.d).reshape((m,) * self.d + (db.record_len,))
        # C-order reshape: axis 0 is the most significant digit, so reverse the digit order
        sel = [np.flatnonzero(_mask_to_bool(q[j], m)) for j in range(self.d)][::-1]
        sub = cube[np.ix_(*sel)]
        return np.bitwise_xor.reduce(sub, axis=tuple(range(self.d)))

    def retrieve(self, db: Database, i: int, rng: random.Random) -> PIRResult:
        m = self.side(db.n)
        qs = self.query(db.n, i, rng)
        ans = [self.answer(db, q) for q in qs]
        rec = np.bitwise_xor.reduce(np.stack(ans), axis=0).tobytes()
        return PIRResult(rec, self.servers * self.d * m, self.servers * 8 * db.record_len,
                         self.servers * m ** self.d // 2 ** self.d)


class PaillierPIR:
    """Single-server computational PIR from additively homomorphic encryption.

    Records are laid out in a ``rows x cols`` matrix.  The client uploads an
    encrypted unit vector selecting the column; the server returns, for every
    row, the homomorphic dot product of its record chunks with that vector.
    The client decrypts only the target row.
    """

    def __init__(self, key_bits: int = 256, seed: int = 0, rows: int | None = 0) -> None:
        self.rng = random.Random(seed)
        self.sk: PrivateKey = keygen(key_bits, self.rng)
        self.pk = self.sk.public
        self.rows = rows  # 0 -> sqrt layout, 1 -> linear layout

    @property
    def chunk_bytes(self) -> int:
        return (self.pk.n.bit_length() - 1) // 8

    def layout(self, n: int) -> tuple[int, int]:
        """Return ``(rows, cols)`` for ``n`` records."""
        r = max(1, math.isqrt(n - 1) + 1 if n > 1 else 1) if not self.rows else self.rows
        r = min(r, n)
        return r, math.ceil(n / r)

    def _chunks(self, db: Database) -> np.ndarray:
        """Records as ints: shape ``(n, T)`` object array of chunk values."""
        cb = self.chunk_bytes
        T = math.ceil(db.record_len / cb)
        out = np.empty((db.n, T), dtype=object)
        for i in range(db.n):
            raw = db.record(i).ljust(T * cb, b"\x00")
            for t in range(T):
                out[i, t] = int.from_bytes(raw[t * cb:(t + 1) * cb], "big")
        return out

    def query(self, n: int, i: int) -> list[int]:
        """Encrypted selection vector over the columns."""
        r, c = self.layout(n)
        col = i % c
        return [self.pk.encrypt(1 if j == col else 0, self.rng) for j in range(c)]

    def answer(self, db: Database, query: list[int]) -> list[list[int]]:
        """Server: per-row, per-chunk encrypted dot products."""
        r, c = self.layout(db.n)
        if len(query) != c:
            raise ValueError("query length does not match layout")
        chunks = self._chunks(db)
        T = chunks.shape[1]
        n2 = self.pk.n2
        out = []
        for row in range(r):
            acc = [1] * T
            for col in range(c):
                idx = row * c + col
                if idx >= db.n:
                    break
                for t in range(T):
                    v = chunks[idx, t]
                    if v:
                        acc[t] = acc[t] * pow(query[col], v, n2) % n2
            out.append(acc)
        return out

    def decode(self, db_len: int, record_len: int, i: int, resp: list[list[int]]) -> bytes:
        """Client: decrypt row ``i // cols``."""
        r, c = self.layout(db_len)
        cb = self.chunk_bytes
        raw = b"".join(self.sk.decrypt(ct).to_bytes(cb, "big") for ct in resp[i // c])
        return raw[:record_len]

    def retrieve(self, db: Database, i: int, rng: random.Random | None = None) -> PIRResult:
        if not 0 <= i < db.n:
            raise IndexError(i)
        q = self.query(db.n, i)
        resp = self.answer(db, q)
        rec = self.decode(db.n, db.record_len, i, resp)
        ct_bits = 2 * self.pk.n.bit_length()
        return PIRResult(rec, len(q) * ct_bits, sum(len(r) for r in resp) * ct_bits,
                         db.n * len(resp[0]))
