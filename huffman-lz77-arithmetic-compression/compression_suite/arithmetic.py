"""Adaptive arithmetic coding with 32-bit integer range arithmetic.

Implements the carry-free Witten-Neal-Cleary coder with pending-bit
underflow handling, an adaptive order-0 model backed by a Fenwick tree and
an optional order-1 context model.
"""
from __future__ import annotations

from .bitio import BitReader, BitWriter

PRECISION = 32
TOP = (1 << PRECISION) - 1
HALF = 1 << (PRECISION - 1)
QUARTER = 1 << (PRECISION - 2)
EOF_SYMBOL = 256
NSYMS = 257
MAX_TOTAL = 1 << 16


class FenwickModel:
    """Adaptive frequency table with O(log n) cumulative queries and updates."""

    def __init__(self, nsyms: int = NSYMS, increment: int = 32) -> None:
        self.n = nsyms
        self.inc = increment
        self.freq = [1] * nsyms
        self._rebuild()

    def _rebuild(self) -> None:
        self.tree = [0] * (self.n + 1)
        for i, f in enumerate(self.freq):
            j = i + 1
            while j <= self.n:
                self.tree[j] += f
                j += j & -j
        self.total = sum(self.freq)
        self._top = 1 << (self.n.bit_length() - 1)

    def cum(self, sym: int) -> int:
        """Sum of frequencies of symbols strictly below ``sym``."""
        s = 0
        j = sym
        while j > 0:
            s += self.tree[j]
            j -= j & -j
        return s

    def find(self, target: int) -> int:
        """Largest symbol whose cumulative frequency is <= ``target``."""
        pos = 0
        step = self._top
        while step:
            nxt = pos + step
            if nxt <= self.n and self.tree[nxt] <= target:
                pos = nxt
                target -= self.tree[nxt]
            step >>= 1
        return pos

    def update(self, sym: int) -> None:
        self.freq[sym] += self.inc
        j = sym + 1
        while j <= self.n:
            self.tree[j] += self.inc
            j += j & -j
        self.total += self.inc
        if self.total >= MAX_TOTAL:
            self.freq = [max(1, f // 2) for f in self.freq]
            self._rebuild()


class ArithmeticEncoder:
    def __init__(self, writer: BitWriter) -> None:
        self.w = writer
        self.low = 0
        self.high = TOP
        self.pending = 0

    def _emit(self, bit: int) -> None:
        self.w.write_bit(bit)
        for _ in range(self.pending):
            self.w.write_bit(1 - bit)
        self.pending = 0

    def encode(self, cum_lo: int, cum_hi: int, total: int) -> None:
        rng = self.high - self.low + 1
        self.high = self.low + rng * cum_hi // total - 1
        self.low = self.low + rng * cum_lo // total
        while True:
            if self.high < HALF:
                self._emit(0)
            elif self.low >= HALF:
                self._emit(1)
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QUARTER and self.high < HALF + QUARTER:
                self.pending += 1
                self.low -= QUARTER
                self.high -= QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1

    def finish(self) -> None:
        self.pending += 1
        self._emit(0 if self.low < QUARTER else 1)


class ArithmeticDecoder:
    def __init__(self, reader: BitReader) -> None:
        self.r = reader
        self.low = 0
        self.high = TOP
        self.value = 0
        for _ in range(PRECISION):
            self.value = (self.value << 1) | self.r.read_bit()

    def target(self, total: int) -> int:
        rng = self.high - self.low + 1
        return ((self.value - self.low + 1) * total - 1) // rng

    def consume(self, cum_lo: int, cum_hi: int, total: int) -> None:
        rng = self.high - self.low + 1
        self.high = self.low + rng * cum_hi // total - 1
        self.low = self.low + rng * cum_lo // total
        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.low -= HALF
                self.high -= HALF
                self.value -= HALF
            elif self.low >= QUARTER and self.high < HALF + QUARTER:
                self.low -= QUARTER
                self.high -= QUARTER
                self.value -= QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1
            self.value = (self.value << 1) | self.r.read_bit()


def compress(data: bytes, order: int = 0) -> bytes:
    """Adaptive arithmetic compression; ``order`` is 0 or 1 (byte context)."""
    if order not in (0, 1):
        raise ValueError("order must be 0 or 1")
    w = BitWriter()
    enc = ArithmeticEncoder(w)
    models: dict[int, FenwickModel] = {}
    ctx = 0
    for sym in list(data) + [EOF_SYMBOL]:
        m = models.get(ctx)
        if m is None:
            m = models[ctx] = FenwickModel()
        lo = m.cum(sym)
        enc.encode(lo, lo + m.freq[sym], m.total)
        m.update(sym)
        if order and sym != EOF_SYMBOL:
            ctx = sym
    enc.finish()
    return bytes([order]) + w.getvalue()


def decompress(blob: bytes) -> bytes:
    order = blob[0]
    dec = ArithmeticDecoder(BitReader(blob[1:], pad_zeros=True))
    models: dict[int, FenwickModel] = {}
    out = bytearray()
    ctx = 0
    while True:
        m = models.get(ctx)
        if m is None:
            m = models[ctx] = FenwickModel()
        sym = m.find(dec.target(m.total))
        lo = m.cum(sym)
        dec.consume(lo, lo + m.freq[sym], m.total)
        if sym == EOF_SYMBOL:
            return bytes(out)
        m.update(sym)
        out.append(sym)
        if order:
            ctx = sym


def ideal_bits(data: bytes, probs: dict[int, float]) -> float:
    """Shannon information content of ``data`` under a fixed model (bits)."""
    import math
    return -sum(math.log2(probs[b]) for b in data)
