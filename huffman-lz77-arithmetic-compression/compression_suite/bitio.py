"""Bit-level reader and writer used by every codec in the suite."""
from __future__ import annotations


class BitWriter:
    """Accumulates bits MSB-first into a bytearray."""

    def __init__(self) -> None:
        self._buf = bytearray()
        self._acc = 0
        self._n = 0

    def write_bit(self, bit: int) -> None:
        self._acc = (self._acc << 1) | (bit & 1)
        self._n += 1
        if self._n == 8:
            self._buf.append(self._acc)
            self._acc = 0
            self._n = 0

    def write_bits(self, value: int, count: int) -> None:
        """Write the low ``count`` bits of ``value``, most significant first."""
        if value < 0 or (count < value.bit_length()):
            raise ValueError("value does not fit in the requested bit count")
        for i in range(count - 1, -1, -1):
            self.write_bit((value >> i) & 1)

    @property
    def bit_length(self) -> int:
        return len(self._buf) * 8 + self._n

    def getvalue(self) -> bytes:
        """Return the stream, zero-padding the final partial byte."""
        out = bytearray(self._buf)
        if self._n:
            out.append(self._acc << (8 - self._n))
        return bytes(out)


class BitReader:
    """Reads bits MSB-first from a bytes object."""

    def __init__(self, data: bytes, pad_zeros: bool = False) -> None:
        self._data = data
        self._pos = 0
        self._pad = pad_zeros

    def read_bit(self) -> int:
        byte = self._pos >> 3
        if byte >= len(self._data):
            if self._pad:
                self._pos += 1
                return 0
            raise EOFError("read past end of bit stream")
        bit = (self._data[byte] >> (7 - (self._pos & 7))) & 1
        self._pos += 1
        return bit

    def read_bits(self, count: int) -> int:
        v = 0
        for _ in range(count):
            v = (v << 1) | self.read_bit()
        return v

    @property
    def position(self) -> int:
        return self._pos
