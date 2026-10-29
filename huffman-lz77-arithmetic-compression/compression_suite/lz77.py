"""LZ77 sliding-window compression with hash-chain match finding.

Tokens are either literals (an int 0-255) or ``(distance, length)`` matches.
The serialised form entropy-codes the literal/length alphabet with Huffman
(a simplified DEFLATE layout).
"""
from __future__ import annotations

from collections import Counter

from .bitio import BitReader, BitWriter
from . import huffman

MIN_MATCH = 3
MAX_MATCH = 258
LEN_SYMBOLS = MAX_MATCH - MIN_MATCH + 1
ALPHABET = 256 + LEN_SYMBOLS

Token = int | tuple[int, int]


def find_tokens(data: bytes, window: int = 4096, max_chain: int = 32,
                lazy: bool = True) -> list[Token]:
    """Parse ``data`` into literals and matches using hash chains over 3-byte keys.

    ``max_chain`` bounds the candidates examined per position; ``lazy`` enables
    one-step lazy matching (defer a match if the next position matches longer).
    """
    n = len(data)
    head: dict[bytes, list[int]] = {}
    tokens: list[Token] = []

    def best(pos: int) -> tuple[int, int]:
        if pos + MIN_MATCH > n:
            return 0, 0
        cands = head.get(data[pos:pos + MIN_MATCH], ())
        bl = bd = 0
        limit = min(MAX_MATCH, n - pos)
        for cand in reversed(cands[-max_chain:]):
            d = pos - cand
            if d > window:
                break
            l = MIN_MATCH
            while l < limit and data[cand + l] == data[pos + l]:
                l += 1
            if l > bl:
                bl, bd = l, d
                if l == limit:
                    break
        return bl, bd

    def insert(pos: int) -> None:
        if pos + MIN_MATCH <= n:
            head.setdefault(data[pos:pos + MIN_MATCH], []).append(pos)

    i = 0
    while i < n:
        l, d = best(i)
        if l >= MIN_MATCH and lazy and i + 1 < n:
            insert(i)
            l2, _ = best(i + 1)
            if l2 > l:
                tokens.append(data[i])
                i += 1
                continue
            tokens.append((d, l))
            for k in range(i + 1, i + l):
                insert(k)
            i += l
            continue
        if l >= MIN_MATCH:
            tokens.append((d, l))
            for k in range(i, i + l):
                insert(k)
            i += l
        else:
            tokens.append(data[i])
            insert(i)
            i += 1
    return tokens


def expand(tokens: list[Token]) -> bytes:
    """Reconstruct bytes from tokens (handles overlapping copies)."""
    out = bytearray()
    for t in tokens:
        if isinstance(t, int):
            out.append(t)
        else:
            d, l = t
            if d <= 0 or d > len(out):
                raise ValueError("match distance out of range")
            start = len(out) - d
            for k in range(l):
                out.append(out[start + k])
    return bytes(out)


def _sym(t: Token) -> int:
    return t if isinstance(t, int) else 256 + t[1] - MIN_MATCH


def compress(data: bytes, window: int = 4096, max_chain: int = 32, lazy: bool = True) -> bytes:
    """LZ77 + Huffman. Distances use a 5-bit width prefix then the low bits."""
    tokens = find_tokens(data, window, max_chain, lazy)
    w = BitWriter()
    w.write_bits(len(tokens), 32)
    if not tokens:
        return w.getvalue()
    lengths = huffman.limit_lengths(huffman.code_lengths(dict(Counter(_sym(t) for t in tokens))), 31)
    huffman.write_lengths(w, lengths, ALPHABET)
    codes = huffman.canonical_codes(lengths)
    for t in tokens:
        c, l = codes[_sym(t)]
        w.write_bits(c, l)
        if not isinstance(t, int):
            d = t[0]
            nb = d.bit_length()
            w.write_bits(nb, 5)
            w.write_bits(d & ((1 << (nb - 1)) - 1), nb - 1)
    return w.getvalue()


def decompress(blob: bytes) -> bytes:
    r = BitReader(blob)
    count = r.read_bits(32)
    if count == 0:
        return b""
    dec = huffman.HuffmanDecoder(huffman.read_lengths(r, ALPHABET))
    tokens: list[Token] = []
    for _ in range(count):
        s = dec.decode(r)
        if s < 256:
            tokens.append(s)
        else:
            nb = r.read_bits(5)
            d = (1 << (nb - 1)) | r.read_bits(nb - 1)
            tokens.append((d, s - 256 + MIN_MATCH))
    return expand(tokens)


def token_stats(tokens: list[Token]) -> dict[str, float]:
    """Summary of a parse: literal count, match count, mean match length."""
    m = [t for t in tokens if not isinstance(t, int)]
    return {
        "literals": len(tokens) - len(m),
        "matches": len(m),
        "mean_match_len": sum(t[1] for t in m) / len(m) if m else 0.0,
    }
