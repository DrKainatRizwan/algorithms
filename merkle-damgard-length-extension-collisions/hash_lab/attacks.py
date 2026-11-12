"""Attacks on Merkle-Damgard hashes.

* Length extension: forge ``H(secret || msg || pad || ext)`` from
  ``H(secret || msg)`` without knowing ``secret``.
* Secret-length discovery by querying a verification oracle.
* Generic collision search on n-bit truncated digests: birthday table and
  Floyd cycle-finding (constant memory, rho method).
"""
from __future__ import annotations

from typing import Callable, Iterator, Optional, Type

from .mdcore import MDHash, md_padding

__all__ = ["length_extend", "recover_secret_length", "truncated_hash",
           "birthday_collision", "rho_collision", "expected_birthday_queries"]


def length_extend(hash_cls: Type[MDHash], digest: bytes, secret_len: int, msg: bytes,
                  extension: bytes) -> tuple[bytes, bytes]:
    """Forge a MAC for ``msg || glue || extension``.

    Returns ``(forged_message, forged_digest)`` such that
    ``H(secret || forged_message) == forged_digest`` for the unknown secret of
    length ``secret_len``.  Only works for full-width digests.
    """
    if len(digest) != hash_cls.digest_size:
        raise ValueError("digest has wrong length for this hash")
    glue = md_padding(secret_len + len(msg), hash_cls.block_size,
                      hash_cls.length_size, hash_cls.byteorder)
    state = hash_cls.unpack_state(digest)
    consumed = secret_len + len(msg) + len(glue)
    forged = hash_cls(extension, state=state, count=consumed).digest()
    return msg + glue + extension, forged


def recover_secret_length(hash_cls: Type[MDHash], digest: bytes, msg: bytes,
                          extension: bytes, oracle: Callable[[bytes, bytes], bool],
                          max_len: int = 128) -> Optional[tuple[int, bytes, bytes]]:
    """Try every plausible secret length against a verification ``oracle``.

    ``oracle(message, tag)`` returns True when the server accepts the pair.
    Returns ``(secret_len, forged_message, forged_digest)`` or ``None``.
    """
    for n in range(max_len + 1):
        forged_msg, forged_tag = length_extend(hash_cls, digest, n, msg, extension)
        if oracle(forged_msg, forged_tag):
            return n, forged_msg, forged_tag
    return None


def truncated_hash(hash_cls: Type[MDHash], bits: int) -> Callable[[bytes], bytes]:
    """Return a function mapping bytes to the leading ``bits`` bits of H.

    The output is a ``ceil(bits/8)``-byte string with unused low bits zeroed.
    """
    if not 1 <= bits <= 8 * hash_cls.digest_size:
        raise ValueError("bits out of range")
    nbytes = (bits + 7) // 8
    drop = 8 * nbytes - bits

    def f(data: bytes) -> bytes:
        top = int.from_bytes(hash_cls.hash(data)[:nbytes], "big")
        return ((top >> drop) << drop).to_bytes(nbytes, "big")

    return f


def _counter_messages() -> Iterator[bytes]:
    i = 0
    while True:
        yield i.to_bytes(8, "big")
        i += 1


def birthday_collision(f: Callable[[bytes], bytes]) -> tuple[bytes, bytes, int]:
    """Table-based birthday attack.  Returns ``(m1, m2, evaluations)``."""
    seen: dict[bytes, bytes] = {}
    for n, m in enumerate(_counter_messages(), 1):
        h = f(m)
        if h in seen:
            return seen[h], m, n
        seen[h] = m
    raise AssertionError("unreachable")


def rho_collision(f: Callable[[bytes], bytes], seed: int = 0,
                  max_restarts: int = 64) -> tuple[bytes, bytes, int]:
    """Floyd/Pollard-rho collision finder with O(1) memory.

    Iterates ``x -> f(x)`` (digest fed back as message).  A collision is two
    distinct points with equal images; Floyd's algorithm locates the cycle
    entry and returns the two predecessors.  Restarts from a new start point if
    the start itself lies on the cycle.  Returns ``(m1, m2, evaluations)``.
    """
    evals = 0

    def step(x: bytes) -> bytes:
        nonlocal evals
        evals += 1
        return f(x)

    for r in range(max_restarts):
        x0 = (seed * 1_000_003 + r).to_bytes(8, "big")
        tort, hare = step(x0), step(step(x0))
        while tort != hare:
            tort, hare = step(tort), step(step(hare))
        # find cycle entry (mu) with both pointers advancing at unit speed
        tort = x0
        if tort == hare:
            continue  # start is on the cycle: no distinct predecessors
        prev_t = prev_h = b""
        while tort != hare:
            prev_t, prev_h = tort, hare
            tort, hare = step(tort), step(hare)
        if prev_t != prev_h:
            return prev_t, prev_h, evals
    raise RuntimeError("no collision found")


def expected_birthday_queries(bits: int) -> float:
    """Approximate expected evaluations before a collision: sqrt(pi/2 * 2^bits)."""
    return (3.141592653589793 / 2 * 2.0**bits) ** 0.5
