"""Proof-of-work primitives: compact targets, work, mining and difficulty retargeting."""
from __future__ import annotations

from .block import Block, BlockHeader
from .merkle import sha256d

MAX_TARGET = (1 << 256) - 1


def target_to_bits(target: int) -> int:
    """Encode a 256-bit target in Bitcoin's 4-byte compact format."""
    if target <= 0:
        return 0
    size = (target.bit_length() + 7) // 8
    mant = target << (8 * (3 - size)) if size <= 3 else target >> (8 * (size - 3))
    if mant & 0x800000:
        mant >>= 8
        size += 1
    return (size << 24) | mant


def bits_to_target(bits: int) -> int:
    size = bits >> 24
    mant = bits & 0x7FFFFF
    if size <= 3:
        return mant >> (8 * (3 - size))
    return mant << (8 * (size - 3))


def work_from_target(target: int) -> int:
    """Expected number of hashes to find a hash <= target."""
    return (1 << 256) // (target + 1)


def difficulty(target: int, limit: int) -> float:
    """Difficulty relative to the easiest allowed target ``limit``."""
    return limit / target


def meets_target(header: BlockHeader, target: int | None = None) -> bool:
    t = bits_to_target(header.bits) if target is None else target
    return header.hash_int <= t


def mine(header: BlockHeader, start_nonce: int = 0, max_tries: int = 1 << 32
         ) -> tuple[BlockHeader, int] | None:
    """Grind nonces until the header hash meets its own ``bits`` target.

    Returns ``(solved_header, hashes_tried)`` or ``None`` if the range is exhausted.
    """
    target = bits_to_target(header.bits)
    prefix = header.serialize()[:-4]
    span = min(max_tries, (1 << 32) - start_nonce)
    for i in range(span):
        n = start_nonce + i
        if int.from_bytes(sha256d(prefix + n.to_bytes(4, "big")), "big") <= target:
            return header.with_nonce(n), i + 1
    return None


def mine_block(block: Block, start_nonce: int = 0) -> tuple[Block, int] | None:
    res = mine(block.header, start_nonce)
    if res is None:
        return None
    header, tries = res
    return Block(header, block.txs), tries


def retarget(old_target: int, actual_timespan: int, expected_timespan: int,
             limit: int, clamp: int = 4) -> int:
    """Bitcoin-style retarget: scale by observed/expected time, clamped to ``clamp``x."""
    actual = max(expected_timespan // clamp, min(expected_timespan * clamp, actual_timespan))
    new = old_target * actual // expected_timespan
    return max(1, min(new, limit))


def expected_attempts(target: int) -> float:
    return float(work_from_target(target))
