"""Core data types: packets, rules and parsing helpers.

Addresses are IPv4 values stored as 32-bit integers. A rule matches a packet
when every one of the five header fields matches; among matching rules the one
with the smallest ``priority`` number wins (first-match semantics).
"""
from __future__ import annotations

from dataclasses import dataclass

MAX_IP = (1 << 32) - 1
ANY_PROTO = -1


@dataclass(frozen=True)
class Packet:
    """A five-tuple packet header."""
    src: int
    dst: int
    sport: int
    dport: int
    proto: int

    def fields(self) -> tuple[int, int, int, int, int]:
        return (self.src, self.dst, self.sport, self.dport, self.proto)


def parse_ip(text: str) -> int:
    """Convert dotted-quad text to a 32-bit integer."""
    parts = text.strip().split(".")
    if len(parts) != 4:
        raise ValueError(f"bad IPv4 address: {text!r}")
    value = 0
    for p in parts:
        octet = int(p)
        if not 0 <= octet <= 255:
            raise ValueError(f"bad octet in {text!r}")
        value = (value << 8) | octet
    return value


def format_ip(value: int) -> str:
    """Convert a 32-bit integer to dotted-quad text."""
    return ".".join(str((value >> s) & 255) for s in (24, 16, 8, 0))


def prefix_mask(length: int) -> int:
    """Network mask with ``length`` leading one bits."""
    if not 0 <= length <= 32:
        raise ValueError("prefix length must be in [0, 32]")
    return (MAX_IP << (32 - length)) & MAX_IP


def parse_prefix(text: str) -> tuple[int, int]:
    """Parse ``a.b.c.d/len`` (or ``*``) into ``(masked_value, length)``."""
    text = text.strip()
    if text in ("*", "any"):
        return 0, 0
    if "/" in text:
        addr, ln = text.split("/")
        length = int(ln)
    else:
        addr, length = text, 32
    return parse_ip(addr) & prefix_mask(length), length


@dataclass(frozen=True)
class Rule:
    """A firewall rule over the five-tuple.

    ``src``/``dst`` are ``(value, prefix_length)`` pairs, ports are inclusive
    ranges and ``proto`` is an exact protocol number or ``ANY_PROTO``.
    """
    src: tuple[int, int]
    dst: tuple[int, int]
    sport: tuple[int, int]
    dport: tuple[int, int]
    proto: int
    action: str
    priority: int

    def __post_init__(self) -> None:
        for v, ln in (self.src, self.dst):
            if v & ~prefix_mask(ln) & MAX_IP:
                raise ValueError("prefix has host bits set")
        for lo, hi in (self.sport, self.dport):
            if not 0 <= lo <= hi <= 65535:
                raise ValueError("invalid port range")

    def ranges(self) -> tuple[tuple[int, int], ...]:
        """Inclusive ``(lo, hi)`` range per dimension (src, dst, sport, dport, proto)."""
        def pr(p: tuple[int, int]) -> tuple[int, int]:
            return p[0], p[0] | (~prefix_mask(p[1]) & MAX_IP)
        proto = (0, 255) if self.proto == ANY_PROTO else (self.proto, self.proto)
        return (pr(self.src), pr(self.dst), self.sport, self.dport, proto)

    def matches(self, p: Packet) -> bool:
        """True if the packet satisfies all five fields."""
        for (lo, hi), x in zip(self.ranges(), p.fields()):
            if x < lo or x > hi:
                return False
        return True

    def covers(self, other: "Rule") -> bool:
        """True if every packet matching ``other`` also matches ``self``."""
        return all(a_lo <= b_lo and b_hi <= a_hi
                   for (a_lo, a_hi), (b_lo, b_hi) in zip(self.ranges(), other.ranges()))

    def overlaps(self, other: "Rule") -> bool:
        """True if some packet matches both rules."""
        return all(max(a_lo, b_lo) <= min(a_hi, b_hi)
                   for (a_lo, a_hi), (b_lo, b_hi) in zip(self.ranges(), other.ranges()))

    def match_rest(self, p: Packet) -> bool:
        """Check only ports and protocol (addresses are checked by the trie)."""
        return (self.sport[0] <= p.sport <= self.sport[1]
                and self.dport[0] <= p.dport <= self.dport[1]
                and (self.proto == ANY_PROTO or self.proto == p.proto))


_PROTOS = {"tcp": 6, "udp": 17, "icmp": 1, "any": ANY_PROTO, "*": ANY_PROTO}


def parse_rule(text: str, priority: int) -> Rule:
    """Parse ``"allow tcp 10.0.0.0/8 * 192.168.1.0/24 80-443"`` style rules.

    Format: ``action proto src sport-or-* dst dport-or-*``; a bare ``*`` for the
    address means any. Ports are ``N``, ``lo-hi`` or ``*``.
    """
    tok = text.split()
    if len(tok) != 6:
        raise ValueError("rule needs: action proto src sport dst dport")
    action, proto, src, sport, dst, dport = tok

    def port(s: str) -> tuple[int, int]:
        if s == "*":
            return 0, 65535
        if "-" in s:
            lo, hi = s.split("-")
            return int(lo), int(hi)
        return int(s), int(s)

    pv = _PROTOS[proto.lower()] if proto.lower() in _PROTOS else int(proto)
    return Rule(parse_prefix(src), parse_prefix(dst), port(sport), port(dport),
                pv, action, priority)
