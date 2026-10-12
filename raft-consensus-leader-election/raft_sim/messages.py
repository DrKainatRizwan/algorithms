"""Raft RPC message types and log entries."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Tuple


@dataclass(frozen=True)
class Entry:
    """A replicated log entry. ``command`` is ``None`` for leader no-ops."""

    term: int
    command: Any = None


@dataclass(frozen=True)
class Message:
    """Base RPC message: every message carries the sender and its term."""

    src: int
    term: int


@dataclass(frozen=True)
class RequestVote(Message):
    last_log_index: int = 0
    last_log_term: int = 0


@dataclass(frozen=True)
class RequestVoteReply(Message):
    granted: bool = False


@dataclass(frozen=True)
class AppendEntries(Message):
    prev_index: int = 0
    prev_term: int = 0
    entries: Tuple[Entry, ...] = field(default_factory=tuple)
    leader_commit: int = 0


@dataclass(frozen=True)
class AppendEntriesReply(Message):
    success: bool = False
    match_index: int = 0
    conflict_index: int = 0
