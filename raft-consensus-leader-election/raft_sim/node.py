"""A single Raft server: election, replication and a key-value state machine."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from .messages import (AppendEntries, AppendEntriesReply, Entry, Message,
                       RequestVote, RequestVoteReply)

FOLLOWER, CANDIDATE, LEADER = "follower", "candidate", "leader"


@dataclass(frozen=True)
class RaftConfig:
    """Timing parameters, in simulated milliseconds."""

    election_min: int = 150
    election_max: int = 300
    heartbeat: int = 50
    max_batch: int = 16


def apply_command(state: Dict[str, Any], command: Any) -> None:
    """Apply ``("set", k, v)`` or ``("del", k)`` to a dict; ``None`` is a no-op."""
    if command is None:
        return
    if command[0] == "set":
        state[command[1]] = command[2]
    elif command[0] == "del":
        state.pop(command[1], None)
    else:
        raise ValueError(f"unknown command {command!r}")


class RaftNode:
    """Raft server following Ongaro & Ousterhout (2014), Figure 2.

    ``current_term``, ``voted_for`` and ``log`` are persistent and survive
    :meth:`restart`; everything else is volatile. Log indices are 1-based.
    """

    def __init__(self, node_id: int, peers: List[int], config: RaftConfig, seed: int = 0):
        self.id = node_id
        self.peers = [p for p in peers if p != node_id]
        self.cluster_size = len(self.peers) + 1
        self.config = config
        self.rng = random.Random(seed * 7919 + node_id)
        # persistent state
        self.current_term = 0
        self.voted_for: Optional[int] = None
        self.log: List[Entry] = []
        self.outbox: List[Tuple[int, Message]] = []
        self._reset_volatile(0)

    # ------------------------------------------------------------------ state
    def _reset_volatile(self, now: int) -> None:
        self.role = FOLLOWER
        self.leader_id: Optional[int] = None
        self.commit_index = 0
        self.last_applied = 0
        self.state: Dict[str, Any] = {}
        self.applied: List[Tuple[int, Any]] = []
        self.votes: Set[int] = set()
        self.next_index: Dict[int, int] = {}
        self.match_index: Dict[int, int] = {}
        self.next_heartbeat = 0
        self._reset_election_timer(now)

    def restart(self, now: int) -> None:
        """Simulate a crash/recovery: keep persistent state, lose the rest."""
        self.outbox.clear()
        self._reset_volatile(now)

    @property
    def last_index(self) -> int:
        return len(self.log)

    def term_at(self, index: int) -> int:
        return 0 if index == 0 else self.log[index - 1].term

    @property
    def majority(self) -> int:
        return self.cluster_size // 2 + 1

    def _reset_election_timer(self, now: int) -> None:
        c = self.config
        self.election_deadline = now + self.rng.randint(c.election_min, c.election_max)

    def _send(self, dst: int, msg: Message) -> None:
        self.outbox.append((dst, msg))

    # ----------------------------------------------------------------- timers
    def tick(self, now: int) -> None:
        if self.role == LEADER:
            if now >= self.next_heartbeat:
                self._broadcast_append(now)
        elif now >= self.election_deadline:
            self._start_election(now)

    def _become_follower(self, term: int, now: int) -> None:
        if term > self.current_term:
            self.current_term = term
            self.voted_for = None
        self.role = FOLLOWER
        self.votes = set()

    def _start_election(self, now: int) -> None:
        self.current_term += 1
        self.role = CANDIDATE
        self.voted_for = self.id
        self.votes = {self.id}
        self.leader_id = None
        self._reset_election_timer(now)
        if len(self.votes) >= self.majority:
            self._become_leader(now)
            return
        for p in self.peers:
            self._send(p, RequestVote(self.id, self.current_term,
                                      self.last_index, self.term_at(self.last_index)))

    def _become_leader(self, now: int) -> None:
        self.role = LEADER
        self.leader_id = self.id
        self.next_index = {p: self.last_index + 1 for p in self.peers}
        self.match_index = {p: 0 for p in self.peers}
        self.log.append(Entry(self.current_term, None))  # no-op commits prior terms
        self._advance_commit()
        self._broadcast_append(now)

    # ------------------------------------------------------------ client API
    def propose(self, command: Any, now: int) -> Optional[int]:
        """Append ``command`` if leader; return its log index, else ``None``."""
        if self.role != LEADER:
            return None
        self.log.append(Entry(self.current_term, command))
        self._advance_commit()
        self._broadcast_append(now)
        return self.last_index

    # ------------------------------------------------------------ replication
    def _broadcast_append(self, now: int) -> None:
        self.next_heartbeat = now + self.config.heartbeat
        for p in self.peers:
            self._send_append(p)

    def _send_append(self, peer: int) -> None:
        prev = self.next_index[peer] - 1
        entries = tuple(self.log[prev:prev + self.config.max_batch])
        self._send(peer, AppendEntries(self.id, self.current_term, prev,
                                       self.term_at(prev), entries, self.commit_index))

    def _advance_commit(self) -> None:
        """Commit the highest index replicated on a majority in the current term."""
        for n in range(self.last_index, self.commit_index, -1):
            if self.term_at(n) != self.current_term:
                break
            count = 1 + sum(1 for p in self.peers if self.match_index.get(p, 0) >= n)
            if count >= self.majority:
                self.commit_index = n
                break
        self._apply_committed()

    def _apply_committed(self) -> None:
        while self.last_applied < self.commit_index:
            self.last_applied += 1
            cmd = self.log[self.last_applied - 1].command
            apply_command(self.state, cmd)
            self.applied.append((self.last_applied, cmd))

    # ------------------------------------------------------------- RPC intake
    def receive(self, msg: Message, now: int) -> None:
        if msg.term > self.current_term:
            self._become_follower(msg.term, now)
        if isinstance(msg, RequestVote):
            self._on_request_vote(msg, now)
        elif isinstance(msg, RequestVoteReply):
            self._on_vote_reply(msg, now)
        elif isinstance(msg, AppendEntries):
            self._on_append(msg, now)
        elif isinstance(msg, AppendEntriesReply):
            self._on_append_reply(msg, now)

    def _on_request_vote(self, m: RequestVote, now: int) -> None:
        up_to_date = (m.last_log_term, m.last_log_index) >= (
            self.term_at(self.last_index), self.last_index)
        grant = (m.term == self.current_term and up_to_date
                 and self.voted_for in (None, m.src))
        if grant:
            self.voted_for = m.src
            self._reset_election_timer(now)
        self._send(m.src, RequestVoteReply(self.id, self.current_term, grant))

    def _on_vote_reply(self, m: RequestVoteReply, now: int) -> None:
        if self.role != CANDIDATE or m.term != self.current_term or not m.granted:
            return
        self.votes.add(m.src)
        if len(self.votes) >= self.majority:
            self._become_leader(now)

    def _on_append(self, m: AppendEntries, now: int) -> None:
        if m.term < self.current_term:
            self._send(m.src, AppendEntriesReply(self.id, self.current_term, False))
            return
        self.role = FOLLOWER
        self.leader_id = m.src
        self._reset_election_timer(now)
        if m.prev_index > self.last_index:
            hint = self.last_index + 1
        elif self.term_at(m.prev_index) != m.prev_term:
            bad = self.term_at(m.prev_index)
            hint = m.prev_index
            while hint > 1 and self.term_at(hint - 1) == bad:
                hint -= 1
        else:
            hint = 0
        if hint:
            self._send(m.src, AppendEntriesReply(self.id, self.current_term, False,
                                                 conflict_index=hint))
            return
        idx = m.prev_index
        for e in m.entries:
            idx += 1
            if idx <= self.last_index:
                if self.log[idx - 1].term == e.term:
                    continue
                del self.log[idx - 1:]  # conflicting suffix is discarded
            self.log.append(e)
        match = m.prev_index + len(m.entries)
        if m.leader_commit > self.commit_index:
            self.commit_index = max(self.commit_index, min(m.leader_commit, match))
            self._apply_committed()
        self._send(m.src, AppendEntriesReply(self.id, self.current_term, True, match))

    def _on_append_reply(self, m: AppendEntriesReply, now: int) -> None:
        if self.role != LEADER or m.term != self.current_term:
            return
        if m.success:
            if m.match_index > self.match_index[m.src]:
                self.match_index[m.src] = m.match_index
            self.next_index[m.src] = self.match_index[m.src] + 1
            self._advance_commit()
            if self.next_index[m.src] <= self.last_index:
                self._send_append(m.src)
        else:
            self.next_index[m.src] = max(1, min(m.conflict_index, self.next_index[m.src] - 1))
            self._send_append(m.src)
