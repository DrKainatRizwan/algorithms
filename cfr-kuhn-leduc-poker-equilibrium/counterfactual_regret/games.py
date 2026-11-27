"""Extensive-form game definitions: Kuhn poker and Leduc hold'em.

Every game exposes the same small interface (see :class:`Game`) so that the
tree compiler in :mod:`counterfactual_regret.tree` can enumerate it.  States
are immutable; ``apply`` returns a new state.  Utilities are always reported
from the point of view of player 0 (the games are zero-sum).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Hashable, Protocol, Sequence

CHANCE = -1
TERMINAL = 2


class Game(Protocol):
    """Minimal interface required by the tree compiler."""

    name: str

    def initial(self) -> Hashable: ...
    def node_kind(self, state: Hashable) -> int:
        """Return 0 or 1 for a decision node, ``CHANCE`` or ``TERMINAL``."""
    def actions(self, state: Hashable) -> Sequence[str]: ...
    def chance_probs(self, state: Hashable) -> Sequence[float]: ...
    def apply(self, state: Hashable, action: str) -> Hashable: ...
    def utility(self, state: Hashable) -> float: ...
    def infoset_key(self, state: Hashable) -> str: ...


# --------------------------------------------------------------------------
# Kuhn poker
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class KuhnState:
    cards: tuple[int, ...] = ()
    history: str = ""


class KuhnPoker:
    """Three-card Kuhn poker (cards 0=J, 1=Q, 2=K, one ante each).

    Actions are ``p`` (pass / check / fold) and ``b`` (bet / call).
    """

    name = "kuhn"
    CARD_NAMES = "JQK"
    _TERMINAL = {"pp", "bp", "bb", "pbp", "pbb"}

    def initial(self) -> KuhnState:
        return KuhnState()

    def node_kind(self, s: KuhnState) -> int:
        if len(s.cards) < 2:
            return CHANCE
        if s.history in self._TERMINAL:
            return TERMINAL
        return len(s.history) % 2

    def actions(self, s: KuhnState) -> Sequence[str]:
        if len(s.cards) < 2:
            return [str(c) for c in range(3) if c not in s.cards]
        return ["p", "b"]

    def chance_probs(self, s: KuhnState) -> Sequence[float]:
        n = 3 - len(s.cards)
        return [1.0 / n] * n

    def apply(self, s: KuhnState, action: str) -> KuhnState:
        if len(s.cards) < 2:
            return KuhnState(s.cards + (int(action),), s.history)
        return KuhnState(s.cards, s.history + action)

    def utility(self, s: KuhnState) -> float:
        h = s.history
        if h == "bp":
            return 1.0
        if h == "pbp":
            return -1.0
        stake = 2.0 if h.endswith("bb") else 1.0
        return stake if s.cards[0] > s.cards[1] else -stake

    def infoset_key(self, s: KuhnState) -> str:
        player = len(s.history) % 2
        return f"{self.CARD_NAMES[s.cards[player]]}:{s.history}"


# --------------------------------------------------------------------------
# Leduc hold'em
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class LeducState:
    priv: tuple[int, ...] = ()
    board: int | None = None
    rounds: tuple[str, ...] = ("",)
    committed: tuple[int, int] = (1, 1)
    player: int = 0
    raises: int = 0
    acted: int = 0
    folded: int = -1
    need_board: bool = False
    over: bool = False


class LeducHoldem:
    """Leduc hold'em with ``ranks`` ranks and two suits.

    One ante each, two betting rounds, raise sizes 2 (pre-board) and 4
    (post-board), at most ``max_raises`` raises per round.  A private pair with
    the board card wins; otherwise the higher rank wins and equal ranks tie.
    """

    name = "leduc"

    def __init__(self, ranks: int = 3, max_raises: int = 2) -> None:
        if ranks < 2:
            raise ValueError("need at least two ranks")
        self.ranks = ranks
        self.max_raises = max_raises
        self.n_cards = 2 * ranks
        self.raise_size = (2, 4)

    def initial(self) -> LeducState:
        return LeducState()

    def _round(self, s: LeducState) -> int:
        return len(s.rounds) - 1

    def node_kind(self, s: LeducState) -> int:
        if s.over:
            return TERMINAL
        if len(s.priv) < 2 or s.need_board:
            return CHANCE
        return s.player

    def _remaining(self, s: LeducState) -> list[int]:
        used = set(s.priv)
        if s.board is not None:
            used.add(s.board)
        return [c for c in range(self.n_cards) if c not in used]

    def actions(self, s: LeducState) -> Sequence[str]:
        if len(s.priv) < 2 or s.need_board:
            return [str(c) for c in self._remaining(s)]
        to_call = max(s.committed) - s.committed[s.player]
        acts = []
        if to_call > 0:
            acts.append("f")
        acts.append("c")
        if s.raises < self.max_raises:
            acts.append("r")
        return acts

    def chance_probs(self, s: LeducState) -> Sequence[float]:
        n = len(self._remaining(s))
        return [1.0 / n] * n

    def apply(self, s: LeducState, action: str) -> LeducState:
        if len(s.priv) < 2:
            return replace(s, priv=s.priv + (int(action),))
        if s.need_board:
            return replace(s, board=int(action), need_board=False,
                           rounds=s.rounds + ("",), player=0, raises=0, acted=0)
        p = s.player
        c = list(s.committed)
        rounds = list(s.rounds)
        rounds[-1] += action
        if action == "f":
            return replace(s, folded=p, over=True, rounds=tuple(rounds))
        round_ends = False
        raises, acted = s.raises, s.acted + 1
        if action == "c":
            if max(c) > c[p]:
                c[p] = max(c)
                round_ends = True
            elif acted >= 2:
                round_ends = True
        elif action == "r":
            c[p] = max(c) + self.raise_size[self._round(s)]
            raises += 1
        else:
            raise ValueError(f"illegal action {action!r}")
        new = replace(s, committed=(c[0], c[1]), rounds=tuple(rounds),
                      raises=raises, acted=acted, player=1 - p)
        if round_ends:
            if self._round(s) == 0:
                return replace(new, need_board=True)
            return replace(new, over=True)
        return new

    def _rank(self, card: int) -> int:
        return card // 2

    def utility(self, s: LeducState) -> float:
        c0, c1 = s.committed
        if s.folded == 0:
            return -float(c0)
        if s.folded == 1:
            return float(c1)
        assert s.board is not None
        r0, r1, rb = (self._rank(s.priv[0]), self._rank(s.priv[1]),
                      self._rank(s.board))
        if r0 == r1:
            return 0.0
        if r0 == rb:
            return float(c1)
        if r1 == rb:
            return -float(c0)
        return float(c1) if r0 > r1 else -float(c0)

    def infoset_key(self, s: LeducState) -> str:
        board = "-" if s.board is None else str(self._rank(s.board))
        return f"{self._rank(s.priv[s.player])}|{board}|{'/'.join(s.rounds)}"
