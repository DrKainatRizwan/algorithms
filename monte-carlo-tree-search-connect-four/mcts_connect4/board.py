"""Bitboard Connect-Four.

Each column uses ROWS+1 bits (the top bit is a sentinel that stays empty), so
shift-based four-in-a-row detection never wraps across columns.
"""
from __future__ import annotations

COLS = 7
ROWS = 6
H = ROWS + 1
_FULL_MASK = sum(((1 << ROWS) - 1) << (c * H) for c in range(COLS))
_TOP = sum(1 << (c * H + ROWS - 1) for c in range(COLS))
_BOTTOM = sum(1 << (c * H) for c in range(COLS))


def _has_four(bits: int) -> bool:
    """Return True if ``bits`` contains four aligned set bits."""
    for shift in (1, H, H - 1, H + 1):
        m = bits & (bits >> shift)
        if m & (m >> (2 * shift)):
            return True
    return False


class Board:
    """Mutable Connect-Four position. Player 0 moves first."""

    __slots__ = ("bits", "mask", "moves")

    def __init__(self) -> None:
        self.bits = [0, 0]
        self.mask = 0
        self.moves = 0

    # -- state ---------------------------------------------------------
    @property
    def to_move(self) -> int:
        return self.moves & 1

    def copy(self) -> "Board":
        b = Board()
        b.bits = list(self.bits)
        b.mask = self.mask
        b.moves = self.moves
        return b

    def key(self) -> tuple[int, int]:
        """Hashable canonical identifier of the position."""
        return (self.bits[self.to_move], self.mask)

    def can_play(self, col: int) -> bool:
        return 0 <= col < COLS and not (self.mask & (1 << (col * H + ROWS - 1)))

    def legal_moves(self) -> list[int]:
        return [c for c in range(COLS) if self.can_play(c)]

    def play(self, col: int) -> None:
        if not self.can_play(col):
            raise ValueError(f"illegal move: column {col}")
        move = (self.mask + (1 << (col * H))) & ~self.mask & (((1 << ROWS) - 1) << (col * H))
        self.bits[self.to_move] |= move
        self.mask |= move
        self.moves += 1

    def undo(self, col: int) -> None:
        """Remove the topmost disc in ``col`` (must belong to the last mover)."""
        colmask = ((1 << ROWS) - 1) << (col * H)
        occupied = self.mask & colmask
        if not occupied:
            raise ValueError("column empty")
        top = 1 << (occupied.bit_length() - 1)
        self.moves -= 1
        self.bits[self.to_move] &= ~top
        self.mask &= ~top

    # -- outcome -------------------------------------------------------
    def winner(self) -> int | None:
        for p in (0, 1):
            if _has_four(self.bits[p]):
                return p
        return None

    def is_full(self) -> bool:
        return self.mask == _FULL_MASK

    def is_terminal(self) -> bool:
        return self.is_full() or self.winner() is not None

    def wins_by_playing(self, col: int, player: int) -> bool:
        """Would ``player`` complete four by dropping in ``col``?"""
        if not self.can_play(col):
            return False
        colmask = ((1 << ROWS) - 1) << (col * H)
        move = (self.mask + (1 << (col * H))) & ~self.mask & colmask
        return _has_four(self.bits[player] | move)

    def result(self) -> float | None:
        """Outcome from player 0's perspective: 1 win, 0 loss, 0.5 draw."""
        w = self.winner()
        if w is not None:
            return 1.0 if w == 0 else 0.0
        return 0.5 if self.is_full() else None

    # -- display -------------------------------------------------------
    def cell(self, col: int, row: int) -> str:
        bit = 1 << (col * H + row)
        if self.bits[0] & bit:
            return "X"
        if self.bits[1] & bit:
            return "O"
        return "."

    def __str__(self) -> str:
        rows = [" ".join(self.cell(c, r) for c in range(COLS)) for r in range(ROWS - 1, -1, -1)]
        return "\n".join(rows + [" ".join(str(c) for c in range(COLS))])

    @classmethod
    def from_moves(cls, moves: list[int]) -> "Board":
        b = cls()
        for m in moves:
            b.play(m)
        return b

    def threats(self, player: int) -> int:
        """Count columns where ``player`` has an immediate win."""
        return sum(self.wins_by_playing(c, player) for c in range(COLS))
