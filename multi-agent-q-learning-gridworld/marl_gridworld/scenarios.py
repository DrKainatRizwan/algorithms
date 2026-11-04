"""Ready-made layouts that stress coordination."""
from __future__ import annotations

from .env import CoopGridWorld, parse_layout

SCENARIOS: dict[str, list[str]] = {
    # two agents cross an open room: their shortest paths intersect
    "crossing": [
        "0..#..b",
        "...#...",
        ".......",
        "...#...",
        "a..#..1",
    ],
    # swap ends of a corridor; the only way to pass is the bay in column 5
    "bay": [
        "#####.#####",
        "b0.......1a",
        "###########",
    ],
    # three agents rotating through a small room with an obstacle
    "triangle": [
        "0.b..",
        ".....",
        "c.#.a",
        "..2.1",
    ],
}


def make_env(name: str, max_steps: int = 60, **kw) -> CoopGridWorld:
    """Build a named scenario."""
    walls, starts, goals = parse_layout(SCENARIOS[name])
    return CoopGridWorld(walls, starts, goals, max_steps=max_steps, **kw)
