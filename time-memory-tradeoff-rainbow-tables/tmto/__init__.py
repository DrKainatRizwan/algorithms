"""Time-memory trade-off cryptanalysis: Hellman, distinguished-point and rainbow tables."""

from .oneway import OneWayFunction, PasswordSpace
from .hellman import HellmanTable, HellmanAttack
from .distinguished import DistinguishedPointTable, DistinguishedPointAttack
from .rainbow import RainbowTable, RainbowAttack
from .analysis import (
    hellman_coverage,
    rainbow_coverage,
    dp_expected_chain_length,
    measure_success,
    AttackStats,
)

__all__ = [
    "OneWayFunction",
    "PasswordSpace",
    "HellmanTable",
    "HellmanAttack",
    "DistinguishedPointTable",
    "DistinguishedPointAttack",
    "RainbowTable",
    "RainbowAttack",
    "hellman_coverage",
    "rainbow_coverage",
    "dp_expected_chain_length",
    "measure_success",
    "AttackStats",
]
