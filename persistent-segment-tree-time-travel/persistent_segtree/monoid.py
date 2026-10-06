"""Monoids used to parameterise segment trees."""
from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Any, Callable


@dataclass(frozen=True)
class Monoid:
    """An associative binary operation with an identity element."""

    identity: Any
    combine: Callable[[Any, Any], Any]
    name: str = "monoid"


SUM = Monoid(0, lambda a, b: a + b, "sum")
MIN = Monoid(float("inf"), min, "min")
MAX = Monoid(float("-inf"), max, "max")
GCD = Monoid(0, gcd, "gcd")
