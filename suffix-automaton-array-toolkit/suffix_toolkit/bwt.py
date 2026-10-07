"""Burrows-Wheeler transform derived from the suffix array."""
from __future__ import annotations

from .suffix_array import build_suffix_array


def bwt_from_suffix_array(s: str, sentinel: str = "\0") -> str:
    """BWT of ``s + sentinel``; the sentinel must not occur in ``s`` and sorts lowest."""
    if sentinel in s:
        raise ValueError("sentinel occurs in input")
    t = s + sentinel
    sa = build_suffix_array(t)
    return "".join(t[i - 1] for i in sa)


def inverse_bwt(b: str, sentinel: str = "\0") -> str:
    """Invert the BWT with the LF-mapping and return the original string."""
    n = len(b)
    order = sorted(range(n), key=lambda i: (b[i], i))
    lf = [0] * n  # last-column row -> first-column row of the same character
    for rank, i in enumerate(order):
        lf[i] = rank
    row = 0  # row 0 is the rotation starting with the sentinel
    out = []
    for _ in range(n - 1):
        out.append(b[row])
        row = lf[row]
    return "".join(reversed(out))
