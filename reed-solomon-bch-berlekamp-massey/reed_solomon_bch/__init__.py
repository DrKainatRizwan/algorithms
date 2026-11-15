"""Reed-Solomon and BCH error-correcting codes built from scratch.

Modules
-------
galois        GF(2^m) arithmetic and polynomial helpers
berlekamp     Berlekamp-Massey and Peterson-Gorenstein-Zierler locator solvers
reed_solomon  Systematic RS codes with errors-and-erasures decoding
bch           Binary primitive BCH codes
channel       Noise models and block interleaving
experiments   Monte-Carlo studies of decoder behaviour
"""

from .galois import GF2m, PRIMITIVE_POLYS
from .berlekamp import berlekamp_massey, pgz_locator
from .reed_solomon import ReedSolomon, DecodingFailure
from .bch import BCH
from .channel import (
    add_symbol_errors,
    add_bit_errors,
    add_burst,
    BlockInterleaver,
)

__all__ = [
    "GF2m",
    "PRIMITIVE_POLYS",
    "berlekamp_massey",
    "pgz_locator",
    "ReedSolomon",
    "DecodingFailure",
    "BCH",
    "add_symbol_errors",
    "add_bit_errors",
    "add_burst",
    "BlockInterleaver",
]
