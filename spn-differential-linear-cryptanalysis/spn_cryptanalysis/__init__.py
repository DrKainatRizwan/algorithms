"""Differential and linear cryptanalysis of a small substitution-permutation network."""
from .analysis import SboxReport, avalanche, sbox_report
from .differential import (Characteristic, best_characteristic,
                           differential_attack)
from .linear import LinearTrail, best_trail, linear_attack, piling_up
from .sbox import (SBOX_AFFINE, SBOX_HEYS, SBOX_PRESENT, ddt, lat,
                   walsh_spectrum)
from .spn import SPN

__all__ = [
    "SPN", "SBOX_HEYS", "SBOX_PRESENT", "SBOX_AFFINE", "ddt", "lat",
    "walsh_spectrum", "Characteristic", "best_characteristic",
    "differential_attack", "LinearTrail", "best_trail", "linear_attack",
    "piling_up", "SboxReport", "sbox_report", "avalanche",
]
