"""hash_lab: Merkle-Damgard hashes from scratch and the attacks that exploit them."""
from .attacks import (birthday_collision, expected_birthday_queries, length_extend,
                      recover_secret_length, rho_collision, truncated_hash)
from .hashes import ALGORITHMS, MD5, SHA1, SHA256, SHA512
from .mac import constant_time_equal, hkdf, hmac, pbkdf2, prefix_mac
from .mdcore import MDHash, md_padding
from .rainbow import RainbowTable

__all__ = ["ALGORITHMS", "MD5", "SHA1", "SHA256", "SHA512", "MDHash", "md_padding",
           "prefix_mac", "hmac", "hkdf", "pbkdf2", "constant_time_equal",
           "length_extend", "recover_secret_length", "truncated_hash",
           "birthday_collision", "rho_collision", "expected_birthday_queries",
           "RainbowTable"]
__version__ = "1.0.0"
