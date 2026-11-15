"""Arithmetic in the binary extension field GF(2^m).

Elements are integers in ``[0, 2^m)`` whose bits are the coefficients of a
polynomial over GF(2), reduced modulo a primitive polynomial.  Multiplication
and division use log/antilog tables built from powers of the primitive
element ``alpha = 2``.

Polynomials over the field are plain ``list[int]`` with the coefficient of
``x**i`` stored at index ``i`` (lowest degree first).
"""

from __future__ import annotations

# Primitive polynomials (including the leading x^m term) for common sizes.
PRIMITIVE_POLYS: dict[int, int] = {
    2: 0b111,
    3: 0b1011,
    4: 0b10011,
    5: 0b100101,
    6: 0b1000011,
    7: 0b10001001,
    8: 0b100011101,
    9: 0b1000010001,
    10: 0b10000001001,
}

Poly = list[int]


class GF2m:
    """The field GF(2^m) with table-driven multiplication."""

    def __init__(self, m: int, prim_poly: int | None = None) -> None:
        if prim_poly is None:
            if m not in PRIMITIVE_POLYS:
                raise ValueError(f"no default primitive polynomial for m={m}")
            prim_poly = PRIMITIVE_POLYS[m]
        self.m = m
        self.size = 1 << m
        self.order = self.size - 1  # multiplicative group order
        self.prim_poly = prim_poly
        # exp table is doubled so products of logs need no modular reduction
        self.exp = [0] * (2 * self.order)
        self.log = [0] * self.size
        x = 1
        for i in range(self.order):
            self.exp[i] = x
            self.log[x] = i
            x <<= 1
            if x & self.size:
                x ^= prim_poly
        if x != 1:
            raise ValueError("polynomial is not primitive")
        for i in range(self.order, 2 * self.order):
            self.exp[i] = self.exp[i - self.order]

    # ------------------------------------------------------------ scalars
    @staticmethod
    def add(a: int, b: int) -> int:
        """Addition (and subtraction) is XOR."""
        return a ^ b

    def mul(self, a: int, b: int) -> int:
        if a == 0 or b == 0:
            return 0
        return self.exp[self.log[a] + self.log[b]]

    def div(self, a: int, b: int) -> int:
        if b == 0:
            raise ZeroDivisionError("division by zero in GF(2^m)")
        if a == 0:
            return 0
        return self.exp[self.log[a] - self.log[b] + self.order]

    def inv(self, a: int) -> int:
        if a == 0:
            raise ZeroDivisionError("zero has no inverse")
        return self.exp[self.order - self.log[a]]

    def pow(self, a: int, e: int) -> int:
        if a == 0:
            return 0 if e > 0 else 1
        return self.exp[(self.log[a] * e) % self.order]

    def alpha_pow(self, e: int) -> int:
        """alpha**e for any integer exponent."""
        return self.exp[e % self.order]

    # --------------------------------------------------------- polynomials
    @staticmethod
    def trim(p: Poly) -> Poly:
        """Drop high-order zero coefficients (keeps at least one entry)."""
        n = len(p)
        while n > 1 and p[n - 1] == 0:
            n -= 1
        return p[:n] if n else [0]

    def poly_add(self, a: Poly, b: Poly) -> Poly:
        n = max(len(a), len(b))
        out = [0] * n
        for i, c in enumerate(a):
            out[i] ^= c
        for i, c in enumerate(b):
            out[i] ^= c
        return out

    def poly_scale(self, a: Poly, c: int) -> Poly:
        return [self.mul(x, c) for x in a]

    def poly_mul(self, a: Poly, b: Poly) -> Poly:
        out = [0] * (len(a) + len(b) - 1)
        for i, x in enumerate(a):
            if x == 0:
                continue
            for j, y in enumerate(b):
                if y:
                    out[i + j] ^= self.mul(x, y)
        return out

    def poly_eval(self, p: Poly, x: int) -> int:
        """Horner evaluation."""
        acc = 0
        for c in reversed(p):
            acc = self.mul(acc, x) ^ c
        return acc

    def poly_divmod(self, num: Poly, den: Poly) -> tuple[Poly, Poly]:
        """Polynomial long division; returns ``(quotient, remainder)``."""
        den = self.trim(den)
        if den == [0]:
            raise ZeroDivisionError("division by zero polynomial")
        rem = list(num)
        dd = len(den) - 1
        if len(rem) <= dd:
            return [0], self.trim(rem)
        quo = [0] * (len(rem) - dd)
        lead_inv = self.inv(den[-1])
        for i in range(len(rem) - 1, dd - 1, -1):
            if rem[i] == 0:
                continue
            f = self.mul(rem[i], lead_inv)
            quo[i - dd] = f
            for j, c in enumerate(den):
                if c:
                    rem[i - dd + j] ^= self.mul(f, c)
        return self.trim(quo), self.trim(rem[:dd] or [0])

    @staticmethod
    def poly_deriv(p: Poly) -> Poly:
        """Formal derivative; in characteristic 2 even-degree terms vanish."""
        if len(p) <= 1:
            return [0]
        return [p[i] if i % 2 == 1 else 0 for i in range(1, len(p))]

    # -------------------------------------------------------- linear algebra
    def solve(self, a: list[list[int]], b: list[int]) -> list[int] | None:
        """Solve ``A x = b`` by Gauss-Jordan elimination; ``None`` if singular."""
        n = len(b)
        m = [row[:] + [b[i]] for i, row in enumerate(a)]
        for col in range(n):
            piv = next((r for r in range(col, n) if m[r][col]), None)
            if piv is None:
                return None
            m[col], m[piv] = m[piv], m[col]
            inv = self.inv(m[col][col])
            m[col] = [self.mul(v, inv) for v in m[col]]
            for r in range(n):
                if r != col and m[r][col]:
                    f = m[r][col]
                    m[r] = [v ^ self.mul(f, w) for v, w in zip(m[r], m[col])]
        return [m[i][n] for i in range(n)]

    # ----------------------------------------------------------- cyclotomy
    def cyclotomic_coset(self, s: int) -> list[int]:
        """Exponents ``s, 2s, 4s, ...`` modulo ``2^m - 1``."""
        coset, x = [], s % self.order
        while x not in coset:
            coset.append(x)
            x = (2 * x) % self.order
        return coset

    def minimal_poly(self, s: int) -> int:
        """Minimal polynomial of alpha**s over GF(2), as a bit-packed int."""
        poly: Poly = [1]
        for c in self.cyclotomic_coset(s):
            poly = self.poly_mul(poly, [self.alpha_pow(c), 1])
        if any(c not in (0, 1) for c in poly):  # pragma: no cover - sanity
            raise AssertionError("minimal polynomial not binary")
        return sum(c << i for i, c in enumerate(poly))
