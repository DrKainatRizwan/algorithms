"""Short Weierstrass curves y^2 = x^3 + ax + b over F_p with several scalar-multiplication strategies."""

from __future__ import annotations

from dataclasses import dataclass

from .field import inv_mod, sqrt_mod

Point = tuple[int, int] | None  # affine point; None is the point at infinity
_Jac = tuple[int, int, int]  # Jacobian (X, Y, Z); Z == 0 is infinity
_JINF: _Jac = (1, 1, 0)


@dataclass(frozen=True)
class Curve:
    """Curve parameters: field prime ``p``, coefficients ``a``, ``b``, generator and group order."""

    name: str
    p: int
    a: int
    b: int
    gx: int
    gy: int
    n: int
    h: int = 1

    @property
    def G(self) -> Point:
        return (self.gx, self.gy)

    @property
    def byte_length(self) -> int:
        return (self.p.bit_length() + 7) // 8

    # ------------------------------------------------------------ predicates
    def is_on_curve(self, P: Point) -> bool:
        if P is None:
            return True
        x, y = P
        if not (0 <= x < self.p and 0 <= y < self.p):
            return False
        return (y * y - (x * x * x + self.a * x + self.b)) % self.p == 0

    # ------------------------------------------------------- affine group law
    def neg(self, P: Point) -> Point:
        return None if P is None else (P[0], (-P[1]) % self.p)

    def add(self, P: Point, Q: Point) -> Point:
        """Affine addition (one inversion per call); used as a reference implementation."""
        if P is None:
            return Q
        if Q is None:
            return P
        p = self.p
        (x1, y1), (x2, y2) = P, Q
        if x1 == x2:
            if (y1 + y2) % p == 0:
                return None
            lam = (3 * x1 * x1 + self.a) * inv_mod(2 * y1, p) % p
        else:
            lam = (y2 - y1) * inv_mod(x2 - x1, p) % p
        x3 = (lam * lam - x1 - x2) % p
        return (x3, (lam * (x1 - x3) - y1) % p)

    def mul_naive(self, k: int, P: Point) -> Point:
        """Textbook affine double-and-add (reference)."""
        if k < 0:
            return self.mul_naive(-k, self.neg(P))
        R: Point = None
        while k:
            if k & 1:
                R = self.add(R, P)
            P = self.add(P, P)
            k >>= 1
        return R

    # ---------------------------------------------------- Jacobian coordinates
    def _to_jac(self, P: Point) -> _Jac:
        return _JINF if P is None else (P[0], P[1], 1)

    def _from_jac(self, J: _Jac) -> Point:
        X, Y, Z = J
        if Z == 0:
            return None
        zi = inv_mod(Z, self.p)
        zi2 = zi * zi % self.p
        return (X * zi2 % self.p, Y * zi2 * zi % self.p)

    def _jdouble(self, J: _Jac) -> _Jac:
        X, Y, Z = J
        p = self.p
        if Z == 0 or Y == 0:
            return _JINF
        YY = Y * Y % p
        S = 4 * X * YY % p
        M = (3 * X * X + self.a * pow(Z, 4, p)) % p
        X3 = (M * M - 2 * S) % p
        Y3 = (M * (S - X3) - 8 * YY * YY) % p
        return (X3, Y3, 2 * Y * Z % p)

    def _jadd(self, A: _Jac, B: _Jac) -> _Jac:
        if A[2] == 0:
            return B
        if B[2] == 0:
            return A
        p = self.p
        X1, Y1, Z1 = A
        X2, Y2, Z2 = B
        Z1Z1, Z2Z2 = Z1 * Z1 % p, Z2 * Z2 % p
        U1, U2 = X1 * Z2Z2 % p, X2 * Z1Z1 % p
        S1, S2 = Y1 * Z2 * Z2Z2 % p, Y2 * Z1 * Z1Z1 % p
        H, R = (U2 - U1) % p, (S2 - S1) % p
        if H == 0:
            return self._jdouble(A) if R == 0 else _JINF
        HH = H * H % p
        HHH = HH * H % p
        V = U1 * HH % p
        X3 = (R * R - HHH - 2 * V) % p
        Y3 = (R * (V - X3) - S1 * HHH) % p
        return (X3, Y3, H * Z1 * Z2 % p)

    def _jneg(self, J: _Jac) -> _Jac:
        return (J[0], (-J[1]) % self.p, J[2])

    # --------------------------------------------------- scalar multiplication
    def mul(self, k: int, P: Point | None = None, method: str = "wnaf") -> Point:
        """Compute ``k * P`` (default ``P`` = generator).

        ``method`` is one of ``"double_and_add"``, ``"ladder"`` (Montgomery ladder, a fixed
        operation pattern per bit) or ``"wnaf"`` (width-4 non-adjacent form).
        """
        if P is None:
            P = self.G
        if P is None:
            return None
        if k < 0:
            return self.mul(-k, self.neg(P), method)
        if method == "double_and_add":
            return self._from_jac(self._mul_dna(k, self._to_jac(P)))
        if method == "ladder":
            return self._from_jac(self._mul_ladder(k, self._to_jac(P)))
        if method == "wnaf":
            return self._from_jac(self._mul_wnaf(k, self._to_jac(P)))
        raise ValueError(f"unknown method {method!r}")

    def _mul_dna(self, k: int, J: _Jac) -> _Jac:
        R = _JINF
        for bit in bin(k)[2:] if k else "":
            R = self._jdouble(R)
            if bit == "1":
                R = self._jadd(R, J)
        return R

    def _mul_ladder(self, k: int, J: _Jac) -> _Jac:
        R0, R1 = _JINF, J
        for bit in bin(k)[2:] if k else "":
            if bit == "0":
                R1 = self._jadd(R0, R1)
                R0 = self._jdouble(R0)
            else:
                R0 = self._jadd(R0, R1)
                R1 = self._jdouble(R1)
        return R0

    @staticmethod
    def wnaf_digits(k: int, w: int = 4) -> list[int]:
        """Width-``w`` NAF of ``k`` (least-significant digit first; digits odd or zero)."""
        digits: list[int] = []
        while k > 0:
            if k & 1:
                d = k % (1 << w)
                if d >= 1 << (w - 1):
                    d -= 1 << w
                k -= d
            else:
                d = 0
            digits.append(d)
            k >>= 1
        return digits

    def _mul_wnaf(self, k: int, J: _Jac, w: int = 4) -> _Jac:
        twoP = self._jdouble(J)
        table = {1: J}
        for i in range(3, 1 << (w - 1), 2):
            table[i] = self._jadd(table[i - 2], twoP)
        R = _JINF
        for d in reversed(self.wnaf_digits(k, w)):
            R = self._jdouble(R)
            if d > 0:
                R = self._jadd(R, table[d])
            elif d < 0:
                R = self._jadd(R, self._jneg(table[-d]))
        return R

    def mul_add(self, u1: int, P: Point, u2: int, Q: Point) -> Point:
        """Compute ``u1*P + u2*Q`` with Shamir's simultaneous double-and-add trick."""
        JP, JQ = self._to_jac(P), self._to_jac(Q)
        JPQ = self._jadd(JP, JQ)
        R = _JINF
        for i in range(max(u1.bit_length(), u2.bit_length()) - 1, -1, -1):
            R = self._jdouble(R)
            b1, b2 = (u1 >> i) & 1, (u2 >> i) & 1
            if b1 and b2:
                R = self._jadd(R, JPQ)
            elif b1:
                R = self._jadd(R, JP)
            elif b2:
                R = self._jadd(R, JQ)
        return self._from_jac(R)

    # ------------------------------------------------------ SEC1 serialisation
    def encode_point(self, P: Point, compressed: bool = True) -> bytes:
        if P is None:
            return b"\x00"
        n = self.byte_length
        x, y = P
        if compressed:
            return bytes([2 + (y & 1)]) + x.to_bytes(n, "big")
        return b"\x04" + x.to_bytes(n, "big") + y.to_bytes(n, "big")

    def decode_point(self, data: bytes) -> Point:
        """Decode a SEC1 point and verify it lies on the curve."""
        n = self.byte_length
        if data == b"\x00":
            return None
        if data[0] in (2, 3) and len(data) == n + 1:
            x = int.from_bytes(data[1:], "big")
            y = sqrt_mod(x ** 3 + self.a * x + self.b, self.p)
            if y is None:
                raise ValueError("x-coordinate is not on the curve")
            if (y & 1) != (data[0] & 1):
                y = self.p - y
            P: Point = (x, y)
        elif data[0] == 4 and len(data) == 2 * n + 1:
            P = (int.from_bytes(data[1:n + 1], "big"), int.from_bytes(data[n + 1:], "big"))
        else:
            raise ValueError("malformed point encoding")
        if not self.is_on_curve(P):
            raise ValueError("point is not on the curve")
        return P

    def lift_x(self, x: int, odd: bool) -> Point:
        """Return the point with abscissa ``x`` and the requested y-parity, or ``None``."""
        y = sqrt_mod(x ** 3 + self.a * x + self.b, self.p)
        if y is None:
            return None
        return (x, y if (y & 1) == int(odd) else self.p - y)


SECP256K1 = Curve(
    "secp256k1",
    p=2 ** 256 - 2 ** 32 - 977, a=0, b=7,
    gx=0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
    gy=0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8,
    n=0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141,
)

P256 = Curve(
    "P-256",
    p=2 ** 256 - 2 ** 224 + 2 ** 192 + 2 ** 96 - 1, a=-3 % (2 ** 256 - 2 ** 224 + 2 ** 192 + 2 ** 96 - 1),
    b=0x5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B,
    gx=0x6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296,
    gy=0x4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5,
    n=0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551,
)

# Textbook toy curve: y^2 = x^3 + 2x + 2 over F_17, prime group order 19.
TOY17 = Curve("toy17", p=17, a=2, b=2, gx=5, gy=1, n=19)

_REGISTRY = {c.name: c for c in (SECP256K1, P256, TOY17)}


def get_curve(name: str) -> Curve:
    """Look up a named curve."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown curve {name!r}; available: {sorted(_REGISTRY)}") from None
