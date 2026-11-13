"""Boolean circuits with a small builder and standard arithmetic gadgets.

Supported gate operations are ``XOR``, ``AND`` and ``NOT``; ``OR`` is
composed so that the cost model (number of AND gates) is explicit. Words are
little-endian lists of wire ids.
"""

from __future__ import annotations

from dataclasses import dataclass, field

XOR, AND, NOT = "XOR", "AND", "NOT"


@dataclass(frozen=True)
class Gate:
    """A single gate; ``b`` is -1 for unary gates."""
    op: str
    a: int
    b: int
    out: int


@dataclass
class Circuit:
    """A topologically ordered Boolean circuit with two input parties."""
    n_wires: int = 0
    gates: list[Gate] = field(default_factory=list)
    inputs_a: list[int] = field(default_factory=list)  # garbler inputs
    inputs_b: list[int] = field(default_factory=list)  # evaluator inputs
    outputs: list[int] = field(default_factory=list)
    _zero: int | None = None

    # -- construction -----------------------------------------------------
    def _new_wire(self) -> int:
        self.n_wires += 1
        return self.n_wires - 1

    def input_a(self, width: int = 1) -> list[int]:
        """Allocate ``width`` input wires owned by party A (garbler)."""
        ws = [self._new_wire() for _ in range(width)]
        self.inputs_a.extend(ws)
        return ws

    def input_b(self, width: int = 1) -> list[int]:
        """Allocate ``width`` input wires owned by party B (evaluator)."""
        ws = [self._new_wire() for _ in range(width)]
        self.inputs_b.extend(ws)
        return ws

    def _gate(self, op: str, a: int, b: int = -1) -> int:
        out = self._new_wire()
        self.gates.append(Gate(op, a, b, out))
        return out

    def xor(self, a: int, b: int) -> int:
        return self._gate(XOR, a, b)

    def and_(self, a: int, b: int) -> int:
        return self._gate(AND, a, b)

    def not_(self, a: int) -> int:
        return self._gate(NOT, a)

    def or_(self, a: int, b: int) -> int:
        """a OR b = a XOR b XOR (a AND b): one AND gate."""
        return self.xor(self.xor(a, b), self.and_(a, b))

    def zero(self) -> int:
        """A constant-0 wire (x XOR x of the first input)."""
        if self._zero is None:
            first = (self.inputs_a + self.inputs_b)[0]
            self._zero = self.xor(first, first)
        return self._zero

    def one(self) -> int:
        return self.not_(self.zero())

    def mux(self, sel: int, x: int, y: int) -> int:
        """``x`` if sel else ``y``: y XOR (sel AND (x XOR y)); one AND."""
        return self.xor(y, self.and_(sel, self.xor(x, y)))

    def set_outputs(self, wires: list[int]) -> None:
        self.outputs = list(wires)

    # -- gadgets ----------------------------------------------------------
    def add(self, xs: list[int], ys: list[int], carry_out: bool = False) -> list[int]:
        """Ripple-carry adder with one AND per bit."""
        carry = self.zero()
        out: list[int] = []
        for a, b in zip(xs, ys):
            axc, bxc = self.xor(a, carry), self.xor(b, carry)
            out.append(self.xor(self.xor(a, b), carry))
            carry = self.xor(carry, self.and_(axc, bxc))
        if carry_out:
            out.append(carry)
        return out

    def greater_than(self, xs: list[int], ys: list[int]) -> int:
        """Unsigned ``x > y`` (Yao's millionaires' comparison), one AND per bit."""
        gt = self.zero()
        for a, b in zip(xs, ys):  # LSB first; later (higher) bits override
            gt = self.xor(gt, self.and_(self.xor(a, b), self.xor(a, gt)))
        return gt

    def equal(self, xs: list[int], ys: list[int]) -> int:
        """``x == y`` via an AND-tree of XNORs."""
        terms = [self.not_(self.xor(a, b)) for a, b in zip(xs, ys)]
        while len(terms) > 1:
            nxt = [self.and_(terms[i], terms[i + 1]) for i in range(0, len(terms) - 1, 2)]
            if len(terms) % 2:
                nxt.append(terms[-1])
            terms = nxt
        return terms[0]

    def multiply(self, xs: list[int], ys: list[int]) -> list[int]:
        """Schoolbook unsigned multiplier producing ``len(xs)+len(ys)`` bits."""
        width = len(xs) + len(ys)
        acc = [self.zero() for _ in range(width)]
        for i, b in enumerate(ys):
            partial = [self.zero()] * i + [self.and_(a, b) for a in xs]
            partial += [self.zero()] * (width - len(partial))
            acc = self.add(acc, partial)
        return acc

    # -- analysis ---------------------------------------------------------
    def count(self, op: str) -> int:
        return sum(1 for g in self.gates if g.op == op)

    def evaluate_plain(self, a_bits: list[int], b_bits: list[int]) -> list[int]:
        """Reference evaluation on cleartext bits."""
        if len(a_bits) != len(self.inputs_a) or len(b_bits) != len(self.inputs_b):
            raise ValueError("wrong number of input bits")
        w: dict[int, int] = dict(zip(self.inputs_a, a_bits))
        w.update(zip(self.inputs_b, b_bits))
        for g in self.gates:
            if g.op == XOR:
                w[g.out] = w[g.a] ^ w[g.b]
            elif g.op == AND:
                w[g.out] = w[g.a] & w[g.b]
            else:
                w[g.out] = 1 - w[g.a]
        return [w[o] for o in self.outputs]


# -- ready-made circuits ------------------------------------------------------
def millionaires(width: int) -> Circuit:
    """Does Alice's ``width``-bit wealth exceed Bob's?"""
    c = Circuit()
    c.set_outputs([c.greater_than(c.input_a(width), c.input_b(width))])
    return c


def adder_circuit(width: int) -> Circuit:
    c = Circuit()
    c.set_outputs(c.add(c.input_a(width), c.input_b(width), carry_out=True))
    return c


def multiplier_circuit(width: int) -> Circuit:
    c = Circuit()
    c.set_outputs(c.multiply(c.input_a(width), c.input_b(width)))
    return c


def equality_circuit(width: int) -> Circuit:
    c = Circuit()
    c.set_outputs([c.equal(c.input_a(width), c.input_b(width))])
    return c


def max_circuit(width: int) -> Circuit:
    """Output the larger of the two parties' values."""
    c = Circuit()
    xs, ys = c.input_a(width), c.input_b(width)
    sel = c.greater_than(xs, ys)
    c.set_outputs([c.mux(sel, a, b) for a, b in zip(xs, ys)])
    return c
