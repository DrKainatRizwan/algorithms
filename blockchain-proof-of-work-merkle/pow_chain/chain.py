"""Block tree, consensus validation, heaviest-chain fork choice and mempool."""
from __future__ import annotations

from dataclasses import dataclass, field

from .block import ZERO_HASH, Block, Transaction, tx_merkle_root
from .merkle import MerkleTree, Proof, verify_proof
from .pow import (bits_to_target, mine_block, retarget, target_to_bits, work_from_target)

EXTENDED, REORG, SIDE_BRANCH, ORPHAN, INVALID, DUPLICATE = (
    "extended", "reorg", "side_branch", "orphan", "invalid", "duplicate")

MTP_WINDOW = 11


@dataclass(frozen=True)
class ConsensusParams:
    genesis_bits: int = target_to_bits(1 << 248)
    pow_limit: int = 1 << 252
    retarget_interval: int = 10
    target_spacing: int = 600
    block_reward: int = 50
    max_block_txs: int = 200
    genesis_alloc: tuple[tuple[str, int], ...] = ()


@dataclass
class ChainNode:
    block: Block
    hash: str
    height: int
    cum_work: int
    parent: "ChainNode | None"
    balances: dict[str, int]
    nonces: dict[str, int]

    @property
    def timestamp(self) -> int:
        return self.block.timestamp


@dataclass
class AddResult:
    status: str
    block_hash: str
    reason: str = ""
    connected: list[Block] = field(default_factory=list)
    disconnected: list[Block] = field(default_factory=list)

    @property
    def reorg_depth(self) -> int:
        return len(self.disconnected)


class ValidationError(Exception):
    pass


def apply_transactions(balances: dict[str, int], nonces: dict[str, int],
                       txs: tuple[Transaction, ...], height: int, params: ConsensusParams
                       ) -> tuple[dict[str, int], dict[str, int]]:
    """Validate a block body against a parent state; returns the child state."""
    if not txs or not txs[0].is_coinbase:
        raise ValidationError("first transaction must be a coinbase")
    if len(txs) - 1 > params.max_block_txs:
        raise ValidationError("too many transactions")
    if len({t.txid for t in txs}) != len(txs):
        raise ValidationError("duplicate transaction")
    bal, non = dict(balances), dict(nonces)
    fees = 0
    for tx in txs[1:]:
        if tx.is_coinbase:
            raise ValidationError("extra coinbase")
        if tx.amount <= 0 or tx.fee < 0:
            raise ValidationError("non-positive amount or negative fee")
        if non.get(tx.sender, 0) != tx.nonce:
            raise ValidationError("bad nonce")
        if bal.get(tx.sender, 0) < tx.amount + tx.fee:
            raise ValidationError("insufficient funds")
        bal[tx.sender] -= tx.amount + tx.fee
        bal[tx.recipient] = bal.get(tx.recipient, 0) + tx.amount
        non[tx.sender] = tx.nonce + 1
        fees += tx.fee
    cb = txs[0]
    if cb.amount != params.block_reward + fees or cb.nonce != height:
        raise ValidationError("bad coinbase")
    bal[cb.recipient] = bal.get(cb.recipient, 0) + cb.amount
    return bal, non


class Blockchain:
    """A node's view of the block tree.

    Every accepted block is kept (including stale branches) together with the
    account state after it, so reorganisations need no replay.  Fork choice is
    greatest cumulative work with first-seen tie-breaking.
    """

    def __init__(self, params: ConsensusParams | None = None) -> None:
        self.params = params or ConsensusParams()
        self.nodes: dict[str, ChainNode] = {}
        self.orphans: dict[str, list[Block]] = {}
        self.genesis = self._make_genesis()
        self.tip = self.genesis

    # ------------------------------------------------------------------ genesis
    def _make_genesis(self) -> ChainNode:
        p = self.params
        txs = [Transaction("", a, amt, 0, 0) for a, amt in p.genesis_alloc]
        txs = txs or [Transaction("", "genesis", 0, 0, 0)]
        block = Block.assemble(ZERO_HASH, txs, 0, p.genesis_bits)
        solved = mine_block(block)
        assert solved is not None
        block = solved[0]
        bal: dict[str, int] = {}
        for t in txs:
            bal[t.recipient] = bal.get(t.recipient, 0) + t.amount
        node = ChainNode(block, block.hash, 0, work_from_target(bits_to_target(p.genesis_bits)),
                         None, bal, {})
        self.nodes[node.hash] = node
        return node

    # ------------------------------------------------------------------ queries
    @property
    def height(self) -> int:
        return self.tip.height

    def main_chain(self) -> list[ChainNode]:
        out, n = [], self.tip
        while n is not None:
            out.append(n)
            n = n.parent
        return out[::-1]

    def on_main_chain(self, h: str) -> bool:
        node = self.nodes.get(h)
        if node is None:
            return False
        return self._ancestor_at(self.tip, node.height) is node

    def balance(self, addr: str) -> int:
        return self.tip.balances.get(addr, 0)

    def next_nonce(self, addr: str) -> int:
        return self.tip.nonces.get(addr, 0)

    def stale_blocks(self) -> list[ChainNode]:
        return [n for n in self.nodes.values() if not self.on_main_chain(n.hash)]

    @staticmethod
    def _ancestor_at(node: ChainNode, height: int) -> ChainNode:
        while node.height > height:
            node = node.parent  # type: ignore[assignment]
        return node

    def median_time_past(self, node: ChainNode) -> int:
        ts, n = [], node
        while n is not None and len(ts) < MTP_WINDOW:
            ts.append(n.timestamp)
            n = n.parent
        ts.sort()
        return ts[len(ts) // 2]

    def expected_bits(self, parent: ChainNode) -> int:
        """Required ``bits`` for a child of ``parent`` (retargets every interval)."""
        p = self.params
        height = parent.height + 1
        if height % p.retarget_interval != 0:
            return parent.block.header.bits
        first = self._ancestor_at(parent, height - p.retarget_interval)
        actual = parent.timestamp - first.timestamp
        expected = p.target_spacing * (p.retarget_interval - 1)
        new = retarget(bits_to_target(parent.block.header.bits), actual, expected, p.pow_limit)
        return target_to_bits(new)

    # ------------------------------------------------------------------ validation
    def _validate(self, block: Block, parent: ChainNode) -> ChainNode:
        h = block.header
        if h.bits != self.expected_bits(parent):
            raise ValidationError("wrong difficulty bits")
        target = bits_to_target(h.bits)
        if h.hash_int > target:
            raise ValidationError("insufficient proof of work")
        if h.merkle_root != tx_merkle_root(block.txs):
            raise ValidationError("merkle root mismatch")
        if h.timestamp <= self.median_time_past(parent):
            raise ValidationError("timestamp not after median time past")
        height = parent.height + 1
        bal, non = apply_transactions(parent.balances, parent.nonces, block.txs, height, self.params)
        return ChainNode(block, block.hash, height, parent.cum_work + work_from_target(target),
                         parent, bal, non)

    # ------------------------------------------------------------------ insertion
    def add_block(self, block: Block) -> AddResult:
        """Insert a block; out-of-order blocks wait in the orphan pool."""
        old_tip = self.tip
        first = self._attach(block)
        if first.status in (INVALID, DUPLICATE, ORPHAN):
            return first
        # Connect any orphans that were waiting on blocks that just arrived.
        queue = [block.hash]
        while queue:
            for child in self.orphans.pop(queue.pop(), []):
                if self._attach(child).status not in (INVALID, DUPLICATE, ORPHAN):
                    queue.append(child.hash)
        res = AddResult(SIDE_BRANCH, block.hash)
        if self.tip is not old_tip:
            res.disconnected, res.connected = self._diff(old_tip, self.tip)
            res.status = REORG if res.disconnected else EXTENDED
        return res

    def _attach(self, block: Block) -> AddResult:
        if block.hash in self.nodes:
            return AddResult(DUPLICATE, block.hash)
        parent = self.nodes.get(block.prev_hash)
        if parent is None:
            if bits_to_target(block.header.bits) < block.header.hash_int:
                return AddResult(INVALID, block.hash, "insufficient proof of work")
            pool = self.orphans.setdefault(block.prev_hash, [])
            if all(b.hash != block.hash for b in pool):
                pool.append(block)
            return AddResult(ORPHAN, block.hash)
        try:
            node = self._validate(block, parent)
        except ValidationError as e:
            return AddResult(INVALID, block.hash, str(e))
        self.nodes[node.hash] = node
        if node.cum_work > self.tip.cum_work:
            self.tip = node
        return AddResult(SIDE_BRANCH, block.hash)

    def _diff(self, old: ChainNode, new: ChainNode) -> tuple[list[Block], list[Block]]:
        a, b = old, new
        down: list[Block] = []
        up: list[Block] = []
        while a.height > b.height:
            down.append(a.block)
            a = a.parent  # type: ignore[assignment]
        while b.height > a.height:
            up.append(b.block)
            b = b.parent  # type: ignore[assignment]
        while a is not b:
            down.append(a.block)
            up.append(b.block)
            a, b = a.parent, b.parent  # type: ignore[assignment]
        return down, up[::-1]

    # ------------------------------------------------------------------ mining
    def block_template(self, miner: str, txs: list[Transaction], timestamp: int) -> Block:
        """Unmined block on the current tip paying ``miner`` reward plus fees."""
        parent = self.tip
        fees = sum(t.fee for t in txs)
        cb = Transaction.coinbase(miner, self.params.block_reward + fees, parent.height + 1)
        ts = max(timestamp, self.median_time_past(parent) + 1)
        return Block.assemble(parent.hash, [cb, *txs], ts, self.expected_bits(parent))

    # ------------------------------------------------------------------ SPV
    def prove_tx(self, txid: str) -> tuple[str, Proof] | None:
        """Merkle proof that a main-chain transaction is in its block."""
        for node in self.main_chain():
            ids = [t.txid for t in node.block.txs]
            if txid in ids:
                tree = MerkleTree([bytes.fromhex(i) for i in ids])
                return node.hash, tree.proof(ids.index(txid))
        return None

    def verify_tx_proof(self, txid: str, block_hash: str, proof: Proof) -> bool:
        """Light-client check using only the header's Merkle root."""
        node = self.nodes.get(block_hash)
        if node is None:
            return False
        return verify_proof(bytes.fromhex(txid), proof, bytes.fromhex(node.block.header.merkle_root))

    def confirmations(self, txid: str) -> int:
        for node in self.main_chain():
            if any(t.txid == txid for t in node.block.txs):
                return self.tip.height - node.height + 1
        return 0


class Mempool:
    """Pending transactions with fee-priority, nonce-respecting selection."""

    def __init__(self) -> None:
        self.txs: dict[str, Transaction] = {}

    def __len__(self) -> int:
        return len(self.txs)

    def add(self, tx: Transaction) -> bool:
        if tx.is_coinbase or tx.txid in self.txs:
            return False
        self.txs[tx.txid] = tx
        return True

    def select(self, chain: Blockchain, limit: int | None = None) -> list[Transaction]:
        limit = chain.params.max_block_txs if limit is None else limit
        bal, non = dict(chain.tip.balances), dict(chain.tip.nonces)
        pending = sorted(self.txs.values(), key=lambda t: (-t.fee, t.nonce, t.txid))
        chosen: list[Transaction] = []
        progress = True
        while progress and len(chosen) < limit:
            progress = False
            for tx in pending:
                if tx.txid in {c.txid for c in chosen} or non.get(tx.sender, 0) != tx.nonce:
                    continue
                if tx.amount <= 0 or bal.get(tx.sender, 0) < tx.amount + tx.fee:
                    continue
                bal[tx.sender] -= tx.amount + tx.fee
                bal[tx.recipient] = bal.get(tx.recipient, 0) + tx.amount
                non[tx.sender] = tx.nonce + 1
                chosen.append(tx)
                progress = True
                break
        return chosen

    def update_after(self, chain: Blockchain, result: AddResult) -> None:
        """Drop confirmed txs and restore those from disconnected blocks."""
        for blk in result.disconnected:
            for tx in blk.txs[1:]:
                self.add(tx)
        for blk in result.connected:
            for tx in blk.txs[1:]:
                self.txs.pop(tx.txid, None)
        for txid, tx in list(self.txs.items()):
            if tx.nonce < chain.tip.nonces.get(tx.sender, 0):
                del self.txs[txid]
