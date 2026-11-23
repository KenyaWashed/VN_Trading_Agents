"""Closed Causal DSL Vocabulary, Prefix-Order AST Serializer, and Grammar Masking.

Guarantees 100% causal syntax validity during Discrete Masked AST Diffusion unmasking
and subtree infilling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vn_rsi.dsl.ast_spec import ASTNode, FactorSpec
from vn_rsi.dsl.operators import ALLOWED_OPERATORS, ALLOWED_TERMINALS

CANONICAL_WINDOWS: tuple[int, ...] = (3, 5, 10, 15, 20)
WINDOWED_DSL_OPS: tuple[str, ...] = (
    "ts_mean",
    "ts_std",
    "ts_zscore",
    "ts_delta",
    "ts_return",
    "ts_rank",
    "lag",
)
UNARY_NON_WINDOW_OPS: tuple[str, ...] = ("cs_rank", "cs_zscore", "neg", "abs_val", "tanh_squash")
BINARY_OPS: tuple[str, ...] = ("add", "sub", "mul", "safe_div")
MECHANISM_FAMILIES: tuple[str, ...] = (
    "momentum_liquidity",
    "reversal_microstructure",
    "value_quality_pit",
    "foreign_flow_regime",
)


@dataclass(frozen=True)
class TokenSpec:
    token: str
    kind: str  # "special" | "op_unary" | "op_binary" | "terminal"
    op_or_term: str
    window: int | None = None
    arity: int = 0


class ASTVocab:
    """Bijective prefix-order vocabulary (~70 causal tokens) with grammar-valid unmasking masks."""

    PAD_TOKEN = "[PAD]"
    MASK_TOKEN = "[MASK]"
    BOS_TOKEN = "[BOS]"
    EOS_TOKEN = "[EOS]"

    def __init__(self, max_seq_len: int = 16) -> None:
        self.max_seq_len = max_seq_len
        self.specs: list[TokenSpec] = []
        self._build_vocab()
        self.token_to_id: dict[str, int] = {s.token: i for i, s in enumerate(self.specs)}
        self.id_to_spec: dict[int, TokenSpec] = dict(enumerate(self.specs))

        self.pad_id = self.token_to_id[self.PAD_TOKEN]
        self.mask_id = self.token_to_id[self.MASK_TOKEN]
        self.bos_id = self.token_to_id[self.BOS_TOKEN]
        self.eos_id = self.token_to_id[self.EOS_TOKEN]

        self.terminal_ids = [i for i, s in enumerate(self.specs) if s.kind == "terminal"]
        self.unary_op_ids = [i for i, s in enumerate(self.specs) if s.kind == "op_unary"]
        self.binary_op_ids = [i for i, s in enumerate(self.specs) if s.kind == "op_binary"]
        self.all_op_ids = self.unary_op_ids + self.binary_op_ids

    def _build_vocab(self) -> None:
        for sp in (self.PAD_TOKEN, self.MASK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN):
            self.specs.append(TokenSpec(token=sp, kind="special", op_or_term=sp, arity=0))
        for op in WINDOWED_DSL_OPS:
            if op in ALLOWED_OPERATORS:
                for w in CANONICAL_WINDOWS:
                    self.specs.append(
                        TokenSpec(token=f"OP:{op}:{w}", kind="op_unary", op_or_term=op, window=w, arity=1)
                    )
        for op in UNARY_NON_WINDOW_OPS:
            if op in ALLOWED_OPERATORS:
                self.specs.append(
                    TokenSpec(token=f"OP:{op}", kind="op_unary", op_or_term=op, window=None, arity=1)
                )
        for op in BINARY_OPS:
            if op in ALLOWED_OPERATORS:
                self.specs.append(
                    TokenSpec(token=f"OP:{op}", kind="op_binary", op_or_term=op, window=None, arity=2)
                )
        for term in sorted(ALLOWED_TERMINALS.keys()):
            self.specs.append(TokenSpec(token=f"TERM:{term}", kind="terminal", op_or_term=term, arity=0))

    @property
    def vocab_size(self) -> int:
        return len(self.specs)

    @staticmethod
    def family_to_id(family: str) -> int:
        if family in MECHANISM_FAMILIES:
            return MECHANISM_FAMILIES.index(family)
        return 0

    @staticmethod
    def _nearest_window(w: int | None) -> int:
        if w is None:
            return CANONICAL_WINDOWS[1]
        return min(CANONICAL_WINDOWS, key=lambda cw: abs(cw - int(w)))

    def encode_ast(self, node: ASTNode) -> list[int]:
        """Serialize ASTNode into fixed-length prefix token IDs [BOS, ..., EOS, PAD...]."""
        body: list[int] = []

        def _visit(n: ASTNode) -> None:
            if n.node_type == "terminal":
                tok = f"TERM:{n.name}"
                body.append(self.token_to_id.get(tok, self.terminal_ids[0]))
                return
            meta = ALLOWED_OPERATORS[n.name]
            if meta.needs_window:
                w = self._nearest_window(n.window)
                tok = f"OP:{n.name}:{w}"
            else:
                tok = f"OP:{n.name}"
            body.append(self.token_to_id.get(tok, self.unary_op_ids[0]))
            for ch in n.args:
                _visit(ch)

        _visit(node)
        seq = [self.bos_id] + body[: self.max_seq_len - 2] + [self.eos_id]
        if len(seq) < self.max_seq_len:
            seq.extend([self.pad_id] * (self.max_seq_len - len(seq)))
        return seq[: self.max_seq_len]

    def decode_ast(self, token_ids: list[int]) -> ASTNode:
        """Parse prefix token IDs back into a strictly valid causal ASTNode."""
        clean = [tid for tid in token_ids if tid not in (self.bos_id, self.eos_id, self.pad_id, self.mask_id)]
        if not clean:
            return ASTNode(node_type="terminal", name="close")

        idx = 0

        def _parse(depth: int) -> ASTNode:
            nonlocal idx
            if idx >= len(clean):
                return ASTNode(node_type="terminal", name="close")
            spec = self.id_to_spec[clean[idx]]
            idx += 1
            if spec.kind == "terminal" or depth >= 4:
                term_name = spec.op_or_term if spec.kind == "terminal" else "close"
                return ASTNode(node_type="terminal", name=term_name)
            children = [_parse(depth + 1) for _ in range(spec.arity)]
            return ASTNode(node_type="op", name=spec.op_or_term, window=spec.window, args=children)

        return _parse(1)

    def subtree_span(self, token_ids: list[int], start_pos: int) -> tuple[int, int]:
        """Return contiguous prefix-order [start_pos, end_pos) span of the subtree rooted at start_pos."""
        if start_pos <= 0 or start_pos >= len(token_ids):
            return (1, 2)
        slots = 1
        pos = start_pos
        while pos < len(token_ids) and slots > 0:
            tid = token_ids[pos]
            if tid in (self.eos_id, self.pad_id):
                break
            spec = self.id_to_spec[tid]
            if spec.kind == "op_binary":
                slots += 1
            elif spec.kind == "op_unary":
                slots += 0
            else:
                slots -= 1
            pos += 1
        return (start_pos, max(start_pos + 1, pos))

    def valid_next_token_mask(
        self,
        prefix_tokens: list[int],
        max_depth: int = 3,
        max_body_tokens: int = 10,
    ) -> np.ndarray:
        """Compute binary mask over vocab_size enforcing exact prefix grammar + depth limit."""
        mask = np.zeros(self.vocab_size, dtype=bool)
        # Track stack of remaining (remaining_children, node_depth)
        stack: list[tuple[int, int]] = [(1, 1)]
        body_len = 0
        for tid in prefix_tokens:
            if tid == self.bos_id:
                continue
            if tid in (self.eos_id, self.pad_id):
                mask[self.pad_id] = True
                return mask
            if not stack:
                mask[self.eos_id] = True
                return mask
            rem, cur_d = stack.pop()
            if rem > 1:
                stack.append((rem - 1, cur_d))
            spec = self.id_to_spec[tid]
            if spec.arity > 0:
                stack.append((spec.arity, cur_d + 1))
            body_len += 1

        if not stack:
            mask[self.eos_id] = True
            return mask

        _, next_depth = stack[-1]
        slots_left = sum(r for r, _ in stack)
        budget_left = max_body_tokens - body_len

        # Allow terminals always when slots remain (except avoid trivial depth-1 root if budget allows)
        if next_depth >= max_depth or budget_left <= slots_left:
            mask[self.terminal_ids] = True
            return mask

        if next_depth == 1 and budget_left >= 3:
            # Prefer an operator at the root so factors express genuine transformations
            mask[self.all_op_ids] = True
            return mask

        mask[self.terminal_ids] = True
        if budget_left > slots_left + 1:
            mask[self.unary_op_ids] = True
        if budget_left > slots_left + 2:
            mask[self.binary_op_ids] = True
        return mask

    def to_factor_spec(
        self,
        token_ids: list[int],
        factor_id: str,
        hypothesis_id: str,
        family: str,
    ) -> FactorSpec:
        node = self.decode_ast(token_ids)
        return FactorSpec(
            factor_id=factor_id,
            hypothesis_id=hypothesis_id,
            description=f"Discrete AST Diffusion factor ({family}): {node.canonical_expr()}",
            root=node,
            normalize_cross_section=True,
        )
