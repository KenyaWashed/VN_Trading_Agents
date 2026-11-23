"""Typed AST specification, causality verifier, and panel evaluator for <=8B constrained decoding."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from vn_rsi.data.qa import join_pit_fundamentals_causal
from vn_rsi.dsl.operators import ALLOWED_OPERATORS, ALLOWED_TERMINALS, apply_operator


class ASTValidationError(ValueError):
    """Raised when an agent-proposed factor specification violates causality or grammar constraints."""


class ASTNode(BaseModel):
    """Typed expression tree node emitted via JSON-schema constrained decoding."""

    node_type: str = Field(description="'terminal' or 'op'")
    name: str = Field(description="Terminal column name or registered causal operator name")
    window: int | None = Field(default=None, description="Positive lookback window (1..120) if operator requires window")
    args: list[ASTNode] = Field(default_factory=list, description="Child AST arguments")

    def depth(self) -> int:
        if not self.args:
            return 1
        return 1 + max(child.depth() for child in self.args)

    def canonical_expr(self) -> str:
        if self.node_type == "terminal":
            return self.name
        child_str = ", ".join(c.canonical_expr() for c in self.args)
        if self.window is not None:
            return f"{self.name}({child_str}, w={self.window})"
        return f"{self.name}({child_str})"

    def structural_fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_expr().encode("utf-8")).hexdigest()[:16]


ASTNode.model_rebuild()


class FactorSpec(BaseModel):
    """Constrained factor specification emitted by the Implementer agent."""

    factor_id: str
    hypothesis_id: str
    description: str
    root: ASTNode
    normalize_cross_section: bool = True

    def spec_hash(self) -> str:
        payload = {
            "expr": self.root.canonical_expr(),
            "normalize_cross_section": self.normalize_cross_section,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


class ASTVerifier:
    """Deterministic verifier enforcing causal grammar, positive windows, and max AST depth."""

    FORBIDDEN_TOKENS = {"future", "lead", "target", "label", "fiscal_period_end", "eval", "exec", "import", "__"}

    def __init__(self, max_depth: int = 4, max_window: int = 120) -> None:
        self.max_depth = max_depth
        self.max_window = max_window

    def verify(self, spec: FactorSpec) -> None:
        if spec.root.depth() > self.max_depth:
            raise ASTValidationError(
                f"AST depth {spec.root.depth()} exceeds max_depth={self.max_depth}: {spec.root.canonical_expr()}"
            )
        self._verify_node(spec.root)

    def _verify_node(self, node: ASTNode) -> None:
        low_name = node.name.lower()
        for bad in self.FORBIDDEN_TOKENS:
            if bad in low_name:
                raise ASTValidationError(f"Forbidden/non-causal identifier '{node.name}' detected in AST")

        if node.node_type == "terminal":
            if node.name not in ALLOWED_TERMINALS:
                raise ASTValidationError(
                    f"Unknown terminal '{node.name}'. Allowed terminals: {sorted(ALLOWED_TERMINALS)}"
                )
            if node.args:
                raise ASTValidationError(f"Terminal node '{node.name}' cannot have child arguments")
            if node.window is not None:
                raise ASTValidationError(f"Terminal node '{node.name}' cannot specify window={node.window}")
            return

        if node.node_type != "op":
            raise ASTValidationError(f"Invalid node_type '{node.node_type}'; expected 'terminal' or 'op'")

        if node.name not in ALLOWED_OPERATORS:
            raise ASTValidationError(
                f"Unknown operator '{node.name}'. Allowed operators: {sorted(ALLOWED_OPERATORS)}"
            )
        meta = ALLOWED_OPERATORS[node.name]
        if len(node.args) != meta.arity:
            raise ASTValidationError(
                f"Operator '{node.name}' requires {meta.arity} child args, got {len(node.args)}"
            )
        if meta.needs_window:
            if node.window is None or int(node.window) < 1 or int(node.window) > self.max_window:
                raise ASTValidationError(
                    f"Operator '{node.name}' requires integer window in [1, {self.max_window}], got {node.window}"
                )
        else:
            if node.window is not None:
                raise ASTValidationError(f"Operator '{node.name}' does not accept window parameter")

        for child in node.args:
            self._verify_node(child)


def factor_spec_json_schema() -> dict[str, Any]:
    """Export JSON Schema for constrained decoding with vLLM / llama.cpp / Ollama structured outputs."""
    return FactorSpec.model_json_schema()


def compute_feature_panel(bars: pd.DataFrame, fundamentals_pit: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Precompute wide (date x symbol) causal terminal matrices from snapshot bars + PIT fundamentals."""
    df = bars.sort_values(["date", "symbol"]).copy()
    pit_joined = join_pit_fundamentals_causal(df, fundamentals_pit)
    for col, src_col in [
        ("pit_roe", "roe"),
        ("pit_pb_ratio", "pb_ratio"),
        ("pit_revenue_yoy", "revenue_yoy"),
    ]:
        df[col] = pit_joined[src_col].to_numpy()
    df["pit_value_quality"] = (
        df["pit_roe"].fillna(0.08) / df["pit_pb_ratio"].clip(lower=0.3).fillna(1.8)
        + 0.25 * df["pit_revenue_yoy"].fillna(0.0)
    )

    panel: dict[str, pd.DataFrame] = {}
    for base_col in (
        "close",
        "open",
        "high",
        "low",
        "volume",
        "turnover_vnd",
        "ref_price",
        "ceiling_price",
        "floor_price",
        "foreign_net_ratio",
        "fol_room_pct",
        "pit_roe",
        "pit_pb_ratio",
        "pit_revenue_yoy",
        "pit_value_quality",
    ):
        panel[base_col] = df.pivot(index="date", columns="symbol", values=base_col).sort_index()

    panel["return_1d"] = panel["close"].pct_change(fill_method=None).fillna(0.0)
    panel["intraday_range"] = (panel["high"] - panel["low"]) / panel["close"].clip(lower=1.0)
    panel["close_to_open"] = (panel["close"] - panel["open"]) / panel["open"].clip(lower=1.0)
    panel["dist_to_ceiling"] = (panel["ceiling_price"] - panel["close"]).clip(lower=0.0) / panel["ref_price"].clip(lower=1.0)
    panel["dist_to_floor"] = (panel["close"] - panel["floor_price"]).clip(lower=0.0) / panel["ref_price"].clip(lower=1.0)
    adv20 = panel["turnover_vnd"].rolling(window=20, min_periods=3).mean().clip(lower=1.0)
    panel["turnover_shock_20d"] = panel["turnover_vnd"] / adv20
    return panel


def _eval_node(node: ASTNode, panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if node.node_type == "terminal":
        return panel[node.name]
    evaluated_args = [_eval_node(child, panel) for child in node.args]
    return apply_operator(node.name, evaluated_args, window=node.window)


def evaluate_factor_spec(
    spec: FactorSpec,
    panel: dict[str, pd.DataFrame],
    verifier: ASTVerifier | None = None,
) -> pd.DataFrame:
    """Verify causality and evaluate FactorSpec into a wide (date x symbol) signal matrix."""
    active_verifier = verifier or ASTVerifier(max_depth=5)
    active_verifier.verify(spec)
    raw = _eval_node(spec.root, panel)
    cleaned = raw.replace([np.inf, -np.inf], np.nan)
    if spec.normalize_cross_section:
        mu = cleaned.mean(axis=1)
        sd = cleaned.std(axis=1) + 1e-8
        cleaned = cleaned.sub(mu, axis=0).div(sd, axis=0)
    return cleaned.clip(lower=-5.0, upper=5.0)
