"""Typed Causal DSL AST Verifier and Operator Causality Tests (Phase 3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from vn_rsi.data import SyntheticVNMarketProvider, load_market_rules
from vn_rsi.data.qa import DataQASuite
from vn_rsi.dsl import (
    ASTNode,
    ASTValidationError,
    ASTVerifier,
    FactorSpec,
    compute_feature_panel,
    evaluate_factor_spec,
    factor_spec_json_schema,
)


def test_ast_verifier_accepts_valid_causal_spec() -> None:
    spec = FactorSpec(
        factor_id="valid_mom_liq",
        hypothesis_id="h1",
        description="Risk-adjusted 10d momentum",
        root=ASTNode(
            node_type="op",
            name="safe_div",
            args=[
                ASTNode(node_type="op", name="ts_return", window=10, args=[ASTNode(node_type="terminal", name="close")]),
                ASTNode(node_type="op", name="ts_std", window=20, args=[ASTNode(node_type="terminal", name="return_1d")]),
            ],
        ),
    )
    verifier = ASTVerifier(max_depth=4)
    verifier.verify(spec)
    schema = factor_spec_json_schema()
    assert schema["title"] == "FactorSpec"


@pytest.mark.parametrize(
    "bad_root,err_match",
    [
        (
            ASTNode(node_type="op", name="lag", window=-1, args=[ASTNode(node_type="terminal", name="close")]),
            "window in \\[1,",
        ),
        (
            ASTNode(node_type="op", name="ts_mean", window=0, args=[ASTNode(node_type="terminal", name="close")]),
            "window in \\[1,",
        ),
        (
            ASTNode(node_type="terminal", name="future_return_5d"),
            "Forbidden/non-causal",
        ),
        (
            ASTNode(node_type="op", name="eval_python", args=[ASTNode(node_type="terminal", name="close")]),
            "Forbidden/non-causal",
        ),
        (
            ASTNode(node_type="op", name="unknown_op", args=[ASTNode(node_type="terminal", name="close")]),
            "Unknown operator",
        ),
    ],
)
def test_ast_verifier_rejects_illegal_or_non_causal_nodes(bad_root: ASTNode, err_match: str) -> None:
    spec = FactorSpec(factor_id="bad", hypothesis_id="h_bad", description="bad", root=bad_root)
    with pytest.raises(ASTValidationError, match=err_match):
        ASTVerifier(max_depth=4).verify(spec)


def test_dsl_evaluation_satisfies_shift_invariance() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    rules = load_market_rules(repo_root / "configs" / "market_rules.yaml")
    bundle = SyntheticVNMarketProvider().fetch_bundle("vn30", "2022-01-03", "2022-09-30", rules, seed=7)
    mini_bars = bundle.bars[bundle.bars["symbol"].isin(["VCB", "FPT", "HPG"])].copy().reset_index(drop=True)
    mini_fund = bundle.fundamentals_pit[bundle.fundamentals_pit["symbol"].isin(["VCB", "FPT", "HPG"])].copy()

    spec = FactorSpec(
        factor_id="combo",
        hypothesis_id="h2",
        description="Foreign flow + PIT quality",
        root=ASTNode(
            node_type="op",
            name="add",
            args=[
                ASTNode(node_type="op", name="ts_mean", window=5, args=[ASTNode(node_type="terminal", name="foreign_net_ratio")]),
                ASTNode(node_type="op", name="cs_zscore", args=[ASTNode(node_type="terminal", name="pit_value_quality")]),
            ],
        ),
    )

    def _eval_fn(b_df):  # type: ignore[no-untyped-def]
        p = compute_feature_panel(b_df, mini_fund)
        m = evaluate_factor_spec(spec, p)
        st = m.stack(future_stack=True)
        return b_df.set_index(["date", "symbol"]).index.map(st).to_series().fillna(0.0)

    assert DataQASuite.verify_feature_causality_shift_test(mini_bars, _eval_fn) is True
