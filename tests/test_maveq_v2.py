"""Tests for MAVE-Q v2 (QD archive, FATS, portfolio credit) and same-topic alpha-mining competitors."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from vn_rsi.data import SnapshotManager, SyntheticVNMarketProvider, load_market_rules
from vn_rsi.dsl.ast_spec import ASTNode, ASTVerifier, FactorSpec
from vn_rsi.eval import SealedEvaluator
from vn_rsi.models.alpha_mining_baselines import (
    Alpha101StyleLibrary,
    GPAlphaMiner,
    GrammarTreeSampler,
    RandomDSLSearch,
    alpha101_style_library,
)
from vn_rsi.models.maveq_v2 import (
    FamilyThompsonBandit,
    MAVEQv2Engine,
    dimension_of,
    dominant_information_source,
    is_dimensionally_stationary,
    negated_spec,
    smoothed_spec,
    turnover_bucket,
)
from vn_rsi.utils import load_yaml


@pytest.fixture(scope="module")
def small_evaluator_factory(tmp_path_factory: pytest.TempPathFactory):
    repo_root = Path(__file__).resolve().parents[1]
    tmp = tmp_path_factory.mktemp("v2")
    rules = load_market_rules(repo_root / "configs" / "market_rules.yaml")
    sandbox_cfg = copy.deepcopy(load_yaml(repo_root / "configs" / "sandbox.yaml"))
    sandbox_cfg["zones"] = {
        "search": {"start": "2021-01-04", "end": "2021-12-31", "agent_access": "full"},
        "validation": {"start": "2022-01-04", "end": "2022-06-30", "agent_access": "scalar"},
        "sealed_test": {"start": "2022-07-15", "end": "2022-12-30", "agent_access": "once"},
    }
    sandbox_cfg["ledger_path"] = str(tmp / "ledger.jsonl")
    bundle = SyntheticVNMarketProvider().fetch_bundle("vn30", "2021-01-04", "2022-12-30", rules, seed=44)
    snap = SnapshotManager(tmp / "snaps").write_snapshot("s_v2", bundle)
    return lambda: SealedEvaluator(snap, rules, sandbox_cfg)


def test_fats_and_sign_repair_wrappers_stay_causal() -> None:
    base = FactorSpec(
        factor_id="f", hypothesis_id="h", description="d",
        root=ASTNode(node_type="op", name="ts_return", window=5, args=[ASTNode(node_type="terminal", name="close")]),
    )
    sm = smoothed_spec(base, 5)
    assert sm.root.name == "ts_mean" and sm.root.window == 5
    assert smoothed_spec(base, 1) is base
    ng = negated_spec(base)
    assert ng.root.name == "neg"
    ASTVerifier(max_depth=5).verify(smoothed_spec(ng, 10))


def test_descriptors_and_bandit() -> None:
    t = lambda n: ASTNode(node_type="terminal", name=n)  # noqa: E731
    assert dimension_of(ASTNode(node_type="op", name="cs_rank", args=[t("high")])) == "!"
    assert is_dimensionally_stationary(ASTNode(node_type="op", name="ts_return", window=5, args=[t("close")]))
    assert not is_dimensionally_stationary(ASTNode(node_type="op", name="add", args=[t("pit_roe"), t("open")]))
    assert is_dimensionally_stationary(ASTNode(node_type="op", name="safe_div", args=[
        ASTNode(node_type="op", name="ts_std", window=20, args=[t("close")]), t("close")]))
    node = ASTNode(node_type="op", name="add", args=[
        ASTNode(node_type="terminal", name="pit_roe"), ASTNode(node_type="terminal", name="pit_pb_ratio")])
    assert dominant_information_source(node) == "fundamental"
    assert turnover_bucket(0.01) == 0 and turnover_bucket(0.5) == 3
    bandit = FamilyThompsonBandit(("a", "b"), np.random.default_rng(0))
    for _ in range(20):
        bandit.update("a", 0.10)
        bandit.update("b", -0.05)
    assert bandit.probs()[0] > 0.8


def test_grammar_sampler_and_alpha101_library_verify() -> None:
    sampler = GrammarTreeSampler(max_depth=4)
    rng = np.random.default_rng(3)
    ver = ASTVerifier(max_depth=5)
    for i in range(50):
        ver.verify(FactorSpec(factor_id=str(i), hypothesis_id="h", description="d", root=sampler.random_tree(rng)))
    for fid, root in alpha101_style_library():
        ver.verify(FactorSpec(factor_id=fid, hypothesis_id="h", description="d", root=root))


def test_maveq_v2_and_competitors_end_to_end(small_evaluator_factory) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = copy.deepcopy(load_yaml(repo_root / "configs" / "maveq_v2.yaml"))
    cfg["loop"].update({"rounds": 2, "latent_candidates_per_round": 12, "screen_per_round": 4,
                        "add_candidates_per_round": 2, "smoothing_grid": [1, 5],
                        "diffu_grpo_epochs_per_round": 2, "hrm_epochs_per_round": 2})
    res = MAVEQv2Engine(small_evaluator_factory(), cfg, preset="cpu_fast").run(run_id="t", score_sealed=True)
    assert res.champion_expressions
    assert "net_sharpe" in res.champion_validation_metrics
    assert "net_sharpe" in res.champion_sealed_test_metrics
    assert len(res.round_logs) == 2

    gp = GPAlphaMiner(small_evaluator_factory(), population=6, generations=2, seed=1).run("t_gp", score_sealed=True)
    assert gp.search_evaluations >= 6 and "net_sharpe" in gp.sealed_test_metrics
    rnd = RandomDSLSearch(small_evaluator_factory(), budget=8, checkpoint_every=4).run("t_rnd")
    assert rnd.search_evaluations == 8
    lib = Alpha101StyleLibrary(small_evaluator_factory()).run("t_lib")
    assert len(lib.champion_expressions) >= 4
