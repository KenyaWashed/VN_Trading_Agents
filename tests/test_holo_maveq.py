"""Unit & causality tests for Holo-MAVE-Q v3 and all 7 same-topic SOTA alpha-mining competitors."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from vn_rsi.data import SnapshotManager, SyntheticVNMarketProvider, load_market_rules
from vn_rsi.dsl.ast_spec import ASTNode
from vn_rsi.eval import SealedEvaluator
from vn_rsi.models.alpha_mining_baselines import (
    Alpha101StyleLibrary,
    AlphaGPTIterativeMiner,
    GPAlphaMiner,
    MAPElitesAlphaMiner,
    MCTSAlphaMiner,
    PPOAlphaMiner,
    RandomDSLSearch,
)
from vn_rsi.models.ast_vocab import ASTVocab
from vn_rsi.models.holo_maveq import (
    HoloHRMReasoner,
    HoloMAVEQEngine,
    TopologicalTreeDiffusion,
    compute_prefix_tree_coords,
    dominant_economic_niche,
    exact_dimensional_exponents,
    is_typed_stationary_ast,
)
from vn_rsi.utils import load_yaml


@pytest.fixture(scope="module")
def mini_evaluator_factory(tmp_path_factory: pytest.TempPathFactory):
    repo_root = Path(__file__).resolve().parents[1]
    tmp = tmp_path_factory.mktemp("holo")
    rules = load_market_rules(repo_root / "configs" / "market_rules.yaml")
    sandbox_cfg = copy.deepcopy(load_yaml(repo_root / "configs" / "sandbox.yaml"))
    sandbox_cfg["zones"] = {
        "search": {"start": "2021-01-04", "end": "2021-12-31", "agent_access": "full"},
        "validation": {"start": "2022-01-04", "end": "2022-06-30", "agent_access": "scalar"},
        "sealed_test": {"start": "2022-07-15", "end": "2022-12-30", "agent_access": "once"},
    }
    sandbox_cfg["ledger_path"] = str(tmp / "ledger.jsonl")
    bundle = SyntheticVNMarketProvider().fetch_bundle("vn30", "2021-01-04", "2022-12-30", rules, seed=55)
    snap = SnapshotManager(tmp / "snaps").write_snapshot("s_holo", bundle)
    return lambda: SealedEvaluator(snap, rules, sandbox_cfg)


def test_exact_dimensional_type_system_and_algebraic_guard() -> None:
    t = lambda n: ASTNode(node_type="terminal", name=n)  # noqa: E731
    op = lambda n, *a, w=None: ASTNode(node_type="op", name=n, window=w, args=list(a))  # noqa: E731

    # turnover_vnd = Price^1 * Volume^1 -> dividing by (close * volume) yields dimensionless (0, 0)!
    vwap_ratio = op("safe_div", t("turnover_vnd"), op("mul", t("close"), t("volume")))
    assert exact_dimensional_exponents(vwap_ratio) == (0, 0)
    assert is_typed_stationary_ast(vwap_ratio)

    # Non-stationary raw price rank is rejected
    assert not is_typed_stationary_ast(op("cs_zscore", t("close")))
    # Algebraic tautologies x - x and x / x are rejected
    assert not is_typed_stationary_ast(op("sub", t("pit_roe"), t("pit_roe")))
    assert not is_typed_stationary_ast(op("safe_div", t("close"), t("close")))
    assert dominant_economic_niche(op("mul", t("foreign_net_ratio"), t("return_1d"))) == "flow"


def test_tree_coords_and_non_commutative_holo_hrm() -> None:
    vocab = ASTVocab(max_seq_len=16)
    t = lambda n: ASTNode(node_type="terminal", name=n)  # noqa: E731
    op = lambda n, *a, w=None: ASTNode(node_type="op", name=n, window=w, args=list(a))  # noqa: E731

    a = op("safe_div", op("ts_return", t("close"), w=15), op("ts_std", t("return_1d"), w=20))
    b = op("safe_div", op("ts_std", t("return_1d"), w=20), op("ts_return", t("close"), w=15))
    seq_a = vocab.encode_ast(a)
    parents, slots, depths = compute_prefix_tree_coords(vocab, seq_a)
    # Root safe_div is at pos 1; left child at pos 2 (slot 1); right child at pos 4 (slot 2)
    assert slots[1] == 0 and slots[2] == 1 and slots[4] == 2
    assert parents[2] == seq_a[1] and parents[4] == seq_a[1]
    assert depths[2] == 2 and depths[3] == 3

    # Holo-HRM must distinguish safe_div(A, B) from safe_div(B, A)!
    hrm = HoloHRMReasoner(preset="cpu_fast", seed=7)
    emb_a = hrm.encode_tree_recursive(a)
    emb_b = hrm.encode_tree_recursive(b)
    assert float(abs(emb_a - emb_b).max()) > 1e-3

    diff = TopologicalTreeDiffusion(preset="cpu_fast", seed=7)
    sp = diff.sample_factor("momentum_liquidity", "f1", "h1", max_depth=4, seed=11)
    sp2 = diff.attribution_guided_infill(sp, weak_start_pos=2, family="value_quality_pit", factor_id="f2", hypothesis_id="h2", seed=12)
    assert sp.root.depth() >= 1 and sp2.root.depth() >= 1


def test_holo_maveq_and_all_7_competitors_end_to_end(mini_evaluator_factory) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = copy.deepcopy(load_yaml(repo_root / "configs" / "holo_maveq.yaml"))
    cfg["loop"].update(
        {"rounds": 2, "latent_candidates_per_round": 16, "screen_per_round": 5, "diffu_grpo_epochs_per_round": 3}
    )
    ev_holo = mini_evaluator_factory()
    res = HoloMAVEQEngine(ev_holo, cfg, preset="cpu_fast").run(run_id="t_holo", score_sealed=True)
    assert res.champion_expressions
    assert "net_sharpe" in res.champion_validation_metrics
    assert "net_sharpe" in res.champion_sealed_test_metrics
    # Verify search_panel isolation: validation/sealed_test dates never exist in engine.search_panel
    eng_check = HoloMAVEQEngine(mini_evaluator_factory(), cfg, preset="cpu_fast")
    assert eng_check.search_panel["close"].index.max() <= eng_check.safe_dates.max() + eng_check.evaluator.horizon_days * eng_check.safe_dates.freq if eng_check.safe_dates.freq else True
    assert str(eng_check.search_panel["close"].index.max().date()) <= "2021-12-31"

    assert PPOAlphaMiner(mini_evaluator_factory(), episodes=2, batch_size=4, seed=1).run("t_ppo").search_evaluations >= 4
    assert MCTSAlphaMiner(mini_evaluator_factory(), budget=6, checkpoint_every=3, seed=1).run("t_mcts").search_evaluations == 6
    assert MAPElitesAlphaMiner(mini_evaluator_factory(), budget=6, init_random=3, checkpoint_every=3, seed=1).run("t_qd").search_evaluations == 6
    assert AlphaGPTIterativeMiner(mini_evaluator_factory(), rounds=2, ideas_per_round=3, seed=1).run("t_agpt").search_evaluations >= 4
    assert GPAlphaMiner(mini_evaluator_factory(), population=5, generations=2, seed=1).run("t_gp").search_evaluations >= 5
    assert RandomDSLSearch(mini_evaluator_factory(), budget=6, checkpoint_every=3, seed=1).run("t_rnd").search_evaluations == 6
    assert len(Alpha101StyleLibrary(mini_evaluator_factory()).run("t_a101").champion_expressions) >= 4


def test_cloud_presets_and_holo_checkpoint_roundtrip(mini_evaluator_factory, tmp_path: Path) -> None:
    from vn_rsi.models.cloud_presets import VALID_PRESETS, detect_cloud_compute, get_preset_spec

    env = detect_cloud_compute()
    assert env["recommended_preset"] in VALID_PRESETS
    for preset in VALID_PRESETS:
        spec = get_preset_spec(preset)
        assert spec.diff_d_model >= 64
        assert spec.hrm_d_hidden >= 64

    repo_root = Path(__file__).resolve().parents[1]
    cfg = copy.deepcopy(load_yaml(repo_root / "configs" / "holo_maveq.yaml"))
    cfg["loop"].update({"rounds": 1, "latent_candidates_per_round": 10, "screen_per_round": 4, "diffu_grpo_epochs_per_round": 2})

    eng1 = HoloMAVEQEngine(mini_evaluator_factory(), cfg, preset="cpu_fast")
    eng1.run(run_id="t_ckpt", score_sealed=False)
    ckpts = eng1.save_checkpoints(tmp_path / "ckpts")
    assert Path(ckpts["tree_mdlm"]).exists()
    assert Path(ckpts["holo_hrm"]).exists()
    assert eng1.diffusion.estimate_training_vram_mb() > 0.0
    assert eng1.hrm.estimate_training_vram_mb() > 0.0

    eng2 = HoloMAVEQEngine(mini_evaluator_factory(), cfg, preset="cpu_fast")
    eng2.load_checkpoints(tmp_path / "ckpts")
    assert float(abs(eng1.diffusion.edge_logits - eng2.diffusion.edge_logits).max()) < 1e-12
    assert float(abs(eng1.hrm.tok_val_ema - eng2.hrm.tok_val_ema).max()) < 1e-12
