"""Tests for diffu-GRPO, kaggle_max_100m scaling config, OrthogonalRewardEngine, and FrontierCombatArena."""

from __future__ import annotations

import copy
from pathlib import Path

from vn_rsi.data import SnapshotManager, SyntheticVNMarketProvider, load_market_rules
from vn_rsi.eval import SealedEvaluator
from vn_rsi.models import (
    ASTDiffusionConfig,
    DiscreteASTDiffusion,
    FrontierCombatArena,
    HRMConfig,
)
from vn_rsi.utils import load_yaml


def test_diffu_grpo_advantage_update_and_kaggle_max_config() -> None:
    cfg_max_d = ASTDiffusionConfig.from_preset("kaggle_max_100m")
    cfg_max_h = HRMConfig.from_preset("kaggle_max_100m")
    assert cfg_max_d.d_model == 768 and cfg_max_d.n_layers == 14
    assert cfg_max_h.d_hidden == 768 and cfg_max_h.n_layers_per_block == 10

    diff = DiscreteASTDiffusion(ASTDiffusionConfig.from_preset("cpu_fast", seed=7))
    s1 = diff.sample_factor("foreign_flow_regime", "f1", "h1", seed=1)
    s2 = diff.sample_factor("reversal_microstructure", "f2", "h2", seed=2)
    s3 = diff.sample_factor("value_quality_pit", "f3", "h3", seed=3)

    rollouts = [
        {"family": "foreign_flow_regime", "factor_spec": s1.model_dump(), "orthogonal_reward": 0.14},
        {"family": "reversal_microstructure", "factor_spec": s2.model_dump(), "orthogonal_reward": -0.08},
        {"family": "value_quality_pit", "factor_spec": s3.model_dump(), "orthogonal_reward": 0.05},
    ]
    stats = diff.fit_diffu_grpo(rollouts, epochs=4)
    assert stats["steps"] == 4
    assert stats["samples"] == 3
    assert stats["max_reward"] == 0.14


def test_frontier_combat_arena_end_to_end(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    rules = load_market_rules(repo_root / "configs" / "market_rules.yaml")
    sandbox_cfg = copy.deepcopy(load_yaml(repo_root / "configs" / "sandbox.yaml"))
    sandbox_cfg["zones"] = {
        "search": {"start": "2021-01-04", "end": "2021-12-31", "agent_access": "full"},
        "validation": {"start": "2022-01-04", "end": "2022-06-30", "agent_access": "scalar"},
        "sealed_test": {"start": "2022-07-15", "end": "2022-12-30", "agent_access": "once"},
    }
    sandbox_cfg["ledger_path"] = str(tmp_path / "combat_ledger.jsonl")

    combat_cfg = copy.deepcopy(load_yaml(repo_root / "configs" / "frontier_combat.yaml"))
    combat_cfg["combat_loop"]["max_rounds"] = 2
    combat_cfg["combat_loop"]["candidates_per_round_latent"] = 10
    combat_cfg["combat_loop"]["top_k_evaluate_on_search"] = 3
    combat_cfg["combat_loop"]["diffu_grpo_epochs_per_round"] = 3
    combat_cfg["combat_loop"]["hrm_epochs_per_round"] = 3

    bundle = SyntheticVNMarketProvider().fetch_bundle("vn30", "2021-01-04", "2022-12-30", rules, seed=44)
    snap = SnapshotManager(tmp_path / "snaps").write_snapshot("s_combat", bundle)
    evaluator = SealedEvaluator(snap, rules, sandbox_cfg)

    arena = FrontierCombatArena(evaluator, combat_cfg, preset="cpu_fast")
    res = arena.run_combat(run_id="test_combat")
    assert len(res.frontier_validation_scores) == 3
    assert len(res.combat_rounds) >= 1
    assert "net_sharpe" in res.champion_validation_metrics
    assert "net_sharpe" in res.champion_sealed_test_metrics
