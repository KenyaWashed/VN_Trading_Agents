"""Tests for <=8B Model Constraint, Hard Budgets, Mutable Surface Guard, and Bi-Level RSI Smoke Run."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from vn_rsi.data import SnapshotManager, SyntheticVNMarketProvider, load_market_rules
from vn_rsi.eval import SealedEvaluator
from vn_rsi.llm import ModelSizeConstraintError, RuleGuidedSmallLMBackend
from vn_rsi.rsi import (
    BiLevelRSIRunner,
    GenomePatchProposal,
    ImmutableSurfaceViolationError,
    MutableSurfaceGuard,
)
from vn_rsi.utils import BudgetConfig, BudgetExceededError, BudgetTracker, load_yaml


def test_8b_model_size_hard_constraint() -> None:
    ok_llm = RuleGuidedSmallLMBackend(model_name="qwen3-8b", parameter_count_billions=8.0)
    assert ok_llm.parameter_count_billions == 8.0

    # >8B in main loop MUST fail
    with pytest.raises(ModelSizeConstraintError, match="violates main-loop <= 8.0B"):
        RuleGuidedSmallLMBackend(model_name="qwen-72b", parameter_count_billions=72.0, is_teacher=False)

    # >8B allowed ONLY when explicitly flagged as teacher with allow_teacher=True
    teacher = RuleGuidedSmallLMBackend(
        model_name="qwen-72b-teacher", parameter_count_billions=72.0, is_teacher=True, allow_teacher=True
    )
    assert teacher.is_teacher is True


def test_budget_tracker_enforces_hard_stops() -> None:
    tracker = BudgetTracker(BudgetConfig(max_tokens=100, max_wall_seconds=60.0, max_gpu_hours=0.01))
    tracker.record_usage(40, 50)
    with pytest.raises(BudgetExceededError, match="Token budget exceeded"):
        tracker.record_usage(10, 5)

    gpu_tracker = BudgetTracker(BudgetConfig(max_tokens=10000, max_wall_seconds=60.0, max_gpu_hours=0.001))
    with pytest.raises(BudgetExceededError, match="GPU-hour budget exceeded"):
        gpu_tracker.add_gpu_seconds(10.0)


def test_mutable_surface_guard_blocks_evaluator_and_cost_edits() -> None:
    guard = MutableSurfaceGuard(["planner_focus_weights", "preferred_windows", "best_of_n"])
    base_genome = {"planner_focus_weights": {"momentum_liquidity": 0.5}, "preferred_windows": [5, 15], "best_of_n": 2}

    valid = GenomePatchProposal(rationale="Increase verifier best-of-n.", updates={"best_of_n": 4})
    updated = guard.apply_patch(base_genome, valid)
    assert updated["best_of_n"] == 4

    for forbidden_key in ("evaluator", "transaction_costs", "sealed_test", "unlisted_secret_knob"):
        bad = GenomePatchProposal(rationale="Hack the evaluation rules.", updates={forbidden_key: 0.0})
        with pytest.raises(ImmutableSurfaceViolationError):
            guard.apply_patch(base_genome, bad)


def test_bilevel_rsi_mini_end_to_end(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    rules = load_market_rules(repo_root / "configs" / "market_rules.yaml")
    sandbox_cfg = copy.deepcopy(load_yaml(repo_root / "configs" / "sandbox.yaml"))
    sandbox_cfg["zones"] = {
        "search": {"start": "2021-01-04", "end": "2021-12-31", "agent_access": "full"},
        "validation": {"start": "2022-01-04", "end": "2022-06-30", "agent_access": "scalar"},
        "sealed_test": {"start": "2022-07-15", "end": "2022-12-30", "agent_access": "once"},
    }
    sandbox_cfg["ledger_path"] = str(tmp_path / "ledger.jsonl")
    agent_cfg = copy.deepcopy(load_yaml(repo_root / "configs" / "agent_v0.yaml"))
    llm_cfg = copy.deepcopy(load_yaml(repo_root / "configs" / "llm.yaml"))
    rsi_cfg = copy.deepcopy(load_yaml(repo_root / "configs" / "rsi.yaml"))
    rsi_cfg["outer_iterations"] = 1
    rsi_cfg["inner_rounds_per_eval"] = 1
    rsi_cfg["inner_hypotheses_per_round"] = 2
    rsi_cfg["selection_seeds"] = [11, 22]
    rsi_cfg["self_training"]["export_sft_trajectories"] = False

    bundle = SyntheticVNMarketProvider().fetch_bundle("vn30", "2021-01-04", "2022-12-30", rules, seed=88)
    snap = SnapshotManager(tmp_path / "snaps").write_snapshot("s_rsi", bundle)
    evaluator = SealedEvaluator(snap, rules, sandbox_cfg)
    llm = RuleGuidedSmallLMBackend()

    runner = BiLevelRSIRunner(evaluator, llm, llm, rsi_cfg, agent_cfg, llm_cfg)
    summary = runner.run_mode("full_rsi_8b", seed_override=99)
    assert summary.archive_size >= 1
    assert len(summary.iteration_logs) == 1
    assert summary.reward_hacking_attempts == 0
