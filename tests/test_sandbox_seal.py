"""Sealed Sandbox Tamper Detection, Data Zone Guards, Single-Shot Test Ledger, and VN Microstructure Tests."""

from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import pytest

from vn_rsi.data import SnapshotManager, SyntheticVNMarketProvider, load_market_rules
from vn_rsi.eval import SealedEvaluator
from vn_rsi.sandbox import (
    DataZoneAccessError,
    EvaluatorSeal,
    SealedTestReuseError,
    TamperDetectedError,
)
from vn_rsi.utils import load_yaml


@pytest.fixture()
def sealed_setup(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    rules = load_market_rules(repo_root / "configs" / "market_rules.yaml")
    sandbox_cfg = copy.deepcopy(load_yaml(repo_root / "configs" / "sandbox.yaml"))
    sandbox_cfg["zones"] = {
        "search": {"start": "2021-01-04", "end": "2021-12-31", "agent_access": "full"},
        "validation": {"start": "2022-01-04", "end": "2022-06-30", "agent_access": "scalar"},
        "sealed_test": {"start": "2022-07-15", "end": "2022-12-30", "agent_access": "once"},
    }
    sandbox_cfg["ledger_path"] = str(tmp_path / "ledger.jsonl")

    bundle = SyntheticVNMarketProvider().fetch_bundle("vn30", "2021-01-04", "2022-12-30", rules, seed=55)
    snap = SnapshotManager(tmp_path / "snaps").write_snapshot("s1", bundle)

    dummy_protected = tmp_path / "protected_rule.yaml"
    dummy_protected.write_text("lock: true\n", encoding="utf-8")

    evaluator = SealedEvaluator(snap, rules, sandbox_cfg, protected_paths=[dummy_protected])
    return evaluator, dummy_protected


def test_data_zone_guard_blocks_validation_and_sealed_test_raw_access(sealed_setup) -> None:
    evaluator, _ = sealed_setup
    s_bars = evaluator.zone_guard.get_agent_visible_bars(evaluator.snapshot.bars, "search")
    assert s_bars["date"].max() <= pd.Timestamp("2021-12-31")

    with pytest.raises(DataZoneAccessError):
        evaluator.zone_guard.get_agent_visible_bars(evaluator.snapshot.bars, "validation")

    with pytest.raises(DataZoneAccessError):
        evaluator.zone_guard.get_agent_visible_bars(evaluator.snapshot.bars, "sealed_test")


def test_sealed_test_ledger_enforces_single_look_per_candidate(sealed_setup) -> None:
    evaluator, _ = sealed_setup
    sig = evaluator.feature_panel["close"].pct_change(10, fill_method=None)

    # Without allow_sealed=True -> blocked
    with pytest.raises(PermissionError):
        evaluator.evaluate_signal_matrix(sig, zone="sealed_test", candidate_id="c1", candidate_hash_override="hash_abc")

    # First authorized evaluation succeeds
    out1 = evaluator.evaluate_signal_matrix(
        sig, zone="sealed_test", candidate_id="c1", candidate_hash_override="hash_abc", allow_sealed=True
    )
    assert "net_sharpe" in out1.metrics

    # Second evaluation of the SAME frozen candidate hash is strictly refused!
    with pytest.raises(SealedTestReuseError, match="already scored"):
        evaluator.evaluate_signal_matrix(
            sig, zone="sealed_test", candidate_id="c1_retry", candidate_hash_override="hash_abc", allow_sealed=True
        )


def test_evaluator_seal_detects_code_or_config_tampering(sealed_setup) -> None:
    evaluator, dummy_protected = sealed_setup
    assert len(evaluator.seal_hash) == 64

    # Simulate illicit modification of protected evaluator/config file
    dummy_protected.write_text("lock: hacked_by_agent\n", encoding="utf-8")
    with pytest.raises(TamperDetectedError, match="tamper check FAILED"):
        _ = evaluator.seal_hash


def test_canary_null_signal_and_seal_standalone(sealed_setup, tmp_path: Path) -> None:
    evaluator, dummy_protected = sealed_setup
    # Restore clean seal via fresh EvaluatorSeal check
    dummy_protected.write_text("lock: true\n", encoding="utf-8")
    seal = EvaluatorSeal([dummy_protected])
    assert seal.verify_intact() == evaluator.seal_hash
    canary = evaluator.run_canary_checks(seed=123)
    assert canary["passed"] is True
