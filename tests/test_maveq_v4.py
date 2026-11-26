"""Causality (future-perturbation shift invariance), VRAM/param budget, and Zero-Val-Burn tests for MAVE-Q v4."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vn_rsi.data import SnapshotManager, SyntheticVNMarketProvider, load_market_rules
from vn_rsi.dsl.ast_spec import compute_feature_panel
from vn_rsi.eval import SealedEvaluator
from vn_rsi.models.maveq_v4_veteran import (
    ControllerParams,
    MAVEQv4Engine,
    VeteranCaseMemory,
    compute_causal_microstructure_state,
)
from vn_rsi.utils import load_yaml


@pytest.fixture(scope="module")
def mini_v4_setup(tmp_path_factory: pytest.TempPathFactory):
    repo_root = Path(__file__).resolve().parents[1]
    tmp = tmp_path_factory.mktemp("v4_test")
    rules = load_market_rules(repo_root / "configs" / "market_rules.yaml")
    sandbox_cfg = copy.deepcopy(load_yaml(repo_root / "configs" / "sandbox.yaml"))
    sandbox_cfg["zones"] = {
        "search": {"start": "2021-01-04", "end": "2021-12-31", "agent_access": "full"},
        "validation": {"start": "2022-01-04", "end": "2022-06-30", "agent_access": "scalar"},
        "sealed_test": {"start": "2022-07-15", "end": "2022-12-30", "agent_access": "once"},
    }
    sandbox_cfg["ledger_path"] = str(tmp / "ledger.jsonl")
    bundle = SyntheticVNMarketProvider().fetch_bundle("vn30", "2021-01-04", "2022-12-30", rules, seed=88)
    snap = SnapshotManager(tmp / "snaps").write_snapshot("s_v4", bundle)
    cfg = copy.deepcopy(load_yaml(repo_root / "configs" / "maveq_v4.yaml"))
    cfg["evolution"].update(
        {"rounds": 2, "latent_candidates_per_round": 16, "screen_per_round": 5, "diffu_grpo_epochs_per_round": 2}
    )
    return snap, rules, sandbox_cfg, cfg, tmp


def test_maveq_v4_zero_lookahead_future_perturbation_invariance(mini_v4_setup) -> None:
    """Perturbing ALL bars strictly after cutoff t* MUST NOT change MAVE-Q v4 signals on dates <= t*."""
    snap, rules, sandbox_cfg, cfg, _ = mini_v4_setup
    ev = SealedEvaluator(snap, rules, sandbox_cfg)
    eng = MAVEQv4Engine(ev, cfg, preset="cpu_fast")
    _ = eng.run(run_id="t_v4_shift", score_sealed=False)

    basis = eng._select_diverse_basis(max_basis=4)
    assert len(basis) >= 1
    weights = [1.0 / len(basis)] * len(basis)
    cp = ControllerParams(name="test_ctrl", trauma_cash_thresh=0.45, dd_cash_thresh=-0.10, bull_top_k_cap=5)

    cutoff = pd.Timestamp("2022-04-01")
    pert_bars = snap.bars.copy()
    fmask = pert_bars["date"] > cutoff
    for col in ("open", "high", "low", "close", "raw_close", "volume", "turnover_vnd", "foreign_net_ratio"):
        pert_bars.loc[fmask, col] = pert_bars.loc[fmask, col] * 2.75 + 500.0

    panel_orig = ev.feature_panel
    panel_pert = compute_feature_panel(pert_bars, snap.fundamentals_pit)

    out1 = eng.build_full_causal_signal(basis, weights, cp, panel_orig)
    out2 = eng.build_full_causal_signal(basis, weights, cp, panel_pert)

    past = pd.DatetimeIndex(panel_orig["close"].index)
    past = past[past <= cutoff]

    s1 = out1.sealed_signal_wide.reindex(index=past).to_numpy(dtype=float)
    s2 = out2.sealed_signal_wide.reindex(index=past).to_numpy(dtype=float)
    both_nan = np.isnan(s1) & np.isnan(s2)
    eq = np.isclose(np.nan_to_num(s1, nan=0.0), np.nan_to_num(s2, nan=0.0), atol=1e-9, rtol=0.0)
    assert bool(np.all(both_nan | eq)), "Future perturbation leaked into MAVE-Q v4 past signals!"

    w1 = out1.kelly_weights_wide.reindex(index=past).to_numpy(dtype=float)
    w2 = out2.kelly_weights_wide.reindex(index=past).to_numpy(dtype=float)
    assert np.allclose(w1, w2, atol=1e-9)


def test_maveq_v4_vram_param_budget_zero_val_burn_and_checkpoint_roundtrip(mini_v4_setup) -> None:
    snap, rules, sandbox_cfg, cfg, tmp = mini_v4_setup
    ev = SealedEvaluator(snap, rules, sandbox_cfg)
    eng = MAVEQv4Engine(ev, cfg, preset="cpu_fast")

    # Search panel isolation: validation/test bars must never enter search_panel
    assert str(eng.search_panel["close"].index.max().date()) <= "2021-12-31"

    res = eng.run(run_id="t_v4_budget", score_sealed=False)
    # Zero-Validation-Burn check: total official evaluator trials strictly <= 12!
    assert res.validation_queries_used <= int(cfg["evolution"]["max_validation_queries"])
    assert res.total_evaluator_trials <= 12
    assert res.search_candidates_screened >= 5
    # Free-tier VRAM & parameter budget check (<15M params, <4500 MB VRAM on cpu_fast/T4)
    assert res.specialist_total_params < 15_000_000
    assert res.estimated_vram_mb < 4500.0
    assert "calmar_ratio" in res.champion_validation_extended
    assert res.champion_sealed_test_metrics.get("note") == "sealed_test_not_requested_dev_mode"

    # Checkpoint round-trip including VeteranCaseMemory
    ckpts = eng.save_checkpoints(tmp / "v4_ckpts")
    for p in ckpts.values():
        assert Path(p).exists()
    eng2 = MAVEQv4Engine(SealedEvaluator(snap, rules, sandbox_cfg), cfg, preset="cpu_fast")
    eng2.load_checkpoints(tmp / "v4_ckpts")
    assert np.allclose(eng.case_memory._z_states, eng2.case_memory._z_states)


def test_veteran_case_memory_nearest_analog_retrieval(mini_v4_setup) -> None:
    snap, rules, sandbox_cfg, _, _ = mini_v4_setup
    ev = SealedEvaluator(snap, rules, sandbox_cfg)
    s_bars = ev.zone_guard.get_agent_visible_bars(snap.bars, "search")
    s_panel = compute_feature_panel(s_bars, snap.fundamentals_pit[snap.fundamentals_pit["announcement_date"] <= pd.Timestamp("2021-12-31")])
    mem = VeteranCaseMemory(k_neighbors=10).fit_on_search(s_panel)
    st = compute_causal_microstructure_state(s_panel)
    analogs = mem.retrieve_nearest_cases(st.iloc[-1].to_numpy(dtype=float), top_k=3)
    assert len(analogs) == 3
    assert all("analog_date" in a and "was_trauma" in a for a in analogs)
