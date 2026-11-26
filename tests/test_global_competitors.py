"""Zero look-ahead causality & end-to-end tests for Global SOTA Competitors & Multi-Market Loader (Task 03)."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vn_rsi.benchmarks.global_competitors import (
    AlphaForgeDoubleEnsembleCompetitor,
    FinAgentCompetitor,
    FinMemCompetitor,
    RDAgentQCompetitor,
    TradingAgentsCompetitor,
    run_global_competitor_arena,
)
from vn_rsi.benchmarks.multi_market import (
    MARKET_PROFILES,
    MultiMarketLoader,
    build_market_rules_for_profile,
)
from vn_rsi.dsl.ast_spec import compute_feature_panel
from vn_rsi.eval import SealedEvaluator
from vn_rsi.utils import load_yaml


@pytest.fixture(scope="module")
def mini_mm_env(tmp_path_factory: pytest.TempPathFactory):
    tmp = tmp_path_factory.mktemp("mm_global")
    loader = MultiMarketLoader(tmp / "snaps")
    env = loader.load_or_build(
        profile="VN_HOSE_T25",
        start_date="2021-01-04",
        end_date="2022-12-30",
        mode="offline_fixture",
        seed=77,
    )
    repo_root = Path(__file__).resolve().parents[1]
    sandbox_cfg = copy.deepcopy(load_yaml(repo_root / "configs" / "sandbox.yaml"))
    sandbox_cfg["zones"] = {
        "search": {"start": "2021-01-04", "end": "2021-12-31", "agent_access": "full"},
        "validation": {"start": "2022-01-04", "end": "2022-06-30", "agent_access": "scalar"},
        "sealed_test": {"start": "2022-07-15", "end": "2022-12-30", "agent_access": "once"},
    }
    sandbox_cfg["ledger_path"] = str(tmp / "ledger.jsonl")
    return env, sandbox_cfg


def _perturb_bars_after(bars: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    pert = bars.copy()
    mask = pert["date"] > cutoff
    for col in ("open", "high", "low", "close", "raw_close", "volume", "turnover_vnd", "foreign_net_ratio"):
        if col in pert.columns:
            pert.loc[mask, col] = pert.loc[mask, col] * 2.31 + 123.4
    return pert


def test_zero_lookahead_shift_invariance_all_5_global_competitors(mini_mm_env) -> None:
    """Perturbing ALL market bars after cutoff date t* MUST NOT change any signal on dates <= t*."""
    env, sandbox_cfg = mini_mm_env
    ev = SealedEvaluator(env.snapshot, env.market_rules, sandbox_cfg)
    panel_orig = ev.feature_panel
    all_dates = pd.DatetimeIndex(panel_orig["close"].index)
    # Cutoff inside validation zone (strictly after search_end so trained models are identical too!)
    cutoff = pd.Timestamp("2022-04-01")
    assert cutoff > pd.Timestamp(sandbox_cfg["zones"]["search"]["end"])

    pert_bars = _perturb_bars_after(env.snapshot.bars, cutoff)
    panel_pert = compute_feature_panel(pert_bars, env.snapshot.fundamentals_pit)
    past_dates = all_dates[all_dates <= cutoff]
    search_end = pd.Timestamp(sandbox_cfg["zones"]["search"]["end"])

    builders = {
        "TradingAgents": lambda p: TradingAgentsCompetitor(ev).build_signal(p),
        "FinMem": lambda p: FinMemCompetitor(ev).build_signal(p),
        "FinAgent": lambda p: FinAgentCompetitor(ev).build_signal(p),
        "RD-Agent(Q)": lambda p: RDAgentQCompetitor(ev, n_rounds=6, seed=7).build_signal(p, search_end=search_end)[0],
        "AlphaForge-DE": lambda p: AlphaForgeDoubleEnsembleCompetitor(ev, n_submodels=2, seed=7).build_signal(
            p, search_end=search_end
        )[0],
    }

    for name, fn in builders.items():
        s1 = fn(panel_orig).reindex(index=past_dates).to_numpy(dtype=float)
        s2 = fn(panel_pert).reindex(index=past_dates).to_numpy(dtype=float)
        both_nan = np.isnan(s1) & np.isnan(s2)
        eq = np.isclose(np.nan_to_num(s1, nan=0.0), np.nan_to_num(s2, nan=0.0), atol=1e-9, rtol=0.0)
        assert bool(np.all(both_nan | eq)), f"Look-ahead leakage detected in {name}!"


def test_global_competitor_arena_dev_mode_never_touches_sealed(mini_mm_env) -> None:
    env, sandbox_cfg = mini_mm_env
    results = run_global_competitor_arena(
        lambda: SealedEvaluator(env.snapshot, env.market_rules, sandbox_cfg),
        run_id="test_dev_arena",
        score_sealed=False,
        seed=42,
    )
    assert set(results.keys()) == {"TradingAgents", "FinMem", "FinAgent", "RD-Agent(Q)", "AlphaForge-DE"}
    for name, res in results.items():
        assert "net_sharpe" in res.validation_metrics, f"{name} missing validation net_sharpe"
        assert "calmar_ratio" in res.validation_extended, f"{name} missing extended calmar_ratio"
        assert res.sealed_test_metrics.get("note") == "sealed_test_not_requested_dev_mode"
        assert res.total_evaluator_trials <= 2


def test_multi_market_loader_all_profiles(tmp_path: Path) -> None:
    loader = MultiMarketLoader(tmp_path / "snaps")
    for prof in MARKET_PROFILES:
        rules = build_market_rules_for_profile(prof)
        env = loader.load_or_build(
            profile=prof,
            start_date="2022-01-03",
            end_date="2023-06-30",
            mode="offline_fixture",
            seed=101,
        )
        assert env.profile == prof
        assert env.is_synthetic is True
        assert len(env.snapshot.bars) > 2000
        assert len(env.snapshot.benchmark) > 100
        if prof == "US_LIQUID_T1":
            assert rules.settlement.min_holding_bars_eod == 1
            assert rules.illiquidity_impact.ceiling_buy_blocked is False
        elif prof == "CSI_EMERGING_T1":
            assert rules.settlement.min_holding_bars_eod == 1
            assert rules.get_exchange_limit("SSE") == pytest.approx(0.10)
        else:
            assert rules.settlement.min_holding_bars_eod == 3
            assert rules.get_exchange_limit("HOSE") == pytest.approx(0.07)
