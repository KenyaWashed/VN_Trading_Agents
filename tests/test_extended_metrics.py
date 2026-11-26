"""Comprehensive unit tests for Extended Quantitative Metrics Suite (Task 02)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from vn_rsi.benchmarks.extended_metrics import (
    REGIME_NAMES,
    calmar_ratio,
    capm_alpha_beta,
    classify_market_regimes,
    compute_extended_metrics,
    compute_ic_decay_curve,
    compute_regime_breakdown,
    cvar_expected_shortfall,
    deflated_sharpe_ratio_ext,
    floor_lock_trap_rate,
    hansen_spa_and_white_rc,
    information_ratio,
    max_drawdown_and_duration,
    omega_ratio,
    paired_block_bootstrap_metrics,
    probabilistic_sharpe_ratio,
    profit_factor,
    ulcer_index,
    win_rate_pct,
)
from vn_rsi.eval.evaluator import EvaluationOutcome


def _make_dates(n: int = 260) -> pd.DatetimeIndex:
    return pd.bdate_range("2023-01-02", periods=n)


def test_core_ratios_and_tail_risk_metrics() -> None:
    rng = np.random.default_rng(7)
    dates = _make_dates(260)
    bm = pd.Series(rng.normal(0.0008, 0.010, size=260), index=dates)
    strat = pd.Series(0.0018 + 0.85 * bm.to_numpy() + rng.normal(0.0, 0.004, size=260), index=dates)

    mdd, dur = max_drawdown_and_duration(strat)
    assert mdd <= 0.0
    assert dur >= 0

    cal = calmar_ratio(strat)
    assert cal > 0.0
    assert omega_ratio(strat) > 1.0
    assert information_ratio(strat, bm) > 0.0

    capm = capm_alpha_beta(strat, bm)
    assert capm["capm_alpha_ann"] > 0.05
    assert 0.55 < capm["capm_beta"] < 1.15
    assert 0.20 < capm["capm_r2"] <= 1.0

    var95, cvar95 = cvar_expected_shortfall(strat, alpha=0.05)
    assert cvar95 <= var95 <= 0.0
    assert ulcer_index(strat) >= 0.0
    assert 40.0 < win_rate_pct(strat) < 75.0
    assert profit_factor(strat) > 1.0


def test_dsr_psr_bootstrap_and_hansen_spa_white_rc() -> None:
    rng = np.random.default_rng(123)
    dates = _make_dates(300)
    bm = pd.Series(rng.normal(0.0002, 0.010, size=300), index=dates)
    strong = pd.Series(bm.to_numpy() + 0.0022 + rng.normal(0.0, 0.004, size=300), index=dates)
    noise = pd.Series(bm.to_numpy() + rng.normal(0.0, 0.006, size=300), index=dates)

    # More trials strictly decreases or preserves DSR
    dsr_1 = deflated_sharpe_ratio_ext(strong, n_trials=1)
    dsr_10 = deflated_sharpe_ratio_ext(strong, n_trials=10)
    dsr_200 = deflated_sharpe_ratio_ext(strong, n_trials=200)
    assert 0.0 <= dsr_200 <= dsr_10 <= dsr_1 <= 1.0

    psr = probabilistic_sharpe_ratio(strong, benchmark_sr_annual=0.0)
    assert psr > 0.90

    boot = paired_block_bootstrap_metrics(strong, bm, n_boot=300, seed=11)
    assert boot["sharpe_diff_mean"] > 0.5
    assert boot["p_one_sided_sharpe"] < 0.05

    spa = hansen_spa_and_white_rc({"strong": strong, "noise": noise}, bm, n_boot=300, seed=11)
    assert spa["best_model"] == "strong"
    assert spa["hansen_spa_p_value"] < 0.05
    assert spa["white_rc_p_value"] < 0.05


def test_ic_decay_curve_and_floor_lock_traps() -> None:
    rng = np.random.default_rng(99)
    dates = _make_dates(120)
    syms = [f"S{i}" for i in range(15)]
    rets = pd.DataFrame(rng.normal(0.0005, 0.015, size=(120, 15)), index=dates, columns=syms)
    close = 100.0 * (1.0 + rets).cumprod()
    # Predictive signal correlated with next 5-day return (for unit-testing IC decay computation)
    true_fwd5 = close.shift(-5) / close - 1.0
    sig = true_fwd5.fillna(0.0) + pd.DataFrame(rng.normal(0.0, 0.04, size=(120, 15)), index=dates, columns=syms)

    decay = compute_ic_decay_curve(sig, close, zone_dates=dates, horizons=(1, 3, 5, 10, 20))
    assert set(decay.keys()) == {"1d", "3d", "5d", "10d", "20d"}
    assert decay["5d"]["rank_ic"] > 0.25
    assert decay["5d"]["icir"] > 5.0

    traps = floor_lock_trap_rate(
        {"blocked_floor_sells": 6, "blocked_ceiling_buys": 4, "t25_settlement_blocks": 12},
        n_days=120,
        rebalance_every_days=3,
        top_k=6,
    )
    assert 0.0 < traps["floor_lock_trap_rate"] < 1.0
    assert 0.0 < traps["t25_lock_rate"] < 1.0


def test_four_regime_classification_and_extended_report_wrapper() -> None:
    dates = _make_dates(200)
    # Construct synthetic benchmark covering all 4 regimes: Bull -> Sideways -> HighVol Crunch -> Bear
    r_bull = np.full(55, 0.0035)
    r_side = np.tile([0.002, -0.002], 25)
    r_crunch = np.tile([-0.035, 0.025, -0.030, 0.020], 12)[:45]
    r_bear = np.full(50, -0.004)
    bm = pd.Series(np.concatenate([r_bull, r_side, r_crunch, r_bear]), index=dates)
    strat = pd.Series(np.where(bm > -0.01, bm + 0.001, 0.0), index=dates)

    regs = classify_market_regimes(bm)
    assert set(regs.unique()).issubset(set(REGIME_NAMES))
    assert "Bull" in regs.values
    assert "Bear_Crash" in regs.values

    breakdown = compute_regime_breakdown(strat, bm)
    assert set(breakdown.keys()) == set(REGIME_NAMES)
    assert breakdown["Bull"].n_days > 10

    outcome = EvaluationOutcome(
        candidate_id="test_strat",
        candidate_hash="abc123",
        zone="validation",
        metrics={
            "rank_ic": 0.08,
            "icir": 4.2,
            "ann_net_return": 0.45,
            "net_sharpe": 1.6,
            "net_sortino": 2.1,
            "max_drawdown": -0.10,
            "daily_turnover": 0.05,
            "deflated_sharpe_prob": 0.75,
        },
        daily_net_returns=strat,
        execution_stats={"blocked_floor_sells": 2, "blocked_ceiling_buys": 1, "t25_settlement_blocks": 3},
    )
    report = compute_extended_metrics(outcome, bm, n_trials=8, n_boot=150)
    d = report.to_dict()
    for key in (
        "calmar_ratio",
        "omega_ratio",
        "information_ratio",
        "capm_alpha_ann",
        "capm_beta",
        "max_drawdown_duration",
        "cvar_95",
        "ulcer_index",
        "floor_lock_trap_rate",
        "win_rate_pct",
        "profit_factor",
        "turnover_adjusted_net_alpha",
        "deflated_sharpe_prob",
        "probabilistic_sharpe_ratio",
        "bootstrap_vs_benchmark",
        "spa_white_rc",
        "regime_breakdown",
    ):
        assert key in d
