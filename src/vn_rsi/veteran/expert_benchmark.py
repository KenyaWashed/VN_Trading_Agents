"""30+ Scenario 10-Year Veteran Cognitive Reasoning & Calibration Benchmark Suite.

Evaluates MAVE-Q v4's VeteranReasoningEngine against three baselines:
1. `NaiveFOMOAgent`: Chases 3-day breakouts & high volume, ignores T+2.5 / PIT traps, overconfident probabilities.
2. `StaticMomentumAnalyst`: Fixed 20d momentum rule with constant uncalibrated confidence.
3. `GenericRuleAnalyst`: Simple drawdown filter without Pre-Mortem floor-lock / PIT / foreign distribution checks.

Objective quantitative metrics across >= 30 multi-regime scenarios:
- Decision Accuracy (%) vs. ground-truth optimal action class (Defensive/Avoid vs. Constructive/Buy)
- Hidden-Risk Detection F1 / Recall (%) across floor-lock, bull-trap, foreign pull-dump, PIT lag, & distress traps
- Causal Consistency Rate (%): strict rejection of unannounced PIT fundamentals & look-ahead traps
- Probability Calibration: Brier Score (lower is better) & Expected Calibration Error (ECE, lower is better)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from vn_rsi.veteran.reasoning_engine import MarketContextSnapshot, VeteranReasoningEngine


@dataclass
class VeteranScenario:
    scenario_id: str
    category: str
    description: str
    context: MarketContextSnapshot
    expected_safe_direction: str  # "AVOID_OR_CASH" | "CONSTRUCTIVE_BUY" | "DEFENSIVE_HOLD"
    expected_hidden_risks: list[str]
    true_outperform_outcome: int  # 1 if constructive trade outperforms net of T+2.5 costs, else 0


def build_30_plus_veteran_scenarios() -> list[VeteranScenario]:
    """Construct 32 realistic, multi-regime, adversarial Vietnamese & emerging-market trading scenarios."""
    scenarios: list[VeteranScenario] = []

    # Group 1 (6 scenarios): Clean Bull-Expansion High-Conviction Leaders (true outcome = 1)
    bull_specs = [
        ("S01", "VCB", "HOSE", 0.055, 0.012, -0.01, 0.72, 0.92, 0.015, 0.09, 1.25, 0.07, 1.4, 0.14),
        ("S02", "FPT", "HOSE", 0.068, 0.018, -0.005, 0.78, 0.88, 0.020, 0.12, 1.30, 0.09, 1.7, 0.12),
        ("S03", "MBB", "HOSE", 0.048, 0.010, -0.015, 0.68, 0.95, 0.012, 0.08, 1.15, 0.06, 1.2, 0.16),
        ("S04", "HPG", "HOSE", 0.062, 0.014, -0.010, 0.74, 0.98, 0.018, 0.10, 1.28, 0.08, 1.1, 0.15),
        ("S05", "ACB", "HOSE", 0.042, 0.009, -0.012, 0.66, 0.90, 0.010, 0.07, 1.10, 0.05, 1.5, 0.18),
        ("S06", "SSI", "HOSE", 0.075, 0.020, 0.0, 0.80, 1.02, 0.022, 0.13, 1.35, 0.10, 0.9, 0.17),
    ]
    for sid, sym, ex, mr20, mr5, dd, br, vr, sr3, sr20, tsh, ff, pz, tp in bull_specs:
        scenarios.append(
            VeteranScenario(
                scenario_id=sid,
                category="bull_clean_leader",
                description=f"Clean bull expansion with broad breadth, steady foreign inflow, and announced PIT quality in {sym}",
                context=MarketContextSnapshot(
                    as_of_date="2023-07-14",
                    symbol=sym,
                    exchange=ex,
                    mkt_r20=mr20,
                    mkt_r5=mr5,
                    mkt_dd60=dd,
                    breadth_20=br,
                    vol_ratio=vr,
                    stock_ret3=sr3,
                    stock_ret20=sr20,
                    turnover_shock_20d=tsh,
                    foreign_net_ratio_5d=ff,
                    pit_value_quality_z=pz,
                    pit_is_announced=True,
                    pit_announcement_lag_days=34,
                    dist_to_floor=0.065,
                    dist_to_ceiling=0.042,
                    intraday_range=0.021,
                    trauma_prob=tp,
                    expected_mkt_5d=0.016,
                ),
                expected_safe_direction="CONSTRUCTIVE_BUY",
                expected_hidden_risks=[],
                true_outperform_outcome=1,
            )
        )

    # Group 2 (6 scenarios): T+2.5 Floor-Lock Cascade & Liquidity Crunch Crashes (true outcome = 0)
    crash_specs = [
        ("S07", "NVL_STRESS", "HOSE", -0.095, -0.048, -0.15, 0.18, 1.75, -0.14, -0.25, 0.45, -0.09, 0.004, 0.08, 0.68),
        ("S08", "FLC_HALT", "HOSE", -0.075, -0.040, -0.13, 0.22, 1.65, -0.12, -0.24, 0.35, -0.07, 0.002, 0.07, 0.64),
        ("S09", "ROS_DELIST", "HOSE", -0.082, -0.035, -0.14, 0.20, 1.55, -0.10, -0.28, 0.30, -0.05, 0.005, 0.06, 0.62),
        ("S10", "VIC", "HOSE", -0.065, -0.032, -0.12, 0.25, 1.60, -0.08, -0.16, 1.40, -0.08, 0.009, 0.055, 0.58),
        ("S11", "VHM", "HOSE", -0.058, -0.028, -0.115, 0.28, 1.52, -0.07, -0.14, 1.35, -0.06, 0.012, 0.05, 0.55),
        ("S12", "HNX_SMALL", "HNX", -0.088, -0.045, -0.16, 0.15, 1.85, -0.16, -0.26, 0.40, -0.04, 0.006, 0.075, 0.72),
    ]
    for sid, sym, ex, mr20, mr5, dd, br, vr, sr3, sr20, tsh, ff, dfl, fhr, tp in crash_specs:
        exp_risks = ["t25_floor_lock_trap"]
        if sr20 <= -0.18:
            exp_risks.append("distress_delisting_halt_risk")
        scenarios.append(
            VeteranScenario(
                scenario_id=sid,
                category="floor_lock_liquidity_crunch",
                description=f"Severe drawdown & floor-lock cascade risk in {sym} under T+2.5 lock",
                context=MarketContextSnapshot(
                    as_of_date="2022-10-21",
                    symbol=sym,
                    exchange=ex,
                    mkt_r20=mr20,
                    mkt_r5=mr5,
                    mkt_dd60=dd,
                    breadth_20=br,
                    vol_ratio=vr,
                    floor_hit_rate_5d=fhr,
                    stock_ret3=sr3,
                    stock_ret20=sr20,
                    turnover_shock_20d=tsh,
                    foreign_net_ratio_5d=ff,
                    pit_value_quality_z=-0.4,
                    dist_to_floor=dfl,
                    dist_to_ceiling=0.12,
                    intraday_range=0.055,
                    trauma_prob=tp,
                    expected_mkt_5d=-0.022,
                ),
                expected_safe_direction="AVOID_OR_CASH",
                expected_hidden_risks=exp_risks,
                true_outperform_outcome=0,
            )
        )

    # Group 3 (5 scenarios): Retail Blow-Off Bull Traps & Ceiling-Locked Chasing (true outcome = 0)
    blowoff_specs = [
        ("S13", "GVR", "HOSE", 0.032, 0.034, -0.02, 0.56, 1.32, 0.135, 0.22, 2.65, -0.04, 0.004, 0.48),
        ("S14", "POW", "HOSE", 0.025, 0.030, -0.03, 0.52, 1.38, 0.115, 0.18, 2.35, -0.03, 0.005, 0.46),
        ("S15", "VRE", "HOSE", 0.018, 0.028, -0.04, 0.48, 1.40, 0.105, 0.15, 2.20, -0.05, 0.006, 0.50),
        ("S16", "UPCOM_HOT", "UPCOM", 0.030, 0.036, -0.02, 0.54, 1.45, 0.210, 0.35, 3.10, -0.02, 0.003, 0.52),
        ("S17", "MWG", "HOSE", 0.015, 0.027, -0.03, 0.50, 1.30, 0.095, 0.14, 2.10, -0.07, 0.007, 0.49),
    ]
    for sid, sym, ex, mr20, mr5, dd, br, vr, sr3, sr20, tsh, ff, dcl, tp in blowoff_specs:
        exp_risks = ["blowoff_bull_trap_t25_supply", "ceiling_unfillable_chase_trap"]
        if ff <= -0.06:
            exp_risks.append("foreign_distribution_pull_and_dump")
        scenarios.append(
            VeteranScenario(
                scenario_id=sid,
                category="blowoff_ceiling_trap",
                description=f"Parabolic 3-day blow-off near ceiling with volume surge in {sym} (T+2.5 supply trap)",
                context=MarketContextSnapshot(
                    as_of_date="2023-09-18",
                    symbol=sym,
                    exchange=ex,
                    mkt_r20=mr20,
                    mkt_r5=mr5,
                    mkt_dd60=dd,
                    breadth_20=br,
                    vol_ratio=vr,
                    stock_ret3=sr3,
                    stock_ret20=sr20,
                    turnover_shock_20d=tsh,
                    foreign_net_ratio_5d=ff,
                    pit_value_quality_z=0.1,
                    dist_to_floor=0.068,
                    dist_to_ceiling=dcl,
                    intraday_range=0.062,
                    trauma_prob=tp,
                    expected_mkt_5d=-0.011,
                ),
                expected_safe_direction="AVOID_OR_CASH",
                expected_hidden_risks=exp_risks,
                true_outperform_outcome=0,
            )
        )

    # Group 4 (5 scenarios): Foreign Institutional Pull-and-Dump ("Kéo xả khối ngoại") (true outcome = 0)
    for i, (sym, sr3, ff, tsh) in enumerate(
        [("MSN", 0.055, -0.12, 1.75), ("VNM", 0.042, -0.10, 1.60), ("SAB", 0.048, -0.09, 1.68), ("STB", 0.060, -0.14, 1.95), ("VJC", 0.038, -0.08, 1.55)],
        start=18,
    ):
        scenarios.append(
            VeteranScenario(
                scenario_id=f"S{i}",
                category="foreign_pull_and_dump",
                description=f"Retail price lift masking heavy foreign institutional distribution in {sym}",
                context=MarketContextSnapshot(
                    as_of_date="2023-11-08",
                    symbol=sym,
                    exchange="HOSE",
                    mkt_r20=0.008,
                    mkt_r5=0.015,
                    mkt_dd60=-0.04,
                    breadth_20=0.46,
                    vol_ratio=1.22,
                    stock_ret3=sr3,
                    stock_ret20=0.03,
                    turnover_shock_20d=tsh,
                    foreign_net_ratio_5d=ff,
                    pit_value_quality_z=-0.2,
                    dist_to_floor=0.052,
                    dist_to_ceiling=0.025,
                    intraday_range=0.038,
                    trauma_prob=0.38,
                    expected_mkt_5d=-0.004,
                ),
                expected_safe_direction="AVOID_OR_CASH",
                expected_hidden_risks=["foreign_distribution_pull_and_dump"],
                true_outperform_outcome=0,
            )
        )

    # Group 5 (5 scenarios): PIT Fundamental Announcement-Lag Look-Ahead Traps (true outcome = 0)
    for i, (sym, lag_d) in enumerate(
        [("TCB", 0), ("VPB", -3), ("HDB", 0), ("TPB", -5), ("GAS", 0)],
        start=23,
    ):
        scenarios.append(
            VeteranScenario(
                scenario_id=f"S{i}",
                category="pit_announcement_lag_trap",
                description=f"Unannounced fiscal quarter-end fundamental jump in {sym} (lag={lag_d}d -> look-ahead hazard)",
                context=MarketContextSnapshot(
                    as_of_date="2024-01-05",
                    symbol=sym,
                    exchange="HOSE",
                    mkt_r20=-0.005,
                    mkt_r5=0.002,
                    mkt_dd60=-0.05,
                    breadth_20=0.45,
                    vol_ratio=1.18,
                    stock_ret3=0.04,
                    stock_ret20=-0.02,
                    turnover_shock_20d=1.45,
                    foreign_net_ratio_5d=-0.07,
                    pit_value_quality_z=2.40,  # Tempting fake/unannounced ROE jump!
                    pit_announcement_lag_days=lag_d,
                    pit_is_announced=False,
                    dist_to_floor=0.040,
                    dist_to_ceiling=0.045,
                    intraday_range=0.032,
                    trauma_prob=0.36,
                    expected_mkt_5d=-0.003,
                ),
                expected_safe_direction="AVOID_OR_CASH",
                expected_hidden_risks=["pit_announcement_lag_trap", "foreign_distribution_pull_and_dump"],
                true_outperform_outcome=0,
            )
        )

    # Group 6 (5 scenarios): Sideways / Post-Pullback Quality Accumulation & Calm Defense (true outcome = 1)
    sideways_specs = [
        ("S28", "BID", 0.006, -0.004, -0.025, 0.51, 0.94, -0.015, 0.045, 0.92, 0.055, 1.35, 0.19),
        ("S29", "CTG", 0.010, -0.002, -0.020, 0.53, 0.96, -0.012, 0.052, 0.88, 0.048, 1.25, 0.20),
        ("S30", "PLX", 0.004, -0.005, -0.030, 0.49, 0.92, -0.018, 0.035, 0.85, 0.042, 1.15, 0.21),
        ("S31", "BVH", 0.008, 0.001, -0.018, 0.52, 0.89, -0.010, 0.040, 0.90, 0.038, 1.20, 0.18),
        ("S32", "VIB", 0.012, 0.003, -0.015, 0.54, 0.93, -0.008, 0.060, 0.95, 0.050, 1.30, 0.17),
    ]
    for sid, sym, mr20, mr5, dd, br, vr, sr3, sr20, tsh, ff, pz, tp in sideways_specs:
        scenarios.append(
            VeteranScenario(
                scenario_id=sid,
                category="post_pullback_quality_accumulation",
                description=f"Calm sideways accumulation after mild 3d pullback with strong announced PIT quality & foreign buying in {sym}",
                context=MarketContextSnapshot(
                    as_of_date="2024-03-12",
                    symbol=sym,
                    exchange="HOSE",
                    mkt_r20=mr20,
                    mkt_r5=mr5,
                    mkt_dd60=dd,
                    breadth_20=br,
                    vol_ratio=vr,
                    stock_ret3=sr3,
                    stock_ret20=sr20,
                    turnover_shock_20d=tsh,
                    foreign_net_ratio_5d=ff,
                    pit_value_quality_z=pz,
                    pit_is_announced=True,
                    pit_announcement_lag_days=36,
                    dist_to_floor=0.062,
                    dist_to_ceiling=0.055,
                    intraday_range=0.018,
                    trauma_prob=tp,
                    expected_mkt_5d=0.009,
                ),
                expected_safe_direction="CONSTRUCTIVE_BUY",
                expected_hidden_risks=[],
                true_outperform_outcome=1,
            )
        )

    return scenarios


def compute_brier_and_ece(probs: list[float], outcomes: list[int], n_bins: int = 5) -> tuple[float, float]:
    """Compute Brier Score and Expected Calibration Error (ECE) in [0, 1] (lower is better)."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if len(p) == 0:
        return 0.0, 0.0
    brier = float(np.mean((p - y) ** 2))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p <= hi if i == n_bins - 1 else p < hi)
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(float(np.mean(p[mask])) - float(np.mean(y[mask])))
    return round(brier, 5), round(ece, 5)


def _baseline_predict(kind: str, ctx: MarketContextSnapshot) -> tuple[str, float, list[str], bool]:
    """Return (direction, win_prob, detected_risks, respected_pit_causality) for comparison baselines."""
    if kind == "naive_fomo":
        # Chases 3d spikes & unannounced ROE jumps, ignores T+2.5 & floor locks, severely overconfident!
        if ctx.stock_ret3 > 0.02 or ctx.pit_value_quality_z > 0.5:
            return "CONSTRUCTIVE_BUY", 0.88, [], ctx.pit_is_announced
        return "AVOID_OR_CASH", 0.72, [], True
    if kind == "static_momentum":
        # Pure 20d momentum rule, uncalibrated confidence, no microstructure trap detection
        if ctx.stock_ret20 > 0.03:
            return "CONSTRUCTIVE_BUY", 0.78, [], True
        return "AVOID_OR_CASH", 0.30, [], True
    # generic_rule: checks macro drawdown only, misses blow-off, foreign pull-dump, and PIT lag
    risks: list[str] = []
    if ctx.mkt_dd60 < -0.10:
        risks.append("t25_floor_lock_trap")
        return "AVOID_OR_CASH", 0.25, risks, True
    if ctx.stock_ret20 > 0.02 and ctx.pit_value_quality_z > 0.0:
        return "CONSTRUCTIVE_BUY", 0.72, risks, ctx.pit_is_announced
    return "AVOID_OR_CASH", 0.45, risks, True


@dataclass
class ExpertBenchmarkReport:
    n_scenarios: int
    system_scores: dict[str, dict[str, Any]]
    veteran_sample_cases: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_scenarios": self.n_scenarios,
            "system_scores": self.system_scores,
            "veteran_sample_cases": self.veteran_sample_cases,
        }


def evaluate_veteran_benchmark(
    engine: VeteranReasoningEngine | None = None,
    scenarios: list[VeteranScenario] | None = None,
) -> ExpertBenchmarkReport:
    """Evaluate VeteranReasoningEngine vs 3 baselines across all 32 complex market scenarios."""
    eng = engine or VeteranReasoningEngine()
    scens = scenarios or build_30_plus_veteran_scenarios()

    def _map_dir(action: str) -> str:
        if action in ("BUY_HIGH_CONVICTION", "BUY_MODERATE"):
            return "CONSTRUCTIVE_BUY"
        if action in ("CASH_SHIELD", "REDUCE_OR_AVOID"):
            return "AVOID_OR_CASH"
        return "DEFENSIVE_HOLD"

    systems = ("MAVE-Q_v4_Veteran", "Generic_Rule_Analyst", "Static_Momentum_Analyst", "Naive_FOMO_Agent")
    scores: dict[str, dict[str, Any]] = {}
    sample_cases: list[dict[str, Any]] = []

    for sys_name in systems:
        correct_dir = 0
        causal_ok = 0
        tp_risk = fp_risk = fn_risk = 0
        probs: list[float] = []
        outcomes: list[int] = []

        for sc in scens:
            if sys_name == "MAVE-Q_v4_Veteran":
                resp = eng.analyze(sc.context)
                pred_dir = _map_dir(resp.action)
                prob = resp.calibrated_win_prob
                det_risks = set(resp.detected_hidden_risks)
                pit_ok = (not sc.context.pit_is_announced and "PIT_UNANNOUNCED_FUNDAMENTAL_LEAKAGE" in resp.causal_violations_flagged) or sc.context.pit_is_announced
                if sc.scenario_id in ("S02", "S07", "S13", "S23"):
                    sample_cases.append(
                        {
                            "scenario_id": sc.scenario_id,
                            "category": sc.category,
                            "response": resp.to_dict(),
                            "markdown": resp.render_markdown(),
                        }
                    )
            else:
                kind = (
                    "naive_fomo"
                    if sys_name == "Naive_FOMO_Agent"
                    else ("static_momentum" if sys_name == "Static_Momentum_Analyst" else "generic_rule")
                )
                pred_dir, prob, det_list, pit_ok = _baseline_predict(kind, sc.context)
                det_risks = set(det_list)

            if pred_dir == sc.expected_safe_direction:
                correct_dir += 1
            if pit_ok:
                causal_ok += 1

            exp_set = set(sc.expected_hidden_risks)
            tp_risk += len(det_risks & exp_set)
            fp_risk += len(det_risks - exp_set)
            fn_risk += len(exp_set - det_risks)
            probs.append(prob)
            outcomes.append(sc.true_outperform_outcome)

        prec = tp_risk / max(1, tp_risk + fp_risk)
        rec = tp_risk / max(1, tp_risk + fn_risk)
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        brier, ece = compute_brier_and_ece(probs, outcomes, n_bins=5)
        scores[sys_name] = {
            "decision_accuracy_pct": round(100.0 * correct_dir / len(scens), 2),
            "causal_consistency_pct": round(100.0 * causal_ok / len(scens), 2),
            "hidden_risk_precision": round(prec, 4),
            "hidden_risk_recall": round(rec, 4),
            "hidden_risk_f1": round(f1, 4),
            "brier_score": brier,
            "ece": ece,
        }

    return ExpertBenchmarkReport(
        n_scenarios=len(scens),
        system_scores=scores,
        veteran_sample_cases=sample_cases,
    )
