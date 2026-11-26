"""Global SOTA Trading Agent & Alpha-Mining Competitor Arena (Strictly Causal, Same Friction).

Faithfully implements the core decision mechanisms of five prominent international quantitative
trading agent & alpha-discovery systems under identical market rules and zero look-ahead:

1. `TradingAgentsCompetitor` (Xiao et al., 2024):
   Structured multi-role debate: Bull Researcher vs. Bear Researcher -> Risk Manager
   (conservative/neutral/aggressive volatility & drawdown gate) -> Trader synthesis.
2. `FinMemCompetitor` (Yu et al., 2024):
   Layered Memory (Shallow 3-5d microstructure, Intermediate 15-20d trend/flow, Deep 60d+ PIT
   fundamental/regime memory with exponential decay) + volatility-adaptive risk character profile.
3. `FinAgentCompetitor` (Zhang et al., KDD 2024):
   Dual-Level Reflection: Low-level signal-level rolling causal performance reflection +
   High-level regime/drawdown portfolio reflection over multi-modal price/volume/fundamental views.
4. `RDAgentQCompetitor` (Li et al., NeurIPS 2025):
   Multi-Armed Bandit (UCB1) scheduling exploration across economic factor hypotheses and
   downstream ML model families (Ridge vs. GBDT) strictly on the `search` zone.
5. `AlphaForgeDoubleEnsembleCompetitor` (Shi et al., AAAI 2025; Zhang et al., ICDM 2020):
   Iterative sample-reweighted & feature-selected neural/linear ensemble trained strictly on the
   `search` zone with dynamic causal IC-momentum combination weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from vn_rsi.benchmarks.extended_metrics import ExtendedMetricsReport, compute_extended_metrics
from vn_rsi.eval.evaluator import EvaluationOutcome, SealedEvaluator
from vn_rsi.eval.metrics import compute_rank_ic_series
from vn_rsi.sandbox.seal import SealedTestReuseError


def _cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean(axis=1)
    sd = df.std(axis=1) + 1e-8
    return df.sub(mu, axis=0).div(sd, axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-4.0, 4.0)


def _extract_causal_signals(panel: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Build strictly causal primitive views (information <= t only) from feature panel."""
    close = panel["close"]
    ret1 = panel["return_1d"].fillna(0.0)
    vol5 = ret1.rolling(5, min_periods=2).std().fillna(0.015)
    vol20 = ret1.rolling(20, min_periods=5).std().fillna(0.015)
    mom5 = close.pct_change(5, fill_method=None).fillna(0.0)
    mom15 = close.pct_change(15, fill_method=None).fillna(0.0)
    mom20 = close.pct_change(20, fill_method=None).fillna(0.0)
    tshock = panel["turnover_shock_20d"].fillna(1.0)
    fflow = panel["foreign_net_ratio"].fillna(0.0)
    pit_vq = panel["pit_value_quality"].fillna(0.0)
    pit_roe = panel["pit_roe"].fillna(0.08)
    pit_pb = panel["pit_pb_ratio"].fillna(1.8).clip(lower=0.3)
    intraday = panel["intraday_range"].fillna(0.02)
    dist_ceil = panel["dist_to_ceiling"].fillna(0.05)
    dist_flr = panel["dist_to_floor"].fillna(0.05)

    return {
        "mom_risk_adj_15": _cs_zscore(mom15 / (vol20 + 0.006)),
        "mom_20": _cs_zscore(mom20),
        "rev_overheat_3": _cs_zscore(-(close.pct_change(3, fill_method=None).fillna(0.0) * tshock.clip(0.5, 2.5))),
        "rev_5": _cs_zscore(-mom5),
        "flow_5": _cs_zscore(fflow.rolling(5, min_periods=2).mean().fillna(0.0)),
        "flow_15": _cs_zscore(fflow.rolling(15, min_periods=3).mean().fillna(0.0)),
        "pit_vq": _cs_zscore(pit_vq),
        "pit_roe_pb": _cs_zscore(pit_roe / pit_pb),
        "low_vol": _cs_zscore(-vol20),
        "micro_safety": _cs_zscore(dist_flr - intraday - 0.5 * (vol5 / (vol20 + 1e-4))),
        "ceil_headroom": _cs_zscore(dist_ceil),
        "liq_calm": _cs_zscore(-tshock.rolling(5, min_periods=2).mean().fillna(1.0)),
    }


def _causal_market_state(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Strictly causal aggregate market state (computed exclusively from <= t bars)."""
    ret1 = panel["return_1d"].fillna(0.0)
    mkt_ret = ret1.mean(axis=1)
    mkt_r20 = (1.0 + mkt_ret).rolling(20, min_periods=5).apply(np.prod, raw=True) - 1.0
    mkt_vol20 = mkt_ret.rolling(20, min_periods=5).std().fillna(0.012)
    mkt_wealth = (1.0 + mkt_ret).cumprod()
    mkt_dd60 = (mkt_wealth / mkt_wealth.rolling(60, min_periods=5).max().clip(lower=1e-8)) - 1.0
    breadth20 = (panel["close"].pct_change(20, fill_method=None) > 0.0).mean(axis=1).fillna(0.5)
    floor_prox = (panel["dist_to_floor"].fillna(0.05) < 0.015).mean(axis=1).rolling(3, min_periods=1).mean()
    return pd.DataFrame(
        {
            "mkt_ret": mkt_ret,
            "mkt_r20": mkt_r20.fillna(0.0),
            "mkt_vol20": mkt_vol20.fillna(0.012),
            "mkt_dd60": mkt_dd60.fillna(0.0),
            "breadth20": breadth20,
            "floor_prox": floor_prox.fillna(0.0),
        },
        index=ret1.index,
    )


@dataclass
class GlobalCompetitorOutcome:
    name: str
    paper_ref: str
    paradigm: str
    total_evaluator_trials: int
    validation_metrics: dict[str, float]
    validation_extended: dict[str, Any]
    sealed_test_metrics: dict[str, Any]
    sealed_test_extended: dict[str, Any]
    validation_daily_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float), repr=False)
    sealed_test_daily_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float), repr=False)
    notes: dict[str, Any] = field(default_factory=dict)


def _benchmark_series_for_zone(evaluator: SealedEvaluator, zone: str) -> pd.Series:
    bm_zone = evaluator.zone_guard._slice_zone(evaluator.snapshot.benchmark, zone).sort_values("date")
    return pd.Series(bm_zone["return_1d"].to_numpy(dtype=float), index=bm_zone["date"])


def _evaluate_signal_global(
    name: str,
    paper_ref: str,
    paradigm: str,
    signal_wide: pd.DataFrame,
    evaluator: SealedEvaluator,
    run_id: str,
    score_sealed: bool,
    notes: dict[str, Any] | None = None,
) -> GlobalCompetitorOutcome:
    val_out: EvaluationOutcome = evaluator.evaluate_signal_matrix(
        signal_wide=signal_wide,
        zone="validation",
        candidate_id=f"{name}_val",
        run_id=run_id,
    )
    bm_val = _benchmark_series_for_zone(evaluator, "validation")
    bars_val = evaluator.zone_guard._slice_zone(evaluator.snapshot.bars, "validation")
    val_ext: ExtendedMetricsReport = compute_extended_metrics(
        val_out,
        bm_val,
        bars_slice=bars_val,
        signal_wide=signal_wide,
        close_wide=evaluator.feature_panel["close"],
        n_trials=evaluator.n_trials_evaluated,
        rebalance_every_days=evaluator.rebalance_every_days,
        top_k=evaluator.top_k,
        n_boot=300,
    )

    test_metrics: dict[str, Any] = {"note": "sealed_test_not_requested_dev_mode"}
    test_ext_dict: dict[str, Any] = {}
    test_rets = pd.Series(dtype=float)
    if score_sealed:
        try:
            test_out = evaluator.evaluate_signal_matrix(
                signal_wide=signal_wide,
                zone="sealed_test",
                candidate_id=f"{name}_sealed",
                run_id=run_id,
                allow_sealed=True,
            )
            test_metrics = dict(test_out.metrics)
            test_rets = test_out.daily_net_returns
            bm_test = _benchmark_series_for_zone(evaluator, "sealed_test")
            bars_test = evaluator.zone_guard._slice_zone(evaluator.snapshot.bars, "sealed_test")
            test_ext_dict = compute_extended_metrics(
                test_out,
                bm_test,
                bars_slice=bars_test,
                signal_wide=signal_wide,
                close_wide=evaluator.feature_panel["close"],
                n_trials=evaluator.n_trials_evaluated,
                rebalance_every_days=evaluator.rebalance_every_days,
                top_k=evaluator.top_k,
                n_boot=300,
            ).to_dict()
        except SealedTestReuseError:
            test_metrics = {"note": "already_scored_in_sealed_ledger"}

    return GlobalCompetitorOutcome(
        name=name,
        paper_ref=paper_ref,
        paradigm=paradigm,
        total_evaluator_trials=int(evaluator.n_trials_evaluated),
        validation_metrics=dict(val_out.metrics),
        validation_extended=val_ext.to_dict(),
        sealed_test_metrics=test_metrics,
        sealed_test_extended=test_ext_dict,
        validation_daily_returns=val_out.daily_net_returns,
        sealed_test_daily_returns=test_rets,
        notes=notes or {},
    )


# ---------------------------------------------------------------------------
# 1. TradingAgents (Xiao et al., 2024): Bull vs Bear Debate + Risk Manager + Trader
# ---------------------------------------------------------------------------
class TradingAgentsCompetitor:
    """Bull vs. Bear Researcher structured debate mediated by Risk Manager & Trader (strictly causal)."""

    def __init__(self, evaluator: SealedEvaluator, seed: int = 2026) -> None:
        self.evaluator = evaluator
        self.seed = seed

    def build_signal(self, panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
        sigs = _extract_causal_signals(panel)
        mkt = _causal_market_state(panel)

        # Bull Researcher thesis: risk-adjusted trend, foreign accumulation, PIT fundamental quality
        bull_thesis = (
            0.40 * sigs["mom_risk_adj_15"]
            + 0.30 * sigs["flow_5"]
            + 0.30 * sigs["pit_vq"]
        )
        # Bear Researcher critique: overheat spike risk, volatility/floor fragility, downside risk
        bear_risk = (
            0.45 * (-sigs["rev_overheat_3"])
            + 0.35 * (-sigs["micro_safety"])
            + 0.20 * (-sigs["low_vol"])
        )

        # Causal Risk Manager debate weighting: Bear argument carries higher weight in drawdowns/high vol
        vol_stress = ((mkt["mkt_vol20"] - 0.012) / 0.012).clip(0.0, 1.5)
        dd_stress = (-mkt["mkt_dd60"] / 0.10).clip(0.0, 1.5)
        bear_weight = (0.35 + 0.30 * vol_stress + 0.25 * dd_stress).clip(0.25, 0.85)
        bull_weight = 1.0 - bear_weight

        # Trader synthesizes debate consensus across assets (strictly causal row-wise broadcast)
        raw_score = bull_thesis.mul(bull_weight, axis=0) - bear_risk.mul(bear_weight, axis=0)
        # Conservative Risk Manager cash veto under severe crash stress
        crash_veto = (mkt["mkt_dd60"] < -0.14) & (mkt["mkt_vol20"] > 0.022)
        out = raw_score.copy()
        out.loc[crash_veto, :] = np.nan
        return out

    def run(self, run_id: str = "global_tradingagents", score_sealed: bool = False) -> GlobalCompetitorOutcome:
        sig = self.build_signal(self.evaluator.feature_panel)
        return _evaluate_signal_global(
            name="TradingAgents",
            paper_ref="Xiao et al. (2024)",
            paradigm="Multi-Agent Bull/Bear Debate + Risk Manager",
            signal_wide=sig,
            evaluator=self.evaluator,
            run_id=run_id,
            score_sealed=score_sealed,
        )


# ---------------------------------------------------------------------------
# 2. FinMem (Yu et al., 2024): Layered Memory + Adaptive Risk Profile
# ---------------------------------------------------------------------------
class FinMemCompetitor:
    """Layered Memory (Shallow/Intermediate/Deep exponential decay) + Volatility-Adaptive Risk Character."""

    def __init__(self, evaluator: SealedEvaluator, seed: int = 2026) -> None:
        self.evaluator = evaluator
        self.seed = seed

    def build_signal(self, panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
        sigs = _extract_causal_signals(panel)
        mkt = _causal_market_state(panel)

        # Shallow memory (3-day fast-decay EWM): immediate reversal + microstructure flow
        shallow_mem = (0.55 * sigs["rev_overheat_3"] + 0.45 * sigs["flow_5"]).ewm(halflife=2.5, min_periods=1).mean()
        # Intermediate memory (15-day EWM): trend confirmation + persistent foreign flow
        inter_mem = (0.55 * sigs["mom_risk_adj_15"] + 0.45 * sigs["flow_15"]).ewm(halflife=10.0, min_periods=1).mean()
        # Deep memory (60-day EWM): structural PIT value-quality & low-vol resilience
        deep_mem = (0.65 * sigs["pit_vq"] + 0.35 * sigs["low_vol"]).ewm(halflife=40.0, min_periods=1).mean()

        # Adaptive Risk Profile (character switches from risk-seeking in calm uptrend to risk-averse in high vol)
        risk_seeking = ((mkt["mkt_r20"] > 0.01) & (mkt["mkt_vol20"] < 0.016)).astype(float)
        risk_averse = ((mkt["mkt_vol20"] > 0.019) | (mkt["mkt_dd60"] < -0.09)).astype(float)

        w_shallow = 0.25 + 0.15 * risk_seeking - 0.10 * risk_averse
        w_inter = 0.40 + 0.10 * risk_seeking - 0.15 * risk_averse
        w_deep = 1.0 - w_shallow - w_inter

        combined = shallow_mem.mul(w_shallow, axis=0) + inter_mem.mul(w_inter, axis=0) + deep_mem.mul(w_deep, axis=0)
        return _cs_zscore(combined)

    def run(self, run_id: str = "global_finmem", score_sealed: bool = False) -> GlobalCompetitorOutcome:
        sig = self.build_signal(self.evaluator.feature_panel)
        return _evaluate_signal_global(
            name="FinMem",
            paper_ref="Yu et al. (2024)",
            paradigm="Layered Memory (Shallow/Inter/Deep) + Adaptive Character",
            signal_wide=sig,
            evaluator=self.evaluator,
            run_id=run_id,
            score_sealed=score_sealed,
        )


# ---------------------------------------------------------------------------
# 3. FinAgent (Zhang et al., KDD 2024): Dual-Level Reflection over Multi-Modal Signals
# ---------------------------------------------------------------------------
class FinAgentCompetitor:
    """Dual-Level Reflection: Low-level signal attribution + High-level regime/drawdown reflection."""

    def __init__(self, evaluator: SealedEvaluator, seed: int = 2026) -> None:
        self.evaluator = evaluator
        self.seed = seed

    def build_signal(self, panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
        sigs = _extract_causal_signals(panel)
        mkt = _causal_market_state(panel)
        ret1 = panel["return_1d"].fillna(0.0)

        # Multi-modal expert views: Price-Trend, Volume-Reversal, Flow, PIT-Fundamental
        experts = {
            "price": sigs["mom_risk_adj_15"],
            "volume": sigs["rev_overheat_3"],
            "flow": sigs["flow_5"],
            "fundamental": sigs["pit_vq"],
        }

        # Low-Level Reflection (strictly causal!): evaluate each expert's lagged prediction at t-2 against realized ret1[t]
        # Note: signal[t-1] predicts ret1[t], so using shift(1) * ret1[t] uses ONLY information available at date t close!
        expert_scores: dict[str, pd.Series] = {}
        for k, mat in experts.items():
            # Cross-sectional long-short proxy realized at t from t-1 signal
            lag_sig = mat.shift(1).fillna(0.0)
            daily_perf = (lag_sig * ret1).mean(axis=1)
            # Rolling 30-day causal reflection Sharpe-like weight
            mu = daily_perf.rolling(30, min_periods=10).mean().fillna(0.0)
            sd = daily_perf.rolling(30, min_periods=10).std().fillna(0.01) + 1e-4
            expert_scores[k] = (mu / sd).clip(-1.0, 2.5)

        score_df = pd.DataFrame(expert_scores, index=ret1.index)
        # Softmax low-level reflection weights (strictly causal at date t)
        exp_w = np.exp(score_df.clip(-2.0, 2.0))
        norm_w = exp_w.div(exp_w.sum(axis=1).clip(lower=1e-8), axis=0)

        blended = sum(experts[k].mul(norm_w[k], axis=0) for k in experts)
        assert isinstance(blended, pd.DataFrame)

        # High-Level Reflection: dampen turnover via causal 3-day EWM and shift defensive if recent market crash
        high_level_sig = blended.ewm(span=3, min_periods=1).mean()
        severe_stress = (mkt["mkt_dd60"] < -0.12) & (mkt["floor_prox"] > 0.04)
        high_level_sig.loc[severe_stress, :] = np.nan
        return high_level_sig

    def run(self, run_id: str = "global_finagent", score_sealed: bool = False) -> GlobalCompetitorOutcome:
        sig = self.build_signal(self.evaluator.feature_panel)
        return _evaluate_signal_global(
            name="FinAgent",
            paper_ref="Zhang et al. (KDD 2024)",
            paradigm="Dual-Level Reflection + Multi-Modal Fusion",
            signal_wide=sig,
            evaluator=self.evaluator,
            run_id=run_id,
            score_sealed=score_sealed,
        )


# ---------------------------------------------------------------------------
# 4. RD-Agent(Q) (Li et al., NeurIPS 2025): MAB Factor + Model Co-Optimization
# ---------------------------------------------------------------------------
class RDAgentQCompetitor:
    """Multi-Armed Bandit (UCB1) scheduling factor subsets + GBDT/Ridge model co-optimization on search zone."""

    def __init__(self, evaluator: SealedEvaluator, n_rounds: int = 8, seed: int = 2026) -> None:
        self.evaluator = evaluator
        self.n_rounds = n_rounds
        self.seed = seed

    def build_signal(
        self,
        panel: dict[str, pd.DataFrame],
        search_end: pd.Timestamp,
        horizon_days: int = 5,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        sigs = _extract_causal_signals(panel)
        close = panel["close"]
        all_dates = pd.DatetimeIndex(close.index)
        syms = list(close.columns)

        # Strictly search-zone training dates (purged by horizon+1 before search_end!)
        search_dates = all_dates[all_dates <= search_end]
        train_cut = max(30, len(search_dates) - horizon_days - 1)
        s_dates = search_dates[:train_cut]
        split_idx = int(len(s_dates) * 0.72)
        tr_dates = s_dates[: max(20, split_idx - horizon_days - 1)]
        dev_dates = s_dates[split_idx:]

        fwd_search = (close.reindex(index=search_dates).shift(-horizon_days) / close.reindex(index=search_dates) - 1.0)

        # Factor-Model co-optimization arms: (feature subset name, model kind)
        factor_sets: dict[str, list[str]] = {
            "core_4": ["mom_risk_adj_15", "rev_overheat_3", "flow_5", "pit_vq"],
            "flow_pit_micro": ["mom_risk_adj_15", "flow_5", "flow_15", "pit_vq", "micro_safety"],
            "full_8": ["mom_risk_adj_15", "rev_overheat_3", "flow_5", "pit_vq", "pit_roe_pb", "low_vol", "micro_safety", "liq_calm"],
        }
        arms = [(fs, mk) for fs in factor_sets for mk in ("ridge", "gbdt")]
        arm_counts = {a: 0 for a in arms}
        arm_rewards = {a: 0.0 for a in arms}
        best_arm = arms[0]
        best_dev_ic = -1e9

        def _fit_and_predict(
            cols: list[str], model_kind: str, fit_dts: pd.DatetimeIndex, pred_dts: pd.DatetimeIndex, alpha_reg: float
        ) -> pd.DataFrame:
            x_tr = np.column_stack([sigs[c].reindex(index=fit_dts, columns=syms).to_numpy(dtype=float).ravel() for c in cols])
            y_tr = fwd_search.reindex(index=fit_dts, columns=syms).to_numpy(dtype=float).ravel()
            valid = np.isfinite(x_tr).all(axis=1) & np.isfinite(y_tr)
            scaler = StandardScaler()
            xs = scaler.fit_transform(x_tr[valid])
            if model_kind == "ridge":
                mdl: Any = Ridge(alpha=alpha_reg, random_state=self.seed)
            else:
                mdl = HistGradientBoostingRegressor(
                    max_iter=45, max_depth=3, learning_rate=0.05, l2_regularization=alpha_reg * 0.1, random_state=self.seed
                )
            mdl.fit(xs, y_tr[valid])
            x_pr = np.column_stack([sigs[c].reindex(index=pred_dts, columns=syms).to_numpy(dtype=float).ravel() for c in cols])
            preds = mdl.predict(scaler.transform(np.nan_to_num(x_pr, nan=0.0))).reshape(len(pred_dts), len(syms))
            return pd.DataFrame(preds, index=pred_dts, columns=syms)

        # UCB1 bandit loop strictly inside search zone (0 validation trials burned!)
        for t_step in range(1, max(len(arms), self.n_rounds) + 1):
            if t_step <= len(arms):
                arm = arms[t_step - 1]
            else:
                arm = max(
                    arms,
                    key=lambda a: (arm_rewards[a] / max(1, arm_counts[a]))
                    + 0.05 * math.sqrt(math.log(t_step) / max(1, arm_counts[a])),
                )
            cols = factor_sets[arm[0]]
            reg = 30.0 if arm_counts[arm] % 2 == 0 else 80.0
            dev_pred = _fit_and_predict(cols, arm[1], tr_dates, dev_dates, reg)
            ic_s = compute_rank_ic_series(dev_pred, fwd_search.reindex(index=dev_dates))
            dev_ic = float(ic_s.mean()) if len(ic_s) > 0 else 0.0
            arm_counts[arm] += 1
            arm_rewards[arm] += dev_ic
            if dev_ic > best_dev_ic:
                best_dev_ic = dev_ic
                best_arm = arm

        # Refit winning arm strictly on full purged search zone (<= search_end) and predict causally for all dates
        final_pred = _fit_and_predict(factor_sets[best_arm[0]], best_arm[1], s_dates, all_dates, 50.0)
        return _cs_zscore(final_pred), {"best_arm": best_arm, "search_dev_ic": round(best_dev_ic, 5)}

    def run(self, run_id: str = "global_rdagentq", score_sealed: bool = False) -> GlobalCompetitorOutcome:
        search_end = pd.Timestamp(self.evaluator.sandbox_cfg["zones"]["search"]["end"])
        sig, notes = self.build_signal(
            self.evaluator.feature_panel, search_end=search_end, horizon_days=self.evaluator.horizon_days
        )
        return _evaluate_signal_global(
            name="RD-Agent(Q)",
            paper_ref="Li et al. (NeurIPS 2025)",
            paradigm="MAB Factor-Model Joint Co-Optimization",
            signal_wide=sig,
            evaluator=self.evaluator,
            run_id=run_id,
            score_sealed=score_sealed,
            notes=notes,
        )


# ---------------------------------------------------------------------------
# 5. AlphaForge / DoubleEnsemble (AAAI 2025 / ICDM 2020): Sample-Reweighted Neural Ensemble
# ---------------------------------------------------------------------------
class AlphaForgeDoubleEnsembleCompetitor:
    """Sample-reweighted neural + linear factor ensemble (AlphaForge / DoubleEnsemble paradigm, strictly causal)."""

    def __init__(self, evaluator: SealedEvaluator, n_submodels: int = 3, seed: int = 2026) -> None:
        self.evaluator = evaluator
        self.n_submodels = n_submodels
        self.seed = seed

    def build_signal(
        self,
        panel: dict[str, pd.DataFrame],
        search_end: pd.Timestamp,
        horizon_days: int = 5,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        sigs = _extract_causal_signals(panel)
        feat_names = [
            "mom_risk_adj_15",
            "rev_overheat_3",
            "flow_5",
            "flow_15",
            "pit_vq",
            "pit_roe_pb",
            "low_vol",
            "micro_safety",
        ]
        close = panel["close"]
        all_dates = pd.DatetimeIndex(close.index)
        syms = list(close.columns)

        search_dates = all_dates[all_dates <= search_end]
        s_dates = search_dates[: max(30, len(search_dates) - horizon_days - 1)]
        fwd_search = close.reindex(index=search_dates).shift(-horizon_days) / close.reindex(index=search_dates) - 1.0

        x_tr = np.column_stack([sigs[c].reindex(index=s_dates, columns=syms).to_numpy(dtype=float).ravel() for c in feat_names])
        y_tr = fwd_search.reindex(index=s_dates, columns=syms).to_numpy(dtype=float).ravel()
        valid = np.isfinite(x_tr).all(axis=1) & np.isfinite(y_tr)
        x_v, y_v = x_tr[valid], y_tr[valid]

        scaler = StandardScaler()
        x_vs = scaler.fit_transform(x_v)
        x_all = np.column_stack([sigs[c].reindex(index=all_dates, columns=syms).to_numpy(dtype=float).ravel() for c in feat_names])
        x_all_s = scaler.transform(np.nan_to_num(x_all, nan=0.0))

        # DoubleEnsemble sample-reweighting loop strictly on search zone
        sample_w = np.ones(len(y_v), dtype=float)
        sub_preds: list[pd.DataFrame] = []
        sub_weights: list[float] = []

        for k in range(self.n_submodels):
            # Alternate compact MLP (AlphaForge neural combiner) and sample-weighted Ridge
            if k == 0:
                mdl: Any = MLPRegressor(
                    hidden_layer_sizes=(24, 12),
                    activation="tanh",
                    alpha=0.5,
                    max_iter=120,
                    early_stopping=False,
                    random_state=self.seed + k,
                )
                # Weighted bootstrap sample according to sample_w
                rng = np.random.default_rng(self.seed + k)
                probs = sample_w / float(np.sum(sample_w))
                idx = rng.choice(len(y_v), size=len(y_v), replace=True, p=probs)
                mdl.fit(x_vs[idx], y_v[idx])
            else:
                mdl = Ridge(alpha=40.0 * k, random_state=self.seed + k)
                mdl.fit(x_vs, y_v, sample_weight=sample_w)

            tr_pred = mdl.predict(x_vs)
            losses = np.abs(y_v - tr_pred)
            # DoubleEnsemble: upweight moderately hard samples while downweighting extreme noise outliers (>95th pct)
            hi_cut = float(np.quantile(losses, 0.95))
            mid = float(np.median(losses)) + 1e-8
            new_w = np.where(losses > hi_cut, 0.35, 0.70 + 0.60 * np.clip(losses / mid, 0.0, 2.0))
            sample_w = 0.5 * sample_w + 0.5 * new_w

            pr_all = mdl.predict(x_all_s).reshape(len(all_dates), len(syms))
            pr_df = pd.DataFrame(pr_all, index=all_dates, columns=syms)
            ic_s = compute_rank_ic_series(pr_df.reindex(index=s_dates), fwd_search.reindex(index=s_dates))
            w_k = max(0.05, float(ic_s.mean()) if len(ic_s) > 0 else 0.05)
            sub_preds.append(_cs_zscore(pr_df))
            sub_weights.append(w_k)

        w_arr = np.asarray(sub_weights, dtype=float)
        w_arr /= float(np.sum(w_arr)) + 1e-12
        comb = sum(df * float(wt) for df, wt in zip(sub_preds, w_arr, strict=True))
        assert isinstance(comb, pd.DataFrame)
        return _cs_zscore(comb), {"n_submodels": self.n_submodels, "sub_weights": [round(float(x), 4) for x in w_arr]}

    def run(self, run_id: str = "global_alphaforge", score_sealed: bool = False) -> GlobalCompetitorOutcome:
        search_end = pd.Timestamp(self.evaluator.sandbox_cfg["zones"]["search"]["end"])
        sig, notes = self.build_signal(
            self.evaluator.feature_panel, search_end=search_end, horizon_days=self.evaluator.horizon_days
        )
        return _evaluate_signal_global(
            name="AlphaForge-DE",
            paper_ref="Shi et al. (AAAI 2025) / DoubleEnsemble",
            paradigm="Sample-Reweighted Neural-Linear Factor Ensemble",
            signal_wide=sig,
            evaluator=self.evaluator,
            run_id=run_id,
            score_sealed=score_sealed,
            notes=notes,
        )


def run_global_competitor_arena(
    evaluator_factory: Any,
    run_id: str = "global_sota_arena",
    score_sealed: bool = False,
    seed: int = 2026,
) -> dict[str, GlobalCompetitorOutcome]:
    """Run all 5 global SOTA trading agent / alpha-mining competitors with isolated evaluators."""
    runners: list[tuple[str, Any]] = [
        ("TradingAgents", lambda ev: TradingAgentsCompetitor(ev, seed=seed)),
        ("FinMem", lambda ev: FinMemCompetitor(ev, seed=seed)),
        ("FinAgent", lambda ev: FinAgentCompetitor(ev, seed=seed)),
        ("RD-Agent(Q)", lambda ev: RDAgentQCompetitor(ev, seed=seed)),
        ("AlphaForge-DE", lambda ev: AlphaForgeDoubleEnsembleCompetitor(ev, seed=seed)),
    ]
    out: dict[str, GlobalCompetitorOutcome] = {}
    for name, ctor in runners:
        ev = evaluator_factory()
        out[name] = ctor(ev).run(run_id=f"{run_id}_{name.lower()}", score_sealed=score_sealed)
    return out
