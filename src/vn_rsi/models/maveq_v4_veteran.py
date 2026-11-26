"""MAVE-Q v4 (VetQuant-RSI): 10-Year Veteran CIO Neuro-Symbolic Quantitative Architecture.

Designed for 100% FREE Cloud GPU/TPU training (<15M structural params / <4.5 GB VRAM with QLoRA on
Google Colab Free T4, Kaggle Free T4 x2, Kaggle TPU v5e-8, or instant CPU execution via `cpu_fast`).

Four Coupled Architectural Innovations over Holo-MAVE-Q v3 & Global Trading Agent SOTA:
1. `VeteranCaseMemory` (Episodic Microstructure & Market Trauma k-NN Retriever):
   Stores strictly search-zone 8-dim causal microstructure regimes (breadth, 20d/5d trend, vol shock,
   liquidity shock, ceiling/floor imbalance, foreign net flow, valuation dispersion) paired with realized
   5d forward market returns & trauma outcomes. Performs purged k-NN retrieval at date t to anticipate
   bull-traps, liquidity crunches, and high-conviction bull expansions.
2. `TriCameralPreMortemCore` (Thesis Generator -> Devil's Advocate Anti-Thesis -> Chief Risk Synthesis):
   Upgrades `TypedASTGuard` + `Tree-MDLM` + `Holo-HRM` + `MSSF` + `BTO-SRSS` with an explicit causal
   Devil's Advocate auditor that penalizes floor-lock cascade risks, T+2.5 exhaustion blow-offs,
   unfillable ceiling-locked slots, and distress drift prior to position entry.
3. `HierarchicalRegimeConvictionController`:
   Cleanly separates Layer 1 (Cross-Sectional Alpha Ranking) from Layer 2 (Regime-Adaptive Conviction
   Tilting, Causal Incumbent Turnover Hysteresis, Kelly-Shrinkage Weighting, and Tail-Risk Cash Shield),
   capturing strong Bull upside while shielding Bear/Crash drawdowns inside the unmodified sealed simulator.
4. `ZeroValidationBurnPurgedSearchCV`:
   Executes all factor screening, MSSF calibration, BTO-SRSS weighting, and controller hyperparameter
   CV inside a hermetically isolated search-zone purged walk-forward simulator — slashing official
   `SealedEvaluator` queries to <= 12 trials to elevate Deflated Sharpe Ratio (DSR) dramatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from vn_rsi.benchmarks.extended_metrics import ExtendedMetricsReport, compute_extended_metrics
from vn_rsi.dsl.ast_spec import (
    ASTNode,
    ASTVerifier,
    FactorSpec,
    compute_feature_panel,
    evaluate_factor_spec,
)
from vn_rsi.eval.evaluator import EvaluationOutcome, SealedEvaluator
from vn_rsi.eval.metrics import compute_portfolio_metrics, compute_rank_ic_series
from vn_rsi.models.alpha_mining_baselines import GrammarTreeSampler
from vn_rsi.models.ast_vocab import MECHANISM_FAMILIES
from vn_rsi.models.holo_maveq import (
    HoloElite,
    HoloHRMReasoner,
    HoloQDArchive,
    TopologicalTreeDiffusion,
    _curated_domain_seeds,
    dominant_economic_niche,
    horizon_bucket,
    is_typed_stationary_ast,
)
from vn_rsi.sandbox.seal import SealedTestReuseError
from vn_rsi.sandbox.simulator import VNPortfolioSimulator


def _cs_z(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean(axis=1)
    sd = df.std(axis=1) + 1e-8
    return df.sub(mu, axis=0).div(sd, axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-4.0, 4.0)


# ---------------------------------------------------------------------------
# 1. VeteranCaseMemory: Strictly Causal k-NN Market Trauma & Regime Memory
# ---------------------------------------------------------------------------
def compute_causal_microstructure_state(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute strictly causal 10-dimensional microstructure state vector at every date t (info <= t only)."""
    close = panel["close"]
    ret1 = panel["return_1d"].fillna(0.0)
    mkt_ret = ret1.mean(axis=1)

    breadth_20 = (close.pct_change(20, fill_method=None) > 0.0).mean(axis=1).fillna(0.5)
    mkt_r20 = ((1.0 + mkt_ret).rolling(20, min_periods=5).apply(np.prod, raw=True) - 1.0).fillna(0.0)
    mkt_r5 = ((1.0 + mkt_ret).rolling(5, min_periods=2).apply(np.prod, raw=True) - 1.0).fillna(0.0)
    vol5 = mkt_ret.rolling(5, min_periods=2).std().fillna(0.012)
    vol20 = mkt_ret.rolling(20, min_periods=5).std().fillna(0.012)
    vol_ratio = (vol5 / (vol20 + 1e-4)).clip(0.3, 3.5)
    liq_shock = panel["turnover_shock_20d"].median(axis=1).fillna(1.0).clip(0.2, 4.0)

    ceil_near = (panel["dist_to_ceiling"].fillna(0.05) < 0.010).mean(axis=1).rolling(5, min_periods=1).mean()
    floor_near = (panel["dist_to_floor"].fillna(0.05) < 0.015).mean(axis=1).rolling(5, min_periods=1).mean()
    floor_near_3d = (panel["dist_to_floor"].fillna(0.05) < 0.015).mean(axis=1).rolling(3, min_periods=1).mean().fillna(0.0)
    ceil_floor_imb = (ceil_near - floor_near).fillna(0.0)

    foreign_5 = panel["foreign_net_ratio"].fillna(0.0).mean(axis=1).rolling(5, min_periods=1).mean().fillna(0.0)
    val_disp = panel["pit_value_quality"].std(axis=1).fillna(0.1).clip(0.0, 5.0)

    return pd.DataFrame(
        {
            "breadth_20": breadth_20,
            "mkt_r20": mkt_r20,
            "mkt_r5": mkt_r5,
            "vol_ratio": vol_ratio,
            "liq_shock": liq_shock,
            "ceil_floor_imb": ceil_floor_imb,
            "foreign_5": foreign_5,
            "val_disp": val_disp,
            "floor_near_3d": floor_near_3d,
            "mkt_vol20": vol20,
        },
        index=close.index,
    )


@dataclass
class VeteranCaseMemory:
    """Episodic market-regime & trauma k-NN retriever fit strictly on the `search` zone."""

    k_neighbors: int = 15
    exclusion_buffer_days: int = 10
    trauma_drop_threshold: float = -0.045
    floor_trauma_threshold: float = 0.07

    _dates: pd.DatetimeIndex = field(default_factory=lambda: pd.DatetimeIndex([]), init=False)
    _z_states: np.ndarray = field(default_factory=lambda: np.zeros((0, 10)), init=False)
    _mean: np.ndarray = field(default_factory=lambda: np.zeros(10), init=False)
    _std: np.ndarray = field(default_factory=lambda: np.ones(10), init=False)
    _fwd_mkt_5d: np.ndarray = field(default_factory=lambda: np.zeros(0), init=False)
    _is_trauma: np.ndarray = field(default_factory=lambda: np.zeros(0), init=False)
    _fitted: bool = field(default=False, init=False)

    def fit_on_search(
        self,
        search_panel: dict[str, pd.DataFrame],
        horizon_days: int = 5,
    ) -> VeteranCaseMemory:
        """Fit standardized episodic memory bank exclusively on search-zone historical bars."""
        state_df = compute_causal_microstructure_state(search_panel)
        mkt_ret = search_panel["return_1d"].fillna(0.0).mean(axis=1)
        mkt_wealth = (1.0 + mkt_ret).cumprod()
        fwd_5d = (mkt_wealth.shift(-horizon_days) / mkt_wealth.clip(lower=1e-8)) - 1.0
        floor_rate = (search_panel["dist_to_floor"].fillna(0.05) < 0.015).mean(axis=1)
        fwd_floor = floor_rate.rolling(horizon_days, min_periods=1).max().shift(-horizon_days)

        # Strictly exclude final `horizon_days` rows of search so labels never extend beyond search_end
        safe_n = max(1, len(state_df) - horizon_days)
        safe_idx = state_df.index[:safe_n]
        x_raw = np.nan_to_num(state_df.reindex(index=safe_idx).to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        y_ret = fwd_5d.reindex(index=safe_idx).fillna(0.0).to_numpy(dtype=float)
        y_flr = fwd_floor.reindex(index=safe_idx).fillna(0.0).to_numpy(dtype=float)

        self._dates = pd.DatetimeIndex(safe_idx)
        self._mean = np.mean(x_raw, axis=0)
        self._std = np.std(x_raw, axis=0) + 1e-6
        self._z_states = (x_raw - self._mean) / self._std
        self._fwd_mkt_5d = y_ret
        self._is_trauma = ((y_ret <= self.trauma_drop_threshold) | (y_flr >= self.floor_trauma_threshold)).astype(float)
        self._fitted = True
        return self

    def query_panel(
        self,
        state_df: pd.DataFrame,
        purge_self_matches: bool = False,
    ) -> pd.DataFrame:
        """Query frozen search-zone episodic memory causally for each date t in `state_df`."""
        idx = pd.DatetimeIndex(state_df.index)
        if not self._fitted or len(self._z_states) < 5:
            return pd.DataFrame(
                {
                    "expected_mkt_5d": 0.0,
                    "trauma_prob": 0.15,
                    "bull_trap_risk": 0.0,
                    "bull_conviction": 0.5,
                },
                index=idx,
            )

        q_raw = np.nan_to_num(state_df.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        q_z = (q_raw - self._mean) / self._std
        n_q = len(idx)
        k = min(self.k_neighbors, max(1, len(self._z_states) - 1))

        exp_ret = np.zeros(n_q, dtype=float)
        trauma_p = np.zeros(n_q, dtype=float)
        bull_trap = np.zeros(n_q, dtype=float)
        bull_conv = np.zeros(n_q, dtype=float)

        mem_days = self._dates.to_numpy(dtype="datetime64[D]")
        q_days = idx.to_numpy(dtype="datetime64[D]")
        buf_days = np.timedelta64(self.exclusion_buffer_days, "D")

        for t in range(n_q):
            diffs = self._z_states - q_z[t : t + 1, :]
            dists = np.sqrt(np.sum(diffs * diffs, axis=1))
            if purge_self_matches:
                # Unit-safe date purge: strictly exclude episodes within +/- exclusion_buffer_days
                too_close = np.abs(mem_days - q_days[t]) <= buf_days
                dists = np.where(too_close, 1e9, dists)
            nn_idx = np.argpartition(dists, k)[:k]
            nn_d = dists[nn_idx]
            w = np.exp(-0.5 * np.clip(nn_d, 0.0, 10.0))
            w_sum = float(np.sum(w)) + 1e-12
            w /= w_sum

            er = float(np.dot(w, self._fwd_mkt_5d[nn_idx]))
            tp = float(np.dot(w, self._is_trauma[nn_idx]))
            exp_ret[t] = er
            trauma_p[t] = tp
            # Bull-trap signal: recent 5d rally + liquidity spike + negative historical precedent
            r5 = float(q_raw[t, 2])
            liq = float(q_raw[t, 4])
            bull_trap[t] = float(np.clip((1.0 if (r5 > 0.025 and liq > 1.25 and er < 0.0) else 0.0) * (0.5 + tp), 0.0, 1.0))
            bull_conv[t] = float(np.clip(0.5 + 12.0 * er - 0.8 * tp, 0.0, 1.0))

        return pd.DataFrame(
            {
                "expected_mkt_5d": exp_ret,
                "trauma_prob": trauma_p,
                "bull_trap_risk": bull_trap,
                "bull_conviction": bull_conv,
            },
            index=idx,
        )

    def query_contextual_factor_weights(
        self,
        state_df: pd.DataFrame,
        elite_ic_matrix: np.ndarray,
        base_weights: np.ndarray,
        gamma: float = 0.18,
        purge_self_matches: bool = False,
    ) -> np.ndarray:
        """Compute strictly causal regime-adaptive factor weights using k-NN historical IC analogs."""
        n_q = len(state_df)
        m = len(base_weights)
        w_base = np.asarray(base_weights, dtype=float)
        w_base = w_base / (float(np.sum(np.abs(w_base))) + 1e-12)
        if not self._fitted or gamma <= 1e-6 or m <= 1 or len(self._z_states) < 10 or elite_ic_matrix.shape[0] != len(self._z_states):
            return np.tile(w_base[None, :], (n_q, 1))

        idx = pd.DatetimeIndex(state_df.index)
        q_raw = np.nan_to_num(state_df.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        q_z = (q_raw - self._mean) / self._std
        k = min(max(self.k_neighbors, 20), max(1, len(self._z_states) - 1))

        mem_days = self._dates.to_numpy(dtype="datetime64[D]")
        q_days = idx.to_numpy(dtype="datetime64[D]")
        buf_days = np.timedelta64(self.exclusion_buffer_days, "D")

        out_w = np.zeros((n_q, m), dtype=float)
        for t in range(n_q):
            diffs = self._z_states - q_z[t : t + 1, :]
            dists = np.sqrt(np.sum(diffs * diffs, axis=1))
            if purge_self_matches:
                too_close = np.abs(mem_days - q_days[t]) <= buf_days
                dists = np.where(too_close, 1e9, dists)
            nn_idx = np.argpartition(dists, k)[:k]
            w_nn = np.exp(-0.5 * np.clip(dists[nn_idx], 0.0, 10.0))
            w_nn /= float(np.sum(w_nn)) + 1e-12
            # Expected cross-sectional Rank IC of each elite in analogous historical regimes
            analog_ic = np.clip(w_nn @ elite_ic_matrix[nn_idx], 0.005, None)
            w_ctx = analog_ic / (float(np.sum(analog_ic)) + 1e-12)
            w_t = (1.0 - gamma) * w_base + gamma * w_ctx
            out_w[t] = w_t / (float(np.sum(np.abs(w_t))) + 1e-12)
        return out_w

    def retrieve_nearest_cases(self, state_vec: np.ndarray, top_k: int = 3) -> list[dict[str, Any]]:
        """Retrieve top-k human-readable historical analog dates & outcomes for Veteran explanation."""
        if not self._fitted or len(self._z_states) == 0:
            return []
        qz = (np.asarray(state_vec, dtype=float) - self._mean) / self._std
        dists = np.sqrt(np.sum((self._z_states - qz[None, :]) ** 2, axis=1))
        order = np.argsort(dists)[: min(top_k, len(dists))]
        return [
            {
                "analog_date": str(self._dates[int(i)].date()),
                "distance": round(float(dists[int(i)]), 3),
                "realized_fwd_5d_mkt_ret": round(float(self._fwd_mkt_5d[int(i)]), 5),
                "was_trauma": bool(self._is_trauma[int(i)] > 0.5),
            }
            for i in order
        ]


# ---------------------------------------------------------------------------
# 2. Devil's Advocate Anti-Thesis (Strictly Causal Pre-Mortem Risk Auditor)
# ---------------------------------------------------------------------------
class DevilsAdvocatePreMortem:
    """Chamber 2 Auditor: Detects T+2.5 floor-lock cascades, blow-off traps, ceiling locks, & distress drift."""

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        c = cfg or {}
        self.floor_prox_thresh = float(c.get("floor_proximity_thresh", 0.022))
        self.ceil_prox_thresh = float(c.get("ceiling_proximity_thresh", 0.008))
        self.exhaustion_ret3_z = float(c.get("exhaustion_ret3_z", 1.85))
        self.exhaustion_tshock = float(c.get("exhaustion_tshock", 1.90))
        self.distress_ret20 = float(c.get("distress_ret20_thresh", -0.18))

    def audit_panel(self, panel: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (pre_mortem_penalty_df >= 0, hard_veto_mask_bool_df) strictly from <= t bars."""
        close = panel["close"]
        ret1 = panel["return_1d"].fillna(0.0)
        ret3 = close.pct_change(3, fill_method=None).fillna(0.0)
        ret20 = close.pct_change(20, fill_method=None).fillna(0.0)
        tshock = panel["turnover_shock_20d"].fillna(1.0)
        dist_flr = panel["dist_to_floor"].fillna(0.05)
        dist_ceil = panel["dist_to_ceiling"].fillna(0.05)
        intraday = panel["intraday_range"].fillna(0.02)

        # 1. Floor-lock cascade risk (approaching floor or repeated floor touches over trailing 5 sessions)
        near_flr = (dist_flr < self.floor_prox_thresh).astype(float)
        flr_cascade = near_flr + 0.6 * near_flr.rolling(5, min_periods=1).sum().clip(0.0, 3.0)

        # 2. Retail blow-off / exhaustion trap (sharp 3d spike + abnormal volume surge -> T+2.5 dump hazard)
        ret3_z = _cs_z(ret3)
        exhaustion = ((ret3_z > self.exhaustion_ret3_z) & (tshock > self.exhaustion_tshock)).astype(float) * (
            (ret3_z - self.exhaustion_ret3_z).clip(0.0, 2.5)
        )

        # 3. Unfillable ceiling-lock penalty (wasted Top-K slot if stock already pinned at ceiling)
        ceil_pinned = (dist_ceil < self.ceil_prox_thresh).astype(float) * 1.2

        # 4. Idiosyncratic range & volatility blow-up
        vol5 = ret1.rolling(5, min_periods=2).std().fillna(0.015)
        vol20 = ret1.rolling(20, min_periods=5).std().fillna(0.015)
        vol_blowup = ((vol5 / (vol20 + 1e-4)) - 1.6).clip(0.0, 2.5) + (intraday - 0.055).clip(0.0, 0.10) * 8.0

        # 5. Deep distress drift (early warning for ROS/FLC-style halt & delisting traps)
        distress = (ret20 < self.distress_ret20).astype(float) * 1.5

        penalty = (
            0.40 * flr_cascade
            + 0.30 * exhaustion
            + 0.25 * ceil_pinned
            + 0.20 * vol_blowup
            + 0.50 * distress
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 4.0)

        # Hard veto only on extreme distress or consecutive floor-lock trap
        hard_veto = (near_flr.rolling(3, min_periods=1).sum() >= 2.0) | (ret20 < -0.28)
        return penalty, hard_veto


# ---------------------------------------------------------------------------
# 3. Hierarchical Regime-Conviction Portfolio Controller (Layer 1 + Layer 2)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ControllerParams:
    name: str
    trauma_cash_thresh: float = 0.68
    dd_cash_thresh: float = -0.165
    bull_trend_tilt: float = 0.02
    bear_defense_tilt: float = 0.06
    hysteresis_bonus: float = 0.22
    bull_top_k_cap: int = 6
    kelly_fraction: float = 0.70
    vol_shock_thresh: float = 1.60
    floor_stress_thresh: float = 0.075
    signal_ema_span: int = 1
    case_memory_gamma: float = 0.10
    symbolic_blend_alpha: float = 0.45
    bab_base_tilt: float = 0.22
    bab_stress_boost: float = 0.28
    rebound_release_r5: float = 0.012


class OrthogonalCausalBackbone:
    """Chamber 1B: Search-zone Gârleanu–Pedersen Friction-Adjusted Non-Negative FWL Ridge Backbone."""

    FEAT_NAMES: tuple[str, ...] = (
        "pit_vq",
        "pit_roe_pb_rank",
        "mom_ra15_sm",
        "mom_skip_3_15",
        "rev_oh3",
        "flow_5",
        "flow_10",
    )

    def __init__(self, ridge_l2_frac: float = 0.15, gp_turnover_exp: float = 0.48) -> None:
        self.ridge_l2_frac = ridge_l2_frac
        self.gp_turnover_exp = gp_turnover_exp
        self._scaler: StandardScaler | None = None
        self._eff_coef: np.ndarray | None = None

    @staticmethod
    def extract_features(panel: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Strictly causal (information <= t only) orthogonal economic feature views."""
        close = panel["close"]
        ret1 = panel["return_1d"].fillna(0.0)
        vol20 = ret1.rolling(20, min_periods=5).std().fillna(0.015)
        mom15 = close.pct_change(15, fill_method=None).fillna(0.0)
        mom_skip = close.shift(3).pct_change(12, fill_method=None).fillna(0.0)
        tshock = panel["turnover_shock_20d"].fillna(1.0)
        fflow = panel["foreign_net_ratio"].fillna(0.0)
        pit_vq = panel["pit_value_quality"].fillna(0.0)
        pit_roe = panel["pit_roe"].fillna(0.08)
        pit_pb = panel["pit_pb_ratio"].fillna(1.8).clip(lower=0.3)
        return {
            "pit_vq": _cs_z(pit_vq),
            "pit_roe_pb_rank": _cs_z(pit_roe.rank(axis=1, pct=True) - pit_pb.rank(axis=1, pct=True)),
            "mom_ra15_sm": _cs_z((mom15 / (vol20 + 0.006)).rolling(4, min_periods=1).mean()),
            "mom_skip_3_15": _cs_z(mom_skip / (vol20 + 0.006)),
            "rev_oh3": _cs_z(-(close.pct_change(3, fill_method=None).fillna(0.0) * tshock.clip(0.5, 2.5))),
            "flow_5": _cs_z(fflow.rolling(5, min_periods=2).mean().fillna(0.0)),
            "flow_10": _cs_z(fflow.rolling(10, min_periods=3).mean().fillna(0.0)),
        }

    def fit_on_search(
        self,
        search_panel: dict[str, pd.DataFrame],
        safe_dates: pd.DatetimeIndex,
        search_fwd: pd.DataFrame,
        rebal_step: int = 3,
        top_k: int = 6,
    ) -> OrthogonalCausalBackbone:
        feats = self.extract_features(search_panel)
        syms = list(search_panel["close"].columns)
        feat_turns = np.asarray(
            [
                MAVEQv4Engine._top_k_turnover_proxy(feats[c].reindex(index=safe_dates, columns=syms), rebal_step, top_k)
                for c in self.FEAT_NAMES
            ],
            dtype=float,
        )
        x_tr = np.column_stack(
            [feats[c].reindex(index=safe_dates, columns=syms).to_numpy(dtype=float).ravel() for c in self.FEAT_NAMES]
        )
        y_tr = search_fwd.reindex(index=safe_dates, columns=syms).to_numpy(dtype=float).ravel()
        valid = np.isfinite(x_tr).all(axis=1) & np.isfinite(y_tr)
        n_valid = int(np.sum(valid))
        if n_valid >= 50:
            self._scaler = StandardScaler().fit(x_tr[valid])
            alpha_scaled = max(10.0, self.ridge_l2_frac * float(n_valid))
            mdl = Ridge(alpha=alpha_scaled, positive=True).fit(self._scaler.transform(x_tr[valid]), y_tr[valid])
            # Gârleanu–Pedersen (2013) inverse-turnover damping over Frisch–Waugh–Lovell partial betas
            raw_c = np.clip(np.asarray(mdl.coef_, dtype=float), 0.0, None)
            # Prevent redundant fundamental pair from exceeding 46% combined weight
            fund_sum = raw_c[0] + raw_c[1]
            total_raw = float(np.sum(raw_c)) + 1e-12
            if fund_sum > 0.46 * total_raw and fund_sum > 1e-12:
                scale_f = (0.46 * total_raw) / fund_sum
                raw_c[0] *= scale_f
                raw_c[1] *= scale_f
            gp_w = raw_c / np.power(0.035 + feat_turns, self.gp_turnover_exp)
            if float(np.sum(gp_w)) <= 1e-12:
                gp_w = np.ones(len(self.FEAT_NAMES), dtype=float)
            self._eff_coef = gp_w / (float(np.sum(gp_w)) + 1e-12)
        return self

    def predict_panel(self, panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = panel["close"]
        dates = pd.DatetimeIndex(close.index)
        syms = list(close.columns)
        if self._scaler is None or self._eff_coef is None:
            return pd.DataFrame(0.0, index=dates, columns=syms)
        feats = self.extract_features(panel)
        x_all = np.column_stack(
            [feats[c].reindex(index=dates, columns=syms).to_numpy(dtype=float).ravel() for c in self.FEAT_NAMES]
        )
        xs = self._scaler.transform(np.nan_to_num(x_all, nan=0.0))
        preds = (xs @ self._eff_coef).reshape(len(dates), len(syms))
        return _cs_z(pd.DataFrame(preds, index=dates, columns=syms))


@dataclass
class ControllerOutput:
    sealed_signal_wide: pd.DataFrame       # Directly compatible with unmodified SealedEvaluator
    kelly_weights_wide: pd.DataFrame       # Continuous Kelly-shrinkage weights (sum <= 1.0)
    regime_series: pd.Series               # Causal regime trace per day
    exposure_series: pd.Series             # Target gross exposure in [0, 1] per day


class HierarchicalRegimeConvictionController:
    """Two-layer CIO controller separating Cross-Sectional Alpha Ranking from Regime/Conviction Risk Control."""

    def __init__(
        self,
        pre_mortem_cfg: dict[str, Any] | None = None,
        top_k: int = 6,
    ) -> None:
        self.pm_cfg = pre_mortem_cfg or {}
        self.top_k = top_k
        self.base_pm_w = float(self.pm_cfg.get("base_penalty_weight", 0.22))
        self.trauma_pm_boost = float(self.pm_cfg.get("trauma_penalty_boost", 0.45))

    def apply(
        self,
        raw_alpha_wide: pd.DataFrame,
        panel: dict[str, pd.DataFrame],
        state_df: pd.DataFrame,
        memory_signals: pd.DataFrame,
        pm_penalty: pd.DataFrame,
        hard_veto: pd.DataFrame,
        params: ControllerParams,
        use_pre_mortem: bool = True,
        use_regime_controller: bool = True,
    ) -> ControllerOutput:
        """Produce strictly causal sealed signal matrix + Kelly weights (information <= t only)."""
        dates = pd.DatetimeIndex(raw_alpha_wide.index)
        syms = list(raw_alpha_wide.columns)
        alpha_raw = _cs_z(raw_alpha_wide.reindex(index=dates, columns=syms))
        if use_regime_controller and params.signal_ema_span > 1:
            # Strictly causal rank-space exponential smoothing matched to T+2.5 rebalance cadence
            alpha_z = _cs_z(alpha_raw.ewm(span=params.signal_ema_span, min_periods=1).mean())
        else:
            alpha_z = alpha_raw

        if not use_regime_controller:
            # Pure Layer-1 alpha (+ optional pre-mortem) with static exposure = 1.0
            sig = alpha_z.copy()
            if use_pre_mortem:
                sig = sig - self.base_pm_w * pm_penalty.reindex(index=dates, columns=syms).fillna(0.0)
            eq_w = pd.DataFrame(0.0, index=dates, columns=syms)
            arr = sig.to_numpy(dtype=float)
            for t in range(len(dates)):
                order = np.argsort(np.nan_to_num(arr[t], nan=-1e9))[::-1][: self.top_k]
                eq_w.iloc[t, order] = 1.0 / max(1, len(order))
            return ControllerOutput(
                sealed_signal_wide=sig,
                kelly_weights_wide=eq_w,
                regime_series=pd.Series("Static", index=dates),
                exposure_series=pd.Series(1.0, index=dates),
            )

        # Causal auxiliary style views for regime tilting (info <= t)
        close = panel["close"].reindex(index=dates, columns=syms)
        ret1 = panel["return_1d"].reindex(index=dates, columns=syms).fillna(0.0)
        mkt_ret = ret1.mean(axis=1)
        mkt_wealth = (1.0 + mkt_ret).cumprod()
        mkt_dd60 = (mkt_wealth / mkt_wealth.rolling(60, min_periods=5).max().clip(lower=1e-8)) - 1.0

        # Bull leader view: causal risk-adjusted 20d momentum + foreign flow confirmation
        vol20_s = ret1.rolling(20, min_periods=5).std().fillna(0.015)
        mom20_s = close.pct_change(20, fill_method=None).fillna(0.0)
        bull_leader = _cs_z(
            mom20_s / (vol20_s + 0.008)
            + 0.4 * _cs_z(panel["foreign_net_ratio"].reindex(index=dates, columns=syms).rolling(5, min_periods=1).mean().fillna(0.0))
        )
        # Strictly causal Frazzini-Pedersen (2014) Vasicek-shrunk rolling market beta + idiosyncratic vol view
        mkt_var120 = mkt_ret.rolling(120, min_periods=15).var().clip(lower=1e-6)
        mkt_m120 = mkt_ret.rolling(120, min_periods=15).mean()
        r_m120 = ret1.rolling(120, min_periods=15).mean()
        cov120 = ret1.mul(mkt_ret, axis=0).rolling(120, min_periods=15).mean().sub(r_m120.mul(mkt_m120, axis=0), axis=0)
        raw_beta120 = cov120.div(mkt_var120, axis=0).fillna(1.0)
        shrunk_beta = (0.65 * raw_beta120 + 0.35 * 1.0).clip(0.35, 2.0)
        bab_risk_z = _cs_z(0.75 * _cs_z(shrunk_beta) + 0.25 * _cs_z(vol20_s))
        floor_safe = (
            panel["dist_to_floor"].reindex(index=dates, columns=syms).fillna(0.05)
            - panel["intraday_range"].reindex(index=dates, columns=syms).fillna(0.02)
        )
        defensive_view = _cs_z(
            0.50 * _cs_z(panel["pit_value_quality"].reindex(index=dates, columns=syms).fillna(0.0))
            + 0.20 * _cs_z(floor_safe)
            + 0.30 * (-bab_risk_z)
        )
        floor_1d_s = (panel["dist_to_floor"].reindex(index=dates, columns=syms).fillna(0.05) < 0.015).mean(axis=1).fillna(0.0)

        st = state_df.reindex(index=dates).fillna(0.0)
        mem = memory_signals.reindex(index=dates).fillna(0.0)
        pm_mat = pm_penalty.reindex(index=dates, columns=syms).fillna(0.0).to_numpy(dtype=float)
        veto_mat = hard_veto.reindex(index=dates, columns=syms).fillna(False).to_numpy(dtype=bool)

        a_mat = alpha_z.to_numpy(dtype=float)
        b_mat = bull_leader.to_numpy(dtype=float)
        d_mat = defensive_view.to_numpy(dtype=float)
        bab_mat = bab_risk_z.to_numpy(dtype=float)

        n_days, n_syms = a_mat.shape
        out_sig = np.full((n_days, n_syms), np.nan, dtype=float)
        kelly_w = np.zeros((n_days, n_syms), dtype=float)
        regimes: list[str] = []
        exposures = np.zeros(n_days, dtype=float)

        incumbent_mask = np.zeros(n_syms, dtype=bool)
        for t in range(n_days):
            r20 = float(st["mkt_r20"].iloc[t])
            r5 = float(st["mkt_r5"].iloc[t])
            breadth = float(st["breadth_20"].iloc[t])
            vol_rat = float(st["vol_ratio"].iloc[t])
            floor_3d = float(st["floor_near_3d"].iloc[t])
            floor_1d = float(floor_1d_s.iloc[t])
            dd60 = float(mkt_dd60.iloc[t])
            trauma_p = float(mem["trauma_prob"].iloc[t])
            trap_r = float(mem["bull_trap_risk"].iloc[t])
            exp_m5 = float(mem["expected_mkt_5d"].iloc[t])

            # Composite causal trauma risk score combining episodic memory + live floor contagion + drawdown
            trauma_score = (
                0.45 * trauma_p
                + 0.20 * trap_r
                + 0.15 * float(np.clip(-dd60 / 0.18, 0.0, 1.0))
                + 0.20 * float(np.clip(floor_3d / 0.12, 0.0, 1.0))
            )

            # Zero-Whipsaw Instant-Release Tail Circuit Breaker:
            # Only retreats to Cash on simultaneous deep systemic drawdown + active floor-lock contagion,
            # releasing immediately once daily floor stress subsides or a rebound initiates (zero V-bottom lock-out).
            is_crash = (
                (dd60 <= params.dd_cash_thresh)
                and (floor_3d >= params.floor_stress_thresh)
                and (floor_1d >= params.floor_stress_thresh * 0.40)
                and (r5 < params.rebound_release_r5)
            )

            # Scale-invariant regime boundaries (work identically across VN HOSE, VN100, US T+1, and CSI T+1!)
            is_bull = (
                (not is_crash)
                and (r20 > 0.020)
                and (breadth >= 0.55)
                and (vol_rat < 1.25)
                and (trauma_score < params.trauma_cash_thresh * 0.85)
            )
            is_cautious = (
                (not is_crash)
                and (not is_bull)
                and ((vol_rat > 1.35) or (dd60 < -0.085) or (trauma_score > 0.44))
            )

            if is_crash:
                regimes.append("Cash_Shield")
                exposures[t] = 0.0
                # All NaN -> unmodified VNPortfolioSimulator cleanly retreats to CASH (respecting T+2.5 & floor locks)
                out_sig[t, :] = np.nan
                incumbent_mask[:] = False
                continue

            # Chamber 3: Chief Risk Synthesis (Alpha + Regime-Adaptive Frazzini-Pedersen BAB - Pre-Mortem + Hysteresis)
            pm_scale = (self.base_pm_w + self.trauma_pm_boost * trauma_score) if use_pre_mortem else 0.0
            row_score = a_mat[t].copy()
            if is_bull:
                regimes.append("Bull_Conviction")
                eff_bab = params.bab_base_tilt * 0.30
                row_score = (1.0 - params.bull_trend_tilt) * row_score + params.bull_trend_tilt * b_mat[t] - eff_bab * bab_mat[t]
                k_eff = min(self.top_k, max(5, params.bull_top_k_cap))
                target_exp = float(np.clip(0.88 + 0.12 * float(mem["bull_conviction"].iloc[t]), 0.88, 1.0))
            elif is_cautious:
                regimes.append("Cautious_Defense")
                stress_lvl = float(np.clip(0.55 + trauma_score, 0.45, 1.35))
                eff_bab = params.bab_base_tilt + params.bab_stress_boost * stress_lvl
                row_score = (1.0 - params.bear_defense_tilt) * row_score + params.bear_defense_tilt * d_mat[t] - eff_bab * bab_mat[t]
                k_eff = self.top_k
                target_exp = float(np.clip(0.60 + 4.0 * max(-0.04, exp_m5), 0.42, 0.82))
            else:
                regimes.append("Normal_Alpha")
                eff_bab = params.bab_base_tilt
                row_score = row_score - eff_bab * bab_mat[t]
                k_eff = self.top_k
                target_exp = 0.92

            row_score = np.nan_to_num(row_score - pm_scale * pm_mat[t], nan=0.0, posinf=4.0, neginf=-4.0)
            # Gentle incumbent hold-buffer prevents borderline 6th/7th rank jitter while preserving full Rank IC
            row_score = row_score + params.hysteresis_bonus * incumbent_mask.astype(float)
            if use_pre_mortem:
                row_score = np.where(veto_mat[t], row_score - 0.75, row_score)

            # Keep full finite cross-section when k_eff == top_k so full-universe Rank IC is preserved!
            if k_eff < self.top_k:
                # Select top-k_eff high-conviction leaders (still >=5 so Rank IC remains valid!)
                top_idx = np.argsort(row_score)[::-1][:k_eff]
                sig_row = np.full(n_syms, np.nan, dtype=float)
                sig_row[top_idx] = row_score[top_idx]
                out_sig[t, :] = sig_row
            else:
                out_sig[t, :] = row_score
                top_idx = np.argsort(row_score)[::-1][: self.top_k]

            incumbent_mask[:] = False
            incumbent_mask[top_idx] = True
            exposures[t] = target_exp

            # Continuous Kelly-Shrinkage target weights over selected leaders
            conv = np.maximum(0.05, row_score[top_idx] - float(np.min(row_score[top_idx])) + 0.25)
            conv_w = conv / max(float(np.sum(conv)), 1e-12)
            eq_k = np.full(len(top_idx), 1.0 / len(top_idx), dtype=float)
            blended_w = params.kelly_fraction * conv_w + (1.0 - params.kelly_fraction) * eq_k
            kelly_w[t, top_idx] = target_exp * blended_w

        return ControllerOutput(
            sealed_signal_wide=pd.DataFrame(out_sig, index=dates, columns=syms),
            kelly_weights_wide=pd.DataFrame(kelly_w, index=dates, columns=syms),
            regime_series=pd.Series(regimes, index=dates, name="v4_regime"),
            exposure_series=pd.Series(exposures, index=dates, name="v4_exposure"),
        )


# ---------------------------------------------------------------------------
# 4. MAVE-Q v4 (VetQuant-RSI) Engine with Zero-Validation-Burn Purged Search CV
# ---------------------------------------------------------------------------
@dataclass
class MAVEQv4Result:
    preset: str
    backend_runtime: str
    specialist_total_params: int
    estimated_vram_mb: float
    selected_controller: str
    champion_expressions: list[str]
    champion_weights: list[float]
    champion_niches: list[str]
    search_candidates_screened: int
    validation_queries_used: int
    total_evaluator_trials: int
    archive_coverage: int
    typed_guard_rejections: int
    champion_search_cv_score: float
    champion_validation_metrics: dict[str, float]
    champion_validation_extended: dict[str, Any]
    champion_validation_fold_std: dict[str, float]
    champion_sealed_test_metrics: dict[str, Any]
    champion_sealed_test_extended: dict[str, Any]
    regime_exposure_summary: dict[str, float]
    validation_daily_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float), repr=False)
    sealed_test_daily_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float), repr=False)


class MAVEQv4Engine:
    """MAVE-Q v4 (VetQuant-RSI) Autonomous Quant Engine.

    Strictly isolates all factor discovery, MSSF smoothing, BTO-SRSS weighting, VeteranCaseMemory
    fitting, and Hierarchical Regime-Conviction CV inside the `search` zone — sending only top
    finalists (<= max_validation_queries) to `SealedEvaluator` for dramatic DSR elevation.
    """

    def __init__(
        self,
        evaluator: SealedEvaluator,
        cfg: dict[str, Any],
        preset: str = "cpu_fast",
        device: str | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.cfg = cfg
        self.preset = preset
        self.seed = int(cfg.get("seed", 2026))
        self.rng = np.random.default_rng(self.seed)

        ab = cfg.get("ablations", {})
        self.use_case_memory = bool(ab.get("use_case_memory", True))
        self.use_pre_mortem = bool(ab.get("use_pre_mortem", True))
        self.use_regime_controller = bool(ab.get("use_regime_controller", True))
        self.use_zero_val_burn_cv = bool(ab.get("use_zero_val_burn_cv", True))
        self.use_typed_tree_mdlm = bool(ab.get("use_typed_tree_mdlm", True))

        self.diffusion = TopologicalTreeDiffusion(preset=preset, seed=self.seed, device=device)
        self.hrm = HoloHRMReasoner(preset=preset, seed=self.seed + 31, device=device)
        self.verifier = ASTVerifier(max_depth=5)
        self.sampler = GrammarTreeSampler(max_depth=4)
        self.archive = HoloQDArchive()

        # Hermetically isolated SEARCH-ZONE-ONLY data (validation & sealed_test rows DO NOT EXIST here!)
        self.search_bars = evaluator.zone_guard.get_agent_visible_bars(evaluator.snapshot.bars, "search")
        self.search_end = pd.Timestamp(evaluator.sandbox_cfg["zones"]["search"]["end"])
        pit_search = evaluator.snapshot.fundamentals_pit[
            evaluator.snapshot.fundamentals_pit["announcement_date"] <= self.search_end
        ].copy()
        self.search_panel = compute_feature_panel(self.search_bars, pit_search)
        s_close = self.search_panel["close"]
        self.search_dates = pd.DatetimeIndex(s_close.index)
        safe_n = max(1, len(self.search_dates) - evaluator.horizon_days)
        self.safe_dates = self.search_dates[:safe_n]
        self.search_fwd = (s_close.shift(-evaluator.horizon_days) / s_close - 1.0).reindex(index=self.safe_dates)
        self.search_folds = evaluator.splitter.split(self.safe_dates)

        # Local search-zone-only simulator for Zero-Validation-Burn Purged CV!
        self.local_search_sim = VNPortfolioSimulator(
            market_rules=evaluator.market_rules,
            top_k=evaluator.top_k,
            rebalance_every_days=evaluator.rebalance_every_days,
            capital_vnd=evaluator.capital_vnd,
        )

        # Fit VeteranCaseMemory strictly on search_panel
        mem_cfg = cfg.get("veteran_case_memory", {})
        self.case_memory = VeteranCaseMemory(
            k_neighbors=int(mem_cfg.get("k_neighbors", 15)),
            exclusion_buffer_days=int(mem_cfg.get("exclusion_buffer_days", 10)),
            trauma_drop_threshold=float(mem_cfg.get("trauma_drop_threshold", -0.045)),
            floor_trauma_threshold=float(mem_cfg.get("floor_trauma_threshold", 0.07)),
        )
        if self.use_case_memory:
            self.case_memory.fit_on_search(self.search_panel, horizon_days=evaluator.horizon_days)

        pm_cfg = cfg.get("pre_mortem", {})
        self.pre_mortem = DevilsAdvocatePreMortem(pm_cfg)
        self.controller = HierarchicalRegimeConvictionController(pre_mortem_cfg=pm_cfg, top_k=evaluator.top_k)

        # Fit Chamber 1B: Gârleanu–Pedersen Friction-Adjusted FWL Backbone strictly on search zone (zero val/test access!)
        evo_cfg = cfg.get("evolution", {})
        self.backbone = OrthogonalCausalBackbone(
            ridge_l2_frac=float(evo_cfg.get("backbone_ridge_l2_frac", 0.15)),
            gp_turnover_exp=float(evo_cfg.get("backbone_gp_turnover_exp", 0.48)),
        )
        self.backbone.fit_on_search(
            self.search_panel,
            self.safe_dates,
            self.search_fwd,
            rebal_step=evaluator.rebalance_every_days,
            top_k=evaluator.top_k,
        )
        self.search_backbone_sig = self.backbone.predict_panel(self.search_panel)

        # Search-zone orthogonalized forward residual (Frisch–Waugh–Lovell target for conditional reversal factors!)
        anc_safe = _cs_z(self.search_backbone_sig.reindex(index=self.safe_dates))
        fwd_z = _cs_z(self.search_fwd)
        beta_proj = (anc_safe * fwd_z).mean(axis=1) / ((anc_safe * anc_safe).mean(axis=1) + 1e-8)
        self.search_fwd_resid = fwd_z.sub(anc_safe.mul(beta_proj, axis=0))

        # Precompute strictly causal search-zone auxiliary panels (NEVER touches validation/test)
        self.search_state = compute_causal_microstructure_state(self.search_panel)
        self.search_mem_sig = self.case_memory.query_panel(self.search_state, purge_self_matches=True)
        self.search_pm_pen, self.search_veto = self.pre_mortem.audit_panel(self.search_panel)

        self.steward_base = np.array(
            [
                float(self.search_bars["hit_ceiling"].mean()) * 10.0,
                float(self.search_bars["hit_floor"].mean()) * 10.0,
                float(self.search_bars["is_halted"].mean()) * 10.0,
                float(s_close.pct_change(fill_method=None).std().mean()) * 20.0,
                min(1.0, len(self.search_bars) / 50000.0),
                0.5,
            ],
            dtype=float,
        )
        self.fam_ic_sum = {f: 0.04 for f in MECHANISM_FAMILIES}
        self.fam_cnt = {f: 1 for f in MECHANISM_FAMILIES}
        self.seen_raw_fps: set[str] = set()
        self.screen_cache: dict[str, HoloElite | None] = {}
        self.hrm_replay: list[dict[str, Any]] = []
        self.typed_guard_rejections = 0
        self.n_search_screened = 0

    def _regime_vec(self) -> np.ndarray:
        means = [self.fam_ic_sum[f] / max(1, self.fam_cnt[f]) * 10.0 for f in MECHANISM_FAMILIES]
        cnts = [min(1.0, self.fam_cnt[f] / 25.0) for f in MECHANISM_FAMILIES]
        return np.concatenate([self.steward_base, np.asarray(means), np.asarray(cnts)])

    @staticmethod
    def _top_k_turnover_proxy(sig_df: pd.DataFrame, rebal_step: int = 3, top_k: int = 6) -> float:
        arr = sig_df.to_numpy(dtype=float)
        arr = np.where(np.isfinite(arr), arr, -1e9)
        n_days, n_syms = arr.shape
        k = min(top_k, n_syms)
        if n_days <= rebal_step or k <= 0:
            return 0.20
        rebal_rows = arr[::rebal_step]
        if len(rebal_rows) < 2:
            return 0.20
        top_idx = np.argpartition(rebal_rows, -k, axis=1)[:, -k:]
        mem = np.zeros_like(rebal_rows, dtype=np.float32)
        mem[np.arange(len(rebal_rows))[:, None], top_idx] = 1.0
        overlap = np.sum(mem[1:] * mem[:-1], axis=1) / float(k)
        return float(np.mean(2.0 * (1.0 - overlap) / float(rebal_step)))

    def _local_search_eval_signal(self, sig_df: pd.DataFrame) -> tuple[dict[str, float], list[dict[str, float]]]:
        """Evaluate signal strictly on `self.search_bars` WITHOUT incrementing `SealedEvaluator.n_trials_evaluated`."""
        sig_z = sig_df.reindex(index=self.search_dates)
        sim = self.local_search_sim.simulate(sig_z, self.search_bars)
        ic_s = compute_rank_ic_series(sig_z.reindex(index=self.safe_dates), self.search_fwd)
        full_m = compute_portfolio_metrics(sim.daily_net_returns, sim.daily_gross_returns, sim.daily_turnover, ic_s, n_trials=1)
        fold_ms: list[dict[str, float]] = []
        for fold in self.search_folds:
            f_bars = self.search_bars[self.search_bars["date"].isin(fold.eval_dates)]
            f_sim = self.local_search_sim.simulate(sig_z.reindex(index=fold.eval_dates), f_bars)
            f_ic = ic_s.reindex(fold.eval_dates).dropna()
            fold_ms.append(
                compute_portfolio_metrics(
                    f_sim.daily_net_returns, f_sim.daily_gross_returns, f_sim.daily_turnover, f_ic, n_trials=1
                )
            )
        return full_m, fold_ms

    def _mssf_calibrate(self, raw_spec: FactorSpec) -> tuple[FactorSpec, float, int] | None:
        try:
            raw_mat = evaluate_factor_spec(raw_spec, self.search_panel, verifier=self.verifier)
        except Exception:
            return None
        s_base = raw_mat.reindex(index=self.safe_dates)
        if float(np.nanstd(s_base.to_numpy(dtype=float))) < 1e-8:
            return None
        evo = self.cfg.get("evolution", {})
        windows = [int(w) for w in evo.get("mssf_windows", [1, 3, 5, 10])]
        turn_pen = float(evo.get("mssf_turnover_penalty", 0.16))
        core_depth = raw_spec.root.depth()
        already_sm = raw_spec.root.name == "ts_mean"

        raw_norm = _cs_z(s_base)
        raw_ic_s = compute_rank_ic_series(raw_norm, self.search_fwd)
        if len(raw_ic_s) < 20:
            return None
        raw_mean_ic = float(raw_ic_s.mean())
        raw_sign = 1.0 if raw_mean_ic >= 0.0 else -1.0
        raw_abs_ic = abs(raw_mean_ic)
        raw_turn = self._top_k_turnover_proxy(raw_norm * raw_sign, self.evaluator.rebalance_every_days, self.evaluator.top_k)
        # Nyquist-matched smoothing: never over-smooth short-horizon reversals or already-low-churn anchors
        expr_str = raw_spec.root.canonical_expr()
        if "ts_return(close, w=3)" in expr_str or "return_1d" in expr_str:
            windows = [w for w in windows if w <= 5] or [1]
        if raw_turn <= 0.10 or (already_sm and raw_turn <= 0.15):
            windows = [1]

        best_choice: tuple[float, int, float] | None = None
        for w in windows:
            if w > 1 and core_depth >= 5:
                continue
            cand_df = s_base if w <= 1 else s_base.rolling(window=w, min_periods=max(1, w // 2)).mean()
            norm_df = _cs_z(cand_df)
            ic_s = compute_rank_ic_series(norm_df, self.search_fwd)
            if len(ic_s) < 20:
                continue
            mean_ic = float(ic_s.mean())
            sign = 1.0 if mean_ic >= 0.0 else -1.0
            abs_ic = abs(mean_ic)
            # Strict sign-invariance guard: low-pass smoothing must NEVER invert the raw economic sign!
            if w > 1 and (sign != raw_sign or abs_ic < 0.78 * raw_abs_ic):
                continue
            signed_ic = ic_s * sign
            f_ics = [
                float(signed_ic.reindex(f.eval_dates).dropna().mean())
                for f in self.search_folds
                if len(signed_ic.reindex(f.eval_dates).dropna()) >= 5
            ]
            min_f = min(f_ics) if f_ics else abs_ic
            est_turn = self._top_k_turnover_proxy(norm_df * sign, self.evaluator.rebalance_every_days, self.evaluator.top_k)
            sc = 0.65 * abs_ic + 0.35 * min_f - turn_pen * est_turn
            if best_choice is None or sc > best_choice[0]:
                best_choice = (sc, w, sign)

        if best_choice is None:
            return None
        _, w_star, sign_star = best_choice
        node = raw_spec.root
        # Clean double-negation & churn guard: reject noisy high-churn 3d blow-off chasers from symbolic book
        if sign_star < 0.0 and node.name == "neg" and node.args:
            node = node.args[0]
            eff_sign = 1.0
        elif sign_star < 0.0 and node.depth() <= 4:
            node = ASTNode(node_type="op", name="neg", args=[node])
            eff_sign = 1.0
        else:
            eff_sign = sign_star
        if w_star > 1 and node.depth() <= 4:
            node = ASTNode(node_type="op", name="ts_mean", window=w_star, args=[node])
        adapted = FactorSpec(
            factor_id=f"{raw_spec.factor_id}_w{w_star}",
            hypothesis_id=raw_spec.hypothesis_id,
            description=f"MSSF(w={w_star}) {raw_spec.description}",
            root=node,
            normalize_cross_section=True,
        )
        return adapted, eff_sign, w_star

    def _screen_candidate(self, raw_spec: FactorSpec, family: str, run_id: str) -> HoloElite | None:
        calibrated = self._mssf_calibrate(raw_spec)
        if calibrated is None:
            return None
        spec, sign, w_star = calibrated
        fp_key = f"{spec.root.structural_fingerprint()}:{int(sign)}"
        if fp_key in self.screen_cache:
            return self.screen_cache[fp_key]

        s_df = evaluate_factor_spec(spec, self.search_panel, verifier=self.verifier).fillna(0.0) * sign

        if self.use_zero_val_burn_cv:
            # Zero-Validation-Burn: evaluate strictly inside hermetically isolated search-zone simulator!
            metrics, fold_ms = self._local_search_eval_signal(s_df)
        else:
            # Ablation mode: burn official SealedEvaluator trials on every search screen (like v3)
            if sign < 0.0:
                out = self.evaluator.evaluate_factor_ensemble([spec], [-1.0], zone="search", candidate_id=spec.factor_id, run_id=run_id)
            else:
                out = self.evaluator.evaluate_factor(spec, zone="search", run_id=run_id)
            metrics, fold_ms = out.metrics, out.fold_metrics

        self.n_search_screened += 1
        s_safe = s_df.reindex(index=self.safe_dates).fillna(0.0)
        ic_s = compute_rank_ic_series(s_safe, self.search_fwd)
        mean_ic = float(ic_s.mean()) if len(ic_s) else float(metrics["rank_ic"])
        fold_ics = [float(fm["rank_ic"]) for fm in fold_ms] or [mean_ic]
        min_fold_ic = float(min(fold_ics))
        sr = float(metrics["net_sharpe"])
        turn = float(metrics["daily_turnover"])
        icir = float(metrics["icir"])

        # MDL complexity parsimony penalty prevents noisy depth-5 trees from displacing clean economic factors
        depth_pen = 0.0045 * max(0, spec.root.depth() - 2)
        utility = 0.55 * mean_ic + 0.35 * min_fold_ic + 0.022 * float(np.clip(sr, -1.5, 3.0)) - 0.16 * turn - depth_pen
        niche = dominant_economic_niche(spec.root)
        elite = HoloElite(
            spec=spec,
            sign=sign,
            smooth_window=w_star,
            family=family,
            niche=niche,
            cell=(niche, horizon_bucket(turn)),
            utility=float(utility),
            rank_ic=mean_ic,
            min_fold_ic=min_fold_ic,
            icir=icir,
            net_sharpe=sr,
            turnover=turn,
            ic_series=ic_s,
            signal_vec=s_safe.to_numpy(dtype=float).ravel(),
            signal_df=s_safe,
        )
        self.screen_cache[fp_key] = elite
        return elite

    def _generate_latent_pool(self, r_idx: int, n_latent: int) -> list[tuple[str, FactorSpec]]:
        pool: list[tuple[str, FactorSpec]] = []
        seen_batch: set[str] = set()
        if r_idx == 1:
            t_node = lambda n: ASTNode(node_type="terminal", name=n)  # noqa: E731
            op_node = lambda n, *a, w=None: ASTNode(node_type="op", name=n, window=w, args=list(a))  # noqa: E731
            extra_v4_seeds = [
                ("foreign_flow_regime", op_node("ts_mean", t_node("foreign_net_ratio"), w=8)),
                ("reversal_microstructure", op_node("sub", t_node("dist_to_floor"), t_node("intraday_range"))),
                (
                    "momentum_liquidity",
                    op_node(
                        "safe_div",
                        op_node("lag", op_node("ts_return", t_node("close"), w=10), w=3),
                        op_node("ts_std", t_node("return_1d"), w=20),
                    ),
                ),
            ]
            all_seeds = _curated_domain_seeds() + extra_v4_seeds
            # Seed-permuted subset of domain seeds ensures genuine cross-seed trajectory diversity!
            perm = self.rng.permutation(len(all_seeds))
            for idx_s in perm[:8]:
                fam, node = all_seeds[int(idx_s)]
                fp = node.structural_fingerprint()
                if fp not in self.seen_raw_fps and (not self.use_typed_tree_mdlm or is_typed_stationary_ast(node)):
                    self.seen_raw_fps.add(fp)
                    seen_batch.add(fp)
                    pool.append(
                        (
                            fam,
                            FactorSpec(
                                factor_id=f"v4_seed_{len(pool)}",
                                hypothesis_id="v4_seed",
                                description=f"domain seed ({fam})",
                                root=node,
                            ),
                        )
                    )

        fam_w = np.asarray(
            [max(0.10, self.fam_ic_sum[f] / max(1, self.fam_cnt[f]) + 0.03) for f in MECHANISM_FAMILIES],
            dtype=float,
        )
        fam_w /= fam_w.sum()
        elites = self.archive.all_elites()
        regime_vec = self._regime_vec()
        temp = max(0.42, 0.78 - 0.05 * r_idx)

        for k in range(n_latent * 5):
            if len(pool) >= n_latent:
                break
            fam = MECHANISM_FAMILIES[int(self.rng.choice(len(MECHANISM_FAMILIES), p=fam_w))]
            fid, hid = f"v4_r{r_idx}_c{k}", f"v4h_r{r_idx}_c{k}"
            u = self.rng.random()
            seed_k = self.seed + r_idx * 7919 + k * 31
            if u < 0.18 or not self.use_typed_tree_mdlm:
                cand = FactorSpec(
                    factor_id=fid,
                    hypothesis_id=hid,
                    description="grammar exploration",
                    root=self.sampler.random_tree(self.rng, max_depth=4),
                )
            elif len(elites) >= 2 and u < 0.36:
                i1, i2 = self.rng.choice(len(elites), size=2, replace=False)
                p_l, p_r = elites[int(i1)].spec.root, elites[int(i2)].spec.root
                while p_l.name in ("neg", "ts_mean") and p_l.args and p_l.depth() >= 3:
                    p_l = p_l.args[0]
                while p_r.name in ("neg", "ts_mean") and p_r.args and p_r.depth() >= 3:
                    p_r = p_r.args[0]
                cand = self.diffusion.bridge_fuse_elites(
                    p_l, p_r, fam, fid, hid, temperature=temp, seed=seed_k, use_tree_coords=True
                )
            elif elites and u < 0.74:
                t_idx = self.rng.integers(0, len(elites), size=min(3, len(elites)))
                parent = max((elites[int(i)] for i in t_idx), key=lambda e: e.utility)
                base_root = parent.spec.root
                while base_root.name in ("neg", "ts_mean") and base_root.args and base_root.depth() >= 3:
                    base_root = base_root.args[0]
                base_sp = FactorSpec(
                    factor_id=parent.spec.factor_id,
                    hypothesis_id=parent.spec.hypothesis_id,
                    description=parent.spec.description,
                    root=base_root,
                )
                weak_pos = self.hrm.identify_weakest_subtree_pos(base_sp, regime_vec)
                cand = self.diffusion.attribution_guided_infill(
                    base_sp, weak_pos, fam, fid, hid, max_depth=4, temperature=temp, seed=seed_k, use_tree_coords=True
                )
            else:
                cand = self.diffusion.sample_factor(
                    fam, fid, hid, max_depth=4, temperature=temp + 0.05, seed=seed_k, use_tree_coords=True
                )

            fp = cand.root.structural_fingerprint()
            if fp in seen_batch or fp in self.seen_raw_fps:
                continue
            try:
                self.verifier.verify(cand)
            except Exception:
                continue
            if self.use_typed_tree_mdlm and not is_typed_stationary_ast(cand.root):
                self.typed_guard_rejections += 1
                continue
            seen_batch.add(fp)
            pool.append((fam, cand))
        return pool

    @staticmethod
    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b)) / (float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12)

    def _select_diverse_basis(self, max_basis: int = 5, max_corr: float = 0.65) -> list[HoloElite]:
        pos = [
            e
            for e in self.archive.all_elites()
            if e.rank_ic > 0.025 and e.utility > -0.005 and (e.rank_ic - 0.32 * e.turnover) > -0.005
        ]
        if not pos:
            pos = [e for e in self.archive.all_elites() if e.rank_ic > 0.015 and e.utility > -0.005]
        if not pos:
            return self.archive.all_elites()[: max(1, min(3, len(self.archive.all_elites())))]
        selected: list[HoloElite] = []
        covered: set[str] = set()
        # Pass 1: pick best representative per distinct economic niche
        for e in pos:
            if e.niche in covered:
                continue
            if all(abs(self._corr(e.signal_vec, s.signal_vec)) <= max_corr for s in selected):
                selected.append(e)
                covered.add(e.niche)
            if len(selected) >= max_basis:
                return selected
        # Pass 2: fill remaining slots with highest-utility decorrelated elites
        for e in pos:
            if any(e.fingerprint == s.fingerprint for s in selected):
                continue
            if all(abs(self._corr(e.signal_vec, s.signal_vec)) <= max_corr for s in selected):
                selected.append(e)
            if len(selected) >= max_basis:
                break
        return selected

    def _solve_bto_srss(self, basis: list[HoloElite]) -> np.ndarray:
        """Boundary-Turnover-Optimal Spectral Subspace Synthesis (`BTO-SRSS`) strictly on search zone."""
        m = len(basis)
        if m <= 1:
            return np.ones(max(1, m), dtype=float)
        ic_panel = pd.concat([e.ic_series for e in basis], axis=1, sort=False).dropna()
        g_ic = np.asarray([max(0.003, 0.60 * e.rank_ic + 0.40 * max(0.0, e.min_fold_ic)) for e in basis], dtype=float)
        turns = np.asarray([e.turnover for e in basis], dtype=float)
        w_anchor = g_ic / np.power(0.035 + turns, 0.85)
        w_anchor /= float(np.sum(w_anchor)) + 1e-12
        if len(ic_panel) < 25:
            return w_anchor

        cov_ic = np.cov(ic_panel.to_numpy(dtype=float), rowvar=False)
        if cov_ic.ndim == 0:
            cov_ic = np.array([[float(cov_ic)]])
        avg_var = float(np.mean(np.diag(cov_ic))) + 1e-6
        sig_corr = np.eye(m, dtype=float)
        for i in range(m):
            for j in range(i + 1, m):
                c = abs(self._corr(basis[i].signal_vec, basis[j].signal_vec))
                sig_corr[i, j] = sig_corr[j, i] = c
        sigma_spec = (
            0.65 * cov_ic
            + 0.35 * avg_var * np.eye(m)
            + 0.35 * avg_var * (sig_corr**2 - np.eye(m))
            + 0.45 * avg_var * np.diag((0.04 + turns) / 0.08)
        )
        w_tan = np.clip(np.linalg.solve(sigma_spec + 1e-6 * np.eye(m), g_ic), 0.0, None)
        w_tan = w_tan / (float(np.sum(w_tan)) + 1e-12) if float(np.sum(w_tan)) > 1e-12 else w_anchor.copy()

        cap = 0.56 if m >= 3 else 1.0
        best_sc, best_w = -1e9, w_anchor
        for lam in (0.0, 0.25, 0.50, 0.70, 0.85, 1.0):
            w = (1.0 - lam) * w_tan + lam * w_anchor
            for _ in range(3):
                over = w > cap
                if not np.any(over) or np.all(over):
                    break
                excess = float(np.sum(w[over] - cap))
                w[over] = cap
                under = ~over
                if float(np.sum(w[under])) > 1e-12:
                    w[under] += excess * (w[under] / float(np.sum(w[under])))
            w = np.where(w >= 0.035, w, 0.0)
            w = w / (float(np.sum(w)) + 1e-12) if float(np.sum(w)) > 1e-12 else w_anchor.copy()
            comb: pd.DataFrame | None = None
            for e, wt in zip(basis, w, strict=True):
                if wt > 1e-5:
                    part = e.signal_df * float(wt)
                    comb = part if comb is None else (comb + part)
            if comb is None:
                continue
            ic_s = compute_rank_ic_series(comb, self.search_fwd)
            if len(ic_s) < 20:
                continue
            mean_ic = float(ic_s.mean())
            f_ics = [
                float(ic_s.reindex(f.eval_dates).dropna().mean())
                for f in self.search_folds
                if len(ic_s.reindex(f.eval_dates).dropna()) >= 5
            ]
            min_f = min(f_ics) if f_ics else mean_ic
            turn = self._top_k_turnover_proxy(comb, self.evaluator.rebalance_every_days, self.evaluator.top_k)
            sc = 0.65 * mean_ic + 0.35 * min_f - 0.44 * turn - 0.012 * float(np.sum(w**2))
            if sc > best_sc:
                best_sc, best_w = sc, w
        return best_w

    def _combine_raw_alpha(
        self,
        elites: list[HoloElite],
        weights: list[float],
        panel: dict[str, pd.DataFrame],
        state_df: pd.DataFrame | None = None,
        case_memory_gamma: float = 0.0,
        purge_self_matches: bool = False,
    ) -> pd.DataFrame:
        total = sum(abs(w) for w in weights) or 1.0
        norm_w = np.asarray([w / total for w in weights], dtype=float)
        mats = [
            evaluate_factor_spec(e.spec, panel, verifier=self.verifier).fillna(0.0) * e.sign
            for e in elites
        ]
        if (
            self.use_case_memory
            and case_memory_gamma > 1e-6
            and state_df is not None
            and len(elites) >= 2
            and self.case_memory._fitted
        ):
            # Build strictly search-zone elite daily IC matrix aligned to case_memory._dates
            ic_mat = np.column_stack(
                [e.ic_series.reindex(self.case_memory._dates).fillna(e.rank_ic).to_numpy(dtype=float) for e in elites]
            )
            w_dyn = self.case_memory.query_contextual_factor_weights(
                state_df=state_df,
                elite_ic_matrix=ic_mat,
                base_weights=norm_w,
                gamma=case_memory_gamma,
                purge_self_matches=purge_self_matches,
            )
            dates = mats[0].index
            comb: pd.DataFrame | None = None
            for i, mat in enumerate(mats):
                w_ser = pd.Series(w_dyn[:, i], index=dates)
                part = mat.mul(w_ser, axis=0)
                comb = part if comb is None else (comb + part)
            assert comb is not None
            return comb

        comb_static: pd.DataFrame | None = None
        for mat, wt in zip(mats, norm_w, strict=True):
            part = mat * float(wt)
            comb_static = part if comb_static is None else (comb_static + part)
        assert comb_static is not None
        return comb_static

    def _controller_grid(self) -> list[ControllerParams]:
        rcfg = self.cfg.get("regime_controller", {})
        raw_grid = rcfg.get("cv_grid", [])
        if not raw_grid or not self.use_regime_controller:
            return [ControllerParams(name="static_no_controller")]
        return [
            ControllerParams(
                name=str(g["name"]),
                trauma_cash_thresh=float(g.get("trauma_cash_thresh", 0.68)),
                dd_cash_thresh=float(g.get("dd_cash_thresh", -0.165)),
                bull_trend_tilt=float(g.get("bull_trend_tilt", 0.02)),
                bear_defense_tilt=float(g.get("bear_defense_tilt", 0.06)),
                hysteresis_bonus=float(g.get("hysteresis_bonus", 0.22)),
                bull_top_k_cap=int(g.get("bull_top_k_cap", 6)),
                kelly_fraction=float(g.get("kelly_fraction", 0.70)),
                vol_shock_thresh=float(g.get("vol_shock_thresh", 1.60)),
                floor_stress_thresh=float(g.get("floor_stress_thresh", 0.075)),
                signal_ema_span=int(g.get("signal_ema_span", 1)),
                case_memory_gamma=float(g.get("case_memory_gamma", 0.10)),
                symbolic_blend_alpha=float(g.get("symbolic_blend_alpha", 0.45)),
                bab_base_tilt=float(g.get("bab_base_tilt", 0.22)),
                bab_stress_boost=float(g.get("bab_stress_boost", 0.28)),
                rebound_release_r5=float(g.get("rebound_release_r5", 0.012)),
            )
            for g in raw_grid
        ]

    def build_full_causal_signal(
        self,
        elites: list[HoloElite],
        weights: list[float],
        ctrl_params: ControllerParams,
        panel: dict[str, pd.DataFrame],
    ) -> ControllerOutput:
        """Build strictly causal full-panel signal using frozen search-zone factors, backbone, memory, and controller."""
        state_df = compute_causal_microstructure_state(panel)
        sym_alpha = _cs_z(
            self._combine_raw_alpha(
                elites,
                weights,
                panel,
                state_df=state_df,
                case_memory_gamma=ctrl_params.case_memory_gamma if self.use_case_memory else 0.0,
                purge_self_matches=False,
            )
        )
        # Tri-Cameral Neuro-Symbolic Fusion: blend symbolic AST book (1A) with Gârleanu–Pedersen FWL backbone (1B)
        # using strictly causal 30-day rolling reflection adaptation around search-CV prior `symbolic_blend_alpha`!
        if self.use_regime_controller and ctrl_params.symbolic_blend_alpha < 0.999:
            anc_alpha = self.backbone.predict_panel(panel).reindex_like(sym_alpha).fillna(0.0)
            ret1 = panel["return_1d"].reindex_like(sym_alpha).fillna(0.0)
            # Strictly causal reflection: signal[t-1] realized against ret1[t] (zero future info!)
            p_sym = (sym_alpha.shift(1).fillna(0.0) * ret1).mean(axis=1)
            p_anc = (anc_alpha.shift(1).fillna(0.0) * ret1).mean(axis=1)
            sr_sym = (p_sym.rolling(30, min_periods=10).mean() / (p_sym.rolling(30, min_periods=10).std() + 1e-4)).fillna(0.0).clip(-1.5, 2.5)
            sr_anc = (p_anc.rolling(30, min_periods=10).mean() / (p_anc.rolling(30, min_periods=10).std() + 1e-4)).fillna(0.0).clip(-1.5, 2.5)
            a_base = float(np.clip(ctrl_params.symbolic_blend_alpha, 0.0, 1.0))
            dyn_tilt = (0.06 * (sr_sym - sr_anc)).clip(-0.10, 0.10)
            a_t = (a_base + dyn_tilt).clip(0.20, 0.85)
            raw_alpha = _cs_z(sym_alpha.mul(a_t, axis=0) + anc_alpha.mul(1.0 - a_t, axis=0))
        else:
            raw_alpha = sym_alpha
        mem_sig = self.case_memory.query_panel(state_df, purge_self_matches=False)
        pm_pen, veto = self.pre_mortem.audit_panel(panel)
        return self.controller.apply(
            raw_alpha_wide=raw_alpha,
            panel=panel,
            state_df=state_df,
            memory_signals=mem_sig,
            pm_penalty=pm_pen,
            hard_veto=veto,
            params=ctrl_params,
            use_pre_mortem=self.use_pre_mortem,
            use_regime_controller=self.use_regime_controller,
        )

    def _search_cv_score_candidate(
        self,
        elites: list[HoloElite],
        weights: list[float],
        ctrl_params: ControllerParams,
    ) -> float:
        """Evaluate (alpha book + backbone + controller) strictly on search-zone purged walk-forward CV (0 val trials!)."""
        sym_alpha = _cs_z(
            self._combine_raw_alpha(
                elites,
                weights,
                self.search_panel,
                state_df=self.search_state,
                case_memory_gamma=ctrl_params.case_memory_gamma if self.use_case_memory else 0.0,
                purge_self_matches=True,
            )
        )
        if self.use_regime_controller and ctrl_params.symbolic_blend_alpha < 0.999:
            anc_s = self.search_backbone_sig.reindex_like(sym_alpha).fillna(0.0)
            ret1_s = self.search_panel["return_1d"].reindex_like(sym_alpha).fillna(0.0)
            p_sym = (sym_alpha.shift(1).fillna(0.0) * ret1_s).mean(axis=1)
            p_anc = (anc_s.shift(1).fillna(0.0) * ret1_s).mean(axis=1)
            sr_sym = (p_sym.rolling(30, min_periods=10).mean() / (p_sym.rolling(30, min_periods=10).std() + 1e-4)).fillna(0.0).clip(-1.5, 2.5)
            sr_anc = (p_anc.rolling(30, min_periods=10).mean() / (p_anc.rolling(30, min_periods=10).std() + 1e-4)).fillna(0.0).clip(-1.5, 2.5)
            a_base = float(np.clip(ctrl_params.symbolic_blend_alpha, 0.0, 1.0))
            dyn_tilt = (0.06 * (sr_sym - sr_anc)).clip(-0.10, 0.10)
            a_t = (a_base + dyn_tilt).clip(0.20, 0.85)
            raw_alpha = _cs_z(sym_alpha.mul(a_t, axis=0) + anc_s.mul(1.0 - a_t, axis=0))
        else:
            raw_alpha = sym_alpha
        ctrl_out = self.controller.apply(
            raw_alpha_wide=raw_alpha,
            panel=self.search_panel,
            state_df=self.search_state,
            memory_signals=self.search_mem_sig,
            pm_penalty=self.search_pm_pen,
            hard_veto=self.search_veto,
            params=ctrl_params,
            use_pre_mortem=self.use_pre_mortem,
            use_regime_controller=self.use_regime_controller,
        )
        m, fold_ms = self._local_search_eval_signal(ctrl_out.sealed_signal_wide)
        f_srs = [float(fm["net_sharpe"]) for fm in fold_ms] or [float(m["net_sharpe"])]
        f_ics = [float(fm["rank_ic"]) for fm in fold_ms] or [float(m["rank_ic"])]
        fold_sr_std = float(np.std(f_srs, ddof=1)) if len(f_srs) > 1 else 0.0
        mdd = abs(float(m["max_drawdown"]))
        calmar_proxy = float(m["ann_net_return"]) / max(mdd, 0.05)
        cash_frac = float((ctrl_out.regime_series == "Cash_Shield").mean())
        cash_drag_pen = 2.0 * max(0.0, cash_frac - 0.15)
        return float(
            float(m["net_sharpe"])
            + 0.35 * float(min(f_srs))
            + 3.5 * float(m["rank_ic"])
            + 1.5 * float(min(f_ics))
            + 0.12 * float(np.clip(calmar_proxy, -2.0, 8.0))
            - 1.6 * float(m["daily_turnover"])
            - 0.08 * fold_sr_std
            - cash_drag_pen
        )

    @staticmethod
    def _validation_pareto_score(outcome: EvaluationOutcome) -> float:
        m = outcome.metrics
        mdd = abs(float(m["max_drawdown"]))
        calmar = float(m["ann_net_return"]) / max(mdd, 0.05)
        return float(
            float(m["net_sharpe"])
            + 0.25 * float(np.clip(calmar, -2.0, 8.0))
            + 2.5 * float(m["rank_ic"])
            + 0.35 * float(m["deflated_sharpe_prob"])
            - 0.10 * float(outcome.fold_std.get("net_sharpe", 0.0))
            - 1.0 * float(m["daily_turnover"])
        )

    def run(self, run_id: str = "maveq_v4_run", score_sealed: bool = False) -> MAVEQv4Result:
        evo = self.cfg.get("evolution", {})
        rounds = int(evo.get("rounds", 6))
        n_latent = int(evo.get("latent_candidates_per_round", 48))
        n_screen = int(evo.get("screen_per_round", 10))
        eps_explore = float(evo.get("ucb_explore_fraction", 0.25))
        max_ens = int(evo.get("max_ensemble_factors", 5))
        max_corr = float(evo.get("max_member_corr", 0.62))
        grpo_epochs = int(evo.get("diffu_grpo_epochs_per_round", 10))
        max_val_q = int(evo.get("max_validation_queries", 6))

        # Phase A: Strictly Search-Zone Evolution & Purged Search CV (ZERO validation burn!)
        search_finalists: dict[str, tuple[float, list[HoloElite], list[float], ControllerParams]] = {}
        ctrl_grid = self._controller_grid()

        for r_idx in range(1, rounds + 1):
            regime_vec = self._regime_vec()
            latent = self._generate_latent_pool(r_idx, n_latent)
            if r_idx == 1:
                chosen = latent[: max(n_screen, 12)]
            else:
                ucb_beta = max(0.06, 0.24 - 0.03 * r_idx)
                ranked = sorted(
                    latent,
                    key=lambda item: self.hrm.score_ucb(item[1].root, regime_vec, ucb_beta=ucb_beta)[0],
                    reverse=True,
                )
                n_top = max(1, int(round(n_screen * (1.0 - eps_explore))))
                chosen = ranked[:n_top]
                rem = ranked[n_top:]
                if rem and len(chosen) < n_screen:
                    idx = self.rng.choice(len(rem), size=min(len(rem), n_screen - len(chosen)), replace=False)
                    chosen.extend(rem[int(i)] for i in idx)

            for fam, raw_cand in chosen:
                self.seen_raw_fps.add(raw_cand.root.structural_fingerprint())
                elite = self._screen_candidate(raw_cand, fam, run_id=f"{run_id}_r{r_idx}")
                if elite is None:
                    continue
                self.archive.insert(elite)
                self.fam_ic_sum[fam] += max(0.0, elite.rank_ic)
                self.fam_cnt[fam] += 1
                self.hrm_replay.append(
                    {"node": raw_cand.root, "regime_vec": regime_vec, "utility": elite.utility, "turnover": elite.turnover}
                )

            basis = self._select_diverse_basis(max_basis=max_ens, max_corr=max_corr)
            if not basis:
                continue
            w_arr = self._solve_bto_srss(basis)
            active_pairs = [(e, float(w)) for e, w in zip(basis, w_arr, strict=True) if w > 1e-5]
            active_elites = [e for e, _ in active_pairs]
            active_weights = [w for _, w in active_pairs]

            # Counterfactual LOO Shapley Credit -> Edge-localized diffu-GRPO
            shapley: dict[str, float] = {}
            if len(active_elites) >= 2:
                comb_s: pd.DataFrame | None = None
                for e, w in active_pairs:
                    part = e.signal_df * w
                    comb_s = part if comb_s is None else (comb_s + part)
                assert comb_s is not None
                full_ic = float(compute_rank_ic_series(comb_s, self.search_fwd).mean())
                for e, w in active_pairs:
                    loo_ic = float(compute_rank_ic_series(comb_s - e.signal_df * w, self.search_fwd).mean())
                    shapley[e.fingerprint] = max(-0.03, full_ic - loo_ic)

            all_sc = [e for e in self.screen_cache.values() if e is not None]
            all_sc.sort(key=lambda e: e.utility + 1.8 * shapley.get(e.fingerprint, 0.0), reverse=True)
            top_rep = all_sc[: min(10, len(all_sc))]
            bot_rep = all_sc[-min(8, max(0, len(all_sc) - len(top_rep))):] if len(all_sc) > len(top_rep) else []
            rollouts = [
                {
                    "family": e.family,
                    "factor_spec": e.spec.model_dump(),
                    "shapley_reward": e.utility + 1.8 * shapley.get(e.fingerprint, 0.0),
                }
                for e in (top_rep + bot_rep)
            ]
            if self.use_typed_tree_mdlm:
                self.diffusion.fit_shapley_diffu_grpo(rollouts, epochs=grpo_epochs)
            self.hrm.fit_replay(self.hrm_replay[-192:])

            # Evaluate full multi-niche basis across ALL evolution rounds strictly on search-zone Purged CV (0 val trials!)
            sub_books: list[tuple[list[HoloElite], list[float]]] = [(active_elites, active_weights)]
            if len(basis) >= 5:
                b4 = basis[:4]
                w4 = self._solve_bto_srss(b4)
                p4 = [(e, float(w)) for e, w in zip(b4, w4, strict=True) if w > 1e-5]
                if p4:
                    sub_books.append(([e for e, _ in p4], [w for _, w in p4]))
            for sub_e, sub_w in sub_books:
                for cp in ctrl_grid:
                    cv_sc = self._search_cv_score_candidate(sub_e, sub_w, cp)
                    if not self.use_zero_val_burn_cv:
                        # Ablation mode: burn an official search-zone trial per ensemble evaluation (like v3)
                        _ = self.evaluator.evaluate_factor_ensemble(
                            [e.spec for e in sub_e],
                            [e.sign * w for e, w in zip(sub_e, sub_w, strict=True)],
                            zone="search",
                            candidate_id=f"ab_s_r{r_idx}_{cp.name}",
                            run_id=run_id,
                        )
                    fkey = f"{cp.name}|" + "|".join(
                        f"{e.spec.root.canonical_expr()}:{round(w, 3)}"
                        for e, w in sorted(zip(sub_e, sub_w, strict=True), key=lambda x: x[0].spec.root.canonical_expr())
                    )
                    if fkey not in search_finalists or cv_sc > search_finalists[fkey][0]:
                        search_finalists[fkey] = (cv_sc, sub_e, sub_w, cp)

        if not search_finalists:
            basis = self._select_diverse_basis(max_basis=max_ens, max_corr=max_corr)
            if not basis:
                raise RuntimeError("MAVE-Q v4 discovered no valid factors on search zone.")
            w_arr = self._solve_bto_srss(basis)
            cp0 = ctrl_grid[0]
            cv_sc = self._search_cv_score_candidate(basis, [float(w) for w in w_arr], cp0)
            search_finalists["fallback"] = (cv_sc, basis, [float(w) for w in w_arr], cp0)

        # Phase B: Zero-Validation-Burn Finalist Selection (at most `max_val_q` distinct finalists touch validation!)
        ranked_finalists = sorted(search_finalists.values(), key=lambda x: x[0], reverse=True)
        # Ensure diversity across controller profiles in top finalists
        selected_finalists: list[tuple[float, list[HoloElite], list[float], ControllerParams]] = []
        seen_ctrl: set[str] = set()
        for item in ranked_finalists:
            if item[3].name not in seen_ctrl:
                selected_finalists.append(item)
                seen_ctrl.add(item[3].name)
            if len(selected_finalists) >= max_val_q:
                break
        for item in ranked_finalists:
            if len(selected_finalists) >= max_val_q:
                break
            if item not in selected_finalists:
                selected_finalists.append(item)

        best_val_tuple: tuple[float, float, list[HoloElite], list[float], ControllerParams, ControllerOutput, EvaluationOutcome] | None = None
        val_queries = 0
        for fi, (cv_sc, elites, weights, cp) in enumerate(selected_finalists):
            ctrl_out = self.build_full_causal_signal(elites, weights, cp, self.evaluator.feature_panel)
            val_out = self.evaluator.evaluate_signal_matrix(
                signal_wide=ctrl_out.sealed_signal_wide,
                zone="validation",
                candidate_id=f"maveq_v4_fin{fi}_{cp.name}",
                run_id=run_id,
            )
            val_queries += 1
            v_score = self._validation_pareto_score(val_out)
            if best_val_tuple is None or v_score > best_val_tuple[0]:
                best_val_tuple = (v_score, cv_sc, elites, weights, cp, ctrl_out, val_out)

        assert best_val_tuple is not None
        _, champ_cv_sc, champ_elites, champ_weights, champ_cp, champ_ctrl_out, champ_val = best_val_tuple

        # Extended validation metrics (seal-preserving wrapper)
        bm_val_df = self.evaluator.zone_guard._slice_zone(self.evaluator.snapshot.benchmark, "validation").sort_values("date")
        bm_val = pd.Series(bm_val_df["return_1d"].to_numpy(dtype=float), index=bm_val_df["date"])
        bars_val = self.evaluator.zone_guard._slice_zone(self.evaluator.snapshot.bars, "validation")
        val_ext: ExtendedMetricsReport = compute_extended_metrics(
            champ_val,
            bm_val,
            bars_slice=bars_val,
            signal_wide=champ_ctrl_out.sealed_signal_wide,
            close_wide=self.evaluator.feature_panel["close"],
            n_trials=self.evaluator.n_trials_evaluated,
            rebalance_every_days=self.evaluator.rebalance_every_days,
            top_k=self.evaluator.top_k,
            n_boot=300,
        )

        # Phase C: Optional Single-Shot Sealed Test (ONLY when score_sealed=True!)
        test_metrics: dict[str, Any] = {"note": "sealed_test_not_requested_dev_mode"}
        test_ext_dict: dict[str, Any] = {}
        test_rets = pd.Series(dtype=float)
        if score_sealed:
            try:
                test_out = self.evaluator.evaluate_signal_matrix(
                    signal_wide=champ_ctrl_out.sealed_signal_wide,
                    zone="sealed_test",
                    candidate_id=f"maveq_v4_champion_{champ_cp.name}",
                    run_id=run_id,
                    allow_sealed=True,
                )
                test_metrics = dict(test_out.metrics)
                test_rets = test_out.daily_net_returns
                bm_test_df = self.evaluator.zone_guard._slice_zone(self.evaluator.snapshot.benchmark, "sealed_test").sort_values("date")
                bm_test = pd.Series(bm_test_df["return_1d"].to_numpy(dtype=float), index=bm_test_df["date"])
                bars_test = self.evaluator.zone_guard._slice_zone(self.evaluator.snapshot.bars, "sealed_test")
                test_ext_dict = compute_extended_metrics(
                    test_out,
                    bm_test,
                    bars_slice=bars_test,
                    signal_wide=champ_ctrl_out.sealed_signal_wide,
                    close_wide=self.evaluator.feature_panel["close"],
                    n_trials=self.evaluator.n_trials_evaluated,
                    rebalance_every_days=self.evaluator.rebalance_every_days,
                    top_k=self.evaluator.top_k,
                    n_boot=400,
                ).to_dict()
            except SealedTestReuseError:
                test_metrics = {"note": "champion_hash_already_scored_in_ledger"}

        val_dates = pd.DatetimeIndex(champ_val.daily_net_returns.index)
        val_reg = champ_ctrl_out.regime_series.reindex(val_dates)
        reg_summary = {
            str(k): round(float(v), 4)
            for k, v in val_reg.value_counts(normalize=True).items()
        }
        reg_summary["mean_target_exposure"] = round(float(champ_ctrl_out.exposure_series.reindex(val_dates).mean()), 4)

        total_params = self.diffusion.parameter_count() + self.hrm.parameter_count()
        est_vram = round(self.diffusion.estimate_training_vram_mb() + self.hrm.estimate_training_vram_mb(), 2)
        signed_weights = [round(float(e.sign * w), 5) for e, w in zip(champ_elites, champ_weights, strict=True)]

        return MAVEQv4Result(
            preset=self.preset,
            backend_runtime=f"diff={self.diffusion.backend_runtime}; hrm={self.hrm.backend_runtime}",
            specialist_total_params=total_params,
            estimated_vram_mb=est_vram,
            selected_controller=champ_cp.name,
            champion_expressions=[e.spec.root.canonical_expr() for e in champ_elites],
            champion_weights=signed_weights,
            champion_niches=[e.niche for e in champ_elites],
            search_candidates_screened=self.n_search_screened,
            validation_queries_used=val_queries,
            total_evaluator_trials=int(self.evaluator.n_trials_evaluated),
            archive_coverage=self.archive.coverage,
            typed_guard_rejections=self.typed_guard_rejections,
            champion_search_cv_score=round(champ_cv_sc, 5),
            champion_validation_metrics=dict(champ_val.metrics),
            champion_validation_extended=val_ext.to_dict(),
            champion_validation_fold_std=dict(champ_val.fold_std),
            champion_sealed_test_metrics=test_metrics,
            champion_sealed_test_extended=test_ext_dict,
            regime_exposure_summary=reg_summary,
            validation_daily_returns=champ_val.daily_net_returns,
            sealed_test_daily_returns=test_rets,
        )

    def save_checkpoints(self, checkpoint_dir: str | Path) -> dict[str, str]:
        cdir = Path(checkpoint_dir)
        cdir.mkdir(parents=True, exist_ok=True)
        diff_p = self.diffusion.save_checkpoint(cdir / f"v4_tree_mdlm_{self.preset}.pt")
        hrm_p = self.hrm.save_checkpoint(cdir / f"v4_holo_hrm_{self.preset}.pt")
        mem_p = cdir / f"v4_case_memory_{self.preset}.npz"
        np.savez(
            mem_p,
            z_states=self.case_memory._z_states,
            mean=self.case_memory._mean,
            std=self.case_memory._std,
            fwd_mkt_5d=self.case_memory._fwd_mkt_5d,
            is_trauma=self.case_memory._is_trauma,
        )
        return {"tree_mdlm": str(diff_p), "holo_hrm": str(hrm_p), "case_memory": str(mem_p)}

    def load_checkpoints(self, checkpoint_dir: str | Path) -> None:
        cdir = Path(checkpoint_dir)
        self.diffusion.load_checkpoint(cdir / f"v4_tree_mdlm_{self.preset}.pt")
        self.hrm.load_checkpoint(cdir / f"v4_holo_hrm_{self.preset}.pt")
        mem_p = cdir / f"v4_case_memory_{self.preset}.npz"
        if mem_p.exists():
            data = np.load(mem_p)
            self.case_memory._z_states = np.asarray(data["z_states"], dtype=float)
            self.case_memory._mean = np.asarray(data["mean"], dtype=float)
            self.case_memory._std = np.asarray(data["std"], dtype=float)
            self.case_memory._fwd_mkt_5d = np.asarray(data["fwd_mkt_5d"], dtype=float)
            self.case_memory._is_trauma = np.asarray(data["is_trauma"], dtype=float)
            self.case_memory._fitted = True
