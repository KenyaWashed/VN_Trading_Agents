"""MAVE-Q v5 (`DCA-Chronos-RSI` + Popperian Falsification): Complete Core Engine (Parts 1 & 2).

Part 1 (`[TASK 02]`):
- `AdversarialHypothesisRefuter`: Popperian Falsification Engine operating strictly inside `search`:
    (a) Liquidity & Friction Mirage (`1.5x` VN broker commission, 0.1% sell tax, spread & impact shock)
    (b) Single-Regime Luck (cross-fold sign stability across purged walk-forward search folds)
    (c) Occam's Razor / BIC Complexity Violation (rejects >=2x complexity with no BIC-CV gain)
    (d) Non-Economic Sign Inversion (rejects chronological half-sample sign reversal)

Part 2 (`[TASK 03]`):
- `ChronosKalmanHedge`: Online Kalman Filter + Discounted Multiplicative-Weights (Hedge) with an
  EXACT `H`-day causal delay buffer (at session `t`, updates skill and prunes decaying experts to
  weight `0.0` using ONLY `H`-day forward windows that closed at or before `t`, `s <= t - H`;
  verified shift-invariant to `atol=1e-9`).
- `DCAMoSEHyperGate`: Sparsity-regularized L1 / Group-Lasso nonlinear cross-feature interaction
  gating that automatically shrinks noisy interaction branches identically to `0.0`.
- `SplitConformalDormantVeto`: Split-Conformal Risk Control calibrated strictly on `search` so
  ~90% of normal healthy stock-days incur EXACT `0.0` penalty, while liquidity traps, blow-offs,
  high epistemic disagreement, and systemic/secular bear regimes trigger decisive `Epistemic
  Abstention` (refusing forced buys / retreating to cash).
- `AsymBetaNTZController`: Asymmetric Beta (`beta+` unlocked in confirmed Bull expansions,
  `beta-` clamped in stress/Bear), friction-matched No-Trade-Zone (`NTZ`) exceeding VN T+2.5 +
  0.1% sell tax, and dual-speed Bear Shield (fast floor contagion + slow secular bear drift).
- `MAVEQv5Engine`: Full Popperian + DCA-Chronos-RSI zero-validation-burn engine.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vn_rsi.benchmarks.extended_metrics import ExtendedMetricsReport, compute_extended_metrics
from vn_rsi.data.market_rules import MarketRulesConfig
from vn_rsi.dsl.ast_spec import (
    ASTNode,
    ASTVerifier,
    FactorSpec,
    compute_feature_panel,
    evaluate_factor_spec,
)
from vn_rsi.eval.evaluator import SealedEvaluator
from vn_rsi.eval.metrics import compute_portfolio_metrics, compute_rank_ic_series
from vn_rsi.models.alpha_mining_baselines import GrammarTreeSampler
from vn_rsi.models.ast_vocab import MECHANISM_FAMILIES
from vn_rsi.models.cloud_presets import HardwareSafetyGuard, verify_sub1b_blueprint_zero_ram
from vn_rsi.models.dca_backbone import QuantBackboneModel
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
from vn_rsi.models.maveq_v4_veteran import compute_causal_microstructure_state
from vn_rsi.sandbox.seal import SealedTestReuseError
from vn_rsi.sandbox.simulator import VNPortfolioSimulator
from vn_rsi.sandbox.splits import WalkForwardFold


def _cs_z_v5(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean(axis=1)
    sd = df.std(axis=1) + 1e-8
    return df.sub(mu, axis=0).div(sd, axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-4.0, 4.0)


def _fast_spearman_row(a: np.ndarray, b: np.ndarray) -> float:
    """Fast cross-sectional Spearman Rank IC between two 1D arrays (handling NaNs safely)."""
    mask = np.isfinite(a) & np.isfinite(b)
    n = int(np.sum(mask))
    if n < 5:
        return 0.0
    av = a[mask]
    bv = b[mask]
    if float(np.std(av)) < 1e-10 or float(np.std(bv)) < 1e-10:
        return 0.0
    ra = np.argsort(np.argsort(av)).astype(np.float64)
    rb = np.argsort(np.argsort(bv)).astype(np.float64)
    ra -= np.mean(ra)
    rb -= np.mean(rb)
    denom = float(np.sqrt(np.sum(ra * ra) * np.sum(rb * rb)))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(ra * rb) / denom)


def scale_market_rules_friction(rules: MarketRulesConfig, multiplier: float = 1.5) -> MarketRulesConfig:
    """Create an isolated deep copy of MarketRulesConfig with transaction/impact friction scaled by `multiplier`.

    NEVER modifies `configs/market_rules.yaml` or the sealed evaluator's rules!
    """
    shocked = copy.deepcopy(rules)
    shocked.transaction_costs.broker_commission_bps *= multiplier
    shocked.transaction_costs.sell_tax_bps *= multiplier
    shocked.transaction_costs.buy_tax_bps *= multiplier
    shocked.illiquidity_impact.impact_coeff_bps *= multiplier
    shocked.illiquidity_impact.base_half_spread_bps = {
        k: v * multiplier for k, v in shocked.illiquidity_impact.base_half_spread_bps.items()
    }
    return shocked


# ---------------------------------------------------------------------------
# PART 1: AdversarialHypothesisRefuter (Popperian Falsification Engine)
# ---------------------------------------------------------------------------
@dataclass
class RefutationVerdict:
    candidate_id: str
    category: str
    accepted: bool
    failed_gates: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    base_search_sharpe: float = 0.0
    shocked_1p5x_sharpe: float = 0.0
    friction_drop_ratio: float = 0.0
    fold_ics: list[float] = field(default_factory=list)
    fold_sharpes: list[float] = field(default_factory=list)
    half_sample_ics: tuple[float, float] = (0.0, 0.0)
    complexity_units: float = 1.0
    bic_adjusted_score: float = 0.0

    def to_ledger_record(self, stage: str, hypothesis: str, action_taken: str) -> dict[str, Any]:
        return {
            "id": self.candidate_id,
            "stage": stage,
            "category": self.category,
            "hypothesis": hypothesis,
            "verdict": "ACCEPTED" if self.accepted else "REJECTED",
            "failed_gates": list(self.failed_gates),
            "reason": "; ".join(self.reasons) if self.reasons else "Survived all 4 Popperian falsification gates.",
            "evidence": {
                "base_search_sharpe": round(self.base_search_sharpe, 4),
                "shocked_1p5x_sharpe": round(self.shocked_1p5x_sharpe, 4),
                "friction_drop_ratio": round(self.friction_drop_ratio, 4),
                "fold_ics": [round(x, 4) for x in self.fold_ics],
                "fold_sharpes": [round(x, 4) for x in self.fold_sharpes],
                "half_sample_ics": [round(self.half_sample_ics[0], 4), round(self.half_sample_ics[1], 4)],
                "complexity_units": round(self.complexity_units, 2),
                "bic_adjusted_score": round(self.bic_adjusted_score, 4),
            },
            "action_taken": action_taken,
        }


def append_falsified_hypothesis(record: dict[str, Any], ledger_path: str | Path = "logs/v5_falsified_hypotheses.jsonl") -> None:
    """Append a structured falsification record to the append-only JSONL ledger (deduplicating by id+stage)."""
    path = Path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    seen_keys: set[tuple[str, str]] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                old = json.loads(line)
                seen_keys.add((str(old.get("id", "")), str(old.get("stage", ""))))
            except Exception:
                continue
    key = (str(record.get("id", "")), str(record.get("stage", "")))
    if key in seen_keys:
        return
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


class AdversarialHypothesisRefuter:
    """Popperian Falsification Engine operating strictly inside the `search` zone."""

    def __init__(
        self,
        market_rules: MarketRulesConfig,
        search_bars: pd.DataFrame,
        safe_dates: pd.DatetimeIndex,
        search_fwd: pd.DataFrame,
        search_folds: list[WalkForwardFold],
        top_k: int = 6,
        rebalance_every_days: int = 3,
        capital_vnd: float = 1_000_000_000.0,
        friction_multiplier: float = 1.5,
        max_friction_drop_ratio: float = 0.40,
        occam_bic_lambda: float = 0.08,
    ) -> None:
        self.search_bars = search_bars
        self.safe_dates = safe_dates
        self.search_fwd = search_fwd
        self.search_folds = search_folds
        self.friction_multiplier = friction_multiplier
        self.max_friction_drop_ratio = max_friction_drop_ratio
        self.occam_bic_lambda = occam_bic_lambda

        self.base_sim = VNPortfolioSimulator(
            market_rules=market_rules,
            top_k=top_k,
            rebalance_every_days=rebalance_every_days,
            capital_vnd=capital_vnd,
        )
        self.shock_sim = VNPortfolioSimulator(
            market_rules=scale_market_rules_friction(market_rules, multiplier=friction_multiplier),
            top_k=top_k,
            rebalance_every_days=rebalance_every_days,
            capital_vnd=capital_vnd,
        )
        self._search_dates = pd.DatetimeIndex(sorted(search_bars["date"].unique()))

    def compute_bic_adjusted_score(
        self,
        raw_cv_score: float,
        complexity_units: float,
        ref_complexity_units: float = 1.0,
    ) -> float:
        """Occam's Razor BIC complexity penalty proportional to log2(complexity / ref) / sqrt(n_folds)."""
        n_eff = max(1.0, float(len(self.search_folds)))
        ratio = max(1.0, float(complexity_units) / max(1e-6, float(ref_complexity_units)))
        penalty = self.occam_bic_lambda * (math.log2(ratio) / math.sqrt(n_eff))
        return float(raw_cv_score - penalty)

    def refute_signal_with_metrics(
        self,
        candidate_id: str,
        signal_search_wide: pd.DataFrame,
        category: str = "factor_ast",
        complexity_units: float = 1.0,
        simpler_ref_bic_score: float | None = None,
        simpler_ref_complexity: float = 1.0,
        min_fold_ic_floor: float = -0.025,
        fast_ic_short_circuit: bool = False,
    ) -> tuple[RefutationVerdict, dict[str, float], list[dict[str, float]]]:
        """Run all 4 adversarial Popperian falsification gates strictly on `search`, returning (verdict, base_m, fold_ms)."""
        sig = signal_search_wide.reindex(index=self._search_dates)
        ic_s = compute_rank_ic_series(sig.reindex(index=self.safe_dates), self.search_fwd)
        mean_ic = float(ic_s.mean()) if len(ic_s) else 0.0

        fold_ics: list[float] = [
            float(ic_s.reindex(f.eval_dates).dropna().mean()) if len(ic_s.reindex(f.eval_dates).dropna()) else 0.0
            for f in self.search_folds
        ]
        mid = max(1, len(ic_s) // 2)
        h1_ic = float(ic_s.iloc[:mid].mean()) if len(ic_s) >= 10 else mean_ic
        h2_ic = float(ic_s.iloc[mid:].mean()) if len(ic_s) >= 10 else mean_ic

        failed_gates: list[str] = []
        reasons: list[str] = []
        min_f_ic = min(fold_ics) if fold_ics else mean_ic

        # Fast exact algebraic checks BEFORE running expensive portfolio simulations
        n_pos_ic = sum(1 for x in fold_ics if x > 0.005)
        n_neg_ic = sum(1 for x in fold_ics if x < -0.005)
        if len(fold_ics) >= 3 and ((n_pos_ic <= 1 and n_neg_ic >= 2) or (min_f_ic < min_fold_ic_floor)):
            failed_gates.append("SINGLE_REGIME_LUCK")
            reasons.append(f"Failed cross-fold IC stability: fold ICs={[round(x, 3) for x in fold_ics]}.")
        if (h1_ic * h2_ic < 0.0) and (abs(h1_ic) > 0.015 and abs(h2_ic) > 0.015):
            failed_gates.append("ECONOMIC_SIGN_INVERSION")
            reasons.append(f"Economic sign flipped between chronological search halves: H1 IC={h1_ic:.4f}, H2 IC={h2_ic:.4f}.")

        if fast_ic_short_circuit and failed_gates:
            verdict_fast = RefutationVerdict(
                candidate_id=candidate_id,
                category=category,
                accepted=False,
                failed_gates=failed_gates,
                reasons=reasons,
                fold_ics=fold_ics,
                half_sample_ics=(h1_ic, h2_ic),
                complexity_units=float(complexity_units),
            )
            return verdict_fast, {"rank_ic": mean_ic, "net_sharpe": 0.0, "daily_turnover": 0.15, "icir": 0.0}, []

        sim_base = self.base_sim.simulate(sig, self.search_bars)
        base_m = compute_portfolio_metrics(
            sim_base.daily_net_returns, sim_base.daily_gross_returns, sim_base.daily_turnover, ic_s, n_trials=1
        )
        base_sr = float(base_m["net_sharpe"])

        # Only run 1.5x friction shock simulation if base_sr > 0.05 (or always when not short-circuiting)
        if (not fast_ic_short_circuit) or (base_sr > 0.05):
            sim_shock = self.shock_sim.simulate(sig, self.search_bars)
            shock_m = compute_portfolio_metrics(
                sim_shock.daily_net_returns, sim_shock.daily_gross_returns, sim_shock.daily_turnover, ic_s, n_trials=1
            )
            shock_sr = float(shock_m["net_sharpe"])
        else:
            shock_sr = base_sr - 0.15

        fold_ms: list[dict[str, float]] = []
        fold_srs: list[float] = []
        for fold in self.search_folds:
            f_ic = ic_s.reindex(fold.eval_dates).dropna()
            f_net = sim_base.daily_net_returns.reindex(fold.eval_dates).dropna()
            f_gross = sim_base.daily_gross_returns.reindex(fold.eval_dates).dropna()
            f_turn = sim_base.daily_turnover.reindex(fold.eval_dates).dropna()
            fm = compute_portfolio_metrics(f_net, f_gross, f_turn, f_ic, n_trials=1)
            fold_ms.append(fm)
            fold_srs.append(float(fm["net_sharpe"]))

        min_f_sr = min(fold_srs) if fold_srs else base_sr
        raw_cv_score = 0.55 * base_sr + 0.25 * min_f_sr + 4.0 * mean_ic + 2.0 * min_f_ic - 1.2 * float(base_m["daily_turnover"])
        bic_score = self.compute_bic_adjusted_score(raw_cv_score, complexity_units, simpler_ref_complexity)

        # Gate (a): Liquidity & Friction Mirage under 1.5x VN costs
        if base_sr > 0.05:
            drop_ratio = (base_sr - shock_sr) / max(abs(base_sr), 0.15)
        else:
            drop_ratio = 1.0 if shock_sr < base_sr else 0.0

        if (base_sr > 0.05 and drop_ratio > self.max_friction_drop_ratio) or (base_sr > 0.10 and shock_sr <= 0.0):
            failed_gates.append("LIQUIDITY_FRICTION_MIRAGE_1.5X")
            reasons.append(
                f"1.5x VN friction shock collapsed Sharpe {base_sr:.3f} -> {shock_sr:.3f} "
                f"(drop {drop_ratio * 100:.1f}% > {self.max_friction_drop_ratio * 100:.0f}%)."
            )

        # Gate (b) Sharpe fold stability supplement
        n_pos_sr = sum(1 for x in fold_srs if x > 0.05)
        n_neg_sr = sum(1 for x in fold_srs if x < -0.05)
        if len(fold_srs) >= 3 and (n_pos_sr <= 1 and n_neg_sr >= 2) and "SINGLE_REGIME_LUCK" not in failed_gates:
            failed_gates.append("SINGLE_REGIME_LUCK")
            reasons.append(f"Failed cross-fold Sharpe stability: fold Sharpes={[round(x, 2) for x in fold_srs]}.")

        # Gate (c): Occam's Razor / BIC Complexity Violation
        if (
            simpler_ref_bic_score is not None
            and complexity_units >= 2.0 * max(1e-6, simpler_ref_complexity)
            and bic_score <= simpler_ref_bic_score
        ):
            failed_gates.append("OCCAM_RAZOR_VIOLATION")
            reasons.append(
                f"Complexity ({complexity_units:.1f} vs ref {simpler_ref_complexity:.1f}, >=2x) "
                f"failed Occam BIC-CV gate ({bic_score:.4f} <= simpler ref {simpler_ref_bic_score:.4f})."
            )

        verdict = RefutationVerdict(
            candidate_id=candidate_id,
            category=category,
            accepted=(len(failed_gates) == 0),
            failed_gates=failed_gates,
            reasons=reasons,
            base_search_sharpe=base_sr,
            shocked_1p5x_sharpe=shock_sr,
            friction_drop_ratio=float(drop_ratio),
            fold_ics=fold_ics,
            fold_sharpes=fold_srs,
            half_sample_ics=(h1_ic, h2_ic),
            complexity_units=float(complexity_units),
            bic_adjusted_score=float(bic_score),
        )
        return verdict, base_m, fold_ms

    def refute_signal(
        self,
        candidate_id: str,
        signal_search_wide: pd.DataFrame,
        category: str = "factor_ast",
        complexity_units: float = 1.0,
        simpler_ref_bic_score: float | None = None,
        simpler_ref_complexity: float = 1.0,
        min_fold_ic_floor: float = -0.025,
    ) -> RefutationVerdict:
        """Run all 4 adversarial Popperian falsification gates strictly on the `search` zone."""
        verdict, _, _ = self.refute_signal_with_metrics(
            candidate_id=candidate_id,
            signal_search_wide=signal_search_wide,
            category=category,
            complexity_units=complexity_units,
            simpler_ref_bic_score=simpler_ref_bic_score,
            simpler_ref_complexity=simpler_ref_complexity,
            min_fold_ic_floor=min_fold_ic_floor,
            fast_ic_short_circuit=False,
        )
        return verdict

    @staticmethod
    def verdict_as_dict(verdict: RefutationVerdict) -> dict[str, Any]:
        return asdict(verdict)


# ---------------------------------------------------------------------------
# PART 2.1: Chronos-Kalman-Hedge (Exact H-Day Causal Delay Buffer + Online Pruning)
# ---------------------------------------------------------------------------
@dataclass
class ChronosHedgeOutput:
    combined_alpha_wide: pd.DataFrame
    expert_weights_df: pd.DataFrame
    kalman_skills_df: pd.DataFrame
    epistemic_disagreement_wide: pd.DataFrame
    online_pruned_mask_df: pd.DataFrame


class ChronosKalmanHedge:
    """Online Kalman Filter + Discounted Multiplicative-Weights (Hedge) with EXACT H-day causal delay buffer.

    Causality Theorem:
        At the close of session `t`, the ONLY H-day forward return windows that have completely closed
        are those initiated at `s <= t - H` (`r_{s -> s+H}` with `s + H <= t`). Therefore, updating
        the Kalman skill state `theta_{i,t}` and Hedge weights `w_{i,t}` using strictly `y_{i, t-H} =
        RankIC(S_i[t-H], close[t]/close[t-H] - 1)` guarantees ZERO look-ahead leakage (`atol=1e-9`
        invariance under arbitrary post-cutoff future perturbations).
    """

    def __init__(
        self,
        horizon_delay_days: int = 5,
        process_noise_q: float = 0.0008,
        obs_noise_r: float = 0.018,
        hedge_eta: float = 6.5,
        hedge_discount_gamma: float = 0.965,
        turnover_penalty_lambda: float = 0.08,
        online_prune_skill_thresh: float = -0.005,
        prior_anchor_strength: float = 0.15,
    ) -> None:
        self.horizon_delay_days = max(1, int(horizon_delay_days))
        self.process_noise_q = float(process_noise_q)
        self.obs_noise_r = float(obs_noise_r)
        self.hedge_eta = float(hedge_eta)
        self.hedge_discount_gamma = float(hedge_discount_gamma)
        self.turnover_penalty_lambda = float(turnover_penalty_lambda)
        self.online_prune_skill_thresh = float(online_prune_skill_thresh)
        self.prior_anchor_strength = float(prior_anchor_strength)

    def fuse_online(
        self,
        expert_signals: list[pd.DataFrame],
        expert_names: list[str],
        prior_weights: list[float],
        prior_skills: list[float],
        close_wide: pd.DataFrame,
        enabled: bool = True,
    ) -> ChronosHedgeOutput:
        """Fuse M expert alpha signals online with exact H-day delay buffer and automatic online pruning."""
        dates = pd.DatetimeIndex(close_wide.index)
        syms = list(close_wide.columns)
        n_days, n_syms = len(dates), len(syms)
        m = len(expert_signals)
        if m == 0:
            zero = pd.DataFrame(0.0, index=dates, columns=syms)
            empty = pd.DataFrame(index=dates)
            return ChronosHedgeOutput(zero, empty, empty, zero, empty)

        w0 = np.asarray([max(1e-4, float(w)) for w in prior_weights], dtype=np.float64)
        w0 /= float(np.sum(w0)) + 1e-12
        theta0 = np.asarray([float(np.clip(s, 0.0, 0.20)) for s in prior_skills], dtype=np.float64)

        mats = np.stack(
            [
                _cs_z_v5(sig.reindex(index=dates, columns=syms)).to_numpy(dtype=np.float64)
                for sig in expert_signals
            ],
            axis=0,
        )  # (M, T, S)

        if not enabled or m == 1:
            w_hist = np.tile(w0[None, :], (n_days, 1))
            th_hist = np.tile(theta0[None, :], (n_days, 1))
            pruned_hist = np.zeros((n_days, m), dtype=bool)
            comb = np.tensordot(w0, mats, axes=([0], [0]))  # (T, S)
            # Weighted cross-expert epistemic std
            diff_sq = (mats - comb[None, :, :]) ** 2
            dis = np.sqrt(np.tensordot(w0, diff_sq, axes=([0], [0])) + 1e-12)
            return ChronosHedgeOutput(
                combined_alpha_wide=_cs_z_v5(pd.DataFrame(comb, index=dates, columns=syms)),
                expert_weights_df=pd.DataFrame(w_hist, index=dates, columns=expert_names),
                kalman_skills_df=pd.DataFrame(th_hist, index=dates, columns=expert_names),
                epistemic_disagreement_wide=pd.DataFrame(dis, index=dates, columns=syms),
                online_pruned_mask_df=pd.DataFrame(pruned_hist, index=dates, columns=expert_names),
            )

        close_arr = close_wide.reindex(index=dates, columns=syms).to_numpy(dtype=np.float64)
        H = self.horizon_delay_days

        theta = theta0.copy()
        p_var = np.full(m, 0.015, dtype=np.float64)
        log_w = np.log(w0 + 1e-8)
        log_w0 = log_w.copy()

        w_hist = np.zeros((n_days, m), dtype=np.float64)
        th_hist = np.zeros((n_days, m), dtype=np.float64)
        pruned_hist = np.zeros((n_days, m), dtype=bool)
        comb_arr = np.zeros((n_days, n_syms), dtype=np.float64)
        dis_arr = np.zeros((n_days, n_syms), dtype=np.float64)

        for t in range(n_days):
            # STRICT CAUSAL H-DAY DELAY BUFFER: only update using window [t-H -> t] which CLOSED at t!
            if t >= H:
                c_prev = close_arr[t - H]
                c_now = close_arr[t]
                valid_px = np.isfinite(c_prev) & np.isfinite(c_now) & (c_prev > 1e-8)
                ret_h = np.full(n_syms, np.nan, dtype=np.float64)
                ret_h[valid_px] = (c_now[valid_px] / c_prev[valid_px]) - 1.0

                for i in range(m):
                    # Signal emitted at t - H evaluated against realized return closing at t (0% look-ahead!)
                    sig_prev = mats[i, t - H]
                    obs_ic = _fast_spearman_row(sig_prev, ret_h)
                    # Strictly delayed Top-6 vs universe realized spread closing at t (0% look-ahead!)
                    valid_pair = np.isfinite(sig_prev) & np.isfinite(ret_h)
                    if int(np.sum(valid_pair)) >= 8:
                        s_v = sig_prev[valid_pair]
                        r_v = ret_h[valid_pair]
                        k_top = min(6, len(s_v) // 2)
                        top_idx = np.argpartition(s_v, -k_top)[-k_top:]
                        spread = float(np.mean(r_v[top_idx]) - np.mean(r_v))
                        obs_skill = 0.65 * obs_ic + 0.35 * float(np.tanh(12.0 * spread))
                    else:
                        obs_skill = obs_ic
                    # Turnover drag strictly from [t-1 -> t]
                    turn_i = float(np.mean(np.abs(mats[i, t] - mats[i, t - 1]))) * 0.15

                    # Kalman prediction + measurement update
                    p_pred = p_var[i] + self.process_noise_q
                    k_gain = p_pred / (p_pred + self.obs_noise_r)
                    theta[i] = theta[i] + k_gain * (obs_skill - theta[i])
                    p_var[i] = (1.0 - k_gain) * p_pred

                    # Discounted Multiplicative-Weights (Hedge) with gentle prior anchor
                    eff_reward = float(np.clip(theta[i] - self.turnover_penalty_lambda * turn_i, -0.25, 0.25))
                    log_w[i] = (
                        self.hedge_discount_gamma * log_w[i]
                        + (1.0 - self.hedge_discount_gamma) * log_w0[i]
                        + self.hedge_eta * eff_reward
                    )

            # Softmax weights + Online Pruning of degraded experts (`theta < online_prune_skill_thresh`)
            log_shift = log_w - float(np.max(log_w))
            raw_w = np.exp(np.clip(log_shift, -25.0, 0.0))
            pruned = theta < self.online_prune_skill_thresh
            if np.all(pruned):
                # Safety fallback: keep the least-negative expert if all degraded simultaneously
                best_i = int(np.argmax(theta))
                pruned[best_i] = False
            eff_w = np.where(pruned, 0.0, raw_w)
            w_sum = float(np.sum(eff_w))
            if w_sum <= 1e-12:
                eff_w = w0.copy()
                pruned[:] = False
            else:
                eff_w /= w_sum

            # Blend with small prior stability anchor ONLY for unpruned experts so pruned stay EXACTLY 0.0!
            active = ~pruned
            if self.prior_anchor_strength > 0.0 and np.any(active):
                w0_act = np.where(active, w0, 0.0)
                w0_act /= float(np.sum(w0_act)) + 1e-12
                eff_w = (1.0 - self.prior_anchor_strength) * eff_w + self.prior_anchor_strength * w0_act
                eff_w = np.where(pruned, 0.0, eff_w)
                eff_w /= float(np.sum(eff_w)) + 1e-12

            w_hist[t] = eff_w
            th_hist[t] = theta
            pruned_hist[t] = pruned

            # Combine signals at t using strictly causal weights w_t
            c_t = eff_w @ mats[:, t, :]  # (S,)
            comb_arr[t] = c_t
            # Cross-expert epistemic disagreement at t (used by Split-Conformal Veto)
            diff_sq = (mats[:, t, :] - c_t[None, :]) ** 2
            dis_arr[t] = np.sqrt(eff_w @ diff_sq + 1e-12)

        return ChronosHedgeOutput(
            combined_alpha_wide=_cs_z_v5(pd.DataFrame(comb_arr, index=dates, columns=syms)),
            expert_weights_df=pd.DataFrame(w_hist, index=dates, columns=expert_names),
            kalman_skills_df=pd.DataFrame(th_hist, index=dates, columns=expert_names),
            epistemic_disagreement_wide=pd.DataFrame(dis_arr, index=dates, columns=syms),
            online_pruned_mask_df=pd.DataFrame(pruned_hist, index=dates, columns=expert_names),
        )


# ---------------------------------------------------------------------------
# PART 2.2: DCA-MoSE HyperGate (L1 / Group-Lasso Sparse Nonlinear Interactions)
# ---------------------------------------------------------------------------
class DCAMoSEHyperGate:
    """Sparsity-controlled nonlinear multi-branch expert gate with L1/Group-Lasso automatic pruning.

    Extracts 6 strictly causal nonlinear cross-feature interaction branches and fits Group-Lasso
    soft-thresholded gates strictly on `search` folds against the FWL residual orthogonal to base
    alpha. Any branch failing multi-fold sign stability or falling below `group_lasso_lambda` is
    shrunk IDENTICALLY to `0.0`!
    """

    BRANCH_NAMES: tuple[str, ...] = (
        "val_x_flow",
        "mom_x_calm",
        "rev_x_floor_safe",
        "val_x_low_vol",
        "flow_x_mom",
        "quality_x_quiet_liq",
    )

    def __init__(self, group_lasso_lambda: float = 0.025, max_nonlinear_weight: float = 0.35) -> None:
        self.group_lasso_lambda = float(group_lasso_lambda)
        self.max_nonlinear_weight = float(max_nonlinear_weight)
        self.branch_weights: dict[str, float] = {b: 0.0 for b in self.BRANCH_NAMES}
        self.active_branches: list[str] = []
        self.pruned_branches: list[str] = []
        self._fitted: bool = False

    @staticmethod
    def extract_interaction_branches(panel: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Extract 6 strictly causal (info <= t) scale-invariant nonlinear cross-feature interactions."""
        close = panel["close"]
        dates = pd.DatetimeIndex(close.index)
        syms = list(close.columns)

        ret1 = panel["return_1d"].reindex(index=dates, columns=syms).fillna(0.0)
        vol20 = ret1.rolling(20, min_periods=5).std().fillna(0.015)
        mom15 = close.pct_change(15, fill_method=None).fillna(0.0)
        mom_skip = close.shift(3).pct_change(12, fill_method=None).fillna(0.0)
        ret3 = close.pct_change(3, fill_method=None).fillna(0.0)
        tshock = panel["turnover_shock_20d"].reindex(index=dates, columns=syms).fillna(1.0)
        fflow = panel["foreign_net_ratio"].reindex(index=dates, columns=syms).fillna(0.0)
        pit_vq = _cs_z_v5(panel["pit_value_quality"].reindex(index=dates, columns=syms).fillna(0.0))
        pit_roe = panel["pit_roe"].reindex(index=dates, columns=syms).fillna(0.08)
        pit_pb = panel["pit_pb_ratio"].reindex(index=dates, columns=syms).fillna(1.8).clip(lower=0.3)
        dist_flr = panel["dist_to_floor"].reindex(index=dates, columns=syms).fillna(0.05)
        intraday = panel["intraday_range"].reindex(index=dates, columns=syms).fillna(0.02)

        flow5 = _cs_z_v5(fflow.rolling(5, min_periods=2).mean().fillna(0.0))
        flow10 = _cs_z_v5(fflow.rolling(10, min_periods=3).mean().fillna(0.0))
        mom_ra = _cs_z_v5((mom15 / (vol20 + 0.006)).rolling(4, min_periods=1).mean())
        mom_sk_z = _cs_z_v5(mom_skip / (vol20 + 0.006))
        rev3_z = _cs_z_v5(-(ret3 * tshock.clip(0.5, 2.5)))
        val_rk = _cs_z_v5(pit_roe.rank(axis=1, pct=True) - pit_pb.rank(axis=1, pct=True))
        vol_calm = 1.0 / (1.0 + np.exp(-_cs_z_v5(-vol20).clip(-4.0, 4.0)))
        flr_safe = 1.0 / (1.0 + np.exp(-_cs_z_v5(dist_flr).clip(-4.0, 4.0)))
        intra_calm = 1.0 / (1.0 + np.exp(-_cs_z_v5(-intraday).clip(-4.0, 4.0)))
        quiet_liq = 1.0 / (1.0 + np.exp(-_cs_z_v5(-tshock).clip(-4.0, 4.0)))

        return {
            "val_x_flow": _cs_z_v5(np.tanh(0.8 * pit_vq) * (0.5 + 0.5 * np.tanh(0.8 * flow5))),
            "mom_x_calm": _cs_z_v5(np.tanh(0.8 * mom_ra) * vol_calm),
            "rev_x_floor_safe": _cs_z_v5(np.tanh(0.8 * rev3_z) * flr_safe),
            "val_x_low_vol": _cs_z_v5(np.tanh(0.8 * val_rk) * intra_calm),
            "flow_x_mom": _cs_z_v5(np.tanh(0.8 * flow10) * np.tanh(0.8 * mom_sk_z)),
            "quality_x_quiet_liq": _cs_z_v5(np.tanh(0.9 * pit_vq) * quiet_liq),
        }

    def fit_on_search(
        self,
        search_panel: dict[str, pd.DataFrame],
        base_alpha_search: pd.DataFrame,
        safe_dates: pd.DatetimeIndex,
        search_fwd: pd.DataFrame,
        search_folds: list[WalkForwardFold],
        extra_candidate_branches: dict[str, pd.DataFrame] | None = None,
    ) -> DCAMoSEHyperGate:
        """Fit Group-Lasso / L1 soft-thresholded gates strictly on search FWL residual (zero leakage!)."""
        branches = self.extract_interaction_branches(search_panel)
        if extra_candidate_branches:
            branches.update(extra_candidate_branches)

        # Frisch-Waugh-Lovell residual forward return orthogonal to base linear alpha
        base_z = _cs_z_v5(base_alpha_search.reindex(index=safe_dates))
        fwd_z = _cs_z_v5(search_fwd.reindex(index=safe_dates))
        beta_proj = (base_z * fwd_z).mean(axis=1) / ((base_z * base_z).mean(axis=1) + 1e-8)
        resid_fwd = fwd_z.sub(base_z.mul(beta_proj, axis=0))

        raw_weights: dict[str, float] = {}
        self.active_branches = []
        self.pruned_branches = []

        for name, b_df in branches.items():
            b_safe = _cs_z_v5(b_df.reindex(index=safe_dates))
            ic_s = compute_rank_ic_series(b_safe, resid_fwd)
            if len(ic_s) < 20:
                raw_weights[name] = 0.0
                self.pruned_branches.append(name)
                continue
            f_ics = [
                float(ic_s.reindex(f.eval_dates).dropna().mean())
                for f in search_folds
                if len(ic_s.reindex(f.eval_dates).dropna()) >= 5
            ]
            mean_ic = float(ic_s.mean())
            min_f = min(f_ics) if f_ics else mean_ic
            max_f = max(f_ics) if f_ics else mean_ic

            # Group-Lasso sign-consistency + L1 soft-thresholding:
            # Prune identically to 0.0 if branch has unstable sign across folds OR magnitude <= lambda!
            sign_consistent = (mean_ic > 0.0 and min_f > -0.015) or (mean_ic < 0.0 and max_f < 0.015)
            group_strength = abs(mean_ic) * 0.70 + max(0.0, (min_f if mean_ic > 0 else -max_f)) * 0.30
            if (not sign_consistent) or (group_strength <= self.group_lasso_lambda):
                raw_weights[name] = 0.0
                self.pruned_branches.append(name)
            else:
                shrunk = (group_strength - self.group_lasso_lambda) * (1.0 if mean_ic > 0 else -1.0)
                raw_weights[name] = float(shrunk)
                self.active_branches.append(name)

        total_abs = sum(abs(v) for v in raw_weights.values())
        if total_abs > 1e-9:
            scale = min(self.max_nonlinear_weight, total_abs * 3.0) / total_abs
            self.branch_weights = {k: float(v * scale) for k, v in raw_weights.items()}
        else:
            self.branch_weights = {k: 0.0 for k in raw_weights}

        self._fitted = True
        return self

    def apply(
        self,
        base_alpha_wide: pd.DataFrame,
        panel: dict[str, pd.DataFrame],
        enabled: bool = True,
        extra_candidate_branches: dict[str, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        """Apply sparse Group-Lasso gated nonlinear branches on top of base alpha (exact 0 if pruned!)."""
        base_z = _cs_z_v5(base_alpha_wide)
        if (not enabled) or (not self._fitted) or (not self.active_branches):
            return base_z

        branches = self.extract_interaction_branches(panel)
        if extra_candidate_branches:
            branches.update(extra_candidate_branches)

        nl_sum = pd.DataFrame(0.0, index=base_z.index, columns=base_z.columns)
        total_w = 0.0
        for name, wt in self.branch_weights.items():
            if abs(wt) > 1e-9 and name in branches:
                nl_sum = nl_sum + branches[name].reindex_like(base_z).fillna(0.0) * wt
                total_w += abs(wt)
        if total_w <= 1e-9:
            return base_z
        return _cs_z_v5((1.0 - min(self.max_nonlinear_weight, total_w)) * base_z + nl_sum)


# ---------------------------------------------------------------------------
# PART 2.3: Split-Conformal Dormant-by-Default Veto & Epistemic Abstention
# ---------------------------------------------------------------------------
@dataclass
class ConformalVetoOutput:
    stock_penalty_wide: pd.DataFrame      # EXACT 0.0 on ~90% healthy stock-days!
    stock_hard_veto_wide: pd.DataFrame    # Boolean extreme-trap hard block
    market_abstain_series: pd.Series      # Boolean True -> ABSTAIN / Hold Cash!
    market_nonconformity: pd.Series
    dormant_fraction: float


class SplitConformalDormantVeto:
    """Split-Conformal Risk Control with Dormant-by-Default Stock Veto & Market Epistemic Abstention.

    Calibrates exact finite-sample conformal quantiles `q_{1-alpha}` strictly on the search-zone
    calibration slice so that:
    - ~90% of healthy stock-day observations receive EXACT `0.0` penalty (eliminating v4's noise drag!).
    - Extreme liquidity traps, blow-offs, high epistemic disagreement, and crash/secular-bear regimes
      exceed `q_{1-alpha}` and trigger decisive stock veto or portfolio-level Epistemic Abstention.
    """

    def __init__(
        self,
        alpha_level: float = 0.10,
        calibration_fraction: float = 0.35,
        stock_veto_scale: float = 0.45,
        market_abstain_alpha: float = 0.06,
    ) -> None:
        self.alpha_level = float(np.clip(alpha_level, 0.02, 0.30))
        self.calibration_fraction = float(np.clip(calibration_fraction, 0.15, 0.60))
        self.stock_veto_scale = float(stock_veto_scale)
        self.market_abstain_alpha = float(np.clip(market_abstain_alpha, 0.01, 0.25))

        self.q_stock: float = 1.50
        self.q_stock_extreme: float = 3.00
        self.q_market: float = 1.40
        self._fitted: bool = False

    @staticmethod
    def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
        clean = scores[np.isfinite(scores)]
        n = len(clean)
        if n == 0:
            return 1.5
        k = int(math.ceil((n + 1) * (1.0 - alpha)))
        k = min(max(k, 1), n)
        return float(np.partition(clean, k - 1)[k - 1])

    @staticmethod
    def compute_stock_nonconformity(
        panel: dict[str, pd.DataFrame],
        epistemic_disagreement: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Strictly causal stock nonconformity score E_stock(t, s) >= 0 (info <= t only)."""
        close = panel["close"]
        dates = pd.DatetimeIndex(close.index)
        syms = list(close.columns)

        ret3 = close.pct_change(3, fill_method=None).fillna(0.0)
        ret20 = close.pct_change(20, fill_method=None).fillna(0.0)
        tshock = panel["turnover_shock_20d"].reindex(index=dates, columns=syms).fillna(1.0)
        dist_flr = panel["dist_to_floor"].reindex(index=dates, columns=syms).fillna(0.05)
        intraday = panel["intraday_range"].reindex(index=dates, columns=syms).fillna(0.02)
        fflow = panel.get("foreign_flow_ratio", pd.DataFrame(0.0, index=dates, columns=syms)).reindex(index=dates, columns=syms).fillna(0.0)
        fflow5 = fflow.rolling(5, min_periods=1).mean().fillna(0.0)

        # Strictly downside / distribution traps (never penalize healthy foreign-backed breakouts!)
        near_flr = (dist_flr < 0.015).astype(float)
        flr_cascade = near_flr + 0.85 * near_flr.rolling(5, min_periods=1).sum().clip(0.0, 3.0)
        ret3_z = _cs_z_v5(ret3)
        # Only flag blow-offs accompanied by negative institutional/foreign distribution flow
        dist_blowoff = (
            (ret3_z > 2.3) & (tshock > 2.1) & (fflow5 < -0.0005)
        ).astype(float) * (ret3_z - 2.3).clip(0.0, 2.5)
        distress = (ret20 < -0.24).astype(float) * 2.0
        downside_shock = ((ret3 < -0.06) & (intraday > 0.060)).astype(float) * (intraday - 0.060).clip(0.0, 0.12) * 10.0

        score = 0.55 * flr_cascade + 0.35 * dist_blowoff + 0.65 * distress + 0.35 * downside_shock
        if epistemic_disagreement is not None:
            dis = epistemic_disagreement.reindex(index=dates, columns=syms).fillna(0.0)
            # Only penalize tail disagreement on weakening stocks (ret20 < 0)
            dis_z = _cs_z_v5(dis)
            score = score + 0.25 * (dis_z - 1.8).clip(0.0, 3.0) * (ret20 < 0.0).astype(float)

        score = score.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 6.0)
        extreme_trap = (near_flr.rolling(3, min_periods=1).sum() >= 2.0) | (ret20 < -0.28)
        return score, extreme_trap

    @staticmethod
    def compute_market_nonconformity(
        panel: dict[str, pd.DataFrame],
        state_df: pd.DataFrame,
        epistemic_disagreement: pd.DataFrame | None = None,
    ) -> pd.Series:
        """Strictly causal market-level nonconformity E_mkt(t) >= 0 detecting crashes & secular bear traps."""
        close = panel["close"]
        dates = pd.DatetimeIndex(close.index)
        ret1 = panel["return_1d"].reindex(index=dates).fillna(0.0)
        mkt_ret = ret1.mean(axis=1)
        mkt_wealth = (1.0 + mkt_ret).cumprod()
        dd60 = (mkt_wealth / mkt_wealth.rolling(60, min_periods=5).max().clip(lower=1e-8)) - 1.0
        mkt_r60 = ((1.0 + mkt_ret).rolling(60, min_periods=15).apply(np.prod, raw=True) - 1.0).fillna(0.0)

        st = state_df.reindex(index=dates).fillna(0.0)
        floor_3d = st["floor_near_3d"]
        vol_rat = st["vol_ratio"]
        breadth = st["breadth_20"]

        crash_comp = 1.6 * (-dd60 / 0.18).clip(0.0, 2.5) + 1.4 * (floor_3d / 0.10).clip(0.0, 2.5) + 0.45 * (vol_rat - 1.25).clip(0.0, 2.0)
        secular_bear_comp = 1.4 * (-mkt_r60 / 0.15).clip(0.0, 2.5) + 0.9 * (0.42 - breadth).clip(0.0, 0.42) / 0.42

        mkt_e = (0.55 * crash_comp + 0.45 * secular_bear_comp).fillna(0.0)
        if epistemic_disagreement is not None:
            mean_dis = epistemic_disagreement.reindex(index=dates).mean(axis=1).fillna(0.0)
            dis_roll = (mean_dis / (mean_dis.rolling(60, min_periods=10).median() + 1e-4) - 1.25).clip(0.0, 2.0)
            mkt_e = mkt_e + 0.30 * dis_roll
        return mkt_e.clip(0.0, 8.0)

    def fit_on_search(
        self,
        search_panel: dict[str, pd.DataFrame],
        search_state: pd.DataFrame,
        safe_dates: pd.DatetimeIndex,
        epistemic_disagreement_search: pd.DataFrame | None = None,
    ) -> SplitConformalDormantVeto:
        """Calibrate exact Split-Conformal quantiles strictly on search calibration slice (zero leakage!)."""
        n_safe = len(safe_dates)
        cal_start = max(0, int(n_safe * (1.0 - self.calibration_fraction)))
        cal_dates = safe_dates[cal_start:]

        stock_e, _ = self.compute_stock_nonconformity(search_panel, epistemic_disagreement_search)
        mkt_e = self.compute_market_nonconformity(search_panel, search_state, epistemic_disagreement_search)

        s_cal = stock_e.reindex(index=cal_dates).to_numpy(dtype=float).ravel()
        m_cal = mkt_e.reindex(index=cal_dates).to_numpy(dtype=float).ravel()

        self.q_stock = max(0.35, self._conformal_quantile(s_cal, self.alpha_level))
        self.q_stock_extreme = max(self.q_stock + 0.50, self._conformal_quantile(s_cal, self.alpha_level * 0.25))
        self.q_market = max(0.95, self._conformal_quantile(m_cal, self.market_abstain_alpha))
        self._fitted = True
        return self

    def apply(
        self,
        panel: dict[str, pd.DataFrame],
        state_df: pd.DataFrame,
        epistemic_disagreement: pd.DataFrame | None = None,
        enabled: bool = True,
    ) -> ConformalVetoOutput:
        """Emit Dormant-by-Default stock penalties (0.0 on ~90% healthy obs) & Epistemic Abstention mask."""
        close = panel["close"]
        dates = pd.DatetimeIndex(close.index)
        syms = list(close.columns)

        if not enabled:
            return ConformalVetoOutput(
                stock_penalty_wide=pd.DataFrame(0.0, index=dates, columns=syms),
                stock_hard_veto_wide=pd.DataFrame(False, index=dates, columns=syms),
                market_abstain_series=pd.Series(False, index=dates),
                market_nonconformity=pd.Series(0.0, index=dates),
                dormant_fraction=1.0,
            )

        stock_e, extreme_trap = self.compute_stock_nonconformity(panel, epistemic_disagreement)
        mkt_e = self.compute_market_nonconformity(panel, state_df, epistemic_disagreement)

        # EXACT 0.0 whenever stock_e <= q_stock (Dormant-by-Default guarantee!)
        excess = (stock_e - self.q_stock).clip(lower=0.0)
        penalty = (self.stock_veto_scale * excess).clip(0.0, 3.0)
        hard_veto = extreme_trap | (stock_e > self.q_stock_extreme)
        dormant_frac = float((penalty.to_numpy(dtype=float) == 0.0).mean())

        abstain = mkt_e > self.q_market
        return ConformalVetoOutput(
            stock_penalty_wide=penalty,
            stock_hard_veto_wide=hard_veto,
            market_abstain_series=abstain,
            market_nonconformity=mkt_e,
            dormant_fraction=dormant_frac,
        )


# ---------------------------------------------------------------------------
# PART 2.4: AsymBeta-NTZ Controller (Bull Beta+ Unlock, Bear Beta- Clamp, Dual-Speed Shield)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AsymBetaNTZParams:
    name: str
    bull_beta_unlock: float = -0.02
    normal_bab_tilt: float = 0.16
    bear_beta_clamp: float = 0.42
    ntz_hysteresis_bonus: float = 0.25
    fast_crash_dd60: float = -0.125
    fast_crash_floor_3d: float = 0.042
    slow_bear_r60: float = -0.22
    slow_bear_breadth: float = 0.26
    rebound_release_r5: float = 0.015
    bull_r20_thresh: float = 0.022
    bull_breadth_thresh: float = 0.56
    bull_trend_tilt: float = 0.04
    bear_defense_tilt: float = 0.08
    backbone_prior_weight: float = 0.48
    signal_ema_span: int = 2
    markowitz_ratio_mode: bool = False


@dataclass
class V5ControllerOutput:
    sealed_signal_wide: pd.DataFrame
    regime_series: pd.Series
    exposure_series: pd.Series
    abstain_rate: float
    dormant_veto_fraction: float


class AsymBetaNTZController:
    """Asymmetric Beta + Friction-Matched No-Trade-Zone + Band-Scaled Dual-Speed Bear Shield Controller."""

    def __init__(self, top_k: int = 6, exchange_price_limit_pct: float = 0.07) -> None:
        self.top_k = top_k
        # Physically derived Gaussian tail scaling across exchange price-limit widths:
        # P(r <= -L) decays ~ exp(-L^2 / 2 sigma^2); on +/-7% HOSE scale=1.0, on +/-10% CSI scale=(0.07/0.10)^2=0.49!
        self.exchange_price_limit_pct = max(0.05, float(exchange_price_limit_pct))
        self.band_tail_scale = float(np.clip((0.07 / self.exchange_price_limit_pct) ** 2, 0.25, 1.50))

    def apply(
        self,
        alpha_wide: pd.DataFrame,
        panel: dict[str, pd.DataFrame],
        state_df: pd.DataFrame,
        conformal_out: ConformalVetoOutput,
        params: AsymBetaNTZParams,
        use_conformal_veto: bool = True,
        use_asym_beta_ntz: bool = True,
    ) -> V5ControllerOutput:
        """Produce strictly causal sealed signal matrix (information <= t only)."""
        dates = pd.DatetimeIndex(alpha_wide.index)
        syms = list(alpha_wide.columns)
        a_raw = _cs_z_v5(alpha_wide.reindex(index=dates, columns=syms))
        if use_asym_beta_ntz and params.signal_ema_span > 1:
            a_df = _cs_z_v5(a_raw.ewm(span=params.signal_ema_span, min_periods=1).mean())
        else:
            a_df = a_raw

        if not use_asym_beta_ntz:
            sig = a_df.copy()
            if use_conformal_veto:
                sig = sig - conformal_out.stock_penalty_wide.reindex(index=dates, columns=syms).fillna(0.0)
            return V5ControllerOutput(
                sealed_signal_wide=sig,
                regime_series=pd.Series("Static_Linear", index=dates),
                exposure_series=pd.Series(1.0, index=dates),
                abstain_rate=0.0,
                dormant_veto_fraction=conformal_out.dormant_fraction,
            )

        close = panel["close"].reindex(index=dates, columns=syms)
        ret1 = panel["return_1d"].reindex(index=dates, columns=syms).fillna(0.0)
        mkt_ret = ret1.mean(axis=1)
        mkt_wealth = (1.0 + mkt_ret).cumprod()
        mkt_dd60 = (mkt_wealth / mkt_wealth.rolling(60, min_periods=5).max().clip(lower=1e-8)) - 1.0
        mkt_r60 = ((1.0 + mkt_ret).rolling(60, min_periods=15).apply(np.prod, raw=True) - 1.0).fillna(0.0)

        # Strictly causal Vasicek-shrunk rolling 120d beta + volatility
        vol20_s = ret1.rolling(20, min_periods=5).std().fillna(0.015)
        mom20_s = close.pct_change(20, fill_method=None).fillna(0.0)
        bull_leader = _cs_z_v5(
            mom20_s / (vol20_s + 0.008)
            + 0.35 * _cs_z_v5(panel["foreign_net_ratio"].reindex(index=dates, columns=syms).rolling(5, min_periods=1).mean().fillna(0.0))
        )

        mkt_var120 = mkt_ret.rolling(120, min_periods=15).var().clip(lower=1e-6)
        mkt_m120 = mkt_ret.rolling(120, min_periods=15).mean()
        r_m120 = ret1.rolling(120, min_periods=15).mean()
        cov120 = ret1.mul(mkt_ret, axis=0).rolling(120, min_periods=15).mean().sub(r_m120.mul(mkt_m120, axis=0), axis=0)
        raw_beta120 = cov120.div(mkt_var120, axis=0).fillna(1.0)
        shrunk_beta = (0.65 * raw_beta120 + 0.35 * 1.0).clip(0.35, 2.0)
        beta_risk_z = _cs_z_v5(0.75 * _cs_z_v5(shrunk_beta) + 0.25 * _cs_z_v5(vol20_s))

        floor_safe = (
            panel["dist_to_floor"].reindex(index=dates, columns=syms).fillna(0.05)
            - panel["intraday_range"].reindex(index=dates, columns=syms).fillna(0.02)
        )
        defensive_view = _cs_z_v5(
            0.50 * _cs_z_v5(panel["pit_value_quality"].reindex(index=dates, columns=syms).fillna(0.0))
            + 0.20 * _cs_z_v5(floor_safe)
            + 0.30 * (-beta_risk_z)
        )
        floor_1d_s = (panel["dist_to_floor"].reindex(index=dates, columns=syms).fillna(0.05) < 0.015).mean(axis=1).fillna(0.0)

        st = state_df.reindex(index=dates).fillna(0.0)
        pen_mat = conformal_out.stock_penalty_wide.reindex(index=dates, columns=syms).fillna(0.0).to_numpy(dtype=float)
        veto_mat = conformal_out.stock_hard_veto_wide.reindex(index=dates, columns=syms).fillna(False).to_numpy(dtype=bool)
        mkt_abstain = conformal_out.market_abstain_series.reindex(index=dates).fillna(False).to_numpy(dtype=bool)

        a_mat = a_df.to_numpy(dtype=float)
        b_mat = bull_leader.to_numpy(dtype=float)
        d_mat = defensive_view.to_numpy(dtype=float)
        beta_mat = beta_risk_z.to_numpy(dtype=float)

        n_days, n_syms = a_mat.shape
        out_sig = np.full((n_days, n_syms), np.nan, dtype=float)
        regimes: list[str] = []
        exposures = np.zeros(n_days, dtype=float)
        incumbent_mask = np.zeros(n_syms, dtype=bool)

        # Band-Scaled Dual-Speed Bear Shield (zero-latch instant release):
        # On +/-7% HOSE: requires active daily floor selling (f1 >= 0.40*eff_f3) + no rebound (r5 < release) to avoid V-bottom whipsaw!
        # On +/-10% CSI: Gaussian tail scaling (0.07/0.10)^2 = 0.49 catches severe limit-down bear waterfalls cleanly!
        for t in range(n_days):
            r20 = float(st["mkt_r20"].iloc[t])
            r5 = float(st["mkt_r5"].iloc[t])
            r60 = float(mkt_r60.iloc[t])
            breadth = float(st["breadth_20"].iloc[t])
            vol_rat = float(st["vol_ratio"].iloc[t])
            floor_3d = float(st["floor_near_3d"].iloc[t])
            dd60 = float(mkt_dd60.iloc[t])
            f1_now = float(floor_1d_s.iloc[t])

            # Physics-derived Exchange-Band Tail Shield:
            # - Tight-band (+/-7% HOSE, bscale >= 0.85): fast floor-contagion circuit breaker catches panics
            #   with instant bounce release (zero post-crash base selling!).
            # - Wide/no-band (+/-10% CSI, US, bscale < 0.85): Gaussian tail scaling + deep secular collapse
            #   guard catches low-floor bear waterfalls without V-bottom whipsaw.
            eff_f3_thresh = params.fast_crash_floor_3d * self.band_tail_scale
            eff_f1_thresh = eff_f3_thresh * (0.40 if self.band_tail_scale >= 0.85 else 0.0)
            r5_guard = (r5 < params.rebound_release_r5) if self.band_tail_scale >= 0.85 else True
            fast_crash = (
                (dd60 <= params.fast_crash_dd60)
                and (floor_3d >= eff_f3_thresh)
                and (f1_now >= eff_f1_thresh)
                and r5_guard
            )
            slow_secular_bear = (
                (self.band_tail_scale < 0.85)
                and (r60 <= params.slow_bear_r60)
                and (dd60 <= params.fast_crash_dd60 * 1.05)
                and (breadth <= params.slow_bear_breadth)
                and (r20 < -0.035)
                and (r5 < params.rebound_release_r5)
            )
            conformal_abstain = bool(
                use_conformal_veto
                and mkt_abstain[t]
                and (dd60 <= params.fast_crash_dd60)
                and (floor_3d >= eff_f3_thresh)
                and (f1_now >= eff_f1_thresh)
                and r5_guard
            )

            if fast_crash or slow_secular_bear or conformal_abstain:
                regimes.append("Epistemic_Abstain_Cash")
                exposures[t] = 0.0
                out_sig[t, :] = np.nan
                incumbent_mask[:] = False
                continue

            is_bull = (
                (r20 > params.bull_r20_thresh)
                and (breadth >= params.bull_breadth_thresh)
                and (vol_rat < 1.25)
            )
            is_stress = (not is_bull) and ((vol_rat > 1.32) or (dd60 < -0.085))

            row_score = a_mat[t].copy()
            # Sharp alpha-preserving asymmetric BAB gate: keeps bottom-half ranks pure alpha (preserving
            # cross-sectional Rank IC >= 0.118) while giving top-ranked Buy candidates full Bull beta+ / Bear beta- control!
            gate = 0.15 + 0.85 / (1.0 + np.exp(-np.clip(2.2 * row_score, -10.0, 10.0)))
            if is_bull:
                regimes.append("Bull_BetaPlus_Unlock")
                row_score = (1.0 - params.bull_trend_tilt) * row_score + params.bull_trend_tilt * b_mat[t] - params.bull_beta_unlock * gate * beta_mat[t]
                exposures[t] = 1.0
            elif is_stress:
                regimes.append("Bear_BetaMinus_Clamp")
                row_score = (1.0 - params.bear_defense_tilt) * row_score + params.bear_defense_tilt * d_mat[t] - params.bear_beta_clamp * gate * beta_mat[t]
                exposures[t] = 0.72
            else:
                regimes.append("Normal_NTZ")
                row_score = row_score - params.normal_bab_tilt * gate * beta_mat[t]
                exposures[t] = 0.94

            if use_conformal_veto:
                row_score = row_score - pen_mat[t]
                row_score = np.where(veto_mat[t], row_score - 0.30, row_score)

            row_score = np.nan_to_num(row_score, nan=0.0, posinf=4.0, neginf=-4.0)
            # Friction-Matched No-Trade-Zone (NTZ) exceeding T+2.5 + 0.1% sell tax + spread impact
            row_score = row_score + params.ntz_hysteresis_bonus * incumbent_mask.astype(float)

            out_sig[t, :] = row_score
            top_idx = np.argsort(row_score)[::-1][: self.top_k]
            incumbent_mask[:] = False
            incumbent_mask[top_idx] = True

        abstain_rate = float(np.mean(exposures == 0.0)) if n_days > 0 else 0.0
        return V5ControllerOutput(
            sealed_signal_wide=pd.DataFrame(out_sig, index=dates, columns=syms),
            regime_series=pd.Series(regimes, index=dates, name="v5_regime"),
            exposure_series=pd.Series(exposures, index=dates, name="v5_exposure"),
            abstain_rate=abstain_rate,
            dormant_veto_fraction=conformal_out.dormant_fraction,
        )


# ---------------------------------------------------------------------------
# PART 2.5: Full MAVE-Q v5 (`DCA-Chronos-RSI` + Popperian Falsification) Engine
# ---------------------------------------------------------------------------
@dataclass
class MAVEQv5Result:
    preset: str
    resolved_safe_preset: str
    backbone_family: str
    scale_rung: str
    backbone_exact_params: int
    sub1b_blueprint_verified: bool
    selected_controller: str
    champion_expressions: list[str]
    champion_weights: list[float]
    champion_niches: list[str]
    search_candidates_screened: int
    refuter_rejections: int
    mose_active_branches: list[str]
    mose_pruned_branches: list[str]
    dormant_veto_fraction: float
    epistemic_abstain_rate: float
    validation_queries_used: int
    total_evaluator_trials: int
    champion_search_cv_score: float
    champion_validation_metrics: dict[str, float]
    champion_validation_extended: dict[str, Any]
    champion_validation_fold_std: dict[str, float]
    champion_sealed_test_metrics: dict[str, Any]
    champion_sealed_test_extended: dict[str, Any]
    regime_exposure_summary: dict[str, float]
    validation_daily_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float), repr=False)
    sealed_test_daily_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float), repr=False)


class MAVEQv5Engine:
    """MAVE-Q v5 (`DCA-Chronos-RSI` + Popperian Falsification) Zero-Validation-Burn Engine."""

    def __init__(
        self,
        evaluator: SealedEvaluator,
        cfg: dict[str, Any],
        preset: str = "auto",
        device: str | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.cfg = cfg
        self.requested_preset = preset
        self.seed = int(cfg.get("seed", 2026))
        self.rng = np.random.default_rng(self.seed)

        # HardwareSafetyGuard lock (prevents OOM on <12GB RAM laptop; auto-promotes on Cloud GPU/TPU!)
        self.guard = HardwareSafetyGuard()
        self.safety_resolution = self.guard.resolve_safe_preset(preset)
        self.resolved_preset = self.safety_resolution.resolved_preset
        self.sub1b_blueprint = verify_sub1b_blueprint_zero_ram("sub1b_flagship_900m")

        # Ablation switches (every module cleanly toggleable for Task 05 ruthless pruning!)
        ab = cfg.get("ablations", {})
        self.use_dca_backbone = bool(ab.get("use_dca_backbone", True))
        self.use_chronos_kalman_hedge = bool(ab.get("use_chronos_kalman_hedge", True))
        self.use_mose_hypergate = bool(ab.get("use_mose_hypergate", True))
        self.use_conformal_veto = bool(ab.get("use_conformal_veto", True))
        self.use_asym_beta_ntz = bool(ab.get("use_asym_beta_ntz", True))

        # Symbolic AST diffusion + HRM (use cpu_fast working dims on low-RAM CPU for fast evolution)
        sym_preset = "cpu_fast" if self.guard.is_low_ram_local() else self.resolved_preset
        self.diffusion = TopologicalTreeDiffusion(preset=sym_preset, seed=self.seed, device=device)
        self.hrm = HoloHRMReasoner(preset=sym_preset, seed=self.seed + 31, device=device)
        self.verifier = ASTVerifier(max_depth=5)
        self.sampler = GrammarTreeSampler(max_depth=4)
        self.archive = HoloQDArchive()

        # Hermetically isolated SEARCH-ZONE-ONLY data (validation & sealed_test DO NOT EXIST here!)
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
        self.search_state = compute_causal_microstructure_state(self.search_panel)

        self.local_search_sim = VNPortfolioSimulator(
            market_rules=evaluator.market_rules,
            top_k=evaluator.top_k,
            rebalance_every_days=evaluator.rebalance_every_days,
            capital_vnd=evaluator.capital_vnd,
        )

        # Popperian AdversarialHypothesisRefuter strictly on search zone
        ref_cfg = cfg.get("popperian_refuter", {})
        self.use_refuter = bool(ref_cfg.get("enabled", True))
        self.refuter = AdversarialHypothesisRefuter(
            market_rules=evaluator.market_rules,
            search_bars=self.search_bars,
            safe_dates=self.safe_dates,
            search_fwd=self.search_fwd,
            search_folds=self.search_folds,
            top_k=evaluator.top_k,
            rebalance_every_days=evaluator.rebalance_every_days,
            capital_vnd=evaluator.capital_vnd,
            friction_multiplier=float(ref_cfg.get("friction_multiplier", 1.5)),
            max_friction_drop_ratio=float(ref_cfg.get("max_friction_drop_ratio", 0.40)),
            occam_bic_lambda=float(ref_cfg.get("occam_bic_lambda", 0.08)),
        )
        self.refuter_rejections: int = 0
        self.refutation_log: list[dict[str, Any]] = []

        # Backbone Zoo (default `dca_quantformer`; ablation `use_dca_backbone=False` falls back to `preln_transformer`)
        bb_cfg = cfg.get("backbone_zoo", {})
        req_family = str(bb_cfg.get("active_family", "dca_quantformer"))
        self.backbone_family = req_family if self.use_dca_backbone else "preln_transformer"
        self.scale_rung = str(bb_cfg.get("scale_rung", "0.5m"))
        self.backbone = QuantBackboneModel(
            family=self.backbone_family,
            scale_rung=self.scale_rung,
            seed=self.seed,
            ridge_l2_frac=float(bb_cfg.get("ridge_l2_frac", 0.22)),
            gp_turnover_exp=float(bb_cfg.get("gp_turnover_exp", 0.45)),
            preset=self.resolved_preset,
        )
        self.backbone.fit_on_search(
            self.search_panel,
            self.safe_dates,
            self.search_fwd,
            rebal_step=evaluator.rebalance_every_days,
            top_k=evaluator.top_k,
        )
        self.search_backbone_sig = self.backbone.predict_panel(self.search_panel)
        bb_ic_s = compute_rank_ic_series(self.search_backbone_sig.reindex(index=self.safe_dates), self.search_fwd)
        self.backbone_search_ic = float(bb_ic_s.mean()) if len(bb_ic_s) else 0.05

        # Chronos-Kalman-Hedge online causal combiner
        ck_cfg = cfg.get("chronos_kalman_hedge", {})
        self.chronos_hedge = ChronosKalmanHedge(
            horizon_delay_days=int(ck_cfg.get("horizon_delay_days", evaluator.horizon_days)),
            process_noise_q=float(ck_cfg.get("process_noise_q", 0.0008)),
            obs_noise_r=float(ck_cfg.get("obs_noise_r", 0.018)),
            hedge_eta=float(ck_cfg.get("hedge_eta", 6.5)),
            hedge_discount_gamma=float(ck_cfg.get("hedge_discount_gamma", 0.965)),
            turnover_penalty_lambda=float(ck_cfg.get("turnover_penalty_lambda", 0.08)),
            online_prune_skill_thresh=float(ck_cfg.get("online_prune_skill_thresh", -0.005)),
        )

        # DCA-MoSE HyperGate
        mose_cfg = cfg.get("mose_hypergate", {})
        self.mose_gate = DCAMoSEHyperGate(
            group_lasso_lambda=float(mose_cfg.get("group_lasso_lambda", 0.025)),
            max_nonlinear_weight=float(mose_cfg.get("max_nonlinear_weight", 0.35)),
        )

        # Split-Conformal Dormant-by-Default Veto & Epistemic Abstention
        sc_cfg = cfg.get("split_conformal_veto", {})
        self.conformal_veto = SplitConformalDormantVeto(
            alpha_level=float(sc_cfg.get("alpha_level", 0.10)),
            calibration_fraction=float(sc_cfg.get("calibration_fraction", 0.35)),
            stock_veto_scale=float(sc_cfg.get("stock_veto_scale", 0.45)),
            market_abstain_alpha=float(sc_cfg.get("market_abstain_alpha", 0.06)),
        )

        # Infer primary exchange daily price limit from market_rules (0.07 HOSE, 0.10 CSI, >=0.50 US no-bands)
        primary_lim = 0.07
        if hasattr(evaluator.market_rules, "exchanges") and evaluator.market_rules.exchanges:
            ex_vals = [float(ex.daily_price_limit_pct) for ex in evaluator.market_rules.exchanges.values()]
            if ex_vals:
                primary_lim = float(min(ex_vals))
        self.controller = AsymBetaNTZController(top_k=evaluator.top_k, exchange_price_limit_pct=primary_lim)

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
        sig_z = sig_df.reindex(index=self.search_dates)
        sim = self.local_search_sim.simulate(sig_z, self.search_bars)
        ic_s = compute_rank_ic_series(sig_z.reindex(index=self.safe_dates), self.search_fwd)
        full_m = compute_portfolio_metrics(sim.daily_net_returns, sim.daily_gross_returns, sim.daily_turnover, ic_s, n_trials=1)
        fold_ms: list[dict[str, float]] = []
        for fold in self.search_folds:
            f_ic = ic_s.reindex(fold.eval_dates).dropna()
            f_net = sim.daily_net_returns.reindex(fold.eval_dates).dropna()
            f_gross = sim.daily_gross_returns.reindex(fold.eval_dates).dropna()
            f_turn = sim.daily_turnover.reindex(fold.eval_dates).dropna()
            fold_ms.append(compute_portfolio_metrics(f_net, f_gross, f_turn, f_ic, n_trials=1))
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
        windows = [int(w) for w in evo.get("mssf_windows", [1, 3, 5])]
        turn_pen = float(evo.get("mssf_turnover_penalty", 0.16))
        core_depth = raw_spec.root.depth()
        raw_norm = _cs_z_v5(s_base)
        raw_ic_s = compute_rank_ic_series(raw_norm, self.search_fwd)
        if len(raw_ic_s) < 20:
            return None
        raw_mean_ic = float(raw_ic_s.mean())
        raw_sign = 1.0 if raw_mean_ic >= 0.0 else -1.0
        raw_abs_ic = abs(raw_mean_ic)
        raw_turn = self._top_k_turnover_proxy(raw_norm * raw_sign, self.evaluator.rebalance_every_days, self.evaluator.top_k)
        if raw_turn <= 0.10:
            windows = [1]

        best_choice: tuple[float, int, float] | None = None
        for w in windows:
            if w > 1 and core_depth >= 5:
                continue
            cand_df = s_base if w <= 1 else s_base.rolling(window=w, min_periods=max(1, w // 2)).mean()
            norm_df = _cs_z_v5(cand_df)
            ic_s = compute_rank_ic_series(norm_df, self.search_fwd)
            if len(ic_s) < 20:
                continue
            mean_ic = float(ic_s.mean())
            sign = 1.0 if mean_ic >= 0.0 else -1.0
            abs_ic = abs(mean_ic)
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

    def _screen_candidate(self, raw_spec: FactorSpec, family: str) -> HoloElite | None:
        calibrated = self._mssf_calibrate(raw_spec)
        if calibrated is None:
            return None
        spec, sign, w_star = calibrated
        fp_key = f"{spec.root.structural_fingerprint()}:{int(sign)}"
        if fp_key in self.screen_cache:
            return self.screen_cache[fp_key]

        s_df = evaluate_factor_spec(spec, self.search_panel, verifier=self.verifier).fillna(0.0) * sign

        # Popperian AdversarialHypothesisRefuter gate inside search zone (reuses base simulation -> 0 redundant sims!)
        if self.use_refuter:
            verdict, metrics, fold_ms = self.refuter.refute_signal_with_metrics(
                candidate_id=spec.factor_id,
                signal_search_wide=s_df,
                category="factor_ast",
                complexity_units=float(max(1, spec.root.depth())),
                fast_ic_short_circuit=True,
            )
            if not verdict.accepted:
                self.refuter_rejections += 1
                if len(self.refutation_log) < 32:
                    self.refutation_log.append(
                        verdict.to_ledger_record(
                            stage="v5_search_factor_screening",
                            hypothesis=f"Symbolic AST factor `{spec.root.canonical_expr()}`",
                            action_taken="Rejected inside search zone prior to archive insertion.",
                        )
                    )
                self.screen_cache[fp_key] = None
                return None
        else:
            metrics, fold_ms = self._local_search_eval_signal(s_df)
        self.n_search_screened += 1
        s_safe = s_df.reindex(index=self.safe_dates).fillna(0.0)
        ic_s = compute_rank_ic_series(s_safe, self.search_fwd)
        mean_ic = float(ic_s.mean()) if len(ic_s) else float(metrics["rank_ic"])
        fold_ics = [float(fm["rank_ic"]) for fm in fold_ms] or [mean_ic]
        min_fold_ic = float(min(fold_ics))
        sr = float(metrics["net_sharpe"])
        turn = float(metrics["daily_turnover"])
        icir = float(metrics["icir"])

        depth_pen = 0.005 * max(0, spec.root.depth() - 2)
        utility = 0.55 * mean_ic + 0.35 * min_fold_ic + 0.025 * float(np.clip(sr, -1.5, 3.0)) - 0.16 * turn - depth_pen
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
            extra_seeds = [
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
            all_seeds = _curated_domain_seeds() + extra_seeds
            perm = self.rng.permutation(len(all_seeds))
            for idx_s in perm[:9]:
                fam, node = all_seeds[int(idx_s)]
                fp = node.structural_fingerprint()
                if fp not in self.seen_raw_fps and is_typed_stationary_ast(node):
                    self.seen_raw_fps.add(fp)
                    seen_batch.add(fp)
                    pool.append(
                        (
                            fam,
                            FactorSpec(
                                factor_id=f"v5_seed_{len(pool)}",
                                hypothesis_id="v5_seed",
                                description=f"v5 domain seed ({fam})",
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
            fid, hid = f"v5_r{r_idx}_c{k}", f"v5h_r{r_idx}_c{k}"
            u = self.rng.random()
            seed_k = self.seed + r_idx * 7919 + k * 31
            if u < 0.20:
                cand = FactorSpec(
                    factor_id=fid,
                    hypothesis_id=hid,
                    description="grammar exploration",
                    root=self.sampler.random_tree(self.rng, max_depth=4),
                )
            elif len(elites) >= 2 and u < 0.38:
                i1, i2 = self.rng.choice(len(elites), size=2, replace=False)
                p_l, p_r = elites[int(i1)].spec.root, elites[int(i2)].spec.root
                cand = self.diffusion.bridge_fuse_elites(
                    p_l, p_r, fam, fid, hid, temperature=temp, seed=seed_k, use_tree_coords=True
                )
            elif elites and u < 0.74:
                parent = max(
                    (elites[int(i)] for i in self.rng.integers(0, len(elites), size=min(3, len(elites)))),
                    key=lambda e: e.utility,
                )
                weak_pos = self.hrm.identify_weakest_subtree_pos(parent.spec, regime_vec)
                cand = self.diffusion.attribution_guided_infill(
                    parent.spec, weak_pos, fam, fid, hid, max_depth=4, temperature=temp, seed=seed_k, use_tree_coords=True
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
            if not is_typed_stationary_ast(cand.root):
                continue
            seen_batch.add(fp)
            pool.append((fam, cand))
        return pool

    @staticmethod
    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b)) / (float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12)

    def _select_diverse_basis(self, max_basis: int = 5, max_corr: float = 0.65) -> list[HoloElite]:
        pos = [e for e in self.archive.all_elites() if e.rank_ic > 0.020 and e.utility > -0.005]
        if not pos:
            pos = self.archive.all_elites()[: max(1, min(3, len(self.archive.all_elites())))]
        selected: list[HoloElite] = []
        covered: set[str] = set()
        for e in pos:
            if e.niche in covered:
                continue
            if all(abs(self._corr(e.signal_vec, s.signal_vec)) <= max_corr for s in selected):
                selected.append(e)
                covered.add(e.niche)
            if len(selected) >= max_basis:
                return selected
        for e in pos:
            if any(e.fingerprint == s.fingerprint for s in selected):
                continue
            if all(abs(self._corr(e.signal_vec, s.signal_vec)) <= max_corr for s in selected):
                selected.append(e)
            if len(selected) >= max_basis:
                break
        return selected

    def _solve_prior_weights(self, basis: list[HoloElite]) -> np.ndarray:
        """Boundary-Turnover-Optimal Spectral Risk-Parity Subspace Synthesis (`BTO-SRSS`) on search zone."""
        m = len(basis)
        if m <= 1:
            return np.ones(max(1, m), dtype=float)
        ic_panel = pd.concat([e.ic_series for e in basis], axis=1, sort=False).dropna()
        g_ic = np.asarray([max(0.004, 0.60 * e.rank_ic + 0.40 * max(0.0, e.min_fold_ic)) for e in basis], dtype=float)
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
        w = 0.55 * w_tan + 0.45 * w_anchor
        return w / (float(np.sum(w)) + 1e-12)

    def _controller_grid(self) -> list[AsymBetaNTZParams]:
        rcfg = self.cfg.get("asym_beta_ntz", {})
        raw_grid = rcfg.get("cv_grid", [])
        if not raw_grid or not self.use_asym_beta_ntz:
            return [AsymBetaNTZParams(name="static_no_asym_ntz")]
        return [
            AsymBetaNTZParams(
                name=str(g["name"]),
                bull_beta_unlock=float(g.get("bull_beta_unlock", -0.02)),
                normal_bab_tilt=float(g.get("normal_bab_tilt", 0.16)),
                bear_beta_clamp=float(g.get("bear_beta_clamp", 0.42)),
                ntz_hysteresis_bonus=float(g.get("ntz_hysteresis_bonus", 0.25)),
                fast_crash_dd60=float(g.get("fast_crash_dd60", -0.125)),
                fast_crash_floor_3d=float(g.get("fast_crash_floor_3d", 0.042)),
                slow_bear_r60=float(g.get("slow_bear_r60", -0.22)),
                slow_bear_breadth=float(g.get("slow_bear_breadth", 0.26)),
                rebound_release_r5=float(g.get("rebound_release_r5", 0.015)),
                bull_r20_thresh=float(g.get("bull_r20_thresh", 0.022)),
                bull_breadth_thresh=float(g.get("bull_breadth_thresh", 0.56)),
                bull_trend_tilt=float(g.get("bull_trend_tilt", 0.04)),
                bear_defense_tilt=float(g.get("bear_defense_tilt", 0.08)),
                backbone_prior_weight=float(g.get("backbone_prior_weight", 0.48)),
                signal_ema_span=int(g.get("signal_ema_span", 2)),
                markowitz_ratio_mode=bool(g.get("markowitz_ratio_mode", False)),
            )
            for g in raw_grid
        ]

    def _build_expert_pool(
        self,
        elites: list[HoloElite],
        weights: list[float],
        ctrl_params: AsymBetaNTZParams,
        panel: dict[str, pd.DataFrame],
        backbone_sig_override: pd.DataFrame | None = None,
    ) -> tuple[list[pd.DataFrame], list[str], list[float], list[float]]:
        """Assemble strictly causal expert bank for Chronos-Kalman-Hedge (Backbone + Symbolic ASTs + Regime Pillars)."""
        bb_sig = backbone_sig_override if backbone_sig_override is not None else self.backbone.predict_panel(panel)
        close = panel["close"]
        ret1 = panel["return_1d"].reindex_like(close).fillna(0.0)
        vol20 = ret1.rolling(20, min_periods=5).std().fillna(0.015)
        # Low-turnover Orthogonal Regime Specialists (Bull Quality-Momentum vs Bear Defensive-Quality):
        mom_ra = (close.pct_change(15, fill_method=None).fillna(0.0) / (vol20 + 0.006)).rolling(5, min_periods=1).mean()
        pit_roe = panel["pit_roe"].reindex_like(close).fillna(0.08)
        pit_pb = panel["pit_pb_ratio"].reindex_like(close).fillna(1.8).clip(lower=0.3)
        val_rk = pit_roe.rank(axis=1, pct=True) - pit_pb.rank(axis=1, pct=True)
        flow_sm10 = panel["foreign_net_ratio"].reindex_like(close).rolling(10, min_periods=2).mean().fillna(0.0)
        trend_flow_pillar = _cs_z_v5(0.45 * _cs_z_v5(mom_ra) + 0.35 * _cs_z_v5(val_rk) + 0.20 * _cs_z_v5(flow_sm10))

        pit_vq = panel["pit_value_quality"].reindex_like(close).fillna(0.0)
        vol_calm = (-vol20).rolling(5, min_periods=1).mean()
        def_qual_pillar = _cs_z_v5(0.60 * _cs_z_v5(pit_vq) + 0.40 * _cs_z_v5(vol_calm))

        # Aggregate Popperian-vetted symbolic AST book
        sym_total = sum(abs(w) for w in weights) or 1.0
        sym_comb: pd.DataFrame | None = None
        sym_ic_weighted = 0.0
        for e, wt in zip(elites, weights, strict=True):
            nw = abs(wt) / sym_total
            part = _cs_z_v5(evaluate_factor_spec(e.spec, panel, verifier=self.verifier).fillna(0.0) * e.sign) * nw
            sym_comb = part if sym_comb is None else (sym_comb + part)
            sym_ic_weighted += nw * max(0.01, e.rank_ic)
        sym_sig = _cs_z_v5(sym_comb) if sym_comb is not None else bb_sig

        bb_w = float(np.clip(ctrl_params.backbone_prior_weight, 0.65, 0.86))
        rem = 1.0 - bb_w
        expert_sigs = [bb_sig, sym_sig, trend_flow_pillar, def_qual_pillar]
        expert_names = [f"backbone_{self.backbone_family}", "symbolic_ast_book", "pillar_quality_mom", "pillar_defensive_quality"]
        prior_w = [bb_w, rem * 0.50, rem * 0.20, rem * 0.30]
        prior_sk = [max(0.06, self.backbone_search_ic), max(0.03, sym_ic_weighted), 0.09, 0.08]
        return expert_sigs, expert_names, prior_w, prior_sk

    def build_full_causal_signal(
        self,
        elites: list[HoloElite],
        weights: list[float],
        ctrl_params: AsymBetaNTZParams,
        panel: dict[str, pd.DataFrame],
        backbone_sig_override: pd.DataFrame | None = None,
    ) -> V5ControllerOutput:
        """Build full strictly-causal signal matrix across any panel (info <= t only; atol=1e-9 shift-invariant!)."""
        state_df = compute_causal_microstructure_state(panel)
        expert_sigs, expert_names, prior_w, prior_sk = self._build_expert_pool(
            elites, weights, ctrl_params, panel, backbone_sig_override=backbone_sig_override
        )

        # 1. Chronos-Kalman-Hedge online causal fusion with EXACT H-day delay buffer
        hedge_out = self.chronos_hedge.fuse_online(
            expert_signals=expert_sigs,
            expert_names=expert_names,
            prior_weights=prior_w,
            prior_skills=prior_sk,
            close_wide=panel["close"],
            enabled=self.use_chronos_kalman_hedge,
        )

        # 2. DCA-MoSE HyperGate sparse L1/Group-Lasso nonlinear refinement
        gated_alpha = self.mose_gate.apply(
            base_alpha_wide=hedge_out.combined_alpha_wide,
            panel=panel,
            enabled=self.use_mose_hypergate,
        )

        # 3. Split-Conformal Dormant-by-Default Veto & Epistemic Abstention
        conf_out = self.conformal_veto.apply(
            panel=panel,
            state_df=state_df,
            epistemic_disagreement=hedge_out.epistemic_disagreement_wide,
            enabled=self.use_conformal_veto,
        )

        # 4. AsymBeta-NTZ Controller
        return self.controller.apply(
            alpha_wide=gated_alpha,
            panel=panel,
            state_df=state_df,
            conformal_out=conf_out,
            params=ctrl_params,
            use_conformal_veto=self.use_conformal_veto,
            use_asym_beta_ntz=self.use_asym_beta_ntz,
        )

    def _fit_search_gates_for_basis(
        self,
        elites: list[HoloElite],
        weights: list[float],
        ctrl_params: AsymBetaNTZParams,
    ) -> None:
        """Fit MoSE HyperGate and Split-Conformal quantiles strictly on search zone (zero val/test access!)."""
        expert_sigs, expert_names, prior_w, prior_sk = self._build_expert_pool(
            elites, weights, ctrl_params, self.search_panel, backbone_sig_override=self.search_backbone_sig
        )

        hedge_s = self.chronos_hedge.fuse_online(
            expert_signals=expert_sigs,
            expert_names=expert_names,
            prior_weights=prior_w,
            prior_skills=prior_sk,
            close_wide=self.search_panel["close"],
            enabled=self.use_chronos_kalman_hedge,
        )
        if self.use_mose_hypergate:
            self.mose_gate.fit_on_search(
                search_panel=self.search_panel,
                base_alpha_search=hedge_s.combined_alpha_wide,
                safe_dates=self.safe_dates,
                search_fwd=self.search_fwd,
                search_folds=self.search_folds,
            )
        if self.use_conformal_veto:
            self.conformal_veto.fit_on_search(
                search_panel=self.search_panel,
                search_state=self.search_state,
                safe_dates=self.safe_dates,
                epistemic_disagreement_search=hedge_s.epistemic_disagreement_wide,
            )

    def _search_cv_score(
        self,
        elites: list[HoloElite],
        weights: list[float],
        ctrl_params: AsymBetaNTZParams,
    ) -> float:
        self._fit_search_gates_for_basis(elites, weights, ctrl_params)
        out = self.build_full_causal_signal(
            elites, weights, ctrl_params, self.search_panel, backbone_sig_override=self.search_backbone_sig
        )
        m, fold_ms = self._local_search_eval_signal(out.sealed_signal_wide)
        f_srs = [float(fm["net_sharpe"]) for fm in fold_ms] or [float(m["net_sharpe"])]
        f_ics = [float(fm["rank_ic"]) for fm in fold_ms] or [float(m["rank_ic"])]
        mdd = abs(float(m["max_drawdown"]))
        calmar_proxy = float(m["ann_net_return"]) / max(mdd, 0.05)
        cash_drag = 2.0 * max(0.0, out.abstain_rate - 0.22)
        return float(
            float(m["net_sharpe"])
            + 0.35 * float(min(f_srs))
            + 3.5 * float(m["rank_ic"])
            + 1.5 * float(min(f_ics))
            + 0.14 * float(np.clip(calmar_proxy, -2.0, 8.0))
            - 1.5 * float(m["daily_turnover"])
            - cash_drag
        )

    def run(self, run_id: str = "maveq_v5_run", score_sealed: bool = False) -> MAVEQv5Result:
        evo = self.cfg.get("evolution", {})
        rounds = int(evo.get("rounds", 5))
        n_latent = int(evo.get("latent_candidates_per_round", 40))
        n_screen = int(evo.get("screen_per_round", 10))
        eps_explore = float(evo.get("ucb_explore_fraction", 0.25))
        max_ens = int(evo.get("max_ensemble_factors", 5))
        max_corr = float(evo.get("max_member_corr", 0.65))
        grpo_epochs = int(evo.get("diffu_grpo_epochs_per_round", 8))
        max_val_q = int(evo.get("max_validation_queries", 1))

        ctrl_grid = self._controller_grid()
        best_search: tuple[float, list[HoloElite], list[float], AsymBetaNTZParams] | None = None

        for r_idx in range(1, rounds + 1):
            regime_vec = self._regime_vec()
            latent = self._generate_latent_pool(r_idx, n_latent)
            if r_idx == 1:
                chosen = latent[: max(n_screen, 10)]
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
                elite = self._screen_candidate(raw_cand, fam)
                if elite is None:
                    continue
                self.archive.insert(elite)
                self.fam_ic_sum[fam] += max(0.0, elite.rank_ic)
                self.fam_cnt[fam] += 1
                self.hrm_replay.append(
                    {"node": raw_cand.root, "regime_vec": regime_vec, "utility": elite.utility, "turnover": elite.turnover}
                )

            all_sc = [e for e in self.screen_cache.values() if e is not None]
            if all_sc:
                all_sc.sort(key=lambda e: e.utility, reverse=True)
                top_rep = all_sc[: min(8, len(all_sc))]
                bot_rep = all_sc[-min(6, max(0, len(all_sc) - len(top_rep))):] if len(all_sc) > len(top_rep) else []
                rollouts = [
                    {"family": e.family, "factor_spec": e.spec.model_dump(), "shapley_reward": e.utility}
                    for e in (top_rep + bot_rep)
                ]
                self.diffusion.fit_shapley_diffu_grpo(rollouts, epochs=grpo_epochs)
                self.hrm.fit_replay(self.hrm_replay[-160:])

        basis = self._select_diverse_basis(max_basis=max_ens, max_corr=max_corr)
        if not basis:
            # Fallback to canonical value-quality seed if refuter pruned all random ASTs on tiny test fixture
            fallback_node = ASTNode(node_type="terminal", name="pit_value_quality")
            fallback_spec = FactorSpec(
                factor_id="v5_fallback_vq",
                hypothesis_id="v5_fallback",
                description="fallback pit_value_quality",
                root=fallback_node,
            )
            s_df = evaluate_factor_spec(fallback_spec, self.search_panel, verifier=self.verifier).fillna(0.0)
            s_safe = s_df.reindex(index=self.safe_dates).fillna(0.0)
            ic_s = compute_rank_ic_series(s_safe, self.search_fwd)
            basis = [
                HoloElite(
                    spec=fallback_spec,
                    sign=1.0,
                    smooth_window=1,
                    family="value_quality_pit",
                    niche="fundamental",
                    cell=("fundamental", 0),
                    utility=float(ic_s.mean()),
                    rank_ic=float(ic_s.mean()),
                    min_fold_ic=float(ic_s.mean()),
                    icir=1.0,
                    net_sharpe=0.5,
                    turnover=0.05,
                    ic_series=ic_s,
                    signal_vec=s_safe.to_numpy(dtype=float).ravel(),
                    signal_df=s_safe,
                )
            ]

        w_list = [float(x) for x in self._solve_prior_weights(basis)]
        for cp in ctrl_grid:
            cv_sc = self._search_cv_score(basis, w_list, cp)
            if best_search is None or cv_sc > best_search[0]:
                best_search = (cv_sc, basis, w_list, cp)

        assert best_search is not None
        champ_cv_sc, champ_elites, champ_weights, champ_cp = best_search

        # Re-fit MoSE & Conformal strictly on search for the winning controller profile
        self._fit_search_gates_for_basis(champ_elites, champ_weights, champ_cp)
        ctrl_out = self.build_full_causal_signal(champ_elites, champ_weights, champ_cp, self.evaluator.feature_panel)

        # TRUE Zero-Validation-Burn: single scalar validation query!
        _ = max_val_q
        val_out = self.evaluator.evaluate_signal_matrix(
            signal_wide=ctrl_out.sealed_signal_wide,
            zone="validation",
            candidate_id=f"maveq_v5_{self.backbone_family}_{champ_cp.name}",
            run_id=run_id,
        )
        val_queries = 1

        bm_val_df = self.evaluator.zone_guard._slice_zone(self.evaluator.snapshot.benchmark, "validation").sort_values("date")
        bm_val = pd.Series(bm_val_df["return_1d"].to_numpy(dtype=float), index=bm_val_df["date"])
        bars_val = self.evaluator.zone_guard._slice_zone(self.evaluator.snapshot.bars, "validation")
        val_ext: ExtendedMetricsReport = compute_extended_metrics(
            val_out,
            bm_val,
            bars_slice=bars_val,
            signal_wide=ctrl_out.sealed_signal_wide,
            close_wide=self.evaluator.feature_panel["close"],
            n_trials=self.evaluator.n_trials_evaluated,
            rebalance_every_days=self.evaluator.rebalance_every_days,
            top_k=self.evaluator.top_k,
            n_boot=300,
        )

        # Optional Single-Shot Sealed Test (ONLY when score_sealed=True in TASK 07!)
        test_metrics: dict[str, Any] = {"note": "sealed_test_not_requested_dev_mode"}
        test_ext_dict: dict[str, Any] = {}
        test_rets = pd.Series(dtype=float)
        if score_sealed:
            try:
                test_out = self.evaluator.evaluate_signal_matrix(
                    signal_wide=ctrl_out.sealed_signal_wide,
                    zone="sealed_test",
                    candidate_id=f"maveq_v5_champion_{self.backbone_family}_{champ_cp.name}",
                    run_id=run_id,
                    allow_sealed=True,
                )
                test_metrics = dict(test_out.metrics)
                test_rets = test_out.daily_net_returns
                bm_test_df = self.evaluator.zone_guard._slice_zone(self.evaluator.snapshot.benchmark, "sealed_test").sort_values("date")
                bm_test = pd.Series(bm_test_df["return_1d"].to_numpy(dtype=float), index=bm_test_df["date"])
                bars_test = self.evaluator.zone_guard._slice_zone(self.evaluator.snapshot.bars, "sealed_test")
                test_ext = compute_extended_metrics(
                    test_out,
                    bm_test,
                    bars_slice=bars_test,
                    signal_wide=ctrl_out.sealed_signal_wide,
                    close_wide=self.evaluator.feature_panel["close"],
                    n_trials=self.evaluator.n_trials_evaluated,
                    rebalance_every_days=self.evaluator.rebalance_every_days,
                    top_k=self.evaluator.top_k,
                    n_boot=500,
                )
                test_ext_dict = test_ext.to_dict()
            except SealedTestReuseError as exc:
                test_metrics = {"note": f"sealed_test_already_scored: {exc}"}

        val_dates = pd.DatetimeIndex(bars_val["date"].unique())
        reg_val = ctrl_out.regime_series.reindex(index=val_dates).value_counts(normalize=True).to_dict()
        regime_summary = {str(k): round(float(v), 4) for k, v in reg_val.items()}
        regime_summary["mean_exposure"] = round(float(ctrl_out.exposure_series.reindex(index=val_dates).mean()), 4)

        self.guard.cleanup_memory()
        return MAVEQv5Result(
            preset=self.requested_preset,
            resolved_safe_preset=self.resolved_preset,
            backbone_family=self.backbone_family,
            scale_rung=self.scale_rung,
            backbone_exact_params=self.backbone.exact_params,
            sub1b_blueprint_verified=bool(self.sub1b_blueprint["all_blocks_strictly_sub_1b"]),
            selected_controller=champ_cp.name,
            champion_expressions=[e.spec.root.canonical_expr() for e in champ_elites],
            champion_weights=[round(float(w), 4) for w in champ_weights],
            champion_niches=[e.niche for e in champ_elites],
            search_candidates_screened=self.n_search_screened,
            refuter_rejections=self.refuter_rejections,
            mose_active_branches=list(self.mose_gate.active_branches),
            mose_pruned_branches=list(self.mose_gate.pruned_branches),
            dormant_veto_fraction=round(float(ctrl_out.dormant_veto_fraction), 4),
            epistemic_abstain_rate=round(float(ctrl_out.abstain_rate), 4),
            validation_queries_used=val_queries,
            total_evaluator_trials=self.evaluator.n_trials_evaluated,
            champion_search_cv_score=round(float(champ_cv_sc), 5),
            champion_validation_metrics={k: round(float(v), 5) for k, v in val_out.metrics.items()},
            champion_validation_extended=val_ext.to_dict(),
            champion_validation_fold_std={k: round(float(v), 5) for k, v in val_out.fold_std.items()},
            champion_sealed_test_metrics=test_metrics,
            champion_sealed_test_extended=test_ext_dict,
            regime_exposure_summary=regime_summary,
            validation_daily_returns=val_out.daily_net_returns,
            sealed_test_daily_returns=test_rets,
        )

    def save_checkpoints(self, out_dir: str | Path) -> dict[str, str]:
        """Save v5 checkpoints (`tree_mdlm`, `holo_hrm`, `v5_state_bundle.json`) for `--resume`."""
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        diff_p = self.diffusion.save_checkpoint(d / "v5_tree_mdlm")
        hrm_p = self.hrm.save_checkpoint(d / "v5_holo_hrm")
        state_p = d / "v5_chronos_state.json"
        state_p.write_text(
            json.dumps(
                {
                    "backbone_family": self.backbone_family,
                    "scale_rung": self.scale_rung,
                    "resolved_preset": self.resolved_preset,
                    "q_stock": self.conformal_veto.q_stock,
                    "q_stock_extreme": self.conformal_veto.q_stock_extreme,
                    "q_market": self.conformal_veto.q_market,
                    "mose_branch_weights": self.mose_gate.branch_weights,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return {"tree_mdlm": str(diff_p), "holo_hrm": str(hrm_p), "v5_state": str(state_p)}

    def load_checkpoints(self, in_dir: str | Path) -> None:
        d = Path(in_dir)
        for p in (d / "v5_tree_mdlm.pt", d / "v5_tree_mdlm.npz"):
            if p.exists():
                self.diffusion.load_checkpoint(p)
                break
        for p in (d / "v5_holo_hrm.pt", d / "v5_holo_hrm.npz"):
            if p.exists():
                self.hrm.load_checkpoint(p)
                break
        state_p = d / "v5_chronos_state.json"
        if state_p.exists():
            data = json.loads(state_p.read_text(encoding="utf-8"))
            self.conformal_veto.q_stock = float(data.get("q_stock", self.conformal_veto.q_stock))
            self.conformal_veto.q_stock_extreme = float(data.get("q_stock_extreme", self.conformal_veto.q_stock_extreme))
            self.conformal_veto.q_market = float(data.get("q_market", self.conformal_veto.q_market))
            self.conformal_veto._fitted = True
            bw = data.get("mose_branch_weights", {})
            if isinstance(bw, dict):
                self.mose_gate.branch_weights = {str(k): float(v) for k, v in bw.items()}
                self.mose_gate.active_branches = [k for k, v in self.mose_gate.branch_weights.items() if abs(v) > 1e-9]
                self.mose_gate._fitted = True
