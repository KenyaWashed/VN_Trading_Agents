"""Human-Owned Sealed Evaluator with Tamper Verification, Walk-Forward Scoring, and Canary Checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vn_rsi.data.market_rules import MarketRulesConfig
from vn_rsi.data.snapshots import MarketSnapshot
from vn_rsi.dsl.ast_spec import ASTVerifier, FactorSpec, compute_feature_panel, evaluate_factor_spec
from vn_rsi.eval.metrics import compute_portfolio_metrics, compute_rank_ic_series
from vn_rsi.sandbox.seal import DataZoneGuard, EvaluatorSeal, SealedTestLedger
from vn_rsi.sandbox.simulator import SimulationResult, VNPortfolioSimulator
from vn_rsi.sandbox.splits import PurgedEmbargoWalkForwardSplitter
from vn_rsi.utils.repro import compute_config_hash


@dataclass
class EvaluationOutcome:
    candidate_id: str
    candidate_hash: str
    zone: str
    metrics: dict[str, float]
    fold_metrics: list[dict[str, float]] = field(default_factory=list)
    fold_std: dict[str, float] = field(default_factory=dict)
    daily_net_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    execution_stats: dict[str, int] = field(default_factory=dict)

    def scalar_view_for_agent(self) -> dict[str, Any]:
        """Restricted score-only summary safe to return to an outer/inner loop without leaking series."""
        return {
            "candidate_id": self.candidate_id,
            "zone": self.zone,
            "rank_ic": self.metrics["rank_ic"],
            "icir": self.metrics["icir"],
            "net_sharpe": self.metrics["net_sharpe"],
            "ann_net_return": self.metrics["ann_net_return"],
            "max_drawdown": self.metrics["max_drawdown"],
            "daily_turnover": self.metrics["daily_turnover"],
            "deflated_sharpe_prob": self.metrics["deflated_sharpe_prob"],
            "fold_net_sharpe_std": self.fold_std.get("net_sharpe", 0.0),
        }


class SealedEvaluator:
    """Central sealed evaluator owned exclusively by human engineering code."""

    def __init__(
        self,
        snapshot: MarketSnapshot,
        market_rules: MarketRulesConfig,
        sandbox_cfg: dict[str, Any],
        protected_paths: list[str | Path] | None = None,
    ) -> None:
        self.snapshot = snapshot
        self.market_rules = market_rules
        self.sandbox_cfg = sandbox_cfg
        self.horizon_days = int(sandbox_cfg.get("prediction_horizon_days", 5))
        self.top_k = int(sandbox_cfg.get("portfolio_top_k", 6))
        self.rebalance_every_days = int(sandbox_cfg.get("rebalance_every_days", 3))
        self.capital_vnd = float(sandbox_cfg.get("capital_vnd", 1_000_000_000.0))

        seal_targets = protected_paths or []
        self.seal = EvaluatorSeal(seal_targets) if seal_targets else None
        self.zone_guard = DataZoneGuard(sandbox_cfg["zones"])
        self.ledger = SealedTestLedger(sandbox_cfg.get("ledger_path", "logs/sealed_test_ledger.jsonl"))

        wf_cfg = sandbox_cfg.get("walk_forward", {})
        self.splitter = PurgedEmbargoWalkForwardSplitter(
            n_folds=int(wf_cfg.get("n_folds", 4)),
            purge_days=max(self.horizon_days + 1, int(wf_cfg.get("purge_days", 6))),
            embargo_days=int(wf_cfg.get("embargo_days", 5)),
        )
        self.simulator = VNPortfolioSimulator(
            market_rules=market_rules,
            top_k=self.top_k,
            rebalance_every_days=self.rebalance_every_days,
            capital_vnd=self.capital_vnd,
        )
        self.verifier = ASTVerifier(max_depth=5)

        self.feature_panel = compute_feature_panel(snapshot.bars, snapshot.fundamentals_pit)
        close_w = self.feature_panel["close"]
        # Strictly future H-day return from t to t+H (used ONLY inside human-owned evaluator for IC & ML targets)
        self.forward_return_wide = close_w.shift(-self.horizon_days) / close_w - 1.0
        self.all_dates = pd.DatetimeIndex(close_w.index)
        self.n_trials_evaluated = 0

    @property
    def seal_hash(self) -> str:
        if self.seal is None:
            return "unsealed_test_instance"
        return self.seal.verify_intact()

    def evaluate_factor(
        self,
        spec: FactorSpec,
        zone: str = "search",
        run_id: str = "agent_run",
        allow_sealed: bool = False,
    ) -> EvaluationOutcome:
        _ = self.seal_hash
        signal_wide = evaluate_factor_spec(spec, self.feature_panel, verifier=self.verifier)
        return self.evaluate_signal_matrix(
            signal_wide=signal_wide,
            zone=zone,
            candidate_id=spec.factor_id,
            candidate_hash_override=spec.spec_hash(),
            run_id=run_id,
            allow_sealed=allow_sealed,
        )

    def evaluate_factor_ensemble(
        self,
        specs: list[FactorSpec],
        weights: list[float] | None = None,
        zone: str = "validation",
        candidate_id: str = "factor_ensemble",
        run_id: str = "agent_run",
        allow_sealed: bool = False,
    ) -> EvaluationOutcome:
        _ = self.seal_hash
        if not specs:
            raise ValueError("Cannot evaluate empty FactorSpec list")
        w = weights or [1.0 / len(specs)] * len(specs)
        total_w = sum(abs(v) for v in w) or 1.0
        norm_w = [v / total_w for v in w]

        combined: pd.DataFrame | None = None
        for sp, wt in zip(specs, norm_w, strict=True):
            mat = evaluate_factor_spec(sp, self.feature_panel, verifier=self.verifier).fillna(0.0) * wt
            combined = mat if combined is None else (combined + mat)

        assert combined is not None
        ens_hash = compute_config_hash(
            {"exprs": [s.root.canonical_expr() for s in specs], "weights": [round(x, 5) for x in norm_w]}
        )[:16]
        return self.evaluate_signal_matrix(
            signal_wide=combined,
            zone=zone,
            candidate_id=candidate_id,
            candidate_hash_override=ens_hash,
            run_id=run_id,
            allow_sealed=allow_sealed,
        )

    def evaluate_signal_matrix(
        self,
        signal_wide: pd.DataFrame,
        zone: str,
        candidate_id: str,
        run_id: str = "eval",
        candidate_hash_override: str | None = None,
        allow_sealed: bool = False,
    ) -> EvaluationOutcome:
        seal_digest = self.seal_hash
        cand_hash = candidate_hash_override or compute_config_hash(
            {"candidate_id": candidate_id, "zone": zone, "mean": float(np.nanmean(signal_wide.to_numpy()))}
        )[:16]

        if zone == "sealed_test":
            if not allow_sealed:
                raise PermissionError(
                    "Refused `sealed_test` evaluation without explicit `allow_sealed=True` final-protocol flag."
                )
            self.ledger.assert_and_record_access(
                candidate_hash=cand_hash,
                candidate_label=candidate_id,
                run_id=run_id,
                evaluator_seal_hash=seal_digest,
            )

        self.n_trials_evaluated += 1
        bars_zone = self.zone_guard._slice_zone(self.snapshot.bars, zone)
        zone_dates = pd.DatetimeIndex(np.sort(bars_zone["date"].unique()))
        # Zero out forward return rows whose horizon extends past the end of this zone (prevents boundary bleed!)
        safe_cutoff_idx = max(0, len(zone_dates) - self.horizon_days)
        safe_ic_dates = zone_dates[:safe_cutoff_idx]

        sig_zone = signal_wide.reindex(index=zone_dates)
        sim: SimulationResult = self.simulator.simulate(sig_zone, bars_zone)
        ic_series = compute_rank_ic_series(
            sig_zone.reindex(index=safe_ic_dates),
            self.forward_return_wide.reindex(index=safe_ic_dates),
        )
        metrics = compute_portfolio_metrics(
            net_returns=sim.daily_net_returns,
            gross_returns=sim.daily_gross_returns,
            turnover=sim.daily_turnover,
            ic_series=ic_series,
            n_trials=self.n_trials_evaluated,
        )

        # Compute purged walk-forward fold variance inside the zone if sufficient dates exist
        fold_metrics: list[dict[str, float]] = []
        if len(zone_dates) >= 80:
            try:
                folds = self.splitter.split(zone_dates)
                for fold in folds:
                    f_bars = bars_zone[bars_zone["date"].isin(fold.eval_dates)]
                    f_sim = self.simulator.simulate(sig_zone.reindex(index=fold.eval_dates), f_bars)
                    f_ic = ic_series.reindex(fold.eval_dates).dropna()
                    fold_metrics.append(
                        compute_portfolio_metrics(
                            f_sim.daily_net_returns,
                            f_sim.daily_gross_returns,
                            f_sim.daily_turnover,
                            f_ic,
                            n_trials=self.n_trials_evaluated,
                        )
                    )
            except ValueError:
                fold_metrics = []

        fold_std: dict[str, float] = {}
        if len(fold_metrics) >= 2:
            for k in ("rank_ic", "net_sharpe", "ann_net_return"):
                fold_std[k] = round(float(np.std([fm[k] for fm in fold_metrics], ddof=1)), 4)

        return EvaluationOutcome(
            candidate_id=candidate_id,
            candidate_hash=cand_hash,
            zone=zone,
            metrics=metrics,
            fold_metrics=fold_metrics,
            fold_std=fold_std,
            daily_net_returns=sim.daily_net_returns,
            execution_stats={
                "blocked_ceiling_buys": sim.blocked_ceiling_buys,
                "blocked_floor_sells": sim.blocked_floor_sells,
                "t25_settlement_blocks": sim.t25_settlement_blocks,
            },
        )

    def evaluate_benchmark_buy_hold(self, zone: str = "validation") -> EvaluationOutcome:
        _ = self.seal_hash
        bm_zone = self.zone_guard._slice_zone(self.snapshot.benchmark, zone).sort_values("date")
        rets = pd.Series(bm_zone["return_1d"].to_numpy(dtype=float), index=bm_zone["date"])
        net = rets.copy()
        turn = pd.Series(0.0, index=bm_zone["date"])
        if len(net) >= 2:
            entry_bps = (
                self.market_rules.transaction_costs.broker_commission_bps
                + self.market_rules.illiquidity_impact.base_half_spread_bps.get("HOSE", 12.0)
            )
            exit_bps = entry_bps + self.market_rules.transaction_costs.sell_tax_bps
            net.iloc[0] -= entry_bps / 10_000.0
            net.iloc[-1] -= exit_bps / 10_000.0
            turn.iloc[0] = 1.0
            turn.iloc[-1] = 1.0
        metrics = compute_portfolio_metrics(net, rets, turn, ic_series=None, n_trials=1)
        return EvaluationOutcome(
            candidate_id="vnindex_buy_hold",
            candidate_hash="vnindex_bm",
            zone=zone,
            metrics=metrics,
            daily_net_returns=net,
        )

    def run_canary_checks(self, seed: int = 999) -> dict[str, Any]:
        """Anti-reward-hacking canary: shuffled random signal MUST produce near-zero Rank IC on validation."""
        seal_digest = self.seal_hash
        rng = np.random.default_rng(seed)
        close = self.feature_panel["close"]
        noise_sig = pd.DataFrame(
            rng.normal(0.0, 1.0, size=close.shape),
            index=close.index,
            columns=close.columns,
        )
        val_bars = self.zone_guard._slice_zone(self.snapshot.bars, "validation")
        val_dates = pd.DatetimeIndex(np.sort(val_bars["date"].unique()))
        safe_dates = val_dates[: max(0, len(val_dates) - self.horizon_days)]
        ic_null = compute_rank_ic_series(
            noise_sig.reindex(index=safe_dates),
            self.forward_return_wide.reindex(index=safe_dates),
        )
        mean_null_ic = float(ic_null.mean()) if len(ic_null) > 0 else 0.0
        max_allowed = float(self.sandbox_cfg.get("canary", {}).get("max_null_ic_abs", 0.045))
        passed = abs(mean_null_ic) <= max_allowed
        return {
            "passed": passed,
            "seal_hash": seal_digest,
            "null_signal_mean_rank_ic": round(mean_null_ic, 5),
            "max_allowed_abs_ic": max_allowed,
        }
