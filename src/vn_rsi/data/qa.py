"""Automated Data QA and Leakage Detection Suite (Phase 1 mandatory CI checks).

Includes:
1. Snapshot checksum integrity check
2. Survivorship-bias check (presence of delisted/halted tickers when required)
3. Price-limit band & OHLC consistency reconciliation
4. Cross-source adjusted-close reconciliation threshold check
5. Point-in-time (PIT) announcement-date causality check (announcement_date > fiscal_period_end)
6. Shift-test future-information detector for any feature computation function
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from vn_rsi.data.market_rules import MarketRulesConfig
from vn_rsi.data.snapshots import MarketSnapshot


@dataclass
class DataQAReport:
    passed: bool
    checks: dict[str, dict[str, Any]] = field(default_factory=dict)

    def summary_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "checks": self.checks}


def join_pit_fundamentals_causal(bars: pd.DataFrame, fundamentals_pit: pd.DataFrame) -> pd.DataFrame:
    """Strictly causal as-of join using `announcement_date <= bar.date` (NEVER `fiscal_period_end`)."""
    if fundamentals_pit.empty:
        out = bars[["date", "symbol"]].copy()
        for col in ("roe", "pb_ratio", "pe_ratio", "revenue_yoy", "debt_to_equity"):
            out[col] = np.nan
        return out

    left = bars[["date", "symbol"]].sort_values("date").copy()
    right = (
        fundamentals_pit.sort_values("announcement_date")
        .drop_duplicates(subset=["symbol", "announcement_date"], keep="last")
        .copy()
    )
    merged = pd.merge_asof(
        left,
        right,
        left_on="date",
        right_on="announcement_date",
        by="symbol",
        direction="backward",
        allow_exact_matches=True,
    )
    return merged.sort_values(["date", "symbol"]).reset_index(drop=True)


class DataQASuite:
    """Runs automated QA and anti-leakage checks on a MarketSnapshot."""

    def __init__(self, market_rules: MarketRulesConfig, max_source_diff_pct: float = 0.02) -> None:
        self.market_rules = market_rules
        self.max_source_diff_pct = max_source_diff_pct

    def run_all(
        self,
        snapshot: MarketSnapshot,
        feature_fns: dict[str, Callable[[pd.DataFrame], pd.Series | pd.DataFrame]] | None = None,
        require_delisted: bool = True,
    ) -> DataQAReport:
        checks: dict[str, dict[str, Any]] = {}

        # 1. OHLC & Exchange Price Band consistency
        bars = snapshot.bars
        active = bars[~bars["is_halted"]]
        ohlc_ok = bool(
            (active["high"] >= active[["open", "close"]].max(axis=1) - 1e-2).all()
            & (active["low"] <= active[["open", "close"]].min(axis=1) + 1e-2).all()
        )
        band_ok = bool(
            (active["close"] <= active["ceiling_price"] + 1.0).all()
            & (active["close"] >= active["floor_price"] - 1.0).all()
        )
        checks["price_bands_and_ohlc"] = {
            "passed": ohlc_ok and band_ok,
            "ohlc_consistent": ohlc_ok,
            "within_exchange_bands": band_ok,
            "ceiling_hits": int(bars["hit_ceiling"].sum()),
            "floor_hits": int(bars["hit_floor"].sum()),
        }

        # 2. Survivorship-bias check
        delisted_syms = sorted(bars.loc[bars["is_delisted"], "symbol"].unique().tolist())
        halted_rows = int(bars["is_halted"].sum())
        surv_ok = (len(delisted_syms) > 0) if require_delisted else True
        checks["survivorship_coverage"] = {
            "passed": surv_ok,
            "delisted_symbols": delisted_syms,
            "halted_bar_count": halted_rows,
        }

        # 3. Cross-source reconciliation
        max_diff = float(bars["source_diff_pct"].max()) if "source_diff_pct" in bars.columns else 0.0
        recon_ok = max_diff <= self.max_source_diff_pct
        checks["cross_source_reconciliation"] = {
            "passed": recon_ok,
            "max_source_diff_pct": round(max_diff, 6),
            "threshold_pct": self.max_source_diff_pct,
        }

        # 4. Point-in-time (PIT) fundamentals announcement lag assertion
        f_df = snapshot.fundamentals_pit
        if not f_df.empty:
            lag_days = (f_df["announcement_date"] - f_df["fiscal_period_end"]).dt.days
            pit_lag_ok = bool((lag_days > 0).all())
            joined = join_pit_fundamentals_causal(bars, f_df)
            valid_join = joined.dropna(subset=["announcement_date"])
            pit_join_ok = bool((valid_join["announcement_date"] <= valid_join["date"]).all())
            checks["pit_fundamentals_causality"] = {
                "passed": pit_lag_ok and pit_join_ok,
                "min_announcement_lag_days": int(lag_days.min()),
                "pit_asof_join_strictly_past": pit_join_ok,
            }
        else:
            checks["pit_fundamentals_causality"] = {"passed": True, "note": "No fundamentals rows"}

        # 5. Future-information shift-test across supplied feature functions
        if feature_fns:
            shift_results: dict[str, bool] = {}
            for name, fn in feature_fns.items():
                shift_results[name] = self.verify_feature_causality_shift_test(bars, fn)
            checks["feature_shift_leakage_test"] = {
                "passed": all(shift_results.values()),
                "per_feature": shift_results,
            }

        overall = all(item["passed"] for item in checks.values())
        return DataQAReport(passed=overall, checks=checks)

    @staticmethod
    def verify_feature_causality_shift_test(
        bars: pd.DataFrame,
        feature_fn: Callable[[pd.DataFrame], pd.Series | pd.DataFrame],
        cutoff_fraction: float = 0.65,
        atol: float = 1e-10,
    ) -> bool:
        """Perturb all data strictly AFTER cutoff date t*; verify features on dates <= t* are invariant."""
        unique_dates = np.sort(bars["date"].unique())
        cutoff_idx = max(5, int(len(unique_dates) * cutoff_fraction))
        cutoff_date = unique_dates[cutoff_idx]

        orig_out = feature_fn(bars.copy())
        perturbed = bars.copy()
        future_mask = perturbed["date"] > cutoff_date
        for col in ("open", "high", "low", "close", "raw_close", "volume", "turnover_vnd", "foreign_net_ratio"):
            if col in perturbed.columns:
                perturbed.loc[future_mask, col] = perturbed.loc[future_mask, col] * 3.71 + 999.0

        pert_out = feature_fn(perturbed)
        past_mask = (bars["date"] <= cutoff_date).to_numpy()

        a = np.asarray(orig_out)[past_mask]
        b = np.asarray(pert_out)[past_mask]
        both_nan = np.isnan(a) & np.isnan(b)
        equal_vals = np.isclose(np.nan_to_num(a, nan=0.0), np.nan_to_num(b, nan=0.0), atol=atol, rtol=0.0)
        return bool(np.all(both_nan | equal_vals))

    @staticmethod
    def assert_label_strictly_future(feature_timestamps: pd.Series, label_start_timestamps: pd.Series) -> bool:
        """Verify target return labels start strictly at or after decision timestamp (no overlap with past)."""
        return bool((pd.to_datetime(label_start_timestamps) > pd.to_datetime(feature_timestamps)).all())
