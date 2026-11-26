"""DataProvider abstraction with multi-source vnstock adapter and realistic VN market simulation fixture.

Hard prohibitions respected:
- No scraping bypassing CAPTCHAs, authentication, rate limits, or robots.txt.
- Rate-limited public requests only when live_vnstock is explicitly selected; experiments read snapshots only.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from vn_rsi.data.market_rules import MarketRulesConfig


@dataclass
class RawMarketBundle:
    bars: pd.DataFrame
    fundamentals_pit: pd.DataFrame
    index_membership: pd.DataFrame
    benchmark: pd.DataFrame
    reconciliation_summary: dict[str, Any]
    intraday_coverage_report: dict[str, Any]
    provider_name: str
    is_synthetic: bool


class DataProvider(ABC):
    """Abstract interface for Vietnamese equity data acquisition (used ONLY at snapshot build time)."""

    @abstractmethod
    def fetch_bundle(
        self,
        universe_tier: str,
        start_date: str,
        end_date: str,
        market_rules: MarketRulesConfig,
        seed: int = 20261123,
    ) -> RawMarketBundle:
        """Fetch OHLCV + corporate action adjustments, PIT fundamentals, index membership, and benchmark."""


VN30_CORE_SPECS: list[tuple[str, str, str, bool]] = [
    ("VCB", "HOSE", "Banking", True),
    ("BID", "HOSE", "Banking", True),
    ("CTG", "HOSE", "Banking", True),
    ("TCB", "HOSE", "Banking", True),
    ("MBB", "HOSE", "Banking", True),
    ("VPB", "HOSE", "Banking", True),
    ("ACB", "HOSE", "Banking", True),
    ("STB", "HOSE", "Banking", True),
    ("HDB", "HOSE", "Banking", True),
    ("TPB", "HOSE", "Banking", True),
    ("FPT", "HOSE", "Technology", False),
    ("HPG", "HOSE", "Materials", False),
    ("VNM", "HOSE", "Consumer", False),
    ("MSN", "HOSE", "Consumer", False),
    ("MWG", "HOSE", "Retail", False),
    ("VIC", "HOSE", "RealEstate", False),
    ("VHM", "HOSE", "RealEstate", False),
    ("VRE", "HOSE", "RealEstate", False),
    ("GAS", "HOSE", "Energy", False),
    ("PLX", "HOSE", "Energy", False),
    ("POW", "HOSE", "Utilities", False),
    ("SSI", "HOSE", "Financials", False),
    ("VJC", "HOSE", "Industrials", False),
    ("SAB", "HOSE", "Consumer", False),
    ("GVR", "HOSE", "Materials", False),
    ("BVH", "HOSE", "Insurance", False),
    ("VIB", "HOSE", "Banking", True),
    # 3 delisted / halted historical names included in universe to prevent survivorship bias!
    ("ROS_DELIST", "HOSE", "RealEstate", False),
    ("FLC_HALT", "HOSE", "RealEstate", False),
    ("NVL_STRESS", "HOSE", "RealEstate", False),
]


def _generate_universe_specs(universe_tier: str) -> list[tuple[str, str, str, bool]]:
    base = list(VN30_CORE_SPECS)
    if universe_tier == "vn30":
        return base
    target = 100 if universe_tier == "vn100" else 220
    sectors = ["Industrials", "Materials", "Consumer", "Financials", "RealEstate", "Energy", "Utilities", "Banking"]
    exchanges = ["HOSE", "HOSE", "HNX", "UPCOM"]
    idx = 1
    while len(base) < target:
        ex = exchanges[idx % len(exchanges)]
        sec = sectors[idx % len(sectors)]
        is_bank = sec == "Banking"
        sym = f"VN{ex[0]}{idx:03d}"
        base.append((sym, ex, sec, is_bank))
        idx += 1
    return base[:target]


class SyntheticVNMarketProvider(DataProvider):
    """Deterministic multi-exchange Vietnamese equity fixture with realistic microstructure & traps.

    Explicitly sets `is_synthetic=True` so generated snapshots are transparently labeled in logs and paper.
    Includes:
    - HOSE (+/-7%), HNX (+/-10%), UPCoM (+/-15%) daily bands + ceiling/floor lock days
    - Corporate action splits/stock-dividends (`raw_close` vs `adj_close`) reconciled across 2 simulated feeds
    - Mid-sample delistings & trading halts (`ROS_DELIST`, `FLC_HALT`) for survivorship-bias testing
    - Point-in-time quarterly fundamentals with realistic announcement lags (25-45 calendar days after quarter-end)
    - Time-varying index membership (`as_of_date`)
    - Realistic weak cross-sectional signals (Rank IC ~ 0.02-0.05) with regime shift across search/val/test
    """

    def fetch_bundle(
        self,
        universe_tier: str,
        start_date: str,
        end_date: str,
        market_rules: MarketRulesConfig,
        seed: int = 20261123,
    ) -> RawMarketBundle:
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range(start=start_date, end=end_date, freq="B")
        n_days = len(dates)
        specs = _generate_universe_specs(universe_tier)
        n_syms = len(specs)

        # Market factor (VN-Index daily return with GARCH-like volatility clustering)
        mkt_vol = np.full(n_days, 0.011)
        mkt_ret = np.zeros(n_days)
        for t in range(1, n_days):
            mkt_vol[t] = np.clip(0.88 * mkt_vol[t - 1] + 0.12 * abs(mkt_ret[t - 1]) + 0.0012, 0.007, 0.032)
            mkt_ret[t] = rng.normal(0.00035, mkt_vol[t])

        bm_close = 900.0 * np.exp(np.cumsum(mkt_ret))
        benchmark_df = pd.DataFrame(
            {
                "date": dates,
                "symbol": "VNINDEX",
                "close": bm_close,
                "return_1d": mkt_ret,
            }
        )

        # Quarterly fundamental schedule & PIT announcement dates
        quarter_ends = pd.date_range(start="2018-09-30", end=end_date, freq="QE")
        fund_rows: list[dict[str, Any]] = []
        pit_quality_map: dict[str, pd.Series] = {}

        for s_idx, (sym, ex, sector, is_bank) in enumerate(specs):
            base_roe = rng.uniform(0.07, 0.24)
            base_pb = rng.uniform(0.9, 3.4)
            sym_fund_records = []
            for qe in quarter_ends:
                roe_q = float(np.clip(base_roe + rng.normal(0.0, 0.025), -0.15, 0.40))
                pb_q = float(np.clip(base_pb + rng.normal(0.0, 0.25), 0.4, 8.0))
                rev_yoy = float(np.clip(rng.normal(0.11, 0.14), -0.45, 0.85))
                de_ratio = float(np.clip(rng.uniform(0.3, 2.5) + (3.5 if is_bank else 0.0), 0.1, 10.0))
                ann_lag_days = int(rng.integers(24, 46))
                ann_date = qe + pd.Timedelta(days=ann_lag_days)
                rec = {
                    "symbol": sym,
                    "fiscal_period_end": qe.normalize(),
                    "announcement_date": ann_date.normalize(),
                    "roe": round(roe_q, 5),
                    "pb_ratio": round(pb_q, 4),
                    "pe_ratio": round(float(np.clip(pb_q / max(roe_q, 0.03), 3.5, 45.0)), 4),
                    "revenue_yoy": round(rev_yoy, 5),
                    "debt_to_equity": round(de_ratio, 4),
                }
                fund_rows.append(rec)
                sym_fund_records.append((ann_date.normalize(), roe_q / pb_q + 0.25 * rev_yoy))

            # Map PIT fundamental quality strictly after announcement_date
            pit_vals = np.zeros(n_days)
            for t, dt in enumerate(dates):
                avail = [val for (ad, val) in sym_fund_records if ad <= dt]
                pit_vals[t] = avail[-1] if avail else 0.05
            pit_quality_map[sym] = pd.Series(pit_vals, index=dates)

        fundamentals_df = pd.DataFrame(fund_rows)

        # Simulate cross-sectional daily bars with realistic VN microstructure & subtle alpha drivers
        betas = rng.uniform(0.65, 1.35, size=n_syms)
        adv_scales = np.array(
            [
                rng.uniform(80e9, 350e9)
                if s[1] == "HOSE" and i < 30
                else (rng.uniform(12e9, 70e9) if s[1] != "UPCOM" else rng.uniform(2e9, 18e9))
                for i, s in enumerate(specs)
            ]
        )

        # State matrices (n_days, n_syms)
        adj_close = np.zeros((n_days, n_syms))
        raw_close = np.zeros((n_days, n_syms))
        adj_close_src_b = np.zeros((n_days, n_syms))
        open_px = np.zeros((n_days, n_syms))
        high_px = np.zeros((n_days, n_syms))
        low_px = np.zeros((n_days, n_syms))
        volume = np.zeros((n_days, n_syms))
        foreign_net_ratio = np.zeros((n_days, n_syms))
        fol_room_pct = np.zeros((n_days, n_syms))
        is_halted = np.zeros((n_days, n_syms), dtype=bool)
        is_delisted = np.zeros((n_days, n_syms), dtype=bool)
        hit_ceiling = np.zeros((n_days, n_syms), dtype=bool)
        hit_floor = np.zeros((n_days, n_syms), dtype=bool)

        init_prices = rng.uniform(14_000.0, 85_000.0, size=n_syms)
        adj_close[0, :] = init_prices
        split_factors = np.ones((n_days, n_syms))

        # Schedule corporate action stock dividends (e.g., 20% stock dividend) on 2 dates
        for s_idx in range(n_syms):
            if s_idx % 3 == 0 and n_days > 500:
                split_day = 320 + (s_idx * 7) % 300
                if split_day < n_days:
                    split_factors[:split_day, s_idx] *= 1.20  # Raw price before split was 1.2x higher

        pit_matrix = np.column_stack([pit_quality_map[s[0]].to_numpy() for s in specs])
        pit_cs_z = (pit_matrix - pit_matrix.mean(axis=1, keepdims=True)) / (
            pit_matrix.std(axis=1, keepdims=True) + 1e-8
        )

        # Delisting / halt dates for realistic survivorship testing
        delist_day_map: dict[int, int] = {}
        halt_window_map: dict[int, tuple[int, int]] = {}
        for s_idx, (sym, _, _, _) in enumerate(specs):
            if sym == "ROS_DELIST":
                delist_day_map[s_idx] = int(n_days * 0.52)  # Delists in 2022
            elif sym == "FLC_HALT":
                halt_window_map[s_idx] = (int(n_days * 0.48), int(n_days * 0.56))
                delist_day_map[s_idx] = int(n_days * 0.56)

        for s_idx, (_, _, _, is_bank) in enumerate(specs):
            fol_cap = (
                market_rules.foreign_ownership.banking_fol_limit_pct
                if is_bank
                else market_rules.foreign_ownership.default_fol_limit_pct
            )
            fol_room_pct[0, s_idx] = rng.uniform(0.02, fol_cap * 0.8)

        daily_rets = np.zeros((n_days, n_syms))

        for t in range(1, n_days):
            dt = dates[t]
            year = dt.year
            # Subtle regime shift: pure short-term momentum weakens slightly post-2023 while liquidity-confirmed
            # medium momentum + foreign flow + PIT value-quality + post-spike T+3 reversal stays informative.
            mom_w = 0.0011 if year <= 2022 else 0.00075
            rev_w = -0.0010 if year <= 2022 else -0.00115
            flow_w = 0.00095
            pit_w = 0.00070

            # Strictly causal signals computed from <= t-1 to drive expected return on day t..t+5
            if t >= 21:
                past_15 = np.sum(daily_rets[t - 15 : t, :], axis=0)
                past_3 = np.sum(daily_rets[t - 3 : t, :], axis=0)
                vol_shk = volume[t - 1, :] / (np.mean(volume[t - 20 : t, :], axis=0) + 1.0)
                flow_5 = np.mean(foreign_net_ratio[t - 5 : t, :], axis=0)

                def _cs_z(arr: np.ndarray) -> np.ndarray:
                    return (arr - np.mean(arr)) / (np.std(arr) + 1e-8)

                liq_confirmed_mom = _cs_z(past_15 / (np.std(daily_rets[t - 15 : t, :], axis=0) + 0.008))
                overheat_rev = _cs_z(past_3 * np.clip(vol_shk, 0.5, 2.5))
                flow_sig = _cs_z(flow_5)
                alpha_drift = (
                    mom_w * liq_confirmed_mom
                    + rev_w * overheat_rev
                    + flow_w * flow_sig
                    + pit_w * pit_cs_z[t - 1, :]
                )
            else:
                alpha_drift = np.zeros(n_syms)

            # Foreign net flow at t (persistent AR(1) + innovation)
            foreign_net_ratio[t, :] = np.clip(
                0.55 * foreign_net_ratio[t - 1, :] + rng.normal(0.0, 0.06, size=n_syms),
                -0.30,
                0.30,
            )

            for s_idx, (sym, ex, _, is_bank) in enumerate(specs):
                if s_idx in delist_day_map and t >= delist_day_map[s_idx]:
                    is_delisted[t, s_idx] = True
                    is_halted[t, s_idx] = True
                    adj_close[t, s_idx] = adj_close[t - 1, s_idx]
                    volume[t, s_idx] = 0.0
                    continue
                if s_idx in halt_window_map and halt_window_map[s_idx][0] <= t < halt_window_map[s_idx][1]:
                    is_halted[t, s_idx] = True
                    adj_close[t, s_idx] = adj_close[t - 1, s_idx]
                    volume[t, s_idx] = 0.0
                    continue

                band = market_rules.get_exchange_limit(ex)
                idio_sigma = 0.018 if ex == "HOSE" else (0.024 if ex == "HNX" else 0.032)
                distress_drag = -0.0045 if sym in {"ROS_DELIST", "FLC_HALT"} else 0.0
                raw_r = betas[s_idx] * mkt_ret[t] + alpha_drift[s_idx] + distress_drag + rng.normal(0.0, idio_sigma)

                # Enforce exchange daily price band [ -band, +band ]
                clipped_r = float(np.clip(raw_r, -band, band))
                if clipped_r >= band - 1e-5:
                    hit_ceiling[t, s_idx] = True
                elif clipped_r <= -band + 1e-5:
                    hit_floor[t, s_idx] = True

                daily_rets[t, s_idx] = clipped_r
                adj_close[t, s_idx] = max(1_000.0, adj_close[t - 1, s_idx] * (1.0 + clipped_r))

                fol_cap = (
                    market_rules.foreign_ownership.banking_fol_limit_pct
                    if is_bank
                    else market_rules.foreign_ownership.default_fol_limit_pct
                )
                fol_room_pct[t, s_idx] = float(
                    np.clip(fol_room_pct[t - 1, s_idx] - 0.01 * foreign_net_ratio[t, s_idx], 0.0, fol_cap)
                )

                turnover_mult = float(np.exp(rng.normal(0.0, 0.38)) * (1.0 + 12.0 * abs(clipped_r)))
                notional = adv_scales[s_idx] * turnover_mult
                sh = max(100.0, round(notional / adj_close[t, s_idx] / 100.0) * 100.0)
                volume[t, s_idx] = sh

        # Construct OHLC, raw_close, and cross-source reconciliation feed
        for s_idx, (_, ex, _, _) in enumerate(specs):
            band = market_rules.get_exchange_limit(ex)
            volume[0, s_idx] = round(adv_scales[s_idx] / adj_close[0, s_idx] / 100.0) * 100.0
            raw_close[:, s_idx] = adj_close[:, s_idx] * split_factors[:, s_idx]
            # Source B has tiny rounding jitter (<0.05%) to exercise cross-source reconciliation
            adj_close_src_b[:, s_idx] = adj_close[:, s_idx] * (1.0 + rng.normal(0.0, 0.00015, size=n_days))

            prev_c = np.concatenate([[adj_close[0, s_idx]], adj_close[:-1, s_idx]])
            gap = np.clip(rng.normal(0.0, 0.004, size=n_days), -band * 0.5, band * 0.5)
            open_px[:, s_idx] = np.where(is_halted[:, s_idx], adj_close[:, s_idx], prev_c * (1.0 + gap))
            hi_wick = np.abs(rng.normal(0.0, 0.006, size=n_days))
            lo_wick = np.abs(rng.normal(0.0, 0.006, size=n_days))
            ceil_px = prev_c * (1.0 + band)
            flr_px = prev_c * (1.0 - band)
            high_px[:, s_idx] = np.where(
                is_halted[:, s_idx],
                adj_close[:, s_idx],
                np.minimum(ceil_px, np.maximum(open_px[:, s_idx], adj_close[:, s_idx]) * (1.0 + hi_wick)),
            )
            low_px[:, s_idx] = np.where(
                is_halted[:, s_idx],
                adj_close[:, s_idx],
                np.maximum(flr_px, np.minimum(open_px[:, s_idx], adj_close[:, s_idx]) * (1.0 - lo_wick)),
            )

        bar_frames: list[pd.DataFrame] = []
        for s_idx, (sym, ex, sector, is_bank) in enumerate(specs):
            prev_adj = np.concatenate([[adj_close[0, s_idx]], adj_close[:-1, s_idx]])
            band = market_rules.get_exchange_limit(ex)
            df_s = pd.DataFrame(
                {
                    "date": dates,
                    "symbol": sym,
                    "exchange": ex,
                    "sector": sector,
                    "is_bank": is_bank,
                    "open": np.round(open_px[:, s_idx], 2),
                    "high": np.round(high_px[:, s_idx], 2),
                    "low": np.round(low_px[:, s_idx], 2),
                    "close": np.round(adj_close[:, s_idx], 2),
                    "raw_close": np.round(raw_close[:, s_idx], 2),
                    "adj_factor": np.round(split_factors[:, s_idx], 6),
                    "ref_price": np.round(prev_adj, 2),
                    "ceiling_price": np.round(prev_adj * (1.0 + band), 2),
                    "floor_price": np.round(prev_adj * (1.0 - band), 2),
                    "volume": volume[:, s_idx],
                    "turnover_vnd": np.round(adj_close[:, s_idx] * volume[:, s_idx], 0),
                    "foreign_net_ratio": np.round(foreign_net_ratio[:, s_idx], 5),
                    "fol_room_pct": np.round(fol_room_pct[:, s_idx], 5),
                    "hit_ceiling": hit_ceiling[:, s_idx],
                    "hit_floor": hit_floor[:, s_idx],
                    "is_halted": is_halted[:, s_idx],
                    "is_delisted": is_delisted[:, s_idx],
                    "source_diff_pct": np.round(
                        np.abs(adj_close[:, s_idx] - adj_close_src_b[:, s_idx]) / adj_close[:, s_idx], 6
                    ),
                }
            )
            bar_frames.append(df_s)

        bars_df = pd.concat(bar_frames, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)

        # Semi-annual point-in-time index membership (constituents ranked by trailing 60d liquidity while listed)
        rebal_dates = pd.date_range(start=start_date, end=end_date, freq="6MS")
        idx_rows: list[dict[str, Any]] = []
        for rd in rebal_dates:
            hist_slice = bars_df[(bars_df["date"] <= rd) & (~bars_df["is_delisted"])]
            if hist_slice.empty:
                ranked_syms = [s[0] for s in specs if not s[0].endswith("_DELIST")]
            else:
                recent = hist_slice[hist_slice["date"] >= (rd - pd.Timedelta(days=90))]
                liq = recent.groupby("symbol")["turnover_vnd"].mean().sort_values(ascending=False)
                ranked_syms = list(liq.index)
            top30 = set(ranked_syms[: min(30, len(ranked_syms))])
            top100 = set(ranked_syms[: min(100, len(ranked_syms))])
            for sym, ex, _, _ in specs:
                idx_rows.append(
                    {
                        "as_of_date": rd.normalize(),
                        "symbol": sym,
                        "exchange": ex,
                        "in_vn30": sym in top30 and ex == "HOSE",
                        "in_vn100": sym in top100 and ex == "HOSE",
                    }
                )
        index_membership_df = pd.DataFrame(idx_rows)

        reconciliation_summary = {
            "sources_compared": ["synthetic_feed_A", "synthetic_feed_B"],
            "max_adj_close_diff_pct": float(bars_df["source_diff_pct"].max()),
            "mean_adj_close_diff_pct": float(bars_df["source_diff_pct"].mean()),
            "corporate_action_splits_reconciled": int((bars_df["adj_factor"] > 1.0).sum()),
        }
        intraday_coverage_report = {
            "1D": {"available": True, "coverage_pct": 1.0, "note": "Full EOD history across universe"},
            "1H": {
                "available": False,
                "coverage_pct": 0.0,
                "note": "Free public endpoints typically retain <= 6-12 months of intraday 1H/15m bars; omitted from multi-year walk-forward to avoid sample truncation bias.",
            },
            "15m": {
                "available": False,
                "coverage_pct": 0.0,
                "note": "Limited free retention; documented honestly per Phase 1 spec.",
            },
        }
        return RawMarketBundle(
            bars=bars_df,
            fundamentals_pit=fundamentals_df,
            index_membership=index_membership_df,
            benchmark=benchmark_df,
            reconciliation_summary=reconciliation_summary,
            intraday_coverage_report=intraday_coverage_report,
            provider_name="SyntheticVNMarketProvider_v1",
            is_synthetic=True,
        )


class VnstockDataProvider(DataProvider):
    """Multi-source rate-limited adapter for `vnstock` (VCI + TCBS cross-check), with explicit fallback."""

    def __init__(self, sources: list[str] | None = None, rate_limit_seconds: float = 1.0) -> None:
        self.sources = sources or ["VCI", "TCBS"]
        self.rate_limit_seconds = max(0.5, rate_limit_seconds)

    def fetch_bundle(
        self,
        universe_tier: str,
        start_date: str,
        end_date: str,
        market_rules: MarketRulesConfig,
        seed: int = 20261123,
    ) -> RawMarketBundle:
        try:
            from vnstock import Vnstock  # type: ignore[import-not-found]
        except Exception as exc:
            raise RuntimeError(
                f"vnstock library is not installed ({exc}). Install optional dependency `pip install -e .[vnstock]` "
                "or use `provider_mode: auto` / `synthetic_fixture`."
            ) from exc

        symbols = [s[0] for s in _generate_universe_specs(universe_tier) if "_" not in s[0]][:30]
        frames: list[pd.DataFrame] = []
        diffs: list[float] = []

        for sym in symbols:
            src_dfs: dict[str, pd.DataFrame] = {}
            for src in self.sources[:2]:
                time.sleep(self.rate_limit_seconds)
                stock = Vnstock().stock(symbol=sym, source=src)
                hist = stock.quote.history(start=start_date, end=end_date, interval="1D")
                if hist is not None and not hist.empty:
                    hist = hist.rename(columns={"time": "date"})
                    hist["date"] = pd.to_datetime(hist["date"]).dt.normalize()
                    src_dfs[src] = hist.sort_values("date").drop_duplicates("date")
            if not src_dfs:
                continue
            primary_src = next(iter(src_dfs))
            df = src_dfs[primary_src].copy()
            if len(src_dfs) >= 2:
                keys = list(src_dfs.keys())
                merged = pd.merge(
                    src_dfs[keys[0]][["date", "close"]],
                    src_dfs[keys[1]][["date", "close"]],
                    on="date",
                    suffixes=("_a", "_b"),
                )
                if not merged.empty:
                    rel = (merged["close_a"] - merged["close_b"]).abs() / merged["close_a"].clip(lower=1.0)
                    diffs.append(float(rel.median()))
            band = market_rules.get_exchange_limit("HOSE")
            prev_c = df["close"].shift(1).fillna(df["close"])
            df["symbol"] = sym
            df["exchange"] = "HOSE"
            df["sector"] = "Listed"
            df["is_bank"] = sym in {"VCB", "BID", "CTG", "TCB", "MBB", "VPB", "ACB", "STB", "HDB", "TPB", "VIB"}
            df["raw_close"] = df["close"]
            df["adj_factor"] = 1.0
            df["ref_price"] = prev_c
            df["ceiling_price"] = prev_c * (1.0 + band)
            df["floor_price"] = prev_c * (1.0 - band)
            df["turnover_vnd"] = df["close"] * df["volume"]
            df["foreign_net_ratio"] = 0.0
            df["fol_room_pct"] = 0.25
            df["hit_ceiling"] = df["close"] >= df["ceiling_price"] * 0.998
            df["hit_floor"] = df["close"] <= df["floor_price"] * 1.002
            df["is_halted"] = df["volume"] <= 0
            df["is_delisted"] = False
            df["source_diff_pct"] = diffs[-1] if diffs else 0.0
            frames.append(df)

        if not frames:
            raise RuntimeError("No historical bars returned from vnstock endpoints.")

        bars_df = pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
        # Benchmark fallback via equal-weighted market proxy if VNINDEX endpoint unavailable
        bm = (
            bars_df.groupby("date")["close"]
            .mean()
            .reset_index()
            .rename(columns={"close": "close"})
            .sort_values("date")
        )
        bm["symbol"] = "VNINDEX"
        bm["return_1d"] = bm["close"].pct_change().fillna(0.0)

        fundamentals_df = pd.DataFrame(
            columns=[
                "symbol",
                "fiscal_period_end",
                "announcement_date",
                "roe",
                "pb_ratio",
                "pe_ratio",
                "revenue_yoy",
                "debt_to_equity",
            ]
        )
        index_df = pd.DataFrame(
            [
                {
                    "as_of_date": pd.Timestamp(start_date),
                    "symbol": sym,
                    "exchange": "HOSE",
                    "in_vn30": True,
                    "in_vn100": True,
                }
                for sym in bars_df["symbol"].unique()
            ]
        )
        return RawMarketBundle(
            bars=bars_df,
            fundamentals_pit=fundamentals_df,
            index_membership=index_df,
            benchmark=bm[["date", "symbol", "close", "return_1d"]],
            reconciliation_summary={
                "sources_compared": self.sources[:2],
                "median_cross_source_diff_pct": float(np.mean(diffs)) if diffs else 0.0,
            },
            intraday_coverage_report={
                "1D": {"available": True, "coverage_pct": 1.0},
                "1H": {"available": False, "note": "Not fetched in EOD snapshot mode; limited history on public APIs"},
            },
            provider_name="VnstockDataProvider",
            is_synthetic=False,
        )


def build_provider(mode: str, sources: list[str] | None = None, rate_limit_seconds: float = 1.0) -> DataProvider:
    if mode == "synthetic_fixture":
        return SyntheticVNMarketProvider()
    if mode == "live_vnstock":
        return VnstockDataProvider(sources=sources, rate_limit_seconds=rate_limit_seconds)
    if mode == "auto":
        try:
            import vnstock  # noqa: F401 # type: ignore[import-not-found]

            return VnstockDataProvider(sources=sources, rate_limit_seconds=rate_limit_seconds)
        except Exception:
            return SyntheticVNMarketProvider()
    raise ValueError(f"Unknown provider_mode '{mode}'")
