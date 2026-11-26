"""Multi-Market Data Loader & Market-Rule Builder for Cross-Market Generalization Benchmarks.

Supports three distinct equity market microstructures:
1. `VN_HOSE_T25`     — Vietnamese Equities (T+2.5 lock, +/-7%/10%/15% limits, 10bps sell tax, 100-share lot)
2. `US_LIQUID_T1`    — US Liquid Equities (T+1, no daily price bands, 0bps sell tax, 2bps comm, 1-share lot)
3. `CSI_EMERGING_T1` — Emerging CSI-300-style Equities (T+1, +/-10% bands, 5bps sell stamp duty, 100-share lot)

Each market supports:
- `offline_fixture`: deterministic realistic microstructure generator (`is_synthetic=True`) for CI/reproducibility
- `live_yfinance` / `live_vnstock`: live empirical fetcher when optional libraries are installed
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vn_rsi.data.market_rules import (
    ExchangeRule,
    ForeignOwnershipRule,
    IlliquidityImpactRule,
    MarketRulesConfig,
    SettlementRule,
    ShortSellingRule,
    TransactionCostRule,
)
from vn_rsi.data.providers import RawMarketBundle, SyntheticVNMarketProvider
from vn_rsi.data.snapshots import MarketSnapshot, SnapshotManager

MARKET_PROFILES: tuple[str, ...] = ("VN_HOSE_T25", "US_LIQUID_T1", "CSI_EMERGING_T1")


def build_market_rules_for_profile(profile: str) -> MarketRulesConfig:
    """Construct validated `MarketRulesConfig` for VN, US Liquid, or CSI-style Emerging market."""
    prof = profile.upper()
    if prof == "VN_HOSE_T25":
        return MarketRulesConfig(
            version="2026.1-vn",
            currency="VND",
            exchanges={
                "HOSE": ExchangeRule(
                    name="Ho Chi Minh City Stock Exchange",
                    daily_price_limit_pct=0.07,
                    first_day_limit_pct=0.20,
                    lot_size=100,
                    source="SSC/HOSE",
                    verified=True,
                ),
                "HNX": ExchangeRule(
                    name="Hanoi Stock Exchange",
                    daily_price_limit_pct=0.10,
                    first_day_limit_pct=0.30,
                    lot_size=100,
                    source="SSC/HNX",
                    verified=True,
                ),
                "UPCOM": ExchangeRule(
                    name="UPCoM",
                    daily_price_limit_pct=0.15,
                    first_day_limit_pct=0.40,
                    lot_size=100,
                    source="SSC/UPCoM",
                    verified=True,
                ),
            },
            settlement=SettlementRule(
                cycle_days=2.5,
                min_holding_bars_eod=3,
                allow_intraday_roundtrip=False,
                source="VSDC T+2.5",
                verified=True,
            ),
            short_selling=ShortSellingRule(
                allow_cash_equity_short=False,
                allow_vn30_futures_hedge=False,
                source="SSC",
                verified=True,
            ),
            transaction_costs=TransactionCostRule(
                broker_commission_bps=15.0,
                sell_tax_bps=10.0,
                buy_tax_bps=0.0,
                source="Circular 111/2013/TT-BTC",
                verified=True,
            ),
            illiquidity_impact=IlliquidityImpactRule(
                model="sqrt_participation_plus_spread",
                base_half_spread_bps={"HOSE": 12.0, "HNX": 25.0, "UPCOM": 45.0},
                impact_coeff_bps=35.0,
                max_adv_participation=0.05,
                ceiling_buy_blocked=True,
                floor_sell_blocked=True,
                source="Conservative emerging-market sqrt impact",
                verified=False,
            ),
            foreign_ownership=ForeignOwnershipRule(
                default_fol_limit_pct=0.49,
                banking_fol_limit_pct=0.30,
                enforce_fol_room_check=True,
                source="Decree 155/2020/ND-CP",
                verified=True,
            ),
        )

    if prof == "US_LIQUID_T1":
        return MarketRulesConfig(
            version="2026.1-us",
            currency="USD",
            exchanges={
                "NYSE": ExchangeRule(
                    name="New York Stock Exchange / NASDAQ",
                    daily_price_limit_pct=0.50,
                    first_day_limit_pct=0.99,
                    lot_size=1,
                    source="SEC T+1 Rule (May 2024)",
                    verified=True,
                ),
                "HOSE": ExchangeRule(
                    name="US Fallback Alias",
                    daily_price_limit_pct=0.50,
                    first_day_limit_pct=0.99,
                    lot_size=1,
                    source="SEC T+1 Rule",
                    verified=True,
                ),
            },
            settlement=SettlementRule(
                cycle_days=1.0,
                min_holding_bars_eod=1,
                allow_intraday_roundtrip=True,
                source="SEC T+1 Settlement",
                verified=True,
            ),
            short_selling=ShortSellingRule(
                allow_cash_equity_short=False,
                allow_vn30_futures_hedge=False,
                source="Long-only cash benchmark parity",
                verified=True,
            ),
            transaction_costs=TransactionCostRule(
                broker_commission_bps=2.0,
                sell_tax_bps=0.0,
                buy_tax_bps=0.0,
                source="Institutional US equity execution",
                verified=True,
            ),
            illiquidity_impact=IlliquidityImpactRule(
                model="sqrt_participation_plus_spread",
                base_half_spread_bps={"NYSE": 3.0, "HOSE": 3.0},
                impact_coeff_bps=12.0,
                max_adv_participation=0.05,
                ceiling_buy_blocked=False,
                floor_sell_blocked=False,
                source="Liquid US large-cap impact calibration",
                verified=False,
            ),
            foreign_ownership=ForeignOwnershipRule(
                default_fol_limit_pct=1.0,
                banking_fol_limit_pct=1.0,
                enforce_fol_room_check=False,
                source="Open US capital market",
                verified=True,
            ),
        )

    if prof == "CSI_EMERGING_T1":
        return MarketRulesConfig(
            version="2026.1-csi",
            currency="CNY",
            exchanges={
                "SSE": ExchangeRule(
                    name="CSI-Style Main Board (+/-10%)",
                    daily_price_limit_pct=0.10,
                    first_day_limit_pct=0.44,
                    lot_size=100,
                    source="CSRC/SSE T+1 +/-10% Trading Rules",
                    verified=True,
                ),
                "HOSE": ExchangeRule(
                    name="CSI Fallback Alias",
                    daily_price_limit_pct=0.10,
                    first_day_limit_pct=0.44,
                    lot_size=100,
                    source="CSRC/SSE",
                    verified=True,
                ),
            },
            settlement=SettlementRule(
                cycle_days=1.0,
                min_holding_bars_eod=1,
                allow_intraday_roundtrip=False,
                source="A-share T+1 Settlement",
                verified=True,
            ),
            short_selling=ShortSellingRule(
                allow_cash_equity_short=False,
                allow_vn30_futures_hedge=False,
                source="Long-only benchmark parity",
                verified=True,
            ),
            transaction_costs=TransactionCostRule(
                broker_commission_bps=5.0,
                sell_tax_bps=5.0,
                buy_tax_bps=0.0,
                source="Stamp duty 0.05% on sales (since Aug 2023)",
                verified=True,
            ),
            illiquidity_impact=IlliquidityImpactRule(
                model="sqrt_participation_plus_spread",
                base_half_spread_bps={"SSE": 8.0, "HOSE": 8.0},
                impact_coeff_bps=22.0,
                max_adv_participation=0.05,
                ceiling_buy_blocked=True,
                floor_sell_blocked=True,
                source="Emerging A-share sqrt impact calibration",
                verified=False,
            ),
            foreign_ownership=ForeignOwnershipRule(
                default_fol_limit_pct=0.30,
                banking_fol_limit_pct=0.30,
                enforce_fol_room_check=True,
                source="QFII/Stock Connect aggregate cap",
                verified=True,
            ),
        )

    raise ValueError(f"Unknown market profile '{profile}'. Valid: {MARKET_PROFILES}")


US_LIQUID_TICKERS: list[tuple[str, str]] = [
    ("AAPL", "Technology"),
    ("MSFT", "Technology"),
    ("NVDA", "Technology"),
    ("AMZN", "Consumer"),
    ("GOOGL", "Technology"),
    ("META", "Technology"),
    ("JPM", "Financials"),
    ("BAC", "Financials"),
    ("XOM", "Energy"),
    ("CVX", "Energy"),
    ("UNH", "Healthcare"),
    ("JNJ", "Healthcare"),
    ("PG", "Consumer"),
    ("HD", "Retail"),
    ("MA", "Financials"),
    ("V", "Financials"),
    ("AVGO", "Technology"),
    ("COST", "Retail"),
    ("MRK", "Healthcare"),
    ("CAT", "Industrials"),
    ("GE", "Industrials"),
    ("NFLX", "Consumer"),
    ("AMD", "Technology"),
    ("WMT", "Retail"),
    ("LLY", "Healthcare"),
    ("PEP", "Consumer"),
    ("KO", "Consumer"),
    ("DIS", "Consumer"),
    ("INTC_STRESS", "Technology"),
    ("SVB_DELIST", "Financials"),
]

CSI_EMERGING_TICKERS: list[tuple[str, str]] = [
    ("CSI_MOUTAI", "Consumer"),
    ("CSI_CATL", "Industrials"),
    ("CSI_PINGAN", "Financials"),
    ("CSI_CMB", "Banking"),
    ("CSI_BYD", "Consumer"),
    ("CSI_MIDEA", "Consumer"),
    ("CSI_ICBC", "Banking"),
    ("CSI_WULIANG", "Consumer"),
    ("CSI_ZIJIN", "Materials"),
    ("CSI_CNOOC", "Energy"),
    ("CSI_YANGTZE", "Utilities"),
    ("CSI_CITIC", "Financials"),
    ("CSI_EASTMON", "Financials"),
    ("CSI_LUXSH", "Technology"),
    ("CSI_SMIC", "Technology"),
    ("CSI_MINDR", "Healthcare"),
    ("CSI_HENG", "Healthcare"),
    ("CSI_LONGI", "Energy"),
    ("CSI_SHENHUA", "Energy"),
    ("CSI_BAOSTEEL", "Materials"),
    ("CSI_HAIER", "Consumer"),
    ("CSI_SANY", "Industrials"),
    ("CSI_COSCO", "Industrials"),
    ("CSI_GREE", "Consumer"),
    ("CSI_BOC", "Banking"),
    ("CSI_ABC", "Banking"),
    ("CSI_CCB", "Banking"),
    ("CSI_NARI", "Utilities"),
    ("CSI_DEV_STRESS", "RealEstate"),
    ("CSI_EVER_DELIST", "RealEstate"),
]


def _generate_synthetic_international_bundle(
    profile: str,
    start_date: str,
    end_date: str,
    rules: MarketRulesConfig,
    seed: int,
) -> RawMarketBundle:
    """Deterministic multi-regime international equity microstructure fixture (US_LIQUID_T1 or CSI_EMERGING_T1)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start_date, end=end_date, freq="B")
    n_days = len(dates)

    if profile == "US_LIQUID_T1":
        specs = US_LIQUID_TICKERS
        exchange = "NYSE"
        bm_sym = "SP500_PROXY"
        drift_base, base_vol, idio_sigma = 0.00042, 0.0105, 0.0165
        mom_coeff, rev_coeff, flow_coeff, pit_coeff = 0.00105, -0.00075, 0.00055, 0.00080
        px_lo, px_hi = 40.0, 420.0
        adv_lo, adv_hi = 1.5e8, 3.0e9
    else:
        specs = CSI_EMERGING_TICKERS
        exchange = "SSE"
        bm_sym = "CSI300_PROXY"
        drift_base, base_vol, idio_sigma = 0.00028, 0.0125, 0.0205
        mom_coeff, rev_coeff, flow_coeff, pit_coeff = 0.00080, -0.00115, 0.00090, 0.00075
        px_lo, px_hi = 8.0, 220.0
        adv_lo, adv_hi = 2.0e8, 2.5e9

    n_syms = len(specs)
    band = rules.get_exchange_limit(exchange)

    # Multi-regime market factor (includes realistic bull, drawdown/crash, and vol clustering)
    mkt_vol = np.full(n_days, base_vol)
    mkt_ret = np.zeros(n_days)
    for t in range(1, n_days):
        # Periodic macro stress pockets for authentic regime testing
        phase = t / max(1, n_days)
        regime_drift = drift_base
        if 0.28 <= phase <= 0.34 or 0.62 <= phase <= 0.68:
            regime_drift = -0.0022
        elif 0.72 <= phase <= 0.95:
            regime_drift = drift_base * 1.65
        mkt_vol[t] = np.clip(0.87 * mkt_vol[t - 1] + 0.13 * abs(mkt_ret[t - 1]) + 0.0010, 0.006, 0.036)
        mkt_ret[t] = rng.normal(regime_drift, mkt_vol[t])

    bm_close = 4000.0 * np.exp(np.cumsum(mkt_ret))
    benchmark_df = pd.DataFrame(
        {"date": dates, "symbol": bm_sym, "close": bm_close, "return_1d": mkt_ret}
    )

    quarter_ends = pd.date_range(start="2018-09-30", end=end_date, freq="QE")
    fund_rows: list[dict[str, Any]] = []
    pit_quality_map: dict[str, pd.Series] = {}
    for sym, _sec in specs:
        base_roe = rng.uniform(0.08, 0.28)
        base_pb = rng.uniform(1.1, 4.8)
        recs = []
        for qe in quarter_ends:
            roe_q = float(np.clip(base_roe + rng.normal(0.0, 0.025), -0.15, 0.45))
            pb_q = float(np.clip(base_pb + rng.normal(0.0, 0.30), 0.5, 10.0))
            rev_yoy = float(np.clip(rng.normal(0.10, 0.14), -0.40, 0.80))
            ann_date = qe + pd.Timedelta(days=int(rng.integers(22, 42)))
            fund_rows.append(
                {
                    "symbol": sym,
                    "fiscal_period_end": qe.normalize(),
                    "announcement_date": ann_date.normalize(),
                    "roe": round(roe_q, 5),
                    "pb_ratio": round(pb_q, 4),
                    "pe_ratio": round(float(np.clip(pb_q / max(roe_q, 0.03), 4.0, 50.0)), 4),
                    "revenue_yoy": round(rev_yoy, 5),
                    "debt_to_equity": round(float(rng.uniform(0.3, 3.0)), 4),
                }
            )
            recs.append((ann_date.normalize(), roe_q / pb_q + 0.25 * rev_yoy))
        vals = np.zeros(n_days)
        for t, dt in enumerate(dates):
            av = [v for ad, v in recs if ad <= dt]
            vals[t] = av[-1] if av else 0.06
        pit_quality_map[sym] = pd.Series(vals, index=dates)

    fundamentals_df = pd.DataFrame(fund_rows)
    pit_mat = np.column_stack([pit_quality_map[s[0]].to_numpy() for s in specs])
    pit_cs_z = (pit_mat - pit_mat.mean(axis=1, keepdims=True)) / (pit_mat.std(axis=1, keepdims=True) + 1e-8)

    betas = rng.uniform(0.70, 1.35, size=n_syms)
    adv_scales = rng.uniform(adv_lo, adv_hi, size=n_syms)
    close_px = np.zeros((n_days, n_syms))
    close_px[0, :] = rng.uniform(px_lo, px_hi, size=n_syms)
    volume = np.zeros((n_days, n_syms))
    volume[0, :] = np.maximum(100.0, np.round(adv_scales / close_px[0, :]))
    foreign_flow = np.zeros((n_days, n_syms))
    fol_room = np.full((n_days, n_syms), 0.35)
    is_halted = np.zeros((n_days, n_syms), dtype=bool)
    is_delisted = np.zeros((n_days, n_syms), dtype=bool)
    hit_ceil = np.zeros((n_days, n_syms), dtype=bool)
    hit_floor = np.zeros((n_days, n_syms), dtype=bool)
    daily_rets = np.zeros((n_days, n_syms))

    delist_idx = {i: int(n_days * 0.55) for i, (sym, _) in enumerate(specs) if "DELIST" in sym}

    for t in range(1, n_days):
        if t >= 21:
            p15 = np.sum(daily_rets[t - 15 : t, :], axis=0)
            p3 = np.sum(daily_rets[t - 3 : t, :], axis=0)
            vshk = volume[t - 1, :] / (np.mean(volume[t - 20 : t, :], axis=0) + 1.0)
            fl5 = np.mean(foreign_flow[t - 5 : t, :], axis=0)

            def _z(a: np.ndarray) -> np.ndarray:
                return (a - float(np.mean(a))) / (float(np.std(a)) + 1e-8)

            alpha_drift = (
                mom_coeff * _z(p15 / (np.std(daily_rets[t - 15 : t, :], axis=0) + 0.008))
                + rev_coeff * _z(p3 * np.clip(vshk, 0.5, 2.5))
                + flow_coeff * _z(fl5)
                + pit_coeff * pit_cs_z[t - 1, :]
            )
        else:
            alpha_drift = np.zeros(n_syms)

        foreign_flow[t, :] = np.clip(0.50 * foreign_flow[t - 1, :] + rng.normal(0.0, 0.05, size=n_syms), -0.3, 0.3)

        for i, (sym, _) in enumerate(specs):
            if i in delist_idx and t >= delist_idx[i]:
                is_delisted[t, i] = True
                is_halted[t, i] = True
                close_px[t, i] = close_px[t - 1, i]
                volume[t, i] = 0.0
                continue
            distress = -0.004 if "DELIST" in sym else 0.0
            raw_r = betas[i] * mkt_ret[t] + alpha_drift[i] + distress + rng.normal(0.0, idio_sigma)
            clipped_r = float(np.clip(raw_r, -band, band))
            if band <= 0.15:
                hit_ceil[t, i] = clipped_r >= band - 1e-5
                hit_floor[t, i] = clipped_r <= -band + 1e-5
            daily_rets[t, i] = clipped_r
            close_px[t, i] = max(1.0, close_px[t - 1, i] * (1.0 + clipped_r))
            vol_mult = float(np.exp(rng.normal(0.0, 0.32)) * (1.0 + 8.0 * abs(clipped_r)))
            volume[t, i] = max(100.0, round(adv_scales[i] * vol_mult / close_px[t, i]))

    frames: list[pd.DataFrame] = []
    for i, (sym, sec) in enumerate(specs):
        prev_c = np.concatenate([[close_px[0, i]], close_px[:-1, i]])
        gap = np.clip(rng.normal(0.0, 0.003, size=n_days), -0.03, 0.03)
        opn = np.where(is_halted[:, i], close_px[:, i], prev_c * (1.0 + gap))
        hi = np.where(is_halted[:, i], close_px[:, i], np.maximum(opn, close_px[:, i]) * (1.0 + np.abs(rng.normal(0.0, 0.005, size=n_days))))
        lo = np.where(is_halted[:, i], close_px[:, i], np.minimum(opn, close_px[:, i]) * (1.0 - np.abs(rng.normal(0.0, 0.005, size=n_days))))
        ceil_p = prev_c * (1.0 + band)
        flr_p = prev_c * (1.0 - band)
        hi = np.minimum(hi, ceil_p)
        lo = np.maximum(lo, flr_p)
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": sym,
                    "exchange": exchange,
                    "sector": sec,
                    "is_bank": sec == "Banking",
                    "open": np.round(opn, 3),
                    "high": np.round(hi, 3),
                    "low": np.round(lo, 3),
                    "close": np.round(close_px[:, i], 3),
                    "raw_close": np.round(close_px[:, i], 3),
                    "adj_factor": 1.0,
                    "ref_price": np.round(prev_c, 3),
                    "ceiling_price": np.round(ceil_p, 3),
                    "floor_price": np.round(flr_p, 3),
                    "volume": volume[:, i],
                    "turnover_vnd": np.round(close_px[:, i] * volume[:, i], 1),
                    "foreign_net_ratio": np.round(foreign_flow[:, i], 5),
                    "fol_room_pct": np.round(fol_room[:, i], 5),
                    "hit_ceiling": hit_ceil[:, i],
                    "hit_floor": hit_floor[:, i],
                    "is_halted": is_halted[:, i],
                    "is_delisted": is_delisted[:, i],
                    "source_diff_pct": 0.0,
                }
            )
        )

    bars_df = pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    idx_df = pd.DataFrame(
        [
            {"as_of_date": dates[0], "symbol": sym, "exchange": exchange, "in_vn30": True, "in_vn100": True}
            for sym, _ in specs
        ]
    )
    return RawMarketBundle(
        bars=bars_df,
        fundamentals_pit=fundamentals_df,
        index_membership=idx_df,
        benchmark=benchmark_df,
        reconciliation_summary={"profile": profile, "mode": "offline_fixture"},
        intraday_coverage_report={"1D": {"available": True, "coverage_pct": 1.0}},
        provider_name=f"MultiMarketFixture_{profile}",
        is_synthetic=True,
    )


def _fetch_live_yfinance_bundle(
    profile: str,
    start_date: str,
    end_date: str,
    rules: MarketRulesConfig,
) -> RawMarketBundle:
    """Optional live empirical loader using `yfinance` for US / international tickers."""
    try:
        import yfinance as yf  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError(
            f"yfinance not installed ({exc}). Install `pip install yfinance` or use mode='offline_fixture'."
        ) from exc

    if profile == "US_LIQUID_T1":
        tickers = [s[0] for s in US_LIQUID_TICKERS if "_" not in s[0]][:20]
        exchange, bm_ticker = "NYSE", "^GSPC"
    else:
        tickers = ["000001.SS", "600519.SS", "601318.SS", "600036.SS", "300750.SZ", "002594.SZ"]
        exchange, bm_ticker = "SSE", "000300.SS"

    band = rules.get_exchange_limit(exchange)
    frames: list[pd.DataFrame] = []
    for tk in tickers:
        hist = yf.download(tk, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if hist is None or hist.empty:
            continue
        hist = hist.reset_index()
        hist.columns = [c[0] if isinstance(c, tuple) else c for c in hist.columns]
        dt_col = "Date" if "Date" in hist.columns else hist.columns[0]
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(hist[dt_col]).dt.tz_localize(None).dt.normalize(),
                "symbol": tk,
                "exchange": exchange,
                "sector": "Listed",
                "is_bank": False,
                "open": hist["Open"].astype(float),
                "high": hist["High"].astype(float),
                "low": hist["Low"].astype(float),
                "close": hist["Close"].astype(float),
                "raw_close": hist["Close"].astype(float),
                "adj_factor": 1.0,
                "volume": hist["Volume"].astype(float).fillna(0.0),
            }
        )
        prev_c = df["close"].shift(1).fillna(df["close"])
        df["ref_price"] = prev_c
        df["ceiling_price"] = prev_c * (1.0 + band)
        df["floor_price"] = prev_c * (1.0 - band)
        df["turnover_vnd"] = df["close"] * df["volume"]
        df["foreign_net_ratio"] = 0.0
        df["fol_room_pct"] = 0.50
        df["hit_ceiling"] = (df["close"] >= df["ceiling_price"] * 0.998) if band <= 0.15 else False
        df["hit_floor"] = (df["close"] <= df["floor_price"] * 1.002) if band <= 0.15 else False
        df["is_halted"] = df["volume"] <= 0
        df["is_delisted"] = False
        df["source_diff_pct"] = 0.0
        frames.append(df)

    if not frames:
        raise RuntimeError(f"yfinance returned no data for profile={profile}")
    bars_df = pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)

    bm_hist = yf.download(bm_ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
    if bm_hist is not None and not bm_hist.empty:
        bm_hist = bm_hist.reset_index()
        bm_hist.columns = [c[0] if isinstance(c, tuple) else c for c in bm_hist.columns]
        dt_col = "Date" if "Date" in bm_hist.columns else bm_hist.columns[0]
        bm = pd.DataFrame(
            {
                "date": pd.to_datetime(bm_hist[dt_col]).dt.tz_localize(None).dt.normalize(),
                "symbol": bm_ticker,
                "close": bm_hist["Close"].astype(float),
            }
        ).sort_values("date")
    else:
        bm = bars_df.groupby("date")["close"].mean().reset_index().sort_values("date")
        bm["symbol"] = bm_ticker
    bm["return_1d"] = bm["close"].pct_change().fillna(0.0)

    fund_df = pd.DataFrame(
        columns=["symbol", "fiscal_period_end", "announcement_date", "roe", "pb_ratio", "pe_ratio", "revenue_yoy", "debt_to_equity"]
    )
    idx_df = pd.DataFrame(
        [
            {"as_of_date": bars_df["date"].min(), "symbol": s, "exchange": exchange, "in_vn30": True, "in_vn100": True}
            for s in bars_df["symbol"].unique()
        ]
    )
    return RawMarketBundle(
        bars=bars_df,
        fundamentals_pit=fund_df,
        index_membership=idx_df,
        benchmark=bm[["date", "symbol", "close", "return_1d"]],
        reconciliation_summary={"profile": profile, "mode": "live_yfinance"},
        intraday_coverage_report={"1D": {"available": True, "coverage_pct": 1.0}},
        provider_name=f"YFinanceProvider_{profile}",
        is_synthetic=False,
    )


@dataclass
class MultiMarketEnvironment:
    profile: str
    snapshot: MarketSnapshot
    market_rules: MarketRulesConfig
    is_synthetic: bool


class MultiMarketLoader:
    """Unified multi-market snapshot builder supporting VN30/VN100, US Liquid T+1, and CSI Emerging T+1."""

    def __init__(self, snapshot_root: str | Path = "data/snapshots") -> None:
        self.snapshot_mgr = SnapshotManager(snapshot_root)

    def load_or_build(
        self,
        profile: str,
        start_date: str = "2019-01-02",
        end_date: str = "2025-12-31",
        mode: str = "offline_fixture",
        universe_tier: str = "vn30",
        seed: int = 20261123,
        snapshot_id_override: str | None = None,
    ) -> MultiMarketEnvironment:
        prof = profile.upper()
        if prof not in MARKET_PROFILES:
            raise ValueError(f"Unsupported profile '{profile}'. Expected one of {MARKET_PROFILES}")

        rules = build_market_rules_for_profile(prof)
        snap_id = snapshot_id_override or f"mm_{prof.lower()}_{universe_tier}_{mode}"

        try:
            snap = self.snapshot_mgr.load_snapshot(snap_id)
            return MultiMarketEnvironment(
                profile=prof, snapshot=snap, market_rules=rules, is_synthetic=snap.is_synthetic
            )
        except FileNotFoundError:
            pass

        if prof == "VN_HOSE_T25":
            bundle = SyntheticVNMarketProvider().fetch_bundle(
                universe_tier=universe_tier,
                start_date=start_date,
                end_date=end_date,
                market_rules=rules,
                seed=seed,
            )
        elif mode == "live_yfinance":
            bundle = _fetch_live_yfinance_bundle(prof, start_date, end_date, rules)
        elif mode == "auto":
            try:
                bundle = _fetch_live_yfinance_bundle(prof, start_date, end_date, rules)
            except Exception:
                bundle = _generate_synthetic_international_bundle(prof, start_date, end_date, rules, seed=seed + 17)
        else:
            bundle = _generate_synthetic_international_bundle(prof, start_date, end_date, rules, seed=seed + 17)

        snap = self.snapshot_mgr.write_snapshot(snap_id, bundle)
        return MultiMarketEnvironment(
            profile=prof, snapshot=snap, market_rules=rules, is_synthetic=snap.is_synthetic
        )
