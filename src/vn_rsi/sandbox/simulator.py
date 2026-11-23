"""Vietnamese Cash-Equity Portfolio Simulator enforcing T+2.5, price bands, board lots, and illiquidity costs.

Research simulation ONLY — zero live order routing or brokerage APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from vn_rsi.data.market_rules import MarketRulesConfig


@dataclass
class SimulationResult:
    daily_net_returns: pd.Series
    daily_gross_returns: pd.Series
    daily_turnover: pd.Series
    daily_cost_drag: pd.Series
    blocked_ceiling_buys: int
    blocked_floor_sells: int
    t25_settlement_blocks: int
    metadata: dict[str, Any]


class VNPortfolioSimulator:
    """Simulates long-only Vietnamese equity execution under realistic exchange constraints.

    Timing convention (leak-free):
    - Signal S[t] is computed at date t close using strictly information <= t.
    - Portfolio trades execute at date t+1 close (or t+1 VWAP/close with spread + impact) and earn returns
      from t+1 onward; equivalently, weights decided at t-1 earn return_1d[t] minus execution costs on weight changes.
    - Shares bought on day tau are locked until tau + min_holding_bars_eod (T+2.5 settlement rule).
    - Buys are blocked if target stock is halted, delisted, or locked at ceiling on execution bar.
    - Sells are blocked if held stock is halted or locked at floor on execution bar (forced hold through distress!).
    """

    def __init__(
        self,
        market_rules: MarketRulesConfig,
        top_k: int = 6,
        rebalance_every_days: int = 3,
        capital_vnd: float = 1_000_000_000.0,
    ) -> None:
        self.market_rules = market_rules
        self.top_k = top_k
        self.rebalance_every_days = max(1, rebalance_every_days)
        self.capital_vnd = capital_vnd

    def simulate(
        self,
        signal_wide: pd.DataFrame,
        bars_slice: pd.DataFrame,
    ) -> SimulationResult:
        df = bars_slice.sort_values(["date", "symbol"]).copy()
        dates = pd.DatetimeIndex(np.sort(df["date"].unique()))
        symbols = sorted(df["symbol"].unique().tolist())
        if len(dates) < 3 or not symbols:
            empty = pd.Series(dtype=float)
            return SimulationResult(empty, empty, empty, empty, 0, 0, 0, {})

        close_w = df.pivot(index="date", columns="symbol", values="close").reindex(index=dates, columns=symbols).ffill()
        ret_w = close_w.pct_change(fill_method=None).fillna(0.0)
        adv_w = (
            df.pivot(index="date", columns="symbol", values="turnover_vnd")
            .reindex(index=dates, columns=symbols)
            .rolling(20, min_periods=1)
            .mean()
            .fillna(1e9)
        )
        halt_w = df.pivot(index="date", columns="symbol", values="is_halted").reindex(index=dates, columns=symbols).fillna(True)
        delist_w = df.pivot(index="date", columns="symbol", values="is_delisted").reindex(index=dates, columns=symbols).fillna(False)
        ceil_w = df.pivot(index="date", columns="symbol", values="hit_ceiling").reindex(index=dates, columns=symbols).fillna(False)
        floor_w = df.pivot(index="date", columns="symbol", values="hit_floor").reindex(index=dates, columns=symbols).fillna(False)
        ex_map = df.drop_duplicates("symbol").set_index("symbol")["exchange"].to_dict()

        sig_aligned = signal_wide.reindex(index=dates, columns=symbols)

        n_days = len(dates)
        n_syms = len(symbols)
        weights = np.zeros(n_syms, dtype=float)
        last_buy_day = np.full(n_syms, -999, dtype=int)

        gross_rets = np.zeros(n_days, dtype=float)
        net_rets = np.zeros(n_days, dtype=float)
        turnovers = np.zeros(n_days, dtype=float)
        costs = np.zeros(n_days, dtype=float)

        blocked_ceil = 0
        blocked_floor = 0
        t25_blocks = 0
        min_hold = self.market_rules.settlement.min_holding_bars_eod

        ret_mat = ret_w.to_numpy(dtype=float)
        close_mat = close_w.to_numpy(dtype=float)
        adv_mat = adv_w.to_numpy(dtype=float)
        halt_mat = halt_w.to_numpy(dtype=bool)
        delist_mat = delist_w.to_numpy(dtype=bool)
        ceil_mat = ceil_w.to_numpy(dtype=bool)
        floor_mat = floor_w.to_numpy(dtype=bool)
        sig_mat = sig_aligned.to_numpy(dtype=float)

        nav = self.capital_vnd

        for t in range(1, n_days):
            # 1. Realize today's gross return from yesterday's post-trade weights
            r_vec = ret_mat[t, :]
            # Delisted stocks still held lose an additional distress liquidation haircut (-15%) once upon delisting
            newly_delisted = delist_mat[t, :] & (~delist_mat[t - 1, :]) & (weights > 0)
            r_effective = np.where(newly_delisted, -0.15, r_vec)

            day_gross = float(np.dot(weights, r_effective))
            gross_rets[t] = day_gross

            # Drift weights by asset return
            denom = 1.0 + day_gross
            if denom > 1e-6:
                weights = weights * (1.0 + r_effective) / denom
            nav = max(1e6, nav * (1.0 + day_gross))

            # Liquidate delisted holdings after penalty
            weights[delist_mat[t, :]] = 0.0

            # 2. Check rebalance schedule using strictly causal signal from t-1 (executed on day t)
            if (t - 1) % self.rebalance_every_days != 0:
                net_rets[t] = day_gross
                continue

            scores = sig_mat[t - 1, :]
            valid_candidates = np.where(
                np.isfinite(scores) & (~halt_mat[t, :]) & (~delist_mat[t, :])
            )[0]

            target_w = np.zeros(n_syms, dtype=float)
            if len(valid_candidates) > 0:
                order = valid_candidates[np.argsort(scores[valid_candidates])[::-1]]
                chosen = order[: min(self.top_k, len(order))]
                # Long-only equal weight across top-K selected stocks
                raw_target = 1.0 / max(1, len(chosen))
                for idx in chosen:
                    ex = ex_map.get(symbols[idx], "HOSE")
                    px = max(close_mat[t, idx], 100.0)
                    # Cap by max ADV participation and round to 100-share board lot
                    max_notional = self.market_rules.illiquidity_impact.max_adv_participation * adv_mat[t, idx]
                    desired_notional = min(raw_target * nav, max_notional)
                    lot_shares = self.market_rules.round_to_board_lot(desired_notional / px, ex)
                    target_w[idx] = min(raw_target, (lot_shares * px) / nav)

            # 3. Enforce T+2.5 settlement lock & ceiling/floor execution blocks
            exec_w = weights.copy()
            day_cost_drag = 0.0
            day_turnover = 0.0

            # Process sells first
            for i in range(n_syms):
                if target_w[i] < exec_w[i] - 1e-5:
                    if (t - last_buy_day[i]) < min_hold:
                        t25_blocks += 1
                        continue
                    if halt_mat[t, i] or (self.market_rules.illiquidity_impact.floor_sell_blocked and floor_mat[t, i]):
                        blocked_floor += 1
                        continue
                    sell_delta = exec_w[i] - target_w[i]
                    notional = sell_delta * nav
                    cost_bps = self.market_rules.one_way_cost_bps(
                        "sell", ex_map.get(symbols[i], "HOSE"), notional, adv_mat[t, i]
                    )
                    day_cost_drag += sell_delta * (cost_bps / 10_000.0)
                    day_turnover += sell_delta
                    exec_w[i] = target_w[i]

            # Available cash weight for buys (long-only, no leverage)
            cash_available = max(0.0, 1.0 - float(np.sum(exec_w)))

            # Process buys within available cash
            buy_indices = [i for i in range(n_syms) if target_w[i] > exec_w[i] + 1e-5]
            total_desired_buy = sum(target_w[i] - exec_w[i] for i in buy_indices)
            scale = min(1.0, cash_available / total_desired_buy) if total_desired_buy > 1e-8 else 0.0

            for i in buy_indices:
                if scale <= 0.0:
                    break
                if halt_mat[t, i] or (self.market_rules.illiquidity_impact.ceiling_buy_blocked and ceil_mat[t, i]):
                    blocked_ceil += 1
                    continue
                buy_delta = (target_w[i] - exec_w[i]) * scale
                if buy_delta <= 1e-5:
                    continue
                notional = buy_delta * nav
                cost_bps = self.market_rules.one_way_cost_bps(
                    "buy", ex_map.get(symbols[i], "HOSE"), notional, adv_mat[t, i]
                )
                day_cost_drag += buy_delta * (cost_bps / 10_000.0)
                day_turnover += buy_delta
                exec_w[i] += buy_delta
                last_buy_day[i] = t

            weights = exec_w
            turnovers[t] = day_turnover
            costs[t] = day_cost_drag
            net_rets[t] = day_gross - day_cost_drag
            nav = max(1e6, nav * (1.0 - day_cost_drag))

        return SimulationResult(
            daily_net_returns=pd.Series(net_rets, index=dates, name="net_return"),
            daily_gross_returns=pd.Series(gross_rets, index=dates, name="gross_return"),
            daily_turnover=pd.Series(turnovers, index=dates, name="turnover"),
            daily_cost_drag=pd.Series(costs, index=dates, name="cost_drag"),
            blocked_ceiling_buys=blocked_ceil,
            blocked_floor_sells=blocked_floor,
            t25_settlement_blocks=t25_blocks,
            metadata={"top_k": self.top_k, "rebalance_every_days": self.rebalance_every_days},
        )
