"""Vietnamese equity market microstructure & regulatory rule configuration (config-driven, never silently hardcoded)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from vn_rsi.utils.repro import load_yaml


class ExchangeRule(BaseModel):
    name: str
    daily_price_limit_pct: float = Field(gt=0.0, le=0.5)
    first_day_limit_pct: float = Field(gt=0.0, le=1.0)
    lot_size: int = Field(ge=1)
    source: str
    verified: bool


class SettlementRule(BaseModel):
    cycle_days: float = Field(ge=0.0)
    min_holding_bars_eod: int = Field(ge=1)
    allow_intraday_roundtrip: bool
    source: str
    verified: bool


class ShortSellingRule(BaseModel):
    allow_cash_equity_short: bool
    allow_vn30_futures_hedge: bool
    source: str
    verified: bool


class TransactionCostRule(BaseModel):
    broker_commission_bps: float = Field(ge=0.0)
    sell_tax_bps: float = Field(ge=0.0)
    buy_tax_bps: float = Field(ge=0.0)
    source: str
    verified: bool


class IlliquidityImpactRule(BaseModel):
    model: str
    base_half_spread_bps: dict[str, float]
    impact_coeff_bps: float = Field(ge=0.0)
    max_adv_participation: float = Field(gt=0.0, le=1.0)
    ceiling_buy_blocked: bool
    floor_sell_blocked: bool
    source: str
    verified: bool


class ForeignOwnershipRule(BaseModel):
    default_fol_limit_pct: float = Field(gt=0.0, le=1.0)
    banking_fol_limit_pct: float = Field(gt=0.0, le=1.0)
    enforce_fol_room_check: bool
    source: str
    verified: bool


class MarketRulesConfig(BaseModel):
    version: str
    currency: str
    exchanges: dict[str, ExchangeRule]
    settlement: SettlementRule
    short_selling: ShortSellingRule
    transaction_costs: TransactionCostRule
    illiquidity_impact: IlliquidityImpactRule
    foreign_ownership: ForeignOwnershipRule

    def get_exchange_limit(self, exchange: str) -> float:
        ex = exchange.upper()
        if ex not in self.exchanges:
            raise KeyError(f"Unknown exchange '{exchange}'. Known: {list(self.exchanges)}")
        return self.exchanges[ex].daily_price_limit_pct

    def round_to_board_lot(self, shares: float, exchange: str) -> int:
        lot = self.exchanges[exchange.upper()].lot_size
        return int(math.floor(max(0.0, shares) / lot) * lot)

    def one_way_cost_bps(
        self,
        side: str,
        exchange: str,
        order_notional_vnd: float,
        adv_20d_vnd: float,
    ) -> float:
        """Compute total one-way cost in bps (commission + statutory sell tax + spread + illiquidity impact)."""
        ex = exchange.upper()
        comm = self.transaction_costs.broker_commission_bps
        tax = (
            self.transaction_costs.sell_tax_bps
            if side.lower() == "sell"
            else self.transaction_costs.buy_tax_bps
        )
        half_spread = self.illiquidity_impact.base_half_spread_bps.get(ex, 30.0)
        safe_adv = max(adv_20d_vnd, 1e6)
        participation = min(max(order_notional_vnd, 0.0) / safe_adv, 1.0)
        impact = self.illiquidity_impact.impact_coeff_bps * math.sqrt(participation)
        return comm + tax + half_spread + impact

    def unverified_items(self) -> list[dict[str, Any]]:
        """List any rule section marked verified=False so reports stay transparent."""
        unverified: list[dict[str, Any]] = []
        for ex_code, ex_rule in self.exchanges.items():
            if not ex_rule.verified:
                unverified.append({"section": f"exchanges.{ex_code}", "source": ex_rule.source})
        for name in ("settlement", "short_selling", "transaction_costs", "illiquidity_impact", "foreign_ownership"):
            sub = getattr(self, name)
            if not sub.verified:
                unverified.append({"section": name, "source": sub.source})
        return unverified


def load_market_rules(path: str | Path) -> MarketRulesConfig:
    return MarketRulesConfig.model_validate(load_yaml(path))
