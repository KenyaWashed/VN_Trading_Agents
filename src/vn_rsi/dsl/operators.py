"""Fixed Causal Operator Registry for Factor Discovery (AQuA-inspired structural leakage prevention).

Every operator in this registry operates strictly on present and past observations (t - w + 1 .. t)
or cross-sectionally at date t using information known at date t close.
No negative shifts or forward-looking windows exist in the grammar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OperatorMeta:
    name: str
    arity: int          # number of child expression arguments (1 or 2)
    needs_window: bool  # whether integer window parameter `w >= 1` is required
    description: str


ALLOWED_TERMINALS: dict[str, str] = {
    "close": "Adjusted closing price at day t",
    "open": "Adjusted opening price at day t",
    "high": "Adjusted high price at day t",
    "low": "Adjusted low price at day t",
    "volume": "Trading volume (shares) at day t",
    "turnover_vnd": "Daily traded value in VND at day t",
    "return_1d": "1-day close-to-close return (t-1 to t)",
    "intraday_range": "(high - low) / close at day t",
    "close_to_open": "(close - open) / open at day t",
    "dist_to_ceiling": "(ceiling_price - close) / ref_price at day t (>= 0)",
    "dist_to_floor": "(close - floor_price) / ref_price at day t (>= 0)",
    "turnover_shock_20d": "turnover_vnd / trailing_20d_mean(turnover_vnd)",
    "foreign_net_ratio": "Net foreign flow ratio reported at day t",
    "fol_room_pct": "Remaining foreign ownership room fraction at day t",
    "pit_roe": "Point-in-time ROE (strictly announcement_date <= t)",
    "pit_pb_ratio": "Point-in-time P/B ratio (strictly announcement_date <= t)",
    "pit_revenue_yoy": "Point-in-time YoY revenue growth (strictly announcement_date <= t)",
    "pit_value_quality": "Point-in-time composite ROE / P/B + 0.25 * revenue_yoy (announcement_date <= t)",
}

ALLOWED_OPERATORS: dict[str, OperatorMeta] = {
    "ts_mean": OperatorMeta("ts_mean", 1, True, "Causal trailing rolling mean over w days"),
    "ts_std": OperatorMeta("ts_std", 1, True, "Causal trailing rolling std over w days"),
    "ts_zscore": OperatorMeta("ts_zscore", 1, True, "Causal time-series z-score over trailing w days"),
    "ts_delta": OperatorMeta("ts_delta", 1, True, "Difference x[t] - x[t - w]"),
    "ts_return": OperatorMeta("ts_return", 1, True, "Relative change (x[t] - x[t - w]) / (|x[t - w]| + eps)"),
    "ts_rank": OperatorMeta("ts_rank", 1, True, "Percentile rank of x[t] within trailing w days"),
    "ts_min": OperatorMeta("ts_min", 1, True, "Causal trailing minimum over w days"),
    "ts_max": OperatorMeta("ts_max", 1, True, "Causal trailing maximum over w days"),
    "lag": OperatorMeta("lag", 1, True, "Strictly lagged value x[t - w] (w >= 1)"),
    "cs_rank": OperatorMeta("cs_rank", 1, False, "Cross-sectional percentile rank across active stocks on date t"),
    "cs_zscore": OperatorMeta("cs_zscore", 1, False, "Cross-sectional z-score across active stocks on date t"),
    "neg": OperatorMeta("neg", 1, False, "Unary negation -x"),
    "abs_val": OperatorMeta("abs_val", 1, False, "Elementwise |x|"),
    "tanh_squash": OperatorMeta("tanh_squash", 1, False, "Outlier-robust hyperbolic tangent tanh(x)"),
    "add": OperatorMeta("add", 2, False, "Elementwise addition a + b"),
    "sub": OperatorMeta("sub", 2, False, "Elementwise subtraction a - b"),
    "mul": OperatorMeta("mul", 2, False, "Elementwise product a * b"),
    "safe_div": OperatorMeta("safe_div", 2, False, "Outlier-safe division a / (sign(b)*max(|b|, 1e-6))"),
}


def apply_operator(
    op_name: str,
    args: list[pd.DataFrame],
    window: int | None = None,
) -> pd.DataFrame:
    """Apply a registered causal operator on wide (date x symbol) DataFrames."""
    if op_name not in ALLOWED_OPERATORS:
        raise ValueError(f"Operator '{op_name}' is not in ALLOWED_OPERATORS")
    meta = ALLOWED_OPERATORS[op_name]
    if len(args) != meta.arity:
        raise ValueError(f"Operator '{op_name}' expects {meta.arity} args, got {len(args)}")

    x = args[0]
    w = int(window or 1)
    if meta.needs_window and w < 1:
        raise ValueError(f"Causal operator '{op_name}' requires positive window >= 1, got {w}")

    if op_name == "ts_mean":
        return x.rolling(window=w, min_periods=max(1, w // 2)).mean()
    if op_name == "ts_std":
        return x.rolling(window=w, min_periods=max(2, w // 2)).std().fillna(0.0)
    if op_name == "ts_zscore":
        mu = x.rolling(window=w, min_periods=max(2, w // 2)).mean()
        sd = x.rolling(window=w, min_periods=max(2, w // 2)).std()
        return (x - mu) / (sd + 1e-8)
    if op_name == "ts_delta":
        return x - x.shift(w)
    if op_name == "ts_return":
        prev = x.shift(w)
        return (x - prev) / (prev.abs() + 1e-8)
    if op_name == "ts_min":
        return x.rolling(window=w, min_periods=max(1, w // 2)).min()
    if op_name == "ts_max":
        return x.rolling(window=w, min_periods=max(1, w // 2)).max()
    if op_name == "ts_rank":
        return x.rolling(window=w, min_periods=max(2, w // 2)).rank(pct=True)
    if op_name == "lag":
        return x.shift(w)
    if op_name == "cs_rank":
        return x.rank(axis=1, pct=True) - 0.5
    if op_name == "cs_zscore":
        mu_cs = x.mean(axis=1)
        sd_cs = x.std(axis=1) + 1e-8
        return x.sub(mu_cs, axis=0).div(sd_cs, axis=0)
    if op_name == "neg":
        return -x
    if op_name == "abs_val":
        return x.abs()
    if op_name == "tanh_squash":
        return pd.DataFrame(np.tanh(np.clip(x.to_numpy(dtype=float), -10.0, 10.0)), index=x.index, columns=x.columns)

    y = args[1]
    if op_name == "add":
        return x + y
    if op_name == "sub":
        return x - y
    if op_name == "mul":
        return x * y
    if op_name == "safe_div":
        denom = np.where(y >= 0, np.maximum(y.abs(), 1e-6), -np.maximum(y.abs(), 1e-6))
        return pd.DataFrame(x.to_numpy(dtype=float) / denom, index=x.index, columns=x.columns)

    raise ValueError(f"Unhandled operator '{op_name}'")
