"""Quantitative Evaluation Metrics: Rank IC, ICIR, Net Sharpe/Sortino, MaxDD, Deflated Sharpe Ratio, Bootstrap CIs."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats


def compute_rank_ic_series(signal_wide: pd.DataFrame, forward_ret_wide: pd.DataFrame) -> pd.Series:
    """Compute daily cross-sectional Spearman Rank Information Coefficient."""
    common_idx = signal_wide.index.intersection(forward_ret_wide.index)
    common_cols = signal_wide.columns.intersection(forward_ret_wide.columns)
    s_df = signal_wide.reindex(index=common_idx, columns=common_cols)
    r_df = forward_ret_wide.reindex(index=common_idx, columns=common_cols)

    s_ranks = s_df.rank(axis=1)
    r_ranks = r_df.rank(axis=1)

    valid_counts = (s_df.notna() & r_df.notna()).sum(axis=1)
    ic = s_ranks.corrwith(r_ranks, axis=1)
    ic = ic.where(valid_counts >= 5, np.nan).dropna()
    return ic


def deflated_sharpe_ratio(
    returns: np.ndarray | pd.Series,
    n_trials: int,
    annualization: float = 252.0,
) -> float:
    """Compute Deflated Sharpe Ratio probability PSR(SR*) in [0, 1] (Bailey & Lopez de Prado, 2014).

    Adjusts observed Sharpe Ratio for:
    1. Multiple testing across `n_trials` candidate strategies (expected max SR under null).
    2. Non-Normal skewness and kurtosis of daily returns.
    3. Finite track record length T.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    t_len = len(r)
    if t_len < 10:
        return 0.0
    sd = float(np.std(r, ddof=1))
    if sd < 1e-12:
        return 0.0

    sr_daily = float(np.mean(r)) / sd
    skew = float(stats.skew(r, bias=False)) if t_len > 3 else 0.0
    kurt = float(stats.kurtosis(r, fisher=False, bias=False)) if t_len > 4 else 3.0

    trials = max(1, int(n_trials))
    if trials == 1:
        sr_star_daily = 0.0
    else:
        euler_gamma = 0.5772156649
        z1 = float(stats.norm.ppf(1.0 - 1.0 / trials))
        z2 = float(stats.norm.ppf(1.0 - 1.0 / (trials * math.e)))
        expected_max_z = (1.0 - euler_gamma) * z1 + euler_gamma * z2
        # Cross-trial daily SR std approximated by 1 / sqrt(T) under null
        sr_star_daily = expected_max_z / math.sqrt(t_len)

    denom_var = 1.0 - skew * sr_daily + ((kurt - 1.0) / 4.0) * (sr_daily**2)
    denom = math.sqrt(max(1e-8, denom_var) / max(1, t_len - 1))
    z_stat = (sr_daily - sr_star_daily) / denom
    return float(stats.norm.cdf(z_stat))


def compute_portfolio_metrics(
    net_returns: pd.Series,
    gross_returns: pd.Series,
    turnover: pd.Series,
    ic_series: pd.Series | None = None,
    n_trials: int = 1,
    annualization: float = 252.0,
) -> dict[str, float]:
    r_net = net_returns.dropna().to_numpy(dtype=float)
    r_gross = gross_returns.dropna().to_numpy(dtype=float)
    turn = turnover.dropna().to_numpy(dtype=float)

    if len(r_net) < 5:
        return {
            "rank_ic": 0.0,
            "icir": 0.0,
            "ann_net_return": 0.0,
            "ann_gross_return": 0.0,
            "net_sharpe": 0.0,
            "net_sortino": 0.0,
            "max_drawdown": 0.0,
            "daily_turnover": 0.0,
            "deflated_sharpe_prob": 0.0,
        }

    mu_net = float(np.mean(r_net))
    sd_net = float(np.std(r_net, ddof=1))
    downside = r_net[r_net < 0.0]
    sd_down = float(np.sqrt(np.mean(downside**2))) if len(downside) > 0 else 1e-8

    ann_net = float((1.0 + mu_net) ** annualization - 1.0) if mu_net > -0.99 else -1.0
    ann_gross = float((1.0 + float(np.mean(r_gross))) ** annualization - 1.0)
    net_sharpe = (mu_net / sd_net) * math.sqrt(annualization) if sd_net > 1e-10 else 0.0
    net_sortino = (mu_net / sd_down) * math.sqrt(annualization) if sd_down > 1e-10 else 0.0

    cum = np.cumprod(1.0 + np.clip(r_net, -0.95, 5.0))
    peaks = np.maximum.accumulate(cum)
    dd = (cum - peaks) / np.maximum(peaks, 1e-8)
    max_dd = float(np.min(dd))

    if ic_series is not None and len(ic_series) >= 3:
        ic_vals = ic_series.to_numpy(dtype=float)
        mean_ic = float(np.mean(ic_vals))
        sd_ic = float(np.std(ic_vals, ddof=1))
        icir = (mean_ic / sd_ic) * math.sqrt(annualization) if sd_ic > 1e-10 else 0.0
    else:
        mean_ic = 0.0
        icir = 0.0

    dsr = deflated_sharpe_ratio(r_net, n_trials=n_trials, annualization=annualization)

    return {
        "rank_ic": round(mean_ic, 5),
        "icir": round(icir, 4),
        "ann_net_return": round(ann_net, 5),
        "ann_gross_return": round(ann_gross, 5),
        "net_sharpe": round(net_sharpe, 4),
        "net_sortino": round(net_sortino, 4),
        "max_drawdown": round(max_dd, 5),
        "daily_turnover": round(float(np.mean(turn)) if len(turn) > 0 else 0.0, 5),
        "deflated_sharpe_prob": round(dsr, 4),
    }


def paired_bootstrap_ci(
    candidate_returns: pd.Series,
    baseline_returns: pd.Series,
    n_boot: int = 500,
    confidence: float = 0.95,
    seed: int = 42,
    block_size: int = 5,
) -> dict[str, float]:
    """Block-bootstrap 95% confidence interval for annualized Sharpe difference (Candidate - Baseline)."""
    common = candidate_returns.index.intersection(baseline_returns.index)
    a = candidate_returns.reindex(common).fillna(0.0).to_numpy(dtype=float)
    b = baseline_returns.reindex(common).fillna(0.0).to_numpy(dtype=float)
    n = len(a)
    if n < 15:
        return {"sharpe_diff_mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_one_sided": 0.5}

    rng = np.random.default_rng(seed)
    n_blocks = max(1, math.ceil(n / block_size))
    diffs = np.zeros(n_boot, dtype=float)

    def _sr(x: np.ndarray) -> float:
        s = float(np.std(x, ddof=1))
        return (float(np.mean(x)) / s) * math.sqrt(252.0) if s > 1e-10 else 0.0

    for i in range(n_boot):
        starts = rng.integers(0, max(1, n - block_size + 1), size=n_blocks)
        idx = np.concatenate([np.arange(st, st + block_size) for st in starts])[:n]
        diffs[i] = _sr(a[idx]) - _sr(b[idx])

    alpha = 1.0 - confidence
    lo = float(np.quantile(diffs, alpha / 2.0))
    hi = float(np.quantile(diffs, 1.0 - alpha / 2.0))
    p_val = float(np.mean(diffs <= 0.0))
    return {
        "sharpe_diff_mean": round(float(np.mean(diffs)), 4),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
        "p_one_sided": round(p_val, 4),
    }
