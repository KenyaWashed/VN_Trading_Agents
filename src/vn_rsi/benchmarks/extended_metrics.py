"""Extended Quantitative Finance & Trading Agent Metrics Suite (Seal-Preserving Wrapper).

Wraps outputs of `SealedEvaluator` / `VNPortfolioSimulator` WITHOUT modifying any SHA-256
hash-locked file in `src/vn_rsi/sandbox/` or `src/vn_rsi/eval/`.

Implements international SOTA benchmark metrics from:
- TradingAgents, FinMem, FinAgent, RD-Agent(Q), AlphaGen, AlphaForge, Qlib
- Bailey & Lopez de Prado (2012, 2014): PSR & Deflated Sharpe Ratio (DSR)
- White (2000) Reality Check & Hansen (2005) Superior Predictive Ability (SPA) test
- Four-regime decomposition: Bull, Bear/Crash, Sideways, High-Volatility Liquidity Crunch
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from vn_rsi.eval.evaluator import EvaluationOutcome
from vn_rsi.eval.metrics import compute_rank_ic_series

REGIME_NAMES: tuple[str, ...] = (
    "Bull",
    "Bear_Crash",
    "Sideways",
    "High_Vol_Liquidity_Crunch",
)


def _clean_array(series: np.ndarray | pd.Series) -> np.ndarray:
    arr = np.asarray(series, dtype=float)
    return arr[np.isfinite(arr)]


def max_drawdown_and_duration(returns: np.ndarray | pd.Series) -> tuple[float, int]:
    """Return (max_drawdown <= 0, max_drawdown_duration_bars >= 0)."""
    r = _clean_array(returns)
    if len(r) == 0:
        return 0.0, 0
    wealth = np.cumprod(1.0 + np.clip(r, -0.95, 5.0))
    peaks = np.maximum.accumulate(wealth)
    dd = (wealth - peaks) / np.maximum(peaks, 1e-12)
    max_dd = float(np.min(dd))

    max_dur = 0
    cur_dur = 0
    running_peak = wealth[0]
    for w in wealth:
        if w >= running_peak - 1e-12:
            running_peak = w
            cur_dur = 0
        else:
            cur_dur += 1
            if cur_dur > max_dur:
                max_dur = cur_dur
    return max_dd, max_dur


def calmar_ratio(returns: np.ndarray | pd.Series, annualization: float = 252.0) -> float:
    """Calmar Ratio = Annualized Geometric/Compound Net Return / |MaxDD|."""
    r = _clean_array(returns)
    if len(r) < 5:
        return 0.0
    mu = float(np.mean(r))
    ann_ret = float((1.0 + mu) ** annualization - 1.0) if mu > -0.99 else -1.0
    max_dd, _ = max_drawdown_and_duration(r)
    denom = abs(max_dd)
    if denom < 1e-6:
        return float(np.clip(ann_ret * 100.0, -50.0, 50.0))
    return float(ann_ret / denom)


def omega_ratio(returns: np.ndarray | pd.Series, threshold: float = 0.0) -> float:
    """Omega Ratio = sum(max(r - threshold, 0)) / sum(max(threshold - r, 0))."""
    r = _clean_array(returns)
    if len(r) < 2:
        return 0.0
    excess = r - threshold
    gains = float(np.sum(np.maximum(excess, 0.0)))
    losses = float(np.sum(np.maximum(-excess, 0.0)))
    if losses < 1e-12:
        return 10.0 if gains > 0.0 else 1.0
    return float(gains / losses)


def information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    annualization: float = 252.0,
) -> float:
    """Annualized Information Ratio relative to benchmark returns."""
    common = returns.index.intersection(benchmark_returns.index)
    if len(common) < 5:
        return 0.0
    active = (
        returns.reindex(common).fillna(0.0).to_numpy(dtype=float)
        - benchmark_returns.reindex(common).fillna(0.0).to_numpy(dtype=float)
    )
    sd = float(np.std(active, ddof=1))
    if sd < 1e-10:
        return 0.0
    return float((float(np.mean(active)) / sd) * math.sqrt(annualization))


def capm_alpha_beta(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    annualization: float = 252.0,
) -> dict[str, float]:
    """OLS CAPM Alpha (annualized), Beta, and R-squared vs benchmark."""
    common = returns.index.intersection(benchmark_returns.index)
    if len(common) < 10:
        return {"capm_alpha_ann": 0.0, "capm_beta": 0.0, "capm_r2": 0.0}
    y = returns.reindex(common).fillna(0.0).to_numpy(dtype=float)
    x = benchmark_returns.reindex(common).fillna(0.0).to_numpy(dtype=float)
    var_x = float(np.var(x, ddof=1))
    if var_x < 1e-12:
        alpha_daily = float(np.mean(y))
        return {"capm_alpha_ann": round(alpha_daily * annualization, 5), "capm_beta": 0.0, "capm_r2": 0.0}
    cov_xy = float(np.cov(x, y, ddof=1)[0, 1])
    beta = cov_xy / var_x
    alpha_daily = float(np.mean(y) - beta * np.mean(x))
    y_hat = alpha_daily + beta * x
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = max(0.0, 1.0 - ss_res / max(ss_tot, 1e-12))
    return {
        "capm_alpha_ann": round(alpha_daily * annualization, 5),
        "capm_beta": round(beta, 4),
        "capm_r2": round(r2, 4),
    }


def cvar_expected_shortfall(
    returns: np.ndarray | pd.Series, alpha: float = 0.05
) -> tuple[float, float]:
    """Return (VaR_95, CVaR_95 / Expected Shortfall) as daily return values (typically <= 0)."""
    r = _clean_array(returns)
    if len(r) < 5:
        return 0.0, 0.0
    var_q = float(np.quantile(r, alpha))
    tail = r[r <= var_q]
    cvar = float(np.mean(tail)) if len(tail) > 0 else var_q
    return var_q, cvar


def ulcer_index(returns: np.ndarray | pd.Series) -> float:
    """Ulcer Index = RMS of percentage drawdowns (measures depth & duration of underwater periods)."""
    r = _clean_array(returns)
    if len(r) < 2:
        return 0.0
    wealth = np.cumprod(1.0 + np.clip(r, -0.95, 5.0))
    peaks = np.maximum.accumulate(wealth)
    dd_pct = 100.0 * (wealth - peaks) / np.maximum(peaks, 1e-12)
    return float(np.sqrt(np.mean(dd_pct**2)))


def win_rate_pct(returns: np.ndarray | pd.Series) -> float:
    """Percentage of active daily sessions with strictly positive net returns."""
    r = _clean_array(returns)
    if len(r) == 0:
        return 0.0
    active = r[np.abs(r) > 1e-12]
    arr = active if len(active) > 0 else r
    return float(100.0 * np.mean(arr > 0.0))


def profit_factor(returns: np.ndarray | pd.Series) -> float:
    """Gross winning sessions divided by gross losing sessions."""
    r = _clean_array(returns)
    if len(r) == 0:
        return 0.0
    pos = float(np.sum(r[r > 0.0]))
    neg = float(np.abs(np.sum(r[r < 0.0])))
    if neg < 1e-12:
        return 10.0 if pos > 0.0 else 0.0
    return float(pos / neg)


def floor_lock_trap_rate(
    execution_stats: dict[str, int] | None,
    n_days: int,
    rebalance_every_days: int = 3,
    top_k: int = 6,
) -> dict[str, float]:
    """Compute T+2.5 & floor-limit lock trap metrics from simulator execution stats."""
    stats_map = execution_stats or {}
    floor_blocks = int(stats_map.get("blocked_floor_sells", 0))
    ceil_blocks = int(stats_map.get("blocked_ceiling_buys", 0))
    t25_blocks = int(stats_map.get("t25_settlement_blocks", 0))
    rebal_slots = max(1, (max(1, n_days) // max(1, rebalance_every_days)) * max(1, top_k))
    return {
        "blocked_floor_sells": float(floor_blocks),
        "blocked_ceiling_buys": float(ceil_blocks),
        "t25_settlement_blocks": float(t25_blocks),
        "floor_lock_trap_rate": round(min(1.0, floor_blocks / float(rebal_slots)), 5),
        "t25_lock_rate": round(min(1.0, t25_blocks / float(rebal_slots)), 5),
    }


def compute_ic_decay_curve(
    signal_wide: pd.DataFrame,
    close_wide: pd.DataFrame,
    zone_dates: pd.DatetimeIndex | None = None,
    horizons: tuple[int, ...] = (1, 3, 5, 10, 20),
    annualization: float = 252.0,
) -> dict[str, dict[str, float]]:
    """Compute cross-sectional Rank IC and ICIR across forward horizons (1d/3d/5d/10d/20d).

    Strictly prevents zone-boundary label bleed by truncating the last `h` dates of `zone_dates`.
    """
    dates = zone_dates if zone_dates is not None else pd.DatetimeIndex(signal_wide.index)
    out: dict[str, dict[str, float]] = {}
    for h in horizons:
        if len(dates) <= h + 5:
            out[f"{h}d"] = {"rank_ic": 0.0, "icir": 0.0, "n_days": 0.0}
            continue
        safe_dates = dates[:-h]
        fwd_h = close_wide.shift(-h) / close_wide - 1.0
        ic_s = compute_rank_ic_series(
            signal_wide.reindex(index=safe_dates),
            fwd_h.reindex(index=safe_dates),
        )
        if len(ic_s) < 3:
            out[f"{h}d"] = {"rank_ic": 0.0, "icir": 0.0, "n_days": float(len(ic_s))}
        else:
            m = float(ic_s.mean())
            sd = float(ic_s.std(ddof=1))
            icir = (m / sd) * math.sqrt(annualization) if sd > 1e-10 else 0.0
            out[f"{h}d"] = {
                "rank_ic": round(m, 5),
                "icir": round(icir, 4),
                "n_days": float(len(ic_s)),
            }
    return out


def probabilistic_sharpe_ratio(
    returns: np.ndarray | pd.Series,
    benchmark_sr_annual: float = 0.0,
    annualization: float = 252.0,
) -> float:
    """Probabilistic Sharpe Ratio PSR(SR_0) in [0, 1] (Bailey & Lopez de Prado, 2012)."""
    r = _clean_array(returns)
    t_len = len(r)
    if t_len < 10:
        return 0.0
    sd = float(np.std(r, ddof=1))
    if sd < 1e-12:
        return 0.0
    sr_daily = float(np.mean(r)) / sd
    sr0_daily = benchmark_sr_annual / math.sqrt(annualization)
    skew = float(stats.skew(r, bias=False)) if t_len > 3 else 0.0
    kurt = float(stats.kurtosis(r, fisher=False, bias=False)) if t_len > 4 else 3.0
    denom_var = 1.0 - skew * sr_daily + ((kurt - 1.0) / 4.0) * (sr_daily**2)
    denom = math.sqrt(max(1e-8, denom_var) / max(1, t_len - 1))
    return float(stats.norm.cdf((sr_daily - sr0_daily) / denom))


def deflated_sharpe_ratio_ext(
    returns: np.ndarray | pd.Series,
    n_trials: int,
    annualization: float = 252.0,
) -> float:
    """Deflated Sharpe Ratio DSR in [0, 1] (Bailey & Lopez de Prado, 2014)."""
    r = _clean_array(returns)
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
        sr_star_daily = expected_max_z / math.sqrt(t_len)
    denom_var = 1.0 - skew * sr_daily + ((kurt - 1.0) / 4.0) * (sr_daily**2)
    denom = math.sqrt(max(1e-8, denom_var) / max(1, t_len - 1))
    return float(stats.norm.cdf((sr_daily - sr_star_daily) / denom))


def paired_block_bootstrap_metrics(
    candidate_returns: pd.Series,
    baseline_returns: pd.Series,
    n_boot: int = 1000,
    block_size: int = 5,
    confidence: float = 0.95,
    seed: int = 42,
    annualization: float = 252.0,
) -> dict[str, float]:
    """Stationary block-bootstrap 95% CI for Sharpe difference and Annualized Return difference."""
    common = candidate_returns.index.intersection(baseline_returns.index)
    a = candidate_returns.reindex(common).fillna(0.0).to_numpy(dtype=float)
    b = baseline_returns.reindex(common).fillna(0.0).to_numpy(dtype=float)
    n = len(a)
    if n < 15:
        return {
            "sharpe_diff_mean": 0.0,
            "sharpe_ci_low": 0.0,
            "sharpe_ci_high": 0.0,
            "p_one_sided_sharpe": 0.5,
            "ann_ret_diff_mean": 0.0,
            "ann_ret_ci_low": 0.0,
            "ann_ret_ci_high": 0.0,
            "p_one_sided_ret": 0.5,
        }

    rng = np.random.default_rng(seed)
    n_blocks = max(1, math.ceil(n / block_size))
    sr_diffs = np.zeros(n_boot, dtype=float)
    ret_diffs = np.zeros(n_boot, dtype=float)
    sqrt_ann = math.sqrt(annualization)

    for i in range(n_boot):
        starts = rng.integers(0, max(1, n - block_size + 1), size=n_blocks)
        idx = np.concatenate([np.arange(st, st + block_size) for st in starts])[:n]
        xa, xb = a[idx], b[idx]
        sa, sb = float(np.std(xa, ddof=1)), float(np.std(xb, ddof=1))
        sra = (float(np.mean(xa)) / sa) * sqrt_ann if sa > 1e-10 else 0.0
        srb = (float(np.mean(xb)) / sb) * sqrt_ann if sb > 1e-10 else 0.0
        sr_diffs[i] = sra - srb
        ra = float((1.0 + float(np.mean(xa))) ** annualization - 1.0)
        rb = float((1.0 + float(np.mean(xb))) ** annualization - 1.0)
        ret_diffs[i] = ra - rb

    alpha = 1.0 - confidence
    return {
        "sharpe_diff_mean": round(float(np.mean(sr_diffs)), 4),
        "sharpe_ci_low": round(float(np.quantile(sr_diffs, alpha / 2.0)), 4),
        "sharpe_ci_high": round(float(np.quantile(sr_diffs, 1.0 - alpha / 2.0)), 4),
        "p_one_sided_sharpe": round(float(np.mean(sr_diffs <= 0.0)), 4),
        "ann_ret_diff_mean": round(float(np.mean(ret_diffs)), 4),
        "ann_ret_ci_low": round(float(np.quantile(ret_diffs, alpha / 2.0)), 4),
        "ann_ret_ci_high": round(float(np.quantile(ret_diffs, 1.0 - alpha / 2.0)), 4),
        "p_one_sided_ret": round(float(np.mean(ret_diffs <= 0.0)), 4),
    }


def hansen_spa_and_white_rc(
    candidate_returns_map: dict[str, pd.Series],
    benchmark_returns: pd.Series,
    n_boot: int = 500,
    block_size: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute White's Reality Check (2000) & Hansen's SPA (2005) consistent p-values.

    Tests H0: max_k E[d_{k,t}] <= 0 where d_{k,t} = r_{k,t} - r_{bm,t} (excess daily return
    of strategy k over benchmark). Rejecting H0 (low p-value) confirms superior predictive
    ability even after accounting for searching over all K competing strategies.
    """
    if not candidate_returns_map:
        return {
            "n_models": 0,
            "best_model": "",
            "white_rc_p_value": 1.0,
            "hansen_spa_p_value": 1.0,
            "hansen_spa_t_stat": 0.0,
            "white_rc_stat": 0.0,
        }

    names = sorted(candidate_returns_map.keys())
    common = benchmark_returns.index
    for nm in names:
        common = common.intersection(candidate_returns_map[nm].index)
    t_len = len(common)
    if t_len < 15:
        return {
            "n_models": len(names),
            "best_model": names[0],
            "white_rc_p_value": 1.0,
            "hansen_spa_p_value": 1.0,
            "hansen_spa_t_stat": 0.0,
            "white_rc_stat": 0.0,
        }

    bm = benchmark_returns.reindex(common).fillna(0.0).to_numpy(dtype=float)
    # Shape: (T, K)
    d_mat = np.column_stack(
        [candidate_returns_map[nm].reindex(common).fillna(0.0).to_numpy(dtype=float) - bm for nm in names]
    )
    d_bar = np.mean(d_mat, axis=0)
    # Newey-West HAC variance estimate (lag L = block_size - 1)
    max_lag = max(1, min(block_size, t_len // 4))
    omega_sq = np.zeros(len(names), dtype=float)
    centered = d_mat - d_bar[None, :]
    for k in range(len(names)):
        c = centered[:, k]
        gamma0 = float(np.dot(c, c) / t_len)
        hac = gamma0
        for lag in range(1, max_lag + 1):
            w_lag = 1.0 - lag / float(max_lag + 1)
            gam = float(np.dot(c[lag:], c[:-lag]) / t_len)
            hac += 2.0 * w_lag * gam
        omega_sq[k] = max(hac, 1e-12)
    omega = np.sqrt(omega_sq)

    sqrt_t = math.sqrt(t_len)
    white_obs = float(np.max(sqrt_t * d_bar))
    t_stats = sqrt_t * d_bar / omega
    spa_obs = float(max(0.0, float(np.max(t_stats))))
    best_idx = int(np.argmax(t_stats))

    # Hansen (2005) consistent recentering threshold: -sqrt(2 * log(log(T)))
    ll_t = math.log(max(math.e, math.log(max(math.e, float(t_len)))))
    hansen_thresh = -math.sqrt(2.0 * ll_t)
    # Sample-dependent null mean for SPA consistent p-value
    g_consistent = np.where(t_stats >= hansen_thresh, d_bar, 0.0)

    rng = np.random.default_rng(seed)
    n_blocks = max(1, math.ceil(t_len / block_size))
    white_boot = np.zeros(n_boot, dtype=float)
    spa_boot = np.zeros(n_boot, dtype=float)

    for b in range(n_boot):
        starts = rng.integers(0, max(1, t_len - block_size + 1), size=n_blocks)
        idx = np.concatenate([np.arange(st, st + block_size) for st in starts])[:t_len]
        d_b = d_mat[idx, :]
        bar_b = np.mean(d_b, axis=0)
        # White RC centers every model at d_bar (least favorable null)
        white_boot[b] = float(np.max(sqrt_t * (bar_b - d_bar)))
        # Hansen SPA centers only non-inferior models at d_bar, keeping clearly inferior ones <= 0
        spa_centered = bar_b - g_consistent
        spa_boot[b] = float(max(0.0, float(np.max(sqrt_t * spa_centered / omega))))

    white_p = float(np.mean(white_boot >= white_obs))
    spa_p = float(np.mean(spa_boot >= spa_obs))
    return {
        "n_models": len(names),
        "best_model": names[best_idx],
        "white_rc_p_value": round(white_p, 4),
        "hansen_spa_p_value": round(spa_p, 4),
        "hansen_spa_t_stat": round(spa_obs, 4),
        "white_rc_stat": round(white_obs, 5),
    }


def classify_market_regimes(
    benchmark_returns: pd.Series,
    bars_slice: pd.DataFrame | None = None,
    vol_window: int = 20,
    trend_window: int = 20,
) -> pd.Series:
    """Classify each trading day into 4 regimes: Bull, Bear_Crash, Sideways, High_Vol_Liquidity_Crunch.

    Strictly causal labeling using trailing information <= t so regimes can be audited consistently.
    """
    bm = benchmark_returns.sort_index().fillna(0.0)
    if bm.empty:
        return pd.Series(dtype=str)

    roll_ret = (1.0 + bm).rolling(trend_window, min_periods=5).apply(np.prod, raw=True) - 1.0
    roll_vol = bm.rolling(vol_window, min_periods=5).std().fillna(bm.std() if len(bm) > 1 else 0.01)
    # Expanding 80th percentile volatility threshold (strictly causal)
    vol_hi_thresh = roll_vol.expanding(min_periods=10).quantile(0.80).bfill()

    wealth = (1.0 + bm).cumprod()
    peak60 = wealth.rolling(60, min_periods=5).max()
    dd60 = (wealth - peak60) / peak60.clip(lower=1e-8)

    floor_rate = pd.Series(0.0, index=bm.index)
    if bars_slice is not None and not bars_slice.empty and "hit_floor" in bars_slice.columns:
        fr = bars_slice.groupby("date")["hit_floor"].mean().reindex(bm.index).fillna(0.0)
        floor_rate = fr.rolling(5, min_periods=1).mean()

    labels: list[str] = []
    for dt in bm.index:
        r20 = float(roll_ret.loc[dt]) if np.isfinite(roll_ret.loc[dt]) else 0.0
        v20 = float(roll_vol.loc[dt]) if np.isfinite(roll_vol.loc[dt]) else 0.0
        v_thr = float(vol_hi_thresh.loc[dt]) if np.isfinite(vol_hi_thresh.loc[dt]) else 0.02
        d60 = float(dd60.loc[dt]) if np.isfinite(dd60.loc[dt]) else 0.0
        flr = float(floor_rate.loc[dt])

        if (v20 >= max(v_thr, 0.016) and (r20 < 0.01 or flr >= 0.03)) or flr >= 0.05:
            labels.append("High_Vol_Liquidity_Crunch")
        elif r20 <= -0.025 or d60 <= -0.08:
            labels.append("Bear_Crash")
        elif r20 >= 0.020:
            labels.append("Bull")
        else:
            labels.append("Sideways")
    return pd.Series(labels, index=bm.index, name="market_regime")


@dataclass
class RegimeMetrics:
    regime: str
    n_days: int
    ann_net_return: float
    benchmark_ann_return: float
    excess_ann_return: float
    net_sharpe: float
    max_drawdown: float
    win_rate_pct: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime,
            "n_days": self.n_days,
            "ann_net_return": self.ann_net_return,
            "benchmark_ann_return": self.benchmark_ann_return,
            "excess_ann_return": self.excess_ann_return,
            "net_sharpe": self.net_sharpe,
            "max_drawdown": self.max_drawdown,
            "win_rate_pct": self.win_rate_pct,
        }


def compute_regime_breakdown(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    bars_slice: pd.DataFrame | None = None,
    annualization: float = 252.0,
) -> dict[str, RegimeMetrics]:
    """Break down strategy performance across Bull, Bear/Crash, Sideways, and High-Vol Liquidity Crunch."""
    common = strategy_returns.index.intersection(benchmark_returns.index)
    if len(common) < 5:
        return {}
    s_ret = strategy_returns.reindex(common).fillna(0.0)
    b_ret = benchmark_returns.reindex(common).fillna(0.0)
    regimes = classify_market_regimes(b_ret, bars_slice=bars_slice)

    out: dict[str, RegimeMetrics] = {}
    for reg in REGIME_NAMES:
        mask = regimes == reg
        n_d = int(mask.sum())
        if n_d < 2:
            out[reg] = RegimeMetrics(reg, n_d, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            continue
        sr_arr = s_ret.loc[mask].to_numpy(dtype=float)
        bm_arr = b_ret.loc[mask].to_numpy(dtype=float)
        mu_s = float(np.mean(sr_arr))
        mu_b = float(np.mean(bm_arr))
        sd_s = float(np.std(sr_arr, ddof=1))
        ann_s = float((1.0 + mu_s) ** annualization - 1.0) if mu_s > -0.99 else -1.0
        ann_b = float((1.0 + mu_b) ** annualization - 1.0) if mu_b > -0.99 else -1.0
        sharpe_s = (mu_s / sd_s) * math.sqrt(annualization) if sd_s > 1e-10 else 0.0
        mdd_s, _ = max_drawdown_and_duration(sr_arr)
        wr_s = win_rate_pct(sr_arr)
        out[reg] = RegimeMetrics(
            regime=reg,
            n_days=n_d,
            ann_net_return=round(ann_s, 5),
            benchmark_ann_return=round(ann_b, 5),
            excess_ann_return=round(ann_s - ann_b, 5),
            net_sharpe=round(sharpe_s, 4),
            max_drawdown=round(mdd_s, 5),
            win_rate_pct=round(wr_s, 2),
        )
    return out


@dataclass
class ExtendedMetricsReport:
    candidate_id: str
    zone: str
    # Core & Risk-adjusted
    rank_ic: float
    icir: float
    ann_net_return: float
    net_sharpe: float
    net_sortino: float
    calmar_ratio: float
    omega_ratio: float
    information_ratio: float
    capm_alpha_ann: float
    capm_beta: float
    capm_r2: float
    # Tail Risk & Survivability
    max_drawdown: float
    max_drawdown_duration: int
    var_95: float
    cvar_95: float
    ulcer_index: float
    floor_lock_trap_rate: float
    t25_lock_rate: float
    # Signal & Trade Quality
    win_rate_pct: float
    profit_factor: float
    daily_turnover: float
    turnover_adjusted_net_alpha: float
    ic_decay: dict[str, dict[str, float]] = field(default_factory=dict)
    # Anti-Overfitting Statistics
    n_trials: int = 1
    deflated_sharpe_prob: float = 0.0
    probabilistic_sharpe_ratio: float = 0.0
    bootstrap_vs_benchmark: dict[str, float] = field(default_factory=dict)
    spa_white_rc: dict[str, Any] = field(default_factory=dict)
    # 4-Regime Breakdown
    regime_breakdown: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "zone": self.zone,
            "rank_ic": self.rank_ic,
            "icir": self.icir,
            "ann_net_return": self.ann_net_return,
            "net_sharpe": self.net_sharpe,
            "net_sortino": self.net_sortino,
            "calmar_ratio": self.calmar_ratio,
            "omega_ratio": self.omega_ratio,
            "information_ratio": self.information_ratio,
            "capm_alpha_ann": self.capm_alpha_ann,
            "capm_beta": self.capm_beta,
            "capm_r2": self.capm_r2,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
            "ulcer_index": self.ulcer_index,
            "floor_lock_trap_rate": self.floor_lock_trap_rate,
            "t25_lock_rate": self.t25_lock_rate,
            "win_rate_pct": self.win_rate_pct,
            "profit_factor": self.profit_factor,
            "daily_turnover": self.daily_turnover,
            "turnover_adjusted_net_alpha": self.turnover_adjusted_net_alpha,
            "ic_decay": self.ic_decay,
            "n_trials": self.n_trials,
            "deflated_sharpe_prob": self.deflated_sharpe_prob,
            "probabilistic_sharpe_ratio": self.probabilistic_sharpe_ratio,
            "bootstrap_vs_benchmark": self.bootstrap_vs_benchmark,
            "spa_white_rc": self.spa_white_rc,
            "regime_breakdown": self.regime_breakdown,
        }


def compute_extended_metrics(
    outcome: EvaluationOutcome,
    benchmark_returns: pd.Series,
    bars_slice: pd.DataFrame | None = None,
    signal_wide: pd.DataFrame | None = None,
    close_wide: pd.DataFrame | None = None,
    n_trials: int = 1,
    rebalance_every_days: int = 3,
    top_k: int = 6,
    peer_returns_map: dict[str, pd.Series] | None = None,
    bootstrap_seed: int = 42,
    n_boot: int = 500,
) -> ExtendedMetricsReport:
    """Wrap a SealedEvaluator `EvaluationOutcome` to compute all extended SOTA quant metrics."""
    r = outcome.daily_net_returns
    m = outcome.metrics
    mdd, mdd_dur = max_drawdown_and_duration(r)
    var95, cvar95 = cvar_expected_shortfall(r, alpha=0.05)
    calmar = calmar_ratio(r)
    omega = omega_ratio(r, threshold=0.0)
    ir = information_ratio(r, benchmark_returns)
    capm = capm_alpha_beta(r, benchmark_returns)
    ui = ulcer_index(r)
    wr = win_rate_pct(r)
    pf = profit_factor(r)
    traps = floor_lock_trap_rate(
        outcome.execution_stats,
        n_days=len(r),
        rebalance_every_days=rebalance_every_days,
        top_k=top_k,
    )
    daily_turn = float(m.get("daily_turnover", 0.0))
    # Turnover-adjusted net CAPM alpha: penalizes high annual turnover drag
    turn_adj_alpha = capm["capm_alpha_ann"] / (1.0 + 4.0 * daily_turn)

    ic_decay: dict[str, dict[str, float]] = {}
    if signal_wide is not None and close_wide is not None:
        ic_decay = compute_ic_decay_curve(
            signal_wide=signal_wide,
            close_wide=close_wide,
            zone_dates=pd.DatetimeIndex(r.index),
        )

    dsr = deflated_sharpe_ratio_ext(r, n_trials=max(1, n_trials))
    psr = probabilistic_sharpe_ratio(r, benchmark_sr_annual=0.0)
    boot_bm = paired_block_bootstrap_metrics(r, benchmark_returns, n_boot=n_boot, seed=bootstrap_seed)

    spa_map = dict(peer_returns_map or {})
    spa_map[outcome.candidate_id] = r
    spa_res = hansen_spa_and_white_rc(spa_map, benchmark_returns, n_boot=min(400, n_boot), seed=bootstrap_seed)

    regimes = compute_regime_breakdown(r, benchmark_returns, bars_slice=bars_slice)

    return ExtendedMetricsReport(
        candidate_id=outcome.candidate_id,
        zone=outcome.zone,
        rank_ic=round(float(m.get("rank_ic", 0.0)), 5),
        icir=round(float(m.get("icir", 0.0)), 4),
        ann_net_return=round(float(m.get("ann_net_return", 0.0)), 5),
        net_sharpe=round(float(m.get("net_sharpe", 0.0)), 4),
        net_sortino=round(float(m.get("net_sortino", 0.0)), 4),
        calmar_ratio=round(calmar, 4),
        omega_ratio=round(omega, 4),
        information_ratio=round(ir, 4),
        capm_alpha_ann=capm["capm_alpha_ann"],
        capm_beta=capm["capm_beta"],
        capm_r2=capm["capm_r2"],
        max_drawdown=round(mdd, 5),
        max_drawdown_duration=int(mdd_dur),
        var_95=round(var95, 5),
        cvar_95=round(cvar95, 5),
        ulcer_index=round(ui, 4),
        floor_lock_trap_rate=traps["floor_lock_trap_rate"],
        t25_lock_rate=traps["t25_lock_rate"],
        win_rate_pct=round(wr, 2),
        profit_factor=round(pf, 4),
        daily_turnover=round(daily_turn, 5),
        turnover_adjusted_net_alpha=round(turn_adj_alpha, 5),
        ic_decay=ic_decay,
        n_trials=max(1, n_trials),
        deflated_sharpe_prob=round(dsr, 4),
        probabilistic_sharpe_ratio=round(psr, 4),
        bootstrap_vs_benchmark=boot_bm,
        spa_white_rc=spa_res,
        regime_breakdown={k: v.to_dict() for k, v in regimes.items()},
    )
