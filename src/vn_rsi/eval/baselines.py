"""Human-authored reference baselines under identical sealed splits and Vietnamese market costs.

Baselines (Section 4):
1. Buy-and-Hold VN-Index
2. Equal-Weight Universe
3. Simple Momentum (20d risk-adjusted)
4. Short-Term Reversal (5d)
5. Purged Walk-Forward Ridge Regression
6. Purged Walk-Forward Gradient Boosted Trees (LightGBM if installed, else HistGradientBoostingRegressor)
7. Purged Walk-Forward Temporal Gated Recurrent / Sequence Network (GRU/Elman sequence model)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from vn_rsi.eval.evaluator import EvaluationOutcome, SealedEvaluator


def _build_tabular_features(panel: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    close = panel["close"]
    ret1 = panel["return_1d"]
    feats = {
        "mom_5": close.pct_change(5, fill_method=None),
        "mom_15": close.pct_change(15, fill_method=None),
        "vol_15": ret1.rolling(15, min_periods=5).std(),
        "tshock": panel["turnover_shock_20d"],
        "fflow_5": panel["foreign_net_ratio"].rolling(5, min_periods=2).mean(),
        "pit_vq": panel["pit_value_quality"],
        "range": panel["intraday_range"],
    }
    return {k: v.replace([np.inf, -np.inf], np.nan).fillna(0.0) for k, v in feats.items()}


def _fit_predict_walk_forward_ml(
    evaluator: SealedEvaluator,
    model_kind: str,
    target_zone: str,
    seed: int = 42,
) -> pd.DataFrame:
    """Fit ML model strictly on causal data prior to target_zone (with purge gap) and predict across target_zone."""
    feats = _build_tabular_features(evaluator.feature_panel)
    feat_names = sorted(feats.keys())
    all_dates = evaluator.all_dates
    horizon = evaluator.horizon_days

    zone_bars = evaluator.zone_guard._slice_zone(evaluator.snapshot.bars, target_zone)
    eval_dates = pd.DatetimeIndex(np.sort(zone_bars["date"].unique()))
    if len(eval_dates) == 0:
        return pd.DataFrame()

    first_eval_date = eval_dates[0]
    eligible_train_dates = all_dates[all_dates < first_eval_date]
    # Strictly purge `horizon + 1` trading days before first_eval_date so forward returns never leak into target_zone
    if len(eligible_train_dates) > horizon + 5:
        train_dates = eligible_train_dates[: -(horizon + 1)]
    else:
        train_dates = eligible_train_dates

    fwd_ret = evaluator.forward_return_wide
    symbols = list(evaluator.feature_panel["close"].columns)

    def _stack_xy(dts: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
        xs = [feats[f].reindex(index=dts, columns=symbols).to_numpy(dtype=float).ravel() for f in feat_names]
        x_mat = np.column_stack(xs)
        y_vec = fwd_ret.reindex(index=dts, columns=symbols).to_numpy(dtype=float).ravel()
        valid = np.isfinite(x_mat).all(axis=1) & np.isfinite(y_vec)
        return x_mat[valid], y_vec[valid]

    x_train, y_train = _stack_xy(train_dates)
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)

    if model_kind == "ridge":
        model: Any = Ridge(alpha=50.0, random_state=seed)
        model.fit(x_train_s, y_train)
    elif model_kind == "gbdt":
        try:
            import lightgbm as lgb  # type: ignore[import-not-found]

            model = lgb.LGBMRegressor(
                n_estimators=80,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=seed,
                verbose=-1,
            )
        except Exception:
            model = HistGradientBoostingRegressor(
                max_iter=80,
                max_depth=4,
                learning_rate=0.05,
                random_state=seed,
            )
        model.fit(x_train_s, y_train)
    elif model_kind == "temporal_gru":
        # Lightweight gated temporal recurrence over 5 lag steps -> Ridge readout (deterministic & CPU-fast)
        rng = np.random.default_rng(seed)
        d_in = len(feat_names)
        d_hid = 16
        w_z = rng.normal(0.0, 0.25 / np.sqrt(d_in), size=(d_in, d_hid))
        u_z = rng.normal(0.0, 0.20 / np.sqrt(d_hid), size=(d_hid, d_hid))
        w_h = rng.normal(0.0, 0.25 / np.sqrt(d_in), size=(d_in, d_hid))
        u_h = rng.normal(0.0, 0.20 / np.sqrt(d_hid), size=(d_hid, d_hid))

        def _encode_gru_states(dts_seq: pd.DatetimeIndex) -> dict[pd.Timestamp, np.ndarray]:
            h = np.zeros((len(symbols), d_hid), dtype=float)
            states: dict[pd.Timestamp, np.ndarray] = {}
            for dt in dts_seq:
                x_t = np.column_stack(
                    [feats[f].loc[dt, symbols].to_numpy(dtype=float) for f in feat_names]
                )
                x_t = np.clip(scaler.transform(np.nan_to_num(x_t, nan=0.0)), -5.0, 5.0)
                z_gate = 1.0 / (1.0 + np.exp(-(x_t @ w_z + h @ u_z)))
                cand = np.tanh(x_t @ w_h + (z_gate * h) @ u_h)
                h = (1.0 - z_gate) * h + z_gate * cand
                states[dt] = h.copy()
            return states

        full_seq_dates = all_dates[all_dates <= eval_dates[-1]]
        all_states = _encode_gru_states(full_seq_dates)
        h_tr = np.vstack([all_states[dt] for dt in train_dates])
        y_tr_all = fwd_ret.reindex(index=train_dates, columns=symbols).to_numpy(dtype=float).ravel()
        valid_tr = np.isfinite(y_tr_all)
        readout = Ridge(alpha=25.0, random_state=seed)
        readout.fit(h_tr[valid_tr], y_tr_all[valid_tr])

        pred_rows = []
        for dt in eval_dates:
            pred_rows.append(readout.predict(all_states[dt]))
        return pd.DataFrame(np.vstack(pred_rows), index=eval_dates, columns=symbols)
    else:
        raise ValueError(f"Unknown ML baseline '{model_kind}'")

    pred_rows = []
    for dt in eval_dates:
        x_dt = np.column_stack([feats[f].loc[dt, symbols].to_numpy(dtype=float) for f in feat_names])
        x_dt_s = scaler.transform(np.nan_to_num(x_dt, nan=0.0))
        pred_rows.append(model.predict(x_dt_s))
    return pd.DataFrame(np.vstack(pred_rows), index=eval_dates, columns=symbols)


def run_all_baselines(
    evaluator: SealedEvaluator,
    zone: str = "validation",
    run_id: str = "baselines",
    score_sealed_test: bool = False,
) -> dict[str, EvaluationOutcome]:
    """Evaluate all 7 mandatory baselines on `validation` (or once on `sealed_test` when explicitly requested)."""
    panel = evaluator.feature_panel
    close = panel["close"]
    ret1 = panel["return_1d"]
    results: dict[str, EvaluationOutcome] = {}

    # 1. Buy-and-Hold VN-Index benchmark
    results["vnindex_buy_hold"] = evaluator.evaluate_benchmark_buy_hold(zone=zone)

    # 2. Equal-Weight Universe
    eq_sig = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    results["equal_weight"] = evaluator.evaluate_signal_matrix(
        eq_sig, zone=zone, candidate_id="baseline_equal_weight", run_id=run_id, allow_sealed=score_sealed_test
    )

    # 3. Momentum 20d risk-adjusted
    mom20 = close.pct_change(20, fill_method=None) / (ret1.rolling(20, min_periods=5).std() + 1e-4)
    results["momentum_20d"] = evaluator.evaluate_signal_matrix(
        mom20, zone=zone, candidate_id="baseline_momentum_20d", run_id=run_id, allow_sealed=score_sealed_test
    )

    # 4. Short-Term Reversal 5d
    rev5 = -close.pct_change(5, fill_method=None)
    results["reversal_5d"] = evaluator.evaluate_signal_matrix(
        rev5, zone=zone, candidate_id="baseline_reversal_5d", run_id=run_id, allow_sealed=score_sealed_test
    )

    # 5-7. Purged Walk-Forward ML / Deep Sequence Baselines
    for ml_name in ("ridge", "gbdt", "temporal_gru"):
        pred_wide = _fit_predict_walk_forward_ml(evaluator, ml_name, target_zone=zone, seed=42)
        results[ml_name] = evaluator.evaluate_signal_matrix(
            pred_wide,
            zone=zone,
            candidate_id=f"baseline_{ml_name}_{zone}",
            run_id=run_id,
            allow_sealed=score_sealed_test,
        )

    return results
