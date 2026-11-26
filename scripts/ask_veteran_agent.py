"""Interactive / One-Shot CLI to Ask the 10-Year Veteran MAVE-Q v4 Agent for Natural-Language Market Analysis.

Loads saved MAVE-Q v4 checkpoints (champion expressions, VeteranCaseMemory, optional QLoRA/DPO adapter)
and answers natural-language strategy/ticker queries on the latest available local snapshot bar
(or custom scenario parameters) using the 5-Step Veteran CIO Protocol.

Research simulation & educational decision-analysis ONLY — NOT investment advice or live trading.

Usage:
    python scripts/ask_veteran_agent.py --symbol FPT --question "Có nên mua FPT giai đoạn này không?"
    python scripts/ask_veteran_agent.py --symbol NVL_STRESS --lang en
    python scripts/ask_veteran_agent.py --symbol VCB --checkpoint-dir artifacts/checkpoints
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vn_rsi.data import SnapshotManager  # noqa: E402
from vn_rsi.dsl.ast_spec import compute_feature_panel  # noqa: E402
from vn_rsi.models.maveq_v4_veteran import (  # noqa: E402
    VeteranCaseMemory,
    compute_causal_microstructure_state,
)
from vn_rsi.utils import load_yaml  # noqa: E402
from vn_rsi.veteran import MarketContextSnapshot, VeteranReasoningEngine  # noqa: E402


def _build_context_from_snapshot(
    symbol: str,
    as_of: str | None,
    question: str,
    lang: str,
    data_cfg: dict[str, Any],
    sandbox_cfg: dict[str, Any],
) -> MarketContextSnapshot:
    snap = SnapshotManager(REPO_ROOT / str(data_cfg.get("snapshot_root", "data/snapshots"))).load_snapshot(
        str(data_cfg["snapshot_id"])
    )
    # Default to end of search/validation zone so CLI never leaks future data
    default_date = str(sandbox_cfg["zones"]["validation"]["end"])
    target_dt = pd.Timestamp(as_of or default_date)

    bars_upto = snap.bars[snap.bars["date"] <= target_dt].copy()
    pit_upto = snap.fundamentals_pit[snap.fundamentals_pit["announcement_date"] <= target_dt].copy()
    if bars_upto.empty:
        raise ValueError(f"No bars found on or before {target_dt.date()}")

    panel = compute_feature_panel(bars_upto, pit_upto)
    close = panel["close"]
    if symbol not in close.columns:
        avail = sorted(close.columns.tolist())[:10]
        raise ValueError(f"Symbol '{symbol}' not found in snapshot. Sample symbols: {avail}")

    state_df = compute_causal_microstructure_state(panel)
    # Fit VeteranCaseMemory strictly on search zone
    search_end = pd.Timestamp(sandbox_cfg["zones"]["search"]["end"])
    s_bars = snap.bars[snap.bars["date"] <= search_end].copy()
    s_pit = snap.fundamentals_pit[snap.fundamentals_pit["announcement_date"] <= search_end].copy()
    mem = VeteranCaseMemory(k_neighbors=12).fit_on_search(compute_feature_panel(s_bars, s_pit))
    mem_sig = mem.query_panel(state_df.iloc[[-1]], purge_self_matches=False)

    dt = close.index[-1]
    st_row = state_df.iloc[-1]
    analogs = mem.retrieve_nearest_cases(st_row.to_numpy(dtype=float), top_k=3)

    ret1 = panel["return_1d"].fillna(0.0)
    mkt_ret = ret1.mean(axis=1)
    mkt_w = (1.0 + mkt_ret).cumprod()
    mkt_dd60 = float((mkt_w.iloc[-1] / max(float(mkt_w.tail(60).max()), 1e-8)) - 1.0)
    dist_flr = panel["dist_to_floor"].fillna(0.05)
    dist_ceil = panel["dist_to_ceiling"].fillna(0.05)
    floor_rate5 = float((dist_flr < 0.015).mean(axis=1).tail(5).mean())
    ceil_rate5 = float((dist_ceil < 0.010).mean(axis=1).tail(5).mean())

    ret3 = float(close[symbol].pct_change(3, fill_method=None).iloc[-1]) if len(close) > 3 else 0.0
    ret20 = float(close[symbol].pct_change(20, fill_method=None).iloc[-1]) if len(close) > 20 else 0.0
    tshock = float(panel["turnover_shock_20d"][symbol].iloc[-1])
    ff5 = float(panel["foreign_net_ratio"][symbol].tail(5).mean())
    pit_row = panel["pit_value_quality"].iloc[-1]
    pit_z = float((pit_row[symbol] - pit_row.mean()) / (pit_row.std() + 1e-8))
    ex = str(bars_upto.loc[bars_upto["symbol"] == symbol, "exchange"].iloc[-1])

    return MarketContextSnapshot(
        as_of_date=str(dt.date()),
        symbol=symbol,
        exchange=ex,
        mkt_r20=float(st_row["mkt_r20"]),
        mkt_r5=float(st_row["mkt_r5"]),
        mkt_dd60=mkt_dd60,
        breadth_20=float(st_row["breadth_20"]),
        vol_ratio=float(st_row["vol_ratio"]),
        floor_hit_rate_5d=floor_rate5,
        ceil_hit_rate_5d=ceil_rate5,
        stock_ret3=0.0 if not np.isfinite(ret3) else ret3,
        stock_ret20=0.0 if not np.isfinite(ret20) else ret20,
        turnover_shock_20d=1.0 if not np.isfinite(tshock) else tshock,
        foreign_net_ratio_5d=0.0 if not np.isfinite(ff5) else ff5,
        pit_value_quality_z=0.0 if not np.isfinite(pit_z) else pit_z,
        pit_announcement_lag_days=30,
        pit_is_announced=True,
        dist_to_floor=float(dist_flr[symbol].iloc[-1]),
        dist_to_ceiling=float(dist_ceil[symbol].iloc[-1]),
        intraday_range=float(panel["intraday_range"][symbol].iloc[-1]),
        trauma_prob=float(mem_sig["trauma_prob"].iloc[-1]),
        expected_mkt_5d=float(mem_sig["expected_mkt_5d"].iloc[-1]),
        nearest_analogs=analogs,
        user_question=question,
        language=lang,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask MAVE-Q v4 10-Year Veteran Agent (Natural-Language CIO Brief)")
    parser.add_argument("--symbol", default="FPT", help="Ticker symbol in snapshot (e.g. FPT, VCB, HPG, NVL_STRESS)")
    parser.add_argument("--as-of", default=None, help="Causal analysis date YYYY-MM-DD (defaults to validation end)")
    parser.add_argument("--question", default="Đánh giá cơ hội và rủi ro T+2.5 cho mã này?", help="Natural-language query")
    parser.add_argument("--lang", default="vi", choices=["vi", "en"], help="Output language (vi | en)")
    parser.add_argument("--checkpoint-dir", default="artifacts/checkpoints", help="Directory containing MAVE-Q v4 checkpoints")
    parser.add_argument("--slm-base-model", default=None, help="Optional HuggingFace base SLM ID if QLoRA/DPO adapter exists")
    parser.add_argument("--json", action="store_true", help="Emit structured JSON instead of Markdown brief")
    args = parser.parse_args()

    data_cfg = load_yaml(REPO_ROOT / "configs/data.yaml")
    sandbox_cfg = load_yaml(REPO_ROOT / "configs/sandbox.yaml")
    ckpt_dir = REPO_ROOT / args.checkpoint_dir

    # Load optional saved champion bundle metadata for transparency
    bundle_meta: dict[str, Any] = {}
    for cand in sorted(ckpt_dir.glob("maveq_v4_bundle_*.json")):
        bundle_meta = json.loads(cand.read_text(encoding="utf-8"))
        break

    adapter_dir = ckpt_dir / "veteran_slm_qlora_dpo" / "final_veteran_adapter"
    engine = VeteranReasoningEngine(
        adapter_dir=adapter_dir if adapter_dir.exists() else None,
        base_model_id=args.slm_base_model,
    )
    ctx = _build_context_from_snapshot(
        symbol=args.symbol.upper(),
        as_of=args.as_of,
        question=args.question,
        lang=args.lang,
        data_cfg=data_cfg,
        sandbox_cfg=sandbox_cfg,
    )
    resp = engine.analyze(ctx)

    if args.json:
        print(json.dumps({"question": args.question, "bundle_loaded": bool(bundle_meta), "analysis": resp.to_dict()}, indent=2, ensure_ascii=False))
        return

    print(f"\n[MAVE-Q v4 VetQuant-RSI | Query]: {args.question}")
    if bundle_meta:
        print(
            f"[Loaded v4 Bundle]: preset={bundle_meta.get('preset')} | controller={bundle_meta.get('selected_controller')} | "
            f"val_sharpe={bundle_meta.get('validation_metrics', {}).get('net_sharpe', 'N/A')}"
        )
    print(resp.render_markdown())
    print("Note: Academic research & simulation only — NOT investment advice.")


if __name__ == "__main__":
    main()
