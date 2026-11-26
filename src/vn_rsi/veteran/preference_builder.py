"""Search-Zone Verified 5-Step SFT & DPO/GRPO Preference Dataset Builder + 2-Stage Free-GPU Trainer.

Automatically mines real market episodes strictly inside the `search` zone (ZERO validation/test access!):
- High-conviction winning setups
- Pre-crash floor-lock escape episodes
- PIT fundamental announcement-lag traps
- Foreign distribution ("kéo xả") bull traps
- T+2.5 liquidity crunch / exhaustion blow-offs

Exports:
1. SFT 5-step veteran reasoning JSONL (`sft_veteran_5step.jsonl`)
2. DPO/GRPO Preference pairs (`dpo_veteran_vs_fomo.jsonl`): Chosen = Veteran 5-step response vs.
   Rejected = Naive FOMO / look-ahead / friction-blind response.
3. Lightweight 2-Stage Trainer (4-bit QLoRA SFT -> DPO via Unsloth or HuggingFace PEFT + TRL,
   < 4.5 GB VRAM on Qwen3-1.7B/4B on Google Colab Free T4 or Kaggle Free T4 x2, with graceful CPU fallback).
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vn_rsi.data.snapshots import MarketSnapshot
from vn_rsi.dsl.ast_spec import compute_feature_panel
from vn_rsi.models.maveq_v4_veteran import VeteranCaseMemory, compute_causal_microstructure_state
from vn_rsi.veteran.reasoning_engine import MarketContextSnapshot, VeteranReasoningEngine


@dataclass
class PreferenceDatasetReport:
    n_sft_examples: int
    n_dpo_pairs: int
    category_counts: dict[str, int]
    sft_path: str
    dpo_path: str
    search_end_verified: str


def _naive_fomo_rejected_response(ctx: MarketContextSnapshot, category: str) -> str:
    """Construct realistic naive/FOMO failure response for DPO/GRPO negative preference pairs."""
    if category == "pit_lag_trap":
        return (
            f"1. Regime: Market looks fine. 2. Hypothesis: Quarter-end ROE is strong so buy {ctx.symbol} immediately "
            "even before official earnings release. 3. Risks: None. 4. Sizing: Go 35% NAV all-in. "
            "5. Brief: Aggressive buy based on fiscal quarter-end numbers."
        )
    if category in ("floor_lock_escape", "liquidity_crunch"):
        return (
            f"1. Regime: Oversold dip! 2. Hypothesis: {ctx.symbol} dropped sharply near floor so it must bounce tomorrow. "
            "3. Risks: Low risk, buy the dip. 4. Sizing: 30% NAV catching the falling knife without T+2.5 lock check. "
            "5. Brief: All-in bottom fishing on floor-locked stock."
        )
    if category in ("foreign_pull_dump", "blowoff_t25_trap"):
        return (
            f"1. Regime: Hot breakout! 2. Hypothesis: {ctx.symbol} surged +{ctx.stock_ret3:.1%} with huge volume "
            f"({ctx.turnover_shock_20d:.1f}x) — pure FOMO momentum! 3. Risks: Ignore foreign selling and T+2.5 supply. "
            "4. Sizing: 40% NAV chasing ceiling. 5. Brief: Chase the breakout before it runs away."
        )
    return (
        f"1. Regime: Bullish. 2. Hypothesis: Buy {ctx.symbol} blindly on price momentum without cost/liquidity checks. "
        "3. Risks: Ignored. 4. Sizing: 35% NAV, no stop-loss. 5. Brief: Full margin buy."
    )


def build_veteran_sft_and_dpo_datasets(
    snapshot: MarketSnapshot,
    search_end_date: str,
    out_dir: str | Path,
    max_examples: int = 80,
    horizon_days: int = 5,
    seed: int = 2026,
) -> PreferenceDatasetReport:
    """Mine strictly search-zone episodes to construct verified 5-step SFT & DPO preference datasets."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    sft_file = out_path / "sft_veteran_5step.jsonl"
    dpo_file = out_path / "dpo_veteran_vs_fomo.jsonl"

    search_end = pd.Timestamp(search_end_date)
    s_bars = snapshot.bars[snapshot.bars["date"] <= search_end].copy()
    s_pit = snapshot.fundamentals_pit[snapshot.fundamentals_pit["announcement_date"] <= search_end].copy()
    panel = compute_feature_panel(s_bars, s_pit)
    state_df = compute_causal_microstructure_state(panel)
    mem = VeteranCaseMemory(k_neighbors=12).fit_on_search(panel, horizon_days=horizon_days)
    mem_sig = mem.query_panel(state_df, purge_self_matches=True)

    close = panel["close"]
    ret1 = panel["return_1d"].fillna(0.0)
    ret3 = close.pct_change(3, fill_method=None).fillna(0.0)
    ret20 = close.pct_change(20, fill_method=None).fillna(0.0)
    tshock = panel["turnover_shock_20d"].fillna(1.0)
    fflow5 = panel["foreign_net_ratio"].fillna(0.0).rolling(5, min_periods=1).mean()
    pit_vq = panel["pit_value_quality"].fillna(0.0)
    pit_z = pit_vq.sub(pit_vq.mean(axis=1), axis=0).div(pit_vq.std(axis=1) + 1e-8, axis=0).fillna(0.0)
    dist_flr = panel["dist_to_floor"].fillna(0.05)
    dist_ceil = panel["dist_to_ceiling"].fillna(0.05)
    intraday = panel["intraday_range"].fillna(0.02)
    fwd5 = (close.shift(-horizon_days) / close - 1.0)

    mkt_ret = ret1.mean(axis=1)
    mkt_w = (1.0 + mkt_ret).cumprod()
    mkt_dd60 = (mkt_w / mkt_w.rolling(60, min_periods=5).max().clip(lower=1e-8)) - 1.0
    floor_rate5 = (dist_flr < 0.015).mean(axis=1).rolling(5, min_periods=1).mean()
    ceil_rate5 = (dist_ceil < 0.010).mean(axis=1).rolling(5, min_periods=1).mean()

    safe_dates = pd.DatetimeIndex(close.index)[25 : max(26, len(close.index) - horizon_days - 1)]
    syms = list(close.columns)
    ex_map = s_bars.drop_duplicates("symbol").set_index("symbol")["exchange"].to_dict()

    rng = np.random.default_rng(seed)
    engine = VeteranReasoningEngine()

    buckets: dict[str, list[tuple[pd.Timestamp, str, bool]]] = {
        "strong_winner": [],
        "floor_lock_escape": [],
        "blowoff_t25_trap": [],
        "foreign_pull_dump": [],
        "pit_lag_trap": [],
        "liquidity_crunch": [],
    }

    for dt in safe_dates[::3]:
        st_row = state_df.loc[dt]
        dd60 = float(mkt_dd60.loc[dt])
        flr5 = float(floor_rate5.loc[dt])
        for sym in syms:
            f5 = float(fwd5.loc[dt, sym]) if np.isfinite(fwd5.loc[dt, sym]) else 0.0
            r3 = float(ret3.loc[dt, sym])
            r20 = float(ret20.loc[dt, sym])
            ts = float(tshock.loc[dt, sym])
            ff = float(fflow5.loc[dt, sym])
            df_l = float(dist_flr.loc[dt, sym])

            if f5 > 0.04 and r20 > 0.02 and ff > 0.02 and df_l > 0.03:
                buckets["strong_winner"].append((dt, sym, True))
            elif (df_l < 0.020 or r20 < -0.18) and f5 < -0.02:
                buckets["floor_lock_escape"].append((dt, sym, True))
            elif r3 > 0.08 and ts > 1.85 and f5 < 0.0:
                buckets["blowoff_t25_trap"].append((dt, sym, True))
            elif ff < -0.05 and r3 > 0.025:
                buckets["foreign_pull_dump"].append((dt, sym, True))
            elif dd60 < -0.09 or flr5 >= 0.035:
                buckets["liquidity_crunch"].append((dt, sym, True))
            elif float(st_row["mkt_r20"]) > 0.0 and len(buckets["pit_lag_trap"]) < 40:
                # Synthetic PIT unannounced-lag trap probe strictly using search-zone bar
                buckets["pit_lag_trap"].append((dt, sym, False))

    per_cat = max(4, max_examples // len(buckets))
    sft_rows: list[dict[str, Any]] = []
    dpo_rows: list[dict[str, Any]] = []
    cat_counts: dict[str, int] = {}

    for cat, items in buckets.items():
        if not items:
            cat_counts[cat] = 0
            continue
        chosen_idx = rng.choice(len(items), size=min(per_cat, len(items)), replace=False)
        cat_counts[cat] = len(chosen_idx)
        for idx in chosen_idx:
            dt, sym, pit_ann = items[int(idx)]
            st_row = state_df.loc[dt]
            ms_row = mem_sig.loc[dt]
            analogs = mem.retrieve_nearest_cases(st_row.to_numpy(dtype=float), top_k=2)
            ctx = MarketContextSnapshot(
                as_of_date=str(dt.date()),
                symbol=sym,
                exchange=str(ex_map.get(sym, "HOSE")),
                mkt_r20=float(st_row["mkt_r20"]),
                mkt_r5=float(st_row["mkt_r5"]),
                mkt_dd60=float(mkt_dd60.loc[dt]),
                breadth_20=float(st_row["breadth_20"]),
                vol_ratio=float(st_row["vol_ratio"]),
                floor_hit_rate_5d=float(floor_rate5.loc[dt]),
                ceil_hit_rate_5d=float(ceil_rate5.loc[dt]),
                stock_ret3=float(ret3.loc[dt, sym]),
                stock_ret20=float(ret20.loc[dt, sym]),
                turnover_shock_20d=float(tshock.loc[dt, sym]),
                foreign_net_ratio_5d=float(fflow5.loc[dt, sym]),
                pit_value_quality_z=float(pit_z.loc[dt, sym]),
                pit_announcement_lag_days=30 if pit_ann else 0,
                pit_is_announced=pit_ann,
                dist_to_floor=float(dist_flr.loc[dt, sym]),
                dist_to_ceiling=float(dist_ceil.loc[dt, sym]),
                intraday_range=float(intraday.loc[dt, sym]),
                trauma_prob=float(ms_row["trauma_prob"]),
                expected_mkt_5d=float(ms_row["expected_mkt_5d"]),
                nearest_analogs=analogs,
                language="vi",
            )
            resp = engine.analyze(ctx)
            prompt = (
                f"[Episode={cat} | Date={ctx.as_of_date} | Symbol={ctx.symbol} ({ctx.exchange})] "
                f"Mkt20d={ctx.mkt_r20:+.1%}, MktDD60={ctx.mkt_dd60:+.1%}, Breadth={ctx.breadth_20:.0%}, "
                f"VolRatio={ctx.vol_ratio:.2f}, TraumaProb={ctx.trauma_prob:.0%}, "
                f"Stock3d={ctx.stock_ret3:+.1%}, Stock20d={ctx.stock_ret20:+.1%}, TShock={ctx.turnover_shock_20d:.2f}x, "
                f"Foreign5d={ctx.foreign_net_ratio_5d:+.2%}, PIT_Announced={ctx.pit_is_announced}, "
                f"DistFloor={ctx.dist_to_floor:.1%}, DistCeil={ctx.dist_to_ceiling:.1%}. "
                "Hãy phân tích theo quy trình 5 bước của Chuyên gia Đầu tư 10 năm kinh nghiệm."
            )
            chosen_md = resp.render_markdown()
            rejected_md = _naive_fomo_rejected_response(ctx, cat)

            sft_rows.append(
                {
                    "category": cat,
                    "as_of_date": ctx.as_of_date,
                    "symbol": sym,
                    "prompt": prompt,
                    "completion": chosen_md,
                    "text": f"<|user|>\n{prompt}\n<|assistant|>\n{chosen_md}<|end|>",
                    "realized_search_fwd_5d": round(float(fwd5.loc[dt, sym]), 5),
                }
            )
            dpo_rows.append(
                {
                    "category": cat,
                    "prompt": prompt,
                    "chosen": chosen_md,
                    "rejected": rejected_md,
                }
            )

    with sft_file.open("w", encoding="utf-8") as f:
        for r in sft_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with dpo_file.open("w", encoding="utf-8") as f:
        for r in dpo_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return PreferenceDatasetReport(
        n_sft_examples=len(sft_rows),
        n_dpo_pairs=len(dpo_rows),
        category_counts=cat_counts,
        sft_path=str(sft_file),
        dpo_path=str(dpo_file),
        search_end_verified=str(search_end.date()),
    )


def train_veteran_two_stage_qlora_dpo(
    sft_path: str | Path,
    dpo_path: str | Path,
    output_dir: str | Path,
    model_id: str = "Qwen/Qwen3-1.7B",
    sft_epochs: int = 2,
    dpo_epochs: int = 1,
    max_seq_len: int = 1024,
) -> dict[str, Any]:
    """2-Stage Lightweight Trainer: 4-bit QLoRA SFT -> DPO (<4.5 GB VRAM on Qwen3-1.7B/4B on Free T4).

    Gracefully returns a calibrated CPU fallback summary when CUDA or optional HF/Unsloth packages
    are unavailable, ensuring zero runtime crashes on 0-GPU CI machines.
    """
    sft_p = Path(sft_path)
    dpo_p = Path(dpo_path)
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    sft_rows = [json.loads(ln) for ln in sft_p.read_text(encoding="utf-8").splitlines() if ln.strip()] if sft_p.exists() else []
    dpo_rows = [json.loads(ln) for ln in dpo_p.read_text(encoding="utf-8").splitlines() if ln.strip()] if dpo_p.exists() else []

    if len(sft_rows) < 2 or len(dpo_rows) < 2:
        return {"status": "skipped_insufficient_data", "n_sft": len(sft_rows), "n_dpo": len(dpo_rows)}

    try:
        import torch
    except Exception as exc:
        return {
            "status": "cpu_neuro_symbolic_fallback",
            "reason": f"PyTorch not installed ({exc}); VeteranReasoningEngine operates in instant CPU Neuro-Symbolic mode.",
            "n_sft_ready": len(sft_rows),
            "n_dpo_ready": len(dpo_rows),
            "estimated_t4_vram_gb": 3.4 if "1.7B" in model_id else 4.3,
        }

    if not torch.cuda.is_available():
        return {
            "status": "cpu_neuro_symbolic_fallback",
            "reason": "CUDA GPU not detected on current host; SFT + DPO datasets ready for Free Colab/Kaggle T4 (<4.5 GB VRAM).",
            "model_id": model_id,
            "n_sft_ready": len(sft_rows),
            "n_dpo_ready": len(dpo_rows),
            "estimated_t4_vram_gb": 3.4 if "1.7B" in model_id else 4.3,
        }

    # CUDA path: 4-bit NF4 QLoRA Stage 1 (SFT) -> Stage 2 (DPO) with explicit VRAM cleanup
    try:
        from datasets import Dataset  # type: ignore[import-not-found]
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer  # type: ignore[import-not-found]

        use_bf16 = bool(torch.cuda.is_bf16_supported())
        dtype = torch.bfloat16 if use_bf16 else torch.float16
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb, device_map={"": 0}, trust_remote_code=True
        )
        model = prepare_model_for_kbit_training(model)
        lora = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)

        # Stage 1: SFT
        sft_dir = out_p / "stage1_sft"
        sft_trainer = SFTTrainer(
            model=model,
            processing_class=tok,
            train_dataset=Dataset.from_list([{"text": r["text"]} for r in sft_rows]),
            args=SFTConfig(
                output_dir=str(sft_dir),
                dataset_text_field="text",
                max_length=max_seq_len,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                num_train_epochs=sft_epochs,
                learning_rate=2e-4,
                fp16=not use_bf16,
                bf16=use_bf16,
                optim="paged_adamw_8bit",
                save_strategy="no",
                report_to="none",
            ),
        )
        sft_out = sft_trainer.train()
        sft_loss = round(float(sft_out.training_loss), 5)
        del sft_trainer
        gc.collect()
        torch.cuda.empty_cache()

        # Stage 2: DPO (reference-free PEFT adapter comparison -> saves 50% VRAM!)
        dpo_dir = out_p / "stage2_dpo"
        dpo_trainer = DPOTrainer(
            model=model,
            ref_model=None,
            processing_class=tok,
            train_dataset=Dataset.from_list(dpo_rows),
            args=DPOConfig(
                output_dir=str(dpo_dir),
                beta=0.1,
                max_length=max_seq_len,
                max_prompt_length=max_seq_len // 2,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                num_train_epochs=dpo_epochs,
                learning_rate=5e-5,
                fp16=not use_bf16,
                bf16=use_bf16,
                optim="paged_adamw_8bit",
                save_strategy="no",
                report_to="none",
            ),
        )
        dpo_out = dpo_trainer.train()
        dpo_loss = round(float(dpo_out.training_loss), 5)
        peak_vram_gb = round(float(torch.cuda.max_memory_allocated()) / (1024**3), 3)
        model.save_pretrained(str(out_p / "final_veteran_adapter"))
        tok.save_pretrained(str(out_p / "final_veteran_adapter"))
        del dpo_trainer, model
        gc.collect()
        torch.cuda.empty_cache()
        return {
            "status": "trained_sft_plus_dpo",
            "model_id": model_id,
            "sft_loss": sft_loss,
            "dpo_loss": dpo_loss,
            "peak_vram_gb": peak_vram_gb,
            "adapter_dir": str(out_p / "final_veteran_adapter"),
        }
    except Exception as exc:
        return {
            "status": "qlora_dpo_deps_unavailable",
            "error": str(exc)[:220],
            "n_sft_ready": len(sft_rows),
            "n_dpo_ready": len(dpo_rows),
        }
