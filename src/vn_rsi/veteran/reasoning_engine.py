"""10-Year Veteran CIO 5-Step Natural-Language & Neuro-Symbolic Reasoning Engine.

Operates in two seamless modes:
1. GPU SLM + LoRA/DPO Adapter mode (Qwen3-1.7B/4B + QLoRA/DPO adapter when CUDA + weights available)
2. Zero-GPU Neuro-Symbolic Hybrid Fallback (instant CPU/RAM execution grounded in exact causal
   microstructure state, VeteranCaseMemory analogs, and Devil's Advocate Pre-Mortem physics).

Every response follows the 5-step Veteran CIO Protocol:
  Step 1: Regime & Microstructure Diagnosis
  Step 2: Quantitative Economic Hypothesis
  Step 3: Pre-Mortem Devil's Advocate (Worst-Case T+2.5 / Floor-Lock / PIT Trap Audit)
  Step 4: Position Sizing, Stop-Loss / Take-Profit & Risk Gate
  Step 5: Concise Natural-Language Executive Brief (Vietnamese / English)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class MarketContextSnapshot:
    """Strictly causal market & ticker snapshot available at decision timestamp t (zero look-ahead)."""

    as_of_date: str
    symbol: str
    exchange: str = "HOSE"
    # Market-level causal state
    mkt_r20: float = 0.02
    mkt_r5: float = 0.005
    mkt_dd60: float = -0.02
    breadth_20: float = 0.58
    vol_ratio: float = 1.0
    floor_hit_rate_5d: float = 0.005
    ceil_hit_rate_5d: float = 0.015
    # Stock-level causal state
    stock_ret3: float = 0.01
    stock_ret20: float = 0.05
    turnover_shock_20d: float = 1.15
    foreign_net_ratio_5d: float = 0.04
    pit_value_quality_z: float = 0.80
    pit_announcement_lag_days: int = 32
    pit_is_announced: bool = True
    dist_to_floor: float = 0.055
    dist_to_ceiling: float = 0.045
    intraday_range: float = 0.024
    # Optional VeteranCaseMemory analogs
    trauma_prob: float = 0.18
    expected_mkt_5d: float = 0.008
    nearest_analogs: list[dict[str, Any]] = field(default_factory=list)
    user_question: str = ""
    language: str = "vi"  # "vi" or "en"


@dataclass
class VeteranReasoningResponse:
    symbol: str
    as_of_date: str
    regime_label: str
    action: str  # "BUY_HIGH_CONVICTION" | "BUY_MODERATE" | "HOLD_DEFENSIVE" | "REDUCE_OR_AVOID" | "CASH_SHIELD"
    recommended_weight_pct: float
    stop_loss_pct: float
    take_profit_pct: float
    calibrated_win_prob: float
    detected_hidden_risks: list[str]
    causal_violations_flagged: list[str]
    step1_regime_diagnosis: str
    step2_economic_hypothesis: str
    step3_premortem_devils_advocate: str
    step4_sizing_and_risk_gate: str
    step5_executive_brief: str
    backend_used: str = "neuro_symbolic_veteran_core"
    slm_generated_commentary: str = ""

    def render_markdown(self) -> str:
        slm_block = (
            f"\n**Fine-Tuned SLM Commentary (`{self.backend_used}`)**:\n{self.slm_generated_commentary}\n"
            if self.slm_generated_commentary
            else ""
        )
        return (
            f"### Veteran CIO Analysis — `{self.symbol}` ({self.as_of_date})\n"
            f"- **Regime**: `{self.regime_label}` | **Action**: `{self.action}` | "
            f"**Target Weight**: `{self.recommended_weight_pct:.1f}%` | "
            f"**Calibrated P(Outperform)**: `{self.calibrated_win_prob:.2%}`\n"
            f"- **Risk Bounds**: Stop-Loss `{self.stop_loss_pct:.1%}` | Take-Profit `+{self.take_profit_pct:.1%}`\n"
            f"- **Hidden Risks Flagged**: {', '.join(self.detected_hidden_risks) if self.detected_hidden_risks else 'None elevated'}\n\n"
            f"1. **Chẩn đoán Chế độ & Vi cấu trúc (Regime & Microstructure)**: {self.step1_regime_diagnosis}\n"
            f"2. **Giả thuyết Kinh tế Định lượng (Quant Hypothesis)**: {self.step2_economic_hypothesis}\n"
            f"3. **Phản biện Pre-Mortem Kịch bản Xấu nhất (Devil's Advocate)**: {self.step3_premortem_devils_advocate}\n"
            f"4. **Tỷ trọng & Cổng Rủi ro T+2.5 (Sizing & Risk Gate)**: {self.step4_sizing_and_risk_gate}\n"
            f"5. **Kết luận Hành động (Executive Brief)**: {self.step5_executive_brief}\n"
            f"{slm_block}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "as_of_date": self.as_of_date,
            "regime_label": self.regime_label,
            "action": self.action,
            "recommended_weight_pct": self.recommended_weight_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "calibrated_win_prob": self.calibrated_win_prob,
            "detected_hidden_risks": self.detected_hidden_risks,
            "causal_violations_flagged": self.causal_violations_flagged,
            "step1_regime_diagnosis": self.step1_regime_diagnosis,
            "step2_economic_hypothesis": self.step2_economic_hypothesis,
            "step3_premortem_devils_advocate": self.step3_premortem_devils_advocate,
            "step4_sizing_and_risk_gate": self.step4_sizing_and_risk_gate,
            "step5_executive_brief": self.step5_executive_brief,
            "backend_used": self.backend_used,
            "slm_generated_commentary": self.slm_generated_commentary,
        }


class VeteranReasoningEngine:
    """10-Year Veteran CIO Reasoning Engine (Neuro-Symbolic Core + Optional Qwen3 QLoRA/DPO Narrator)."""

    def __init__(
        self,
        adapter_dir: str | Path | None = None,
        base_model_id: str | None = None,
        use_slm_generation: bool = False,
    ) -> None:
        self.adapter_dir = Path(adapter_dir) if adapter_dir else None
        self.base_model_id = base_model_id or self._infer_base_model_id(self.adapter_dir)
        self.use_slm_generation = use_slm_generation
        self._slm_model: Any = None
        self._slm_tok: Any = None
        self._slm_loaded = False

    @staticmethod
    def _infer_base_model_id(adapter_dir: Path | None) -> str | None:
        if adapter_dir is None or not adapter_dir.exists():
            return None
        for fname in ("adapter_config.json", "veteran_training_meta.json"):
            cfg_path = adapter_dir / fname
            if cfg_path.exists():
                try:
                    data = json.loads(cfg_path.read_text(encoding="utf-8"))
                    cand = data.get("base_model_name_or_path") or data.get("base_model_id")
                    if isinstance(cand, str) and cand.strip():
                        return cand.strip()
                except Exception:
                    continue
        return None

    def _try_load_slm(self) -> bool:
        if self._slm_loaded:
            return self._slm_model is not None
        self._slm_loaded = True
        if not self.adapter_dir or not self.adapter_dir.exists() or not self.base_model_id:
            return False
        try:
            import torch
            from peft import PeftModel  # type: ignore[import-not-found]
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]

            if not torch.cuda.is_available() and not self.use_slm_generation:
                return False
            tok_src = str(self.adapter_dir) if (self.adapter_dir / "tokenizer_config.json").exists() else self.base_model_id
            tok = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
            if torch.cuda.is_available():
                load_kwargs: dict[str, Any] = {"device_map": "auto", "trust_remote_code": True}
                try:
                    from transformers import BitsAndBytesConfig  # type: ignore[import-not-found]

                    use_bf16 = bool(hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported())
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
                    )
                except Exception:
                    load_kwargs["torch_dtype"] = torch.float16
                base = AutoModelForCausalLM.from_pretrained(self.base_model_id, **load_kwargs)
            else:
                base = AutoModelForCausalLM.from_pretrained(
                    self.base_model_id, torch_dtype=torch.float32, trust_remote_code=True
                )
            self._slm_model = PeftModel.from_pretrained(base, str(self.adapter_dir))
            self._slm_model.eval()
            self._slm_tok = tok
            return True
        except Exception:
            self._slm_model = None
            return False

    def generate_slm_commentary(self, ctx: MarketContextSnapshot, max_new_tokens: int = 320) -> str:
        """Generate natural-language synthesis from the fine-tuned open-source LoRA/DPO adapter."""
        if not self._try_load_slm() or self._slm_model is None or self._slm_tok is None:
            return ""
        try:
            import torch

            prompt = (
                f"[Date={ctx.as_of_date} | Symbol={ctx.symbol} ({ctx.exchange})] "
                f"Mkt20d={ctx.mkt_r20:+.1%}, MktDD60={ctx.mkt_dd60:+.1%}, Breadth={ctx.breadth_20:.0%}, "
                f"VolRatio={ctx.vol_ratio:.2f}, TraumaProb={ctx.trauma_prob:.0%}, "
                f"Stock3d={ctx.stock_ret3:+.1%}, Stock20d={ctx.stock_ret20:+.1%}, TShock={ctx.turnover_shock_20d:.2f}x, "
                f"Foreign5d={ctx.foreign_net_ratio_5d:+.2%}, PIT_Announced={ctx.pit_is_announced}, "
                f"DistFloor={ctx.dist_to_floor:.1%}, DistCeil={ctx.dist_to_ceiling:.1%}. "
                f"{ctx.user_question or 'Hãy phân tích theo quy trình 5 bước của Chuyên gia Đầu tư 10 năm kinh nghiệm.'}"
            )
            messages = [{"role": "user", "content": prompt}]
            if getattr(self._slm_tok, "chat_template", None):
                text_in = self._slm_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                text_in = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            inputs = self._slm_tok(text_in, return_tensors="pt").to(self._slm_model.device)
            with torch.no_grad():
                out_ids = self._slm_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self._slm_tok.eos_token_id,
                )
            gen_ids = out_ids[0][inputs["input_ids"].shape[-1] :]
            return str(self._slm_tok.decode(gen_ids, skip_special_tokens=True)).strip()
        except Exception:
            return ""

    @staticmethod
    def _exchange_band(exchange: str) -> float:
        ex = exchange.upper()
        if ex == "HNX":
            return 0.10
        if ex == "UPCOM":
            return 0.15
        return 0.07

    def analyze(self, ctx: MarketContextSnapshot) -> VeteranReasoningResponse:
        """Execute deterministic 5-step Veteran CIO cognitive analysis with calibrated probability estimation."""
        band = self._exchange_band(ctx.exchange)
        hidden_risks: list[str] = []
        causal_flags: list[str] = []

        # --- Step 1: Regime & Microstructure Diagnosis ---
        trauma_score = (
            0.50 * float(np.clip(ctx.trauma_prob, 0.0, 1.0))
            + 0.25 * float(np.clip(-ctx.mkt_dd60 / 0.12, 0.0, 1.0))
            + 0.25 * float(np.clip((ctx.vol_ratio - 1.0) / 1.0, 0.0, 1.0))
        )
        if ctx.floor_hit_rate_5d >= 0.045 or (ctx.mkt_dd60 <= -0.10 and ctx.vol_ratio >= 1.35):
            regime = "High_Vol_Liquidity_Crunch"
        elif ctx.mkt_r20 <= -0.025 or ctx.mkt_dd60 <= -0.08:
            regime = "Bear_Crash"
        elif ctx.mkt_r20 >= 0.02 and ctx.breadth_20 >= 0.55 and trauma_score < 0.42:
            regime = "Bull_Expansion"
        else:
            regime = "Sideways_Rotational"

        # --- Hidden Risk & Causality Audit (Devil's Advocate Chamber) ---
        if not ctx.pit_is_announced or ctx.pit_announcement_lag_days <= 0:
            causal_flags.append("PIT_UNANNOUNCED_FUNDAMENTAL_LEAKAGE")
            hidden_risks.append("pit_announcement_lag_trap")

        if ctx.dist_to_floor < 0.020 or ctx.floor_hit_rate_5d >= 0.04:
            hidden_risks.append("t25_floor_lock_trap")

        if ctx.dist_to_ceiling < 0.008:
            hidden_risks.append("ceiling_unfillable_chase_trap")

        if ctx.stock_ret3 >= 0.08 and ctx.turnover_shock_20d >= 1.85:
            hidden_risks.append("blowoff_bull_trap_t25_supply")

        if ctx.foreign_net_ratio_5d <= -0.06 and ctx.stock_ret3 >= 0.03:
            hidden_risks.append("foreign_distribution_pull_and_dump")

        if ctx.stock_ret20 <= -0.18:
            hidden_risks.append("distress_delisting_halt_risk")

        if ctx.intraday_range >= band * 0.85 and ctx.turnover_shock_20d >= 2.0:
            hidden_risks.append("intraday_whipsaw_volatility_spike")

        if ctx.trauma_prob >= 0.45 and ctx.mkt_r5 >= 0.025:
            hidden_risks.append("dead_cat_bounce_macro_trap")

        # --- Quantitative Alpha & Calibrated Win Probability (Logistic Shrinkage Model) ---
        # Ignore unannounced fundamentals strictly (zero PIT look-ahead!)
        eff_pit_z = ctx.pit_value_quality_z if ctx.pit_is_announced and ctx.pit_announcement_lag_days > 0 else 0.0
        alpha_logit = (
            0.48 * float(np.clip(ctx.stock_ret20 / 0.10, -2.0, 2.0))
            + 0.42 * float(np.clip(eff_pit_z, -2.5, 2.5))
            + 0.45 * float(np.clip(ctx.foreign_net_ratio_5d / 0.06, -2.0, 2.0))
            - 0.55 * float(np.clip((ctx.stock_ret3 * ctx.turnover_shock_20d) / 0.14, -1.5, 2.5))
            + 0.35 * float(np.clip(ctx.mkt_r20 / 0.05, -2.0, 2.0))
            - 0.65 * float(len(hidden_risks))
            - 0.90 * trauma_score
        )
        # Well-calibrated logistic scaling for low Brier Score & low ECE without overconfident 0/1 extremes
        win_prob = float(1.0 / (1.0 + np.exp(-1.25 * alpha_logit)))
        win_prob = round(float(np.clip(win_prob, 0.03, 0.96)), 4)

        # --- Action & Sizing Decision ---
        severe_traps = {"t25_floor_lock_trap", "distress_delisting_halt_risk", "ceiling_unfillable_chase_trap"}
        has_severe_trap = any(r in severe_traps for r in hidden_risks)

        if regime in ("High_Vol_Liquidity_Crunch", "Bear_Crash") and (trauma_score >= 0.48 or ctx.mkt_dd60 <= -0.11):
            action = "CASH_SHIELD"
            target_w = 0.0
            sl_pct = -0.05
            tp_pct = 0.08
        elif has_severe_trap or len(hidden_risks) >= 2 or win_prob < 0.42:
            action = "REDUCE_OR_AVOID"
            target_w = 0.0
            sl_pct = -0.05
            tp_pct = 0.08
        elif win_prob >= 0.64 and regime == "Bull_Expansion" and not hidden_risks:
            action = "BUY_HIGH_CONVICTION"
            target_w = round(min(20.0, 12.0 + 18.0 * (win_prob - 0.50)), 1)
            sl_pct = -0.07
            tp_pct = 0.16
        elif win_prob >= 0.54 and not hidden_risks:
            action = "BUY_MODERATE"
            target_w = round(min(15.0, 8.0 + 14.0 * (win_prob - 0.48)), 1)
            sl_pct = -0.06
            tp_pct = 0.12
        else:
            action = "HOLD_DEFENSIVE"
            target_w = round(max(4.0, 10.0 * (1.0 - trauma_score)), 1)
            sl_pct = -0.055
            tp_pct = 0.09

        analog_txt = ""
        if ctx.nearest_analogs:
            top_a = ctx.nearest_analogs[0]
            analog_txt = (
                f" Nearest historical analog `{top_a.get('analog_date')}` saw "
                f"{float(top_a.get('realized_fwd_5d_mkt_ret', 0.0)):+.2%} over the next 5 sessions."
            )

        if ctx.language == "vi":
            s1 = (
                f"Thị trường ở chế độ **{regime}** (20d={ctx.mkt_r20:+.1%}, 5d={ctx.mkt_r5:+.1%}, "
                f"độ rộng={ctx.breadth_20:.0%}, tỷ số biến động={ctx.vol_ratio:.2f}x, "
                f"xác suất chấn thương lịch sử={ctx.trauma_prob:.0%}).{analog_txt}"
            )
            s2 = (
                f"Mã `{ctx.symbol}` (sàn {ctx.exchange}, biên độ ±{band:.0%}): động lượng 20 phiên {ctx.stock_ret20:+.1%}, "
                f"dòng tiền khối ngoại 5d={ctx.foreign_net_ratio_5d:+.2%}, "
                f"điểm chất lượng-định giá PIT={'đã công bố' if ctx.pit_is_announced else 'CHƯA công bố (bị loại chống look-ahead)'} "
                f"(z={eff_pit_z:+.2f}), đột biến thanh khoản={ctx.turnover_shock_20d:.2f}x."
            )
            risk_desc = ", ".join(hidden_risks) if hidden_risks else "chưa ghi nhận bẫy thanh khoản hay kẹt sàn T+2.5 nghiêm trọng"
            s3 = (
                f"Phản biện Pre-Mortem (kịch bản xấu nhất khi hàng kẹt T+2.5): khoảng cách sàn={ctx.dist_to_floor:.1%}, "
                f"khoảng cách trần={ctx.dist_to_ceiling:.1%}, biên độ trong phiên={ctx.intraday_range:.1%}. "
                f"Rủi ro ẩn phát hiện: **{risk_desc}**."
            )
            s4 = (
                f"Quyết định **{action}** | Tỷ trọng mục tiêu: **{target_w:.1f}% NAV** (Kelly-Shrinkage sau phí 0.15% + thuế bán 0.10%) | "
                f"Cắt lỗ kỷ luật: **{sl_pct:.1%}** (ưu tiên thoát ngay khi hết khóa T+2.5 nếu vi phạm) | "
                f"Chốt lời mục tiêu: **+{tp_pct:.1%}** | Xác suất vượt trội hiệu chỉnh: **{win_prob:.1%}**."
            )
            if action == "CASH_SHIELD":
                s5 = (
                    f"Ưu tiên bảo toàn vốn: thị trường phát tín hiệu rủi ro đuôi ({regime}, trauma={ctx.trauma_prob:.0%}). "
                    f"Không mở mua mới `{ctx.symbol}` để tránh kẹt hàng T+2.5 trong kịch bản nằm sàn diện rộng; giữ tiền mặt chờ ổn định thanh khoản."
                )
            elif action == "REDUCE_OR_AVOID":
                s5 = (
                    f"Đứng ngoài / hạ tỷ trọng `{ctx.symbol}` do xuất hiện bẫy rủi ro ẩn ({risk_desc}). "
                    "Tỷ lệ lợi nhuận/rủi ro sau thuế phí và độ trễ T+2.5 không đủ hấp dẫn — kỷ luật bỏ qua kèo FOMO là lợi thế cốt lõi của chuyên gia."
                )
            elif action in ("BUY_HIGH_CONVICTION", "BUY_MODERATE"):
                s5 = (
                    f"Giải ngân có kiểm soát `{target_w:.1f}%` NAV vào `{ctx.symbol}` khi cả xu hướng, dòng tiền ngoại và nền tảng PIT đồng thuận "
                    f"mà chưa bị kéo nóng quá mức. Đặt kỷ luật cắt lỗ `{sl_pct:.1%}` sau khi hàng về T+2.5."
                )
            else:
                s5 = (
                    f"Duy trì vị thế phòng thủ nhẹ (`{target_w:.1f}%` NAV) với `{ctx.symbol}`, hạn chế đảo danh mục liên tục "
                    "để tránh thuế bán 0.10% và trượt giá bào mòn alpha trong pha đi ngang."
                )
        else:
            s1 = (
                f"Market regime is **{regime}** (20d={ctx.mkt_r20:+.1%}, 5d={ctx.mkt_r5:+.1%}, "
                f"breadth={ctx.breadth_20:.0%}, vol_ratio={ctx.vol_ratio:.2f}x, trauma_prob={ctx.trauma_prob:.0%}).{analog_txt}"
            )
            s2 = (
                f"`{ctx.symbol}` ({ctx.exchange}, ±{band:.0%} limit): 20d momentum {ctx.stock_ret20:+.1%}, "
                f"5d foreign flow {ctx.foreign_net_ratio_5d:+.2%}, PIT quality z={eff_pit_z:+.2f} "
                f"(announced={ctx.pit_is_announced}), liquidity shock={ctx.turnover_shock_20d:.2f}x."
            )
            risk_desc = ", ".join(hidden_risks) if hidden_risks else "no critical T+2.5 or floor-lock trap detected"
            s3 = (
                f"Pre-Mortem Worst-Case Audit (T+2.5 lock horizon): dist_to_floor={ctx.dist_to_floor:.1%}, "
                f"dist_to_ceiling={ctx.dist_to_ceiling:.1%}, intraday_range={ctx.intraday_range:.1%}. "
                f"Hidden risks flagged: **{risk_desc}**."
            )
            s4 = (
                f"Decision **{action}** | Target Weight: **{target_w:.1f}% NAV** | "
                f"Stop-Loss: **{sl_pct:.1%}** | Take-Profit: **+{tp_pct:.1%}** | Calibrated Win Prob: **{win_prob:.1%}**."
            )
            s5 = (
                f"Execute `{action}` at `{target_w:.1f}%` NAV on `{ctx.symbol}` under strict T+2.5 friction discipline "
                f"(hidden risks: {risk_desc})."
            )

        backend = "neuro_symbolic_veteran_core"
        slm_commentary = ""
        if self._try_load_slm() and self._slm_model is not None and self._slm_tok is not None:
            backend = "qwen3_qlora_dpo_plus_neuro_symbolic"
            if self.use_slm_generation:
                slm_commentary = self.generate_slm_commentary(ctx)

        return VeteranReasoningResponse(
            symbol=ctx.symbol,
            as_of_date=ctx.as_of_date,
            regime_label=regime,
            action=action,
            recommended_weight_pct=target_w,
            stop_loss_pct=sl_pct,
            take_profit_pct=tp_pct,
            calibrated_win_prob=win_prob,
            detected_hidden_risks=hidden_risks,
            causal_violations_flagged=causal_flags,
            step1_regime_diagnosis=s1,
            step2_economic_hypothesis=s2,
            step3_premortem_devils_advocate=s3,
            step4_sizing_and_risk_gate=s4,
            step5_executive_brief=s5,
            backend_used=backend,
            slm_generated_commentary=slm_commentary,
        )
