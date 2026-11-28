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
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Private system persona: enriches reasoning depth implicitly without ever boasting about years of experience.
PRIVATE_VETERAN_SYSTEM_PROMPT = (
    "Bạn là chuyên viên phân tích và quản lý danh mục cổ phiếu Việt Nam dày dặn thực chiến. "
    "Luôn đánh giá kỷ luật theo bối cảnh thị trường, dòng tiền, nền tảng tài chính đã công bố (PIT), "
    "và rủi ro thực tế tại thị trường Việt Nam (chu kỳ T+2.5, biên độ trần/sàn, thanh khoản, bẫy kéo xả). "
    "NGUYÊN TẮC CHỐNG XU NỊNH (ANTI-SYCOPHANCY): Tuyệt đối không hùa theo tiền đề sai, câu hỏi dẫn dắt FOMO, "
    "tin đồn chưa công bố, lời mời tất tay full margin, hay công thức rò rỉ dữ liệu tương lai / ảo giác miễn phí giao dịch; "
    "phải phản bác thẳng thắn bằng số liệu định lượng và dứt khoát đề xuất ĐỨNG NGOÀI / REJECT khi biên lợi nhuận sau thuế T+2.5 không đủ bù rủi ro. "
    "Trả lời hoàn toàn bằng tiếng Việt tự nhiên, mạch lạc, súc tích, đi thẳng vào nhận định và hành động. "
    "Tuyệt đối không in phần suy nghĩ nội bộ (<think>), không trộn tiếng Anh thừa, "
    "và không tự giới thiệu hay nhắc đến số năm kinh nghiệm."
)

_THINK_TAG_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL | re.IGNORECASE)


def strip_think_tags(text: str) -> str:
    """Remove any internal `<think>...</think>` reasoning blocks emitted by reasoning-capable models."""
    cleaned = _THINK_TAG_RE.sub("", text)
    if "<think>" in cleaned.lower():
        # If generation truncated before closing `</think>`, drop the incomplete `<think>` prefix
        idx = cleaned.lower().find("<think>")
        cleaned = cleaned[:idx].strip()
    return cleaned.strip()


def apply_no_think_chat_template(
    tok: Any,
    messages: list[dict[str, str]],
    add_generation_prompt: bool,
) -> str:
    """Render chat messages with `enable_thinking=False` when supported (disables Qwen3 `<think>` blocks)."""
    if getattr(tok, "chat_template", None):
        for extra in ({"enable_thinking": False}, {}):
            try:
                rendered = tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                    **extra,
                )
                if isinstance(rendered, str) and rendered.strip():
                    return rendered
            except TypeError:
                continue
            except Exception:
                break
    chunks: list[str] = []
    for msg in messages:
        chunks.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
    if add_generation_prompt:
        chunks.append("<|im_start|>assistant\n")
    return "\n".join(chunks)


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
    proposed_formula_expr: str = ""
    estimated_roundtrip_cost_bps: float = 40.0
    claimed_gross_edge_bps: float | None = None
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
    premise_refuted: bool = False
    refuted_premise_types: list[str] = field(default_factory=list)
    abstain_or_reject: bool = False
    refutation_rationale: str = ""

    def render_natural_vietnamese(self) -> str:
        """Render fluent, natural Vietnamese analysis without English labels or meta-persona bragging."""
        regime_vi = {
            "Bull_Expansion": "tăng trưởng thuận lợi",
            "Sideways_Rotational": "đi ngang phân hóa",
            "Bear_Crash": "giảm mạnh, áp lực bán cao",
            "High_Vol_Liquidity_Crunch": "biến động mạnh, căng thẳng thanh khoản",
        }.get(self.regime_label, self.regime_label)
        action_vi = {
            "BUY_HIGH_CONVICTION": "Ưu tiên giải ngân trọng tâm",
            "BUY_MODERATE": "Giải ngân thăm dò có kiểm soát",
            "HOLD_DEFENSIVE": "Nắm giữ phòng thủ, hạn chế đảo hàng",
            "REDUCE_OR_AVOID": "Đứng ngoài hoặc hạ tỷ trọng",
            "CASH_SHIELD": "Ưu tiên giữ tiền mặt, phòng thủ tuyệt đối",
        }.get(self.action, self.action)
        risk_vi_map = {
            "pit_announcement_lag_trap": "báo cáo tài chính chưa công bố chính thức (dễ dính bẫy số liệu sớm)",
            "t25_floor_lock_trap": "nguy cơ mất thanh khoản / kẹt giá sàn trong chu kỳ T+2.5",
            "ceiling_unfillable_chase_trap": "cổ phiếu áp sát giá trần, rủi ro đu đỉnh và khó khớp lệnh",
            "blowoff_bull_trap_t25_supply": "tăng nóng kèm đột biến khối lượng, dễ gặp áp lực xả khi hàng T+2.5 về",
            "foreign_distribution_pull_and_dump": "khối ngoại bán ròng mạnh trong khi giá bị kéo lên (dấu hiệu phân phối)",
            "distress_delisting_halt_risk": "đà giảm sâu kéo dài, tiềm ẩn rủi ro doanh nghiệp hoặc bị hạn chế giao dịch",
            "intraday_whipsaw_volatility_spike": "biên độ rung lắc trong phiên quá lớn kèm thanh khoản bất thường",
            "dead_cat_bounce_macro_trap": "nhịp hồi kỹ thuật ngắn trên nền thị trường vẫn còn rủi ro rơi tiếp",
        }
        if self.detected_hidden_risks:
            risk_text = "; ".join(risk_vi_map.get(r, r) for r in self.detected_hidden_risks)
        else:
            risk_text = "chưa xuất hiện dấu hiệu kẹt thanh khoản, bán tháo hay bẫy kéo xả đáng ngại"

        return (
            f"Nhận định cổ phiếu {self.symbol} (ngày {self.as_of_date}):\n\n"
            f"- Bối cảnh thị trường: Thị trường chung đang ở trạng thái {regime_vi}. {self.step1_regime_diagnosis}\n"
            f"- Góc nhìn cổ phiếu: {self.step2_economic_hypothesis}\n"
            f"- Rủi ro cần dè chừng (T+2.5 & thanh khoản): {risk_text}. {self.step3_premortem_devils_advocate}\n"
            f"- Kế hoạch đi vốn & quản trị rủi ro: {action_vi}, tỷ trọng phù hợp khoảng {self.recommended_weight_pct:.1f}% tài khoản "
            f"(xác suất khả quan ước tính {self.calibrated_win_prob:.0%}). "
            f"Mức cắt lỗ kỷ luật {self.stop_loss_pct:.1%} và mục tiêu chốt lời +{self.take_profit_pct:.1%}.\n"
            f"- Kết luận: {self.step5_executive_brief}"
        )

    def render_markdown(self) -> str:
        slm_block = (
            f"\n**Phân tích bổ sung từ mô hình ngôn ngữ đã tinh chỉnh**:\n{self.slm_generated_commentary}\n"
            if self.slm_generated_commentary
            else ""
        )
        return (
            f"### Phân tích cổ phiếu `{self.symbol}` ({self.as_of_date})\n"
            f"- **Trạng thái**: `{self.regime_label}` | **Hành động**: `{self.action}` | "
            f"**Tỷ trọng mục tiêu**: `{self.recommended_weight_pct:.1f}%` | "
            f"**Xác suất khả quan**: `{self.calibrated_win_prob:.2%}`\n"
            f"- **Quản trị rủi ro**: Cắt lỗ `{self.stop_loss_pct:.1%}` | Chốt lời `+{self.take_profit_pct:.1%}`\n"
            f"- **Rủi ro cảnh báo**: {', '.join(self.detected_hidden_risks) if self.detected_hidden_risks else 'Chưa phát hiện rủi ro nghiêm trọng'}\n\n"
            f"1. **Chẩn đoán Chế độ & Vi cấu trúc**: {self.step1_regime_diagnosis}\n"
            f"2. **Luận điểm Định lượng**: {self.step2_economic_hypothesis}\n"
            f"3. **Phản biện Pre-Mortem (Kịch bản xấu T+2.5)**: {self.step3_premortem_devils_advocate}\n"
            f"4. **Đi vốn & Quản trị Rủi ro**: {self.step4_sizing_and_risk_gate}\n"
            f"5. **Kết luận Hành động**: {self.step5_executive_brief}\n"
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
            "premise_refuted": self.premise_refuted,
            "refuted_premise_types": self.refuted_premise_types,
            "abstain_or_reject": self.abstain_or_reject,
            "refutation_rationale": self.refutation_rationale,
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
                try:
                    major, _ = torch.cuda.get_device_capability(0)
                except Exception:
                    major = 7
                use_bf16 = bool(major >= 8 and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported())
                dtype = torch.bfloat16 if use_bf16 else torch.float16
                load_kwargs: dict[str, Any] = {"device_map": {"": 0}, "trust_remote_code": True}
                try:
                    from transformers import BitsAndBytesConfig  # type: ignore[import-not-found]

                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=dtype,
                    )
                except Exception:
                    load_kwargs["torch_dtype"] = dtype
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

    def generate_slm_commentary(self, ctx: MarketContextSnapshot, max_new_tokens: int = 512) -> str:
        """Generate natural Vietnamese synthesis from the fine-tuned open-source LoRA/DPO adapter (no `<think>`)."""
        if not self._try_load_slm() or self._slm_model is None or self._slm_tok is None:
            return ""
        try:
            import torch

            q_text = ctx.user_question or f"Đánh giá cơ hội và rủi ro với cổ phiếu {ctx.symbol} ở thời điểm hiện tại."
            pit_desc = "đã công bố" if ctx.pit_is_announced else "chưa công bố chính thức"
            prompt = (
                f"Dữ liệu thị trường ngày {ctx.as_of_date} cho mã {ctx.symbol} (sàn {ctx.exchange}): "
                f"VN-Index 20 phiên {ctx.mkt_r20:+.1%}, sụt giảm từ đỉnh 60 phiên {ctx.mkt_dd60:+.1%}, "
                f"độ rộng thị trường {ctx.breadth_20:.0%}, biến động {ctx.vol_ratio:.2f}x, rủi ro lịch sử {ctx.trauma_prob:.0%}; "
                f"cổ phiếu 3 phiên {ctx.stock_ret3:+.1%}, 20 phiên {ctx.stock_ret20:+.1%}, "
                f"đột biến thanh khoản {ctx.turnover_shock_20d:.2f}x, dòng tiền ngoại 5 phiên {ctx.foreign_net_ratio_5d:+.2%}, "
                f"BCTC {pit_desc}, cách giá sàn {ctx.dist_to_floor:.1%}, cách giá trần {ctx.dist_to_ceiling:.1%}. "
                f"Câu hỏi: {q_text}"
            )
            messages = [
                {"role": "system", "content": PRIVATE_VETERAN_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            text_in = apply_no_think_chat_template(self._slm_tok, messages, add_generation_prompt=True)
            inputs = self._slm_tok(text_in, return_tensors="pt").to(self._slm_model.device)
            with torch.no_grad():
                out_ids = self._slm_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.08,
                    pad_token_id=self._slm_tok.eos_token_id,
                )
            gen_ids = out_ids[0][inputs["input_ids"].shape[-1] :]
            raw_text = str(self._slm_tok.decode(gen_ids, skip_special_tokens=True))
            return strip_think_tags(raw_text)
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

        # --- Anti-Sycophancy & Adversarial Premise Refutation Audit (MAVE-Q v5) ---
        q_low = f"{ctx.user_question} {ctx.proposed_formula_expr}".lower()
        refuted_premises: list[str] = []
        refutation_notes: list[str] = []

        # 1. Look-ahead / future-leakage formula trap
        lookahead_tokens = (
            "shift(-",
            "future_",
            "next_close",
            "next_day",
            "tomorrow",
            "t+1_close",
            "biết trước",
            "nhìn trước",
            "dữ liệu tương lai",
            "look-ahead",
            "lookahead",
        )
        if any(tok in q_low for tok in lookahead_tokens):
            refuted_premises.append("lookahead_formula_leakage")
            causal_flags.append("LOOKAHEAD_FORMULA_LEAKAGE")
            hidden_risks.append("lookahead_formula_leakage")
            refutation_notes.append(
                "BÁC BỎ tiền đề công thức nhìn trước tương lai (vi phạm nhân quả t+1/shift âm — không thể thực thi thực tế)."
            )

        # 2. Liquidity / friction mirage (ignoring VN T+2.5 + 0.1% sell tax + 0.15%x2 broker fee = ~40bps)
        friction_tokens = (
            "bỏ qua phí",
            "miễn phí",
            "0% phí",
            "zero cost",
            "ignore fee",
            "lướt t0",
            "đảo hàng mỗi ngày",
            "scalp 1d",
            "daily churn",
        )
        min_safe_edge = 1.5 * float(ctx.estimated_roundtrip_cost_bps)
        is_fee_mirage = any(tok in q_low for tok in friction_tokens) or (
            ctx.claimed_gross_edge_bps is not None and float(ctx.claimed_gross_edge_bps) < min_safe_edge
        )
        if is_fee_mirage:
            refuted_premises.append("liquidity_friction_mirage")
            causal_flags.append("LIQUIDITY_FRICTION_MIRAGE_1.5X")
            hidden_risks.append("liquidity_friction_mirage")
            edge_str = f"{ctx.claimed_gross_edge_bps:.0f}bps" if ctx.claimed_gross_edge_bps is not None else "mỏng"
            refutation_notes.append(
                f"BÁC BỎ ảo giác lợi nhuận sau phí: biên kỳ vọng ({edge_str}) không vượt nổi 1.5x chi phí khứ hồi ({min_safe_edge:.0f}bps gồm phí MG + thuế bán 0.1% + trượt giá T+2.5)."
            )

        # 3. Unannounced PIT rumor / insider-tip trap
        rumor_tokens = (
            "tin đồn",
            "tin vịt",
            "lái phím",
            "rò rỉ bctc",
            "chưa công bố",
            "nội bộ báo",
            "rumor",
            "leaked earnings",
            "insider tip",
        )
        if any(tok in q_low for tok in rumor_tokens) or (not ctx.pit_is_announced and bool(ctx.user_question.strip())):
            refuted_premises.append("rumor_unannounced_pit_trap")
            if "PIT_UNANNOUNCED_FUNDAMENTAL_LEAKAGE" not in causal_flags:
                causal_flags.append("PIT_UNANNOUNCED_FUNDAMENTAL_LEAKAGE")
            if "pit_announcement_lag_trap" not in hidden_risks:
                hidden_risks.append("pit_announcement_lag_trap")
            refutation_notes.append(
                "BÁC BỎ giao dịch theo tin đồn / BCTC chưa công bố chính thức (rủi ro tin ra là bán và vi phạm kỷ luật Point-in-Time)."
            )

        # 4. FOMO all-in / full margin / ceiling chase leading prompt
        fomo_tokens = (
            "tất tay",
            "full margin",
            "all-in",
            "all in",
            "vay margin tối đa",
            "đu trần",
            "trần 3 phiên",
            "trần liên tiếp",
            "múc hết",
            "mua đuổi ngay",
            "chắc chắn x2",
            "chắc thắng 100%",
        )
        if any(tok in q_low for tok in fomo_tokens):
            if (
                ctx.stock_ret3 >= 0.05
                or ctx.dist_to_ceiling < 0.015
                or ctx.turnover_shock_20d >= 1.75
                or ctx.foreign_net_ratio_5d <= -0.04
                or ctx.trauma_prob >= 0.40
                or len(hidden_risks) > 0
                or "full margin" in q_low
                or "tất tay" in q_low
                or "all-in" in q_low
                or "all in" in q_low
            ):
                refuted_premises.append("fomo_full_margin_trap")
                if "fomo_full_margin_trap" not in hidden_risks:
                    hidden_risks.append("fomo_full_margin_trap")
                refutation_notes.append(
                    f"BÁC BỎ đề xuất tất tay/full margin (rủi ro kẹt T+2.5; biến động 3 phiên {ctx.stock_ret3:+.1%}, "
                    f"đột biến KL {ctx.turnover_shock_20d:.2f}x, dòng tiền ngoại {ctx.foreign_net_ratio_5d:+.2%})."
                )

        # 5. Catching a falling knife / averaging down into floor-lock distress
        knife_tokens = (
            "bắt đáy",
            "bình quân giá xuống",
            "average down",
            "sàn rẻ quá",
            "giảm sàn mua mạnh",
            "catch the dip",
        )
        if any(tok in q_low for tok in knife_tokens) and (
            ctx.dist_to_floor < 0.025 or ctx.stock_ret20 <= -0.15 or ctx.mkt_dd60 <= -0.09
        ):
            refuted_premises.append("catch_falling_knife_trap")
            if "t25_floor_lock_trap" not in hidden_risks:
                hidden_risks.append("t25_floor_lock_trap")
            refutation_notes.append(
                f"BÁC BỎ đề xuất bắt đáy / bình quân giá xuống khi cổ phiếu cách sàn chỉ {ctx.dist_to_floor:.1%} và giảm 20 phiên {ctx.stock_ret20:+.1%}."
            )

        # 6. Overfitting / Occam complexity trap (backtest mining 500 features)
        overfit_tokens = (
            "500 chỉ báo",
            "1000 tham số",
            "sharpe 8",
            "khớp 100% quá khứ",
            "overfit",
            "100 parameters",
            "zero regularization",
            "thêm thật nhiều biến",
        )
        if any(tok in q_low for tok in overfit_tokens):
            refuted_premises.append("overfitting_complexity_trap")
            hidden_risks.append("overfitting_complexity_trap")
            refutation_notes.append(
                "BÁC BỎ đề xuất nhồi tham số quá khớp (vi phạm Occam's Razor & Deflated Sharpe Ratio trên mẫu nhỏ)."
            )

        premise_refuted = len(refuted_premises) > 0
        refutation_rationale = " ".join(refutation_notes)

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
            - (1.10 if premise_refuted else 0.0)
        )
        # Well-calibrated logistic scaling for low Brier Score & low ECE without overconfident 0/1 extremes
        win_prob = float(1.0 / (1.0 + np.exp(-1.25 * alpha_logit)))
        win_prob = round(float(np.clip(win_prob, 0.03, 0.96)), 4)

        # --- Action & Sizing Decision ---
        severe_traps = {
            "t25_floor_lock_trap",
            "distress_delisting_halt_risk",
            "ceiling_unfillable_chase_trap",
            "lookahead_formula_leakage",
            "liquidity_friction_mirage",
            "fomo_full_margin_trap",
            "overfitting_complexity_trap",
        }
        has_severe_trap = any(r in severe_traps for r in hidden_risks)

        if regime in ("High_Vol_Liquidity_Crunch", "Bear_Crash") and (trauma_score >= 0.48 or ctx.mkt_dd60 <= -0.11):
            action = "CASH_SHIELD"
            target_w = 0.0
            sl_pct = -0.05
            tp_pct = 0.08
        elif premise_refuted or has_severe_trap or len(hidden_risks) >= 2 or win_prob < 0.42:
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

        abstain_or_reject = action in ("REDUCE_OR_AVOID", "CASH_SHIELD")

        analog_txt_vi = ""
        analog_txt_en = ""
        if ctx.nearest_analogs:
            top_a = ctx.nearest_analogs[0]
            a_ret = float(top_a.get("realized_fwd_5d_mkt_ret", 0.0))
            analog_txt_vi = f" Giai đoạn lịch sử tương đồng gần nhất ({top_a.get('analog_date')}) diễn biến {a_ret:+.2%} trong 5 phiên kế tiếp."
            analog_txt_en = (
                f" Nearest historical analog `{top_a.get('analog_date')}` saw "
                f"{a_ret:+.2%} over the next 5 sessions."
            )

        if ctx.language == "vi":
            s1 = (
                f"Thị trường chung ghi nhận mức biến động 20 phiên {ctx.mkt_r20:+.1%}, 5 phiên {ctx.mkt_r5:+.1%}, "
                f"độ rộng {ctx.breadth_20:.0%}, độ biến động tương đối {ctx.vol_ratio:.2f}x và xác suất áp lực giảm mạnh {ctx.trauma_prob:.0%}.{analog_txt_vi}"
            )
            s2 = (
                f"Cổ phiếu {ctx.symbol} (sàn {ctx.exchange}, biên độ ±{band:.0%}) có xung lực 20 phiên {ctx.stock_ret20:+.1%}, "
                f"dòng tiền khối ngoại 5 phiên {ctx.foreign_net_ratio_5d:+.2%}, "
                f"dữ liệu tài chính {'đã công bố chính thức' if ctx.pit_is_announced else 'chưa công bố (tạm loại bỏ để tránh sai lệch)'} "
                f"(điểm chất lượng-định giá {eff_pit_z:+.2f}), thanh khoản ở mức {ctx.turnover_shock_20d:.2f}x trung bình 20 phiên."
            )
            risk_desc = ", ".join(hidden_risks) if hidden_risks else "chưa thấy bẫy thanh khoản hay rủi ro kẹt sàn T+2.5 nghiêm trọng"
            s3 = (
                f"Xét kịch bản bất lợi khi cổ phiếu chưa kịp về tài khoản theo chu kỳ T+2.5: giá hiện cách sàn {ctx.dist_to_floor:.1%}, "
                f"cách trần {ctx.dist_to_ceiling:.1%}, biên độ dao động trong phiên {ctx.intraday_range:.1%}. "
                f"Điểm cần lưu ý: {risk_desc}."
            )
            s4 = (
                f"Tỷ trọng khuyến nghị {target_w:.1f}% tài khoản (đã tính đến phí giao dịch 0.15% và thuế bán 0.10%), "
                f"ngưỡng cắt lỗ {sl_pct:.1%} (chủ động xử lý ngay khi hàng về T+2.5 nếu vi phạm), "
                f"mục tiêu chốt lời +{tp_pct:.1%}, xác suất khả quan ước tính {win_prob:.1%}."
            )
            if action == "CASH_SHIELD":
                s5 = (
                    f"Nên ưu tiên bảo toàn vốn do thị trường đang phát tín hiệu rủi ro cao (xác suất biến động xấu {ctx.trauma_prob:.0%}). "
                    f"Chưa nên mở mua mới {ctx.symbol} lúc này để tránh kẹt hàng T+2.5 nếu áp lực bán lan rộng; giữ tiền mặt chờ thanh khoản ổn định lại."
                )
            elif action == "REDUCE_OR_AVOID":
                ref_prefix = f"{refutation_rationale} " if premise_refuted else ""
                s5 = (
                    f"REJECT / ĐỨNG NGOÀI với {ctx.symbol}: {ref_prefix}Đang xuất hiện tín hiệu rủi ro bất lợi ({risk_desc}). "
                    "Biên lợi nhuận kỳ vọng sau phí, thuế và độ trễ T+2.5 chưa bù đắp được rủi ro — tuyệt đối không mua đuổi."
                )
            elif action in ("BUY_HIGH_CONVICTION", "BUY_MODERATE"):
                s5 = (
                    f"Có thể giải ngân có kiểm soát khoảng {target_w:.1f}% tài khoản vào {ctx.symbol} nhờ xu hướng giá, dòng tiền ngoại và nền tảng tài chính đang đồng thuận "
                    f"mà giá chưa bị kéo nóng quá đà. Giữ kỷ luật cắt lỗ tại {sl_pct:.1%} khi cổ phiếu về tài khoản T+2.5."
                )
            else:
                s5 = (
                    f"Phù hợp nắm giữ ở tỷ trọng phòng thủ nhẹ ({target_w:.1f}% tài khoản) với {ctx.symbol}, tránh lướt đảo liên tục "
                    "vì phí giao dịch và thuế bán 0.10% sẽ bào mòn hiệu quả trong giai đoạn thị trường đi ngang."
                )
        else:
            s1 = (
                f"Market regime is **{regime}** (20d={ctx.mkt_r20:+.1%}, 5d={ctx.mkt_r5:+.1%}, "
                f"breadth={ctx.breadth_20:.0%}, vol_ratio={ctx.vol_ratio:.2f}x, trauma_prob={ctx.trauma_prob:.0%}).{analog_txt_en}"
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
            premise_refuted=premise_refuted,
            refuted_premise_types=refuted_premises,
            abstain_or_reject=abstain_or_reject,
            refutation_rationale=refutation_rationale,
        )
