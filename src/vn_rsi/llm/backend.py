"""Interchangeable LLMBackend enforcing <=8B main-loop constraint + small-model scaffolding.

Hard constraints (Section 1.1 - 1.3):
- Every LLM in the main loop MUST be <= 8B parameters (e.g., Qwen3-8B, Llama-3.1-8B, quantized 4/8-bit).
- Larger teacher models (>8B) are BLOCKED in main loop unless `allow_teacher=True` is explicitly set
  for flagged teacher ablation / distillation.
- Supports JSON-schema constrained decoding, retry with error feedback, and deterministic verifier best-of-n.
- Gracefully degrades to CPU deterministic <=8B scaffolding simulator when local GPU/Ollama endpoint is offline,
  recording `backend_type` transparently in every log.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np
from pydantic import BaseModel, ValidationError

from vn_rsi.utils.budget import BudgetTracker


class ModelSizeConstraintError(ValueError):
    """Raised when attempting to instantiate a >8B model in the main research loop."""


@dataclass
class LLMGenerationResult:
    parsed: dict[str, Any]
    raw_text: str
    prompt_tokens: int
    completion_tokens: int
    retries_used: int
    backend_name: str
    is_teacher: bool


class LLMBackend(ABC):
    """Unified interface for <=8B local backends (Ollama/vLLM/llama.cpp) and offline CI scaffolding."""

    def __init__(
        self,
        model_name: str,
        parameter_count_billions: float,
        is_teacher: bool = False,
        allow_teacher: bool = False,
        max_allowed_params_b: float = 8.0,
    ) -> None:
        if parameter_count_billions > max_allowed_params_b and not (is_teacher and allow_teacher):
            raise ModelSizeConstraintError(
                f"Model '{model_name}' ({parameter_count_billions}B) violates main-loop <= {max_allowed_params_b}B "
                "constraint! Larger models are permitted ONLY when `is_teacher=True` and `allow_teacher=True`."
            )
        self.model_name = model_name
        self.parameter_count_billions = parameter_count_billions
        self.is_teacher = is_teacher

    @abstractmethod
    def _raw_complete(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any],
        temperature: float,
        seed: int,
    ) -> tuple[str, int, int, float]:
        """Return (json_string, prompt_tokens, completion_tokens, elapsed_seconds)."""

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_cls: type[BaseModel],
        budget: BudgetTracker,
        temperature: float = 0.3,
        seed: int = 42,
        max_retries: int = 3,
        best_of_n: int = 1,
        verifier_fn: Callable[[BaseModel], float] | None = None,
    ) -> tuple[BaseModel, LLMGenerationResult]:
        """Constrained JSON generation with schema-error feedback retries and verifier-guided best-of-n."""
        schema_dict = schema_cls.model_json_schema()
        candidates: list[tuple[float, BaseModel, LLMGenerationResult]] = []
        last_error = ""

        n_draws = max(1, best_of_n)
        for draw_idx in range(n_draws):
            feedback_prompt = user_prompt
            for attempt in range(max_retries):
                budget.check()
                raw_txt, p_tok, c_tok, dt = self._raw_complete(
                    system_prompt=system_prompt,
                    user_prompt=feedback_prompt,
                    json_schema=schema_dict,
                    temperature=temperature,
                    seed=seed + draw_idx * 31 + attempt,
                )
                budget.record_usage(p_tok, c_tok, call_seconds=dt)
                try:
                    data = json.loads(raw_txt)
                    validated = schema_cls.model_validate(data)
                    score = verifier_fn(validated) if verifier_fn is not None else 0.0
                    gen_res = LLMGenerationResult(
                        parsed=validated.model_dump(),
                        raw_text=raw_txt,
                        prompt_tokens=p_tok,
                        completion_tokens=c_tok,
                        retries_used=attempt,
                        backend_name=self.backend_id,
                        is_teacher=self.is_teacher,
                    )
                    candidates.append((score, validated, gen_res))
                    break
                except (json.JSONDecodeError, ValidationError, ValueError) as err:
                    last_error = str(err)
                    feedback_prompt = (
                        f"{user_prompt}\n\n[SCHEMA/VERIFIER ERROR ON ATTEMPT {attempt + 1}]: {last_error}\n"
                        "Fix the JSON strictly according to the schema and causal rules."
                    )

        if not candidates:
            raise RuntimeError(
                f"LLMBackend '{self.backend_id}' failed structured generation after {max_retries} retries: {last_error}"
            )

        candidates.sort(key=lambda item: item[0], reverse=True)
        _, best_model, best_meta = candidates[0]
        return best_model, best_meta

    @property
    @abstractmethod
    def backend_id(self) -> str:
        """Identifier recorded in experiment logs."""


class OpenAICompatLocalBackend(LLMBackend):
    """Connects to local Ollama / vLLM / llama.cpp OpenAI-compatible server serving <=8B open weights."""

    def __init__(
        self,
        model_name: str,
        parameter_count_billions: float,
        endpoint_url: str,
        max_tokens: int = 768,
        is_teacher: bool = False,
        allow_teacher: bool = False,
        timeout_seconds: float = 45.0,
    ) -> None:
        super().__init__(
            model_name=model_name,
            parameter_count_billions=parameter_count_billions,
            is_teacher=is_teacher,
            allow_teacher=allow_teacher,
        )
        self.endpoint_url = endpoint_url
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds

    @property
    def backend_id(self) -> str:
        prefix = "teacher_http" if self.is_teacher else "local_http_8b"
        return f"{prefix}:{self.model_name}"

    def _raw_complete(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any],
        temperature: float,
        seed: int,
    ) -> tuple[str, int, int, float]:
        t0 = time.monotonic()
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": f"{system_prompt}\nRespond ONLY with valid JSON matching schema: {json.dumps(json_schema)}",
                },
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "seed": seed,
            "response_format": {"type": "json_object"},
        }
        with httpx.Client(timeout=self.timeout_seconds) as client:
            resp = client.post(self.endpoint_url, json=payload)
            resp.raise_for_status()
            body = resp.json()
        dt = time.monotonic() - t0
        content = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})
        p_tok = int(usage.get("prompt_tokens", max(32, len(system_prompt + user_prompt) // 4)))
        c_tok = int(usage.get("completion_tokens", max(24, len(content) // 4)))
        return content, p_tok, c_tok, dt


class RuleGuidedSmallLMBackend(LLMBackend):
    """Deterministic CPU-friendly <=8B grammar-constrained policy backend for reproducible offline runs & CI.

    Faithfully conditions proposals on prompt context (mechanism family, genome focus weights, preferred windows,
    memory beliefs, error feedback, and temperature/seed), including realistic noisy/suboptimal draws that
    best-of-n verification and outer-loop selection must filter. Explicitly tagged in logs as
    `deterministic_8b_scaffolding` (or `teacher_sim_reference` when simulating teacher ablation).
    """

    def __init__(
        self,
        model_name: str = "qwen3-8b-grammar-scaffold-cpu",
        parameter_count_billions: float = 8.0,
        is_teacher: bool = False,
        allow_teacher: bool = False,
    ) -> None:
        super().__init__(
            model_name=model_name,
            parameter_count_billions=parameter_count_billions,
            is_teacher=is_teacher,
            allow_teacher=allow_teacher,
        )

    @property
    def backend_id(self) -> str:
        if self.is_teacher:
            return f"teacher_sim_reference:{self.model_name}"
        return f"deterministic_8b_scaffolding:{self.model_name}"

    def _raw_complete(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any],
        temperature: float,
        seed: int,
    ) -> tuple[str, int, int, float]:
        t0 = time.monotonic()
        rng = np.random.default_rng(seed)
        title = json_schema.get("title", "")

        # Parse JSON context hint embedded in user_prompt if present
        ctx: dict[str, Any] = {}
        if "CONTEXT_JSON:" in user_prompt:
            raw_ctx = user_prompt.split("CONTEXT_JSON:", 1)[1].strip().split("\n")[0]
            try:
                ctx = json.loads(raw_ctx)
            except Exception:
                ctx = {}

        has_err = "[SCHEMA/VERIFIER ERROR" in user_prompt
        if title == "FalsifiableHypothesis":
            out = self._propose_hypothesis(ctx, rng, temperature)
        elif title == "FactorSpec":
            out = self._propose_factor_ast(ctx, rng, temperature, has_err=has_err)
        elif title == "GenomePatchProposal":
            out = self._propose_genome_patch(ctx, rng, temperature)
        else:
            out = {}

        txt = json.dumps(out, sort_keys=True)
        p_tok = max(48, (len(system_prompt) + len(user_prompt)) // 4)
        c_tok = max(32, len(txt) // 4)
        dt = max(0.0005, time.monotonic() - t0)
        return txt, p_tok, c_tok, dt

    def _propose_hypothesis(
        self, ctx: dict[str, Any], rng: np.random.Generator, temperature: float
    ) -> dict[str, Any]:
        family = str(ctx.get("family", "momentum_liquidity"))
        hyp_id = str(ctx.get("hypothesis_id", f"hyp_{rng.integers(1000, 9999)}"))

        templates: dict[str, tuple[str, str, int, list[str], list[str]]] = {
            "momentum_liquidity": (
                "Medium-horizon price momentum scaled by inverse return volatility predicts 5-day continuation on HOSE/HNX.",
                "Gradual information diffusion under +/-7% daily price limits delays full price discovery across T+2.5 cycles.",
                1,
                ["close", "return_1d", "turnover_shock_20d"],
                ["ts_return", "ts_std", "safe_div", "cs_zscore"],
            ),
            "reversal_microstructure": (
                "Short-horizon 3-to-5 day price spikes accompanied by retail turnover shocks mean-revert after T+2.5 settlement unlock.",
                "Retail T+2.5 lockup creates temporary supply-demand imbalance; once shares settle on T+2 afternoon, profit-taking induces reversal.",
                -1,
                ["close", "turnover_shock_20d", "dist_to_ceiling"],
                ["ts_return", "mul", "neg", "cs_zscore"],
            ),
            "value_quality_pit": (
                "Stocks with high point-in-time ROE-to-P/B and YoY revenue growth outperform peers post-announcement.",
                "Post-earnings announcement drift in retail-dominated Vietnamese equities under-reacts to verified quarterly fundamentals.",
                1,
                ["pit_value_quality", "pit_roe", "pit_pb_ratio"],
                ["ts_mean", "cs_zscore", "add"],
            ),
            "foreign_flow_regime": (
                "Persistent multi-day net foreign buying relative to remaining FOL room predicts positive excess returns.",
                "Institutional foreign inflows exhibit multi-day execution persistence due to board-lot and ADV participation caps.",
                1,
                ["foreign_net_ratio", "fol_room_pct", "close"],
                ["ts_mean", "cs_zscore", "add"],
            ),
        }
        stmt, mech, sign, terms, ops = templates.get(family, templates["momentum_liquidity"])
        # Small-model stochastic noise at high exploration temperature
        if not self.is_teacher and temperature > 0.55 and rng.random() < 0.25:
            sign = -sign
        return {
            "hypothesis_id": hyp_id,
            "mechanism_family": family,
            "statement": stmt,
            "economic_mechanism": mech,
            "expected_sign": sign,
            "refutation_criteria": "Reject if search-zone Spearman Rank IC has opposite sign or |Rank IC| < min_search_rank_ic.",
            "suggested_terminals": terms,
            "suggested_operators": ops,
        }

    def _propose_factor_ast(
        self, ctx: dict[str, Any], rng: np.random.Generator, temperature: float, has_err: bool = False
    ) -> dict[str, Any]:
        family = str(ctx.get("family", "momentum_liquidity"))
        hyp_id = str(ctx.get("hypothesis_id", "hyp_0"))
        factor_id = str(ctx.get("factor_id", f"fac_{hyp_id}"))
        windows: list[int] = [int(w) for w in ctx.get("preferred_windows", [5, 10, 20])]
        max_depth = 2 if has_err else int(ctx.get("max_depth", 3))
        w_short = min(windows) if windows else 5
        w_med = sorted(windows)[len(windows) // 2] if windows else 15
        w_long = max(windows) if windows else 20

        variant_roll = float(rng.random())
        if self.is_teacher:
            # Teacher upper-bound has cleaner prior alignment
            variant_roll = 0.85

        root: dict[str, Any]
        if family == "momentum_liquidity":
            if max_depth >= 3 and variant_roll > 0.35:
                root = {
                    "node_type": "op",
                    "name": "safe_div",
                    "args": [
                        {"node_type": "op", "name": "ts_return", "window": w_med, "args": [{"node_type": "terminal", "name": "close"}]},
                        {"node_type": "op", "name": "ts_std", "window": w_long, "args": [{"node_type": "terminal", "name": "return_1d"}]},
                    ],
                }
            else:
                root = {
                    "node_type": "op",
                    "name": "ts_return",
                    "window": int(rng.choice(windows)),
                    "args": [{"node_type": "terminal", "name": "close"}],
                }
        elif family == "reversal_microstructure":
            w_rev = min(5, w_short)
            if max_depth >= 4 and variant_roll > 0.30:
                root = {
                    "node_type": "op",
                    "name": "neg",
                    "args": [
                        {
                            "node_type": "op",
                            "name": "mul",
                            "args": [
                                {"node_type": "op", "name": "ts_return", "window": max(3, w_rev), "args": [{"node_type": "terminal", "name": "close"}]},
                                {"node_type": "terminal", "name": "turnover_shock_20d"},
                            ],
                        }
                    ],
                }
            else:
                root = {
                    "node_type": "op",
                    "name": "neg",
                    "args": [
                        {"node_type": "op", "name": "ts_return", "window": max(3, w_rev), "args": [{"node_type": "terminal", "name": "close"}]}
                    ],
                }
        elif family == "value_quality_pit":
            if max_depth >= 3 and variant_roll > 0.40:
                root = {
                    "node_type": "op",
                    "name": "add",
                    "args": [
                        {"node_type": "op", "name": "cs_zscore", "args": [{"node_type": "terminal", "name": "pit_value_quality"}]},
                        {"node_type": "op", "name": "ts_mean", "window": w_short, "args": [{"node_type": "terminal", "name": "foreign_net_ratio"}]},
                    ],
                }
            else:
                root = {
                    "node_type": "op",
                    "name": "cs_zscore",
                    "args": [{"node_type": "terminal", "name": "pit_value_quality"}],
                }
        else:  # foreign_flow_regime
            if max_depth >= 3 and variant_roll > 0.35:
                root = {
                    "node_type": "op",
                    "name": "add",
                    "args": [
                        {"node_type": "op", "name": "ts_mean", "window": w_short, "args": [{"node_type": "terminal", "name": "foreign_net_ratio"}]},
                        {"node_type": "op", "name": "cs_zscore", "args": [{"node_type": "terminal", "name": "pit_value_quality"}]},
                    ],
                }
            else:
                root = {
                    "node_type": "op",
                    "name": "ts_mean",
                    "window": int(rng.choice(windows)),
                    "args": [{"node_type": "terminal", "name": "foreign_net_ratio"}],
                }

        # Occasional noisy wrapper when temperature is high (will be repaired via schema retry if exceeding max_depth)
        if not has_err and temperature > 0.60 and rng.random() < 0.20:
            root = {"node_type": "op", "name": "tanh_squash", "args": [root]}

        return {
            "factor_id": factor_id,
            "hypothesis_id": hyp_id,
            "description": f"Causal DSL expression for {family}",
            "root": root,
            "normalize_cross_section": True,
        }

    def _propose_genome_patch(
        self, ctx: dict[str, Any], rng: np.random.Generator, temperature: float
    ) -> dict[str, Any]:
        family_perf: dict[str, float] = ctx.get("family_ic_summary", {})
        current_genome: dict[str, Any] = ctx.get("parent_genome", {})
        iter_idx = int(ctx.get("iteration", 0))

        # Reweight planner_focus_weights toward empirically stronger causal mechanism families (with exploration floor)
        base_weights = dict(
            current_genome.get(
                "planner_focus_weights",
                {
                    "momentum_liquidity": 0.25,
                    "reversal_microstructure": 0.25,
                    "value_quality_pit": 0.25,
                    "foreign_flow_regime": 0.25,
                },
            )
        )
        for fam in list(base_weights.keys()):
            obs_ic = float(family_perf.get(fam, 0.01))
            step = 0.12 if self.is_teacher else 0.08
            noise = float(rng.normal(0.0, 0.025 * max(0.2, temperature)))
            base_weights[fam] = float(np.clip(base_weights[fam] + step * np.sign(obs_ic - 0.008) + noise, 0.10, 0.55))
        tot = sum(base_weights.values()) or 1.0
        norm_weights = {k: round(v / tot, 4) for k, v in base_weights.items()}

        window_candidates = [[3, 5, 15], [4, 10, 15], [3, 8, 15, 20], [5, 12, 20]]
        chosen_windows = window_candidates[iter_idx % len(window_candidates)]
        best_n = 4 if (self.is_teacher or iter_idx >= 1) else int(rng.choice([2, 3, 4]))
        new_temp = round(float(np.clip(0.28 + rng.normal(0.0, 0.04), 0.18, 0.48)), 3)
        new_depth = 4 if iter_idx >= 1 else 3

        updates: dict[str, Any] = {
            "planner_focus_weights": norm_weights,
            "preferred_windows": chosen_windows,
            "best_of_n": best_n,
            "exploration_temperature": new_temp,
            "implementer_max_ast_depth": new_depth,
            "memory_decay_rate": round(float(np.clip(0.92 + rng.normal(0.0, 0.02), 0.85, 0.97)), 3),
        }
        return {
            "rationale": "Shift search budget toward validated microstructure/foreign-flow/PIT mechanisms, refine lookback windows to match T+2.5 cycle (3-15d), and increase verifier best-of-n.",
            "updates": updates,
        }


def build_llm_backend(llm_cfg: dict[str, Any], use_teacher: bool = False) -> LLMBackend:
    """Factory selecting configured local <=8B server or graceful CPU scaffolding fallback."""
    if use_teacher:
        t_cfg = llm_cfg.get("teacher_backend", {})
        if t_cfg.get("enabled", False):
            return OpenAICompatLocalBackend(
                model_name=str(t_cfg["name"]),
                parameter_count_billions=float(t_cfg["parameter_count_billions"]),
                endpoint_url=str(t_cfg["endpoint_url"]),
                is_teacher=True,
                allow_teacher=True,
            )
        return RuleGuidedSmallLMBackend(
            model_name=f"{t_cfg.get('name', 'teacher-72b')}-sim",
            parameter_count_billions=float(t_cfg.get("parameter_count_billions", 72.0)),
            is_teacher=True,
            allow_teacher=True,
        )

    mode = str(llm_cfg.get("active_backend", "local_8b_sim_or_ollama"))
    m_cfg = llm_cfg.get("model", {})
    model_name = str(m_cfg.get("name", "qwen3:8b-q4_K_M"))
    params_b = float(m_cfg.get("parameter_count_billions", 8.0))
    max_allowed = float(m_cfg.get("max_allowed_main_loop_params_b", 8.0))
    endpoint = str(m_cfg.get("endpoint_url", "http://127.0.0.1:11434/v1/chat/completions"))
    max_toks = int(m_cfg.get("max_tokens", 768))

    if mode in {"ollama", "vllm", "llamacpp"}:
        return OpenAICompatLocalBackend(
            model_name=model_name,
            parameter_count_billions=params_b,
            endpoint_url=endpoint,
            max_tokens=max_toks,
            is_teacher=False,
            allow_teacher=False,
        )

    if mode == "local_8b_sim_or_ollama":
        # Probe local server quickly; degrade gracefully to deterministic <=8B scaffolding if offline
        try:
            with httpx.Client(timeout=0.6) as client:
                probe = client.get(endpoint.rsplit("/v1/", 1)[0])
                if probe.status_code < 500:
                    return OpenAICompatLocalBackend(
                        model_name=model_name,
                        parameter_count_billions=params_b,
                        endpoint_url=endpoint,
                        max_tokens=max_toks,
                    )
        except Exception:
            pass

    return RuleGuidedSmallLMBackend(
        model_name=f"{model_name}-cpu-scaffold",
        parameter_count_billions=params_b,
        is_teacher=False,
        allow_teacher=False,
        # Enforces <= max_allowed in super().__init__
    ) if params_b <= max_allowed else OpenAICompatLocalBackend(
        model_name=model_name,
        parameter_count_billions=params_b,
        endpoint_url=endpoint,
    )
