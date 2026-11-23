"""Modular Micro-RSI Backend combining Tiny SLM (Ideation) + Discrete AST Diffusion (Synthesis) + HRM (Planning/Scoring).

Plugs directly into `MultiAgentResearchPipeline` and `BiLevelRSIRunner` without modifying the Sealed Sandbox.
Supports weight-level recursive self-improvement (`self_improve_weights`) strictly on search-zone trajectories.
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np

from vn_rsi.dsl.ast_spec import FactorSpec
from vn_rsi.llm.backend import LLMBackend, RuleGuidedSmallLMBackend
from vn_rsi.models.ast_diffusion import ASTDiffusionConfig, DiscreteASTDiffusion
from vn_rsi.models.hrm_reasoner import HRMConfig, HRMReasoner


class ModularMicroRSIBackend(LLMBackend):
    """Decomposed 3-model backend staying strictly under <= 4.1B total active parameters (free-GPU friendly)."""

    def __init__(
        self,
        preset: str = "cpu_fast",
        narrator_backend: LLMBackend | None = None,
        diffusion_device: str | None = None,
        hrm_device: str | None = None,
        seed: int = 42,
    ) -> None:
        self.preset = preset
        self.narrator = narrator_backend or RuleGuidedSmallLMBackend(
            model_name="qwen3-1.7b-narrator-scaffold",
            parameter_count_billions=1.7,
        )
        self.diffusion = DiscreteASTDiffusion(
            config=ASTDiffusionConfig.from_preset(preset, seed=seed),
            device=diffusion_device,
        )
        self.hrm = HRMReasoner(
            config=HRMConfig.from_preset(preset, seed=seed + 17),
            device=hrm_device,
        )
        self.elite_pool: dict[str, list[FactorSpec]] = {}

        total_b = round(
            self.narrator.parameter_count_billions
            + (self.diffusion.parameter_count() + self.hrm.parameter_count()) / 1e9,
            4,
        )
        super().__init__(
            model_name=f"modular_micro_rsi[{preset}]",
            parameter_count_billions=total_b,
            is_teacher=False,
            allow_teacher=False,
            max_allowed_params_b=8.0,
        )

    @property
    def backend_id(self) -> str:
        return (
            f"modular_micro_rsi:{self.preset}("
            f"slm={self.narrator.parameter_count_billions}B,"
            f"diff={self.diffusion.parameter_count()},"
            f"hrm={self.hrm.parameter_count()})"
        )

    def register_verified_search_winner(self, family: str, spec: FactorSpec) -> None:
        """Store verified search-zone FactorSpec as parent template for Discrete Diffusion subtree infilling."""
        pool = self.elite_pool.setdefault(family, [])
        pool.append(spec)
        if len(pool) > 8:
            del pool[0]

    def self_improve_weights(
        self,
        verified_trajectories: list[dict[str, Any]],
        hrm_observations: list[dict[str, Any]],
        epochs: int = 6,
    ) -> dict[str, Any]:
        """True weight-level RSI on micro-models (<20M params), strictly restricted to search-zone evidence."""
        diff_stats = self.diffusion.fit_on_trajectories(verified_trajectories, epochs=epochs)
        hrm_stats = self.hrm.train_on_search_observations(hrm_observations, epochs=epochs)
        return {"diffusion_update": diff_stats, "hrm_update": hrm_stats}

    def _raw_complete(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any],
        temperature: float,
        seed: int,
    ) -> tuple[str, int, int, float]:
        t0 = time.monotonic()
        title = json_schema.get("title", "")

        # Role A: Natural-language falsifiable hypothesis ideation -> Tiny SLM Narrator (Qwen3-1.7B / 4B)
        if title == "FalsifiableHypothesis":
            return self.narrator._raw_complete(system_prompt, user_prompt, json_schema, temperature, seed)

        ctx: dict[str, Any] = {}
        if "CONTEXT_JSON:" in user_prompt:
            raw_ctx = user_prompt.split("CONTEXT_JSON:", 1)[1].strip().split("\n")[0]
            try:
                ctx = json.loads(raw_ctx)
            except Exception:
                ctx = {}

        # Role B: Causal AST synthesis / subtree infilling + HRM surrogate pre-ranking -> Discrete AST Diffusion + HRM
        if title == "FactorSpec":
            family = str(ctx.get("family", "momentum_liquidity"))
            hyp_id = str(ctx.get("hypothesis_id", "hyp_0"))
            factor_id = str(ctx.get("factor_id", f"fac_{hyp_id}"))
            max_depth = int(ctx.get("max_depth", 3))
            regime_vec = self.hrm.encode_regime_state(ctx.get("steward"), ctx.get("family_ic_summary"))
            rng = np.random.default_rng(seed)

            # Generate 3 micro-candidates via Discrete AST Diffusion (mixing fresh draws & subtree infilling)
            # and pick the highest HRM z_L surrogate Rank IC prediction before touching SealedEvaluator!
            cands: list[tuple[float, FactorSpec]] = []
            elites = self.elite_pool.get(family, [])
            for k in range(3):
                if elites and k == 0:
                    parent = elites[int(rng.integers(0, len(elites)))]
                    sp = self.diffusion.infill_mutate_factor(
                        parent_spec=parent,
                        family=family,
                        factor_id=factor_id,
                        hypothesis_id=hyp_id,
                        max_depth=max_depth,
                        temperature=max(0.30, temperature),
                        seed=seed + k * 19,
                    )
                else:
                    sp = self.diffusion.sample_factor(
                        family=family,
                        factor_id=factor_id,
                        hypothesis_id=hyp_id,
                        max_depth=max_depth,
                        temperature=max(0.35, temperature),
                        seed=seed + k * 19,
                    )
                toks = self.diffusion.vocab.encode_ast(sp.root)
                sur_score = self.hrm.score_candidate_ast(regime_vec, toks)
                cands.append((sur_score, sp))

            cands.sort(key=lambda item: item[0], reverse=True)
            best_spec = cands[0][1]
            txt = json.dumps(best_spec.model_dump(), sort_keys=True)
            dt = max(0.0005, time.monotonic() - t0)
            return txt, max(32, len(user_prompt) // 4), max(24, len(txt) // 4), dt

        # Role C: Outer-Loop Meta-Proposal -> HRM High-Level Module (z_H)
        if title == "GenomePatchProposal":
            fam_ic = ctx.get("family_ic_summary", {})
            iteration = int(ctx.get("iteration", 0))
            regime_vec = self.hrm.encode_regime_state(None, fam_ic)
            updates = self.hrm.propose_outer_policy(regime_vec, fam_ic, iteration=iteration)
            payload = {
                "rationale": (
                    f"HRM z_H multi-timescale update (iter {iteration}): reallocate focus across VN microstructure "
                    "families and adjust causal AST depth + verifier best-of-n."
                ),
                "updates": updates,
            }
            txt = json.dumps(payload, sort_keys=True)
            dt = max(0.0005, time.monotonic() - t0)
            return txt, max(32, len(user_prompt) // 4), max(24, len(txt) // 4), dt

        return self.narrator._raw_complete(system_prompt, user_prompt, json_schema, temperature, seed)
