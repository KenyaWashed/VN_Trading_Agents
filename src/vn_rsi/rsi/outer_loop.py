"""Bi-Level Recursive Self-Improvement Loop, Statistical Acceptance Gate, Ignition & OOD Checks (Phase 4)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from vn_rsi.agents.pipeline import MultiAgentResearchPipeline
from vn_rsi.dsl.ast_spec import FactorSpec
from vn_rsi.eval.evaluator import SealedEvaluator
from vn_rsi.llm.backend import LLMBackend
from vn_rsi.rsi.archive import ArchiveEntry, DGMArchive
from vn_rsi.rsi.mutable_surface import (
    GenomePatchProposal,
    ImmutableSurfaceViolationError,
    MutableSurfaceGuard,
)
from vn_rsi.utils.budget import BudgetConfig, BudgetTracker


@dataclass
class OuterIterationLog:
    iteration: int
    parent_id: str
    child_id: str
    rationale: str
    parent_mean_score: float
    child_mean_score: float
    mean_improvement: float
    win_fraction: float
    p_value_one_sided: float
    accepted: bool
    rejection_reason: str
    tokens_cumulative: int


@dataclass
class RSIRunSummary:
    mode_name: str
    seed_entry: ArchiveEntry
    best_entry: ArchiveEntry
    iteration_logs: list[OuterIterationLog]
    accepted_rewrites: int
    rejected_rewrites: int
    reward_hacking_attempts: int
    outer_budget_snapshot: dict[str, Any]
    archive_size: int
    extra_diagnostics: dict[str, Any] = field(default_factory=dict)


class BiLevelRSIRunner:
    """Implements AIDE^2 + DGM + AQuA bi-level self-improvement scaled down to <=8B models."""

    def __init__(
        self,
        evaluator: SealedEvaluator,
        inner_llm: LLMBackend,
        outer_proposer_llm: LLMBackend,
        rsi_cfg: dict[str, Any],
        agent_cfg: dict[str, Any],
        llm_cfg: dict[str, Any],
    ) -> None:
        self.evaluator = evaluator
        self.inner_llm = inner_llm
        self.outer_proposer_llm = outer_proposer_llm
        self.rsi_cfg = rsi_cfg
        self.agent_cfg = agent_cfg
        self.llm_cfg = llm_cfg

        self.surface_guard = MutableSurfaceGuard(list(rsi_cfg["mutable_surface"]))

    @staticmethod
    def composite_validation_score(net_sharpe: float, rank_ic: float) -> float:
        """Private outer-loop selection objective combining risk-adjusted net return and cross-sectional IC."""
        return float(net_sharpe + 10.0 * rank_ic)

    def _evaluate_genome_multi_seed(
        self,
        variant_id: str,
        parent_id: str | None,
        generation: int,
        genome: dict[str, Any],
        seeds: list[int],
        outer_budget: BudgetTracker,
        enable_memory: bool = True,
        enable_auditor: bool = True,
        sft_export_path: str | None = None,
    ) -> ArchiveEntry:
        b_cfg = self.llm_cfg.get("budgets", {})
        vram = float(self.llm_cfg.get("hardware", {}).get("configured_vram_gb", 0.0))
        inner_rounds = int(self.rsi_cfg.get("inner_rounds_per_eval", 2))
        inner_hyps = int(self.rsi_cfg.get("inner_hypotheses_per_round", 3))
        acc_cfg = self.agent_cfg.get("acceptance_criteria", {})

        scores: list[float] = []
        sharpes: list[float] = []
        ics: list[float] = []
        rep_specs: list[FactorSpec] = []
        rep_weights: list[float] = []
        agg_fam: dict[str, list[float]] = {}

        for sd in seeds:
            outer_budget.check()
            inner_budget = BudgetTracker(
                BudgetConfig(
                    max_tokens=int(b_cfg.get("inner_loop_max_tokens", 60_000)),
                    max_wall_seconds=float(b_cfg.get("inner_loop_max_wall_seconds", 180.0)),
                    max_gpu_hours=float(b_cfg.get("inner_loop_max_gpu_hours", 0.25)),
                    configured_vram_gb=vram,
                ),
                scope_name=f"inner_{variant_id}_s{sd}",
            )
            pipe = MultiAgentResearchPipeline(
                evaluator=self.evaluator,
                llm=self.inner_llm,
                genome=genome,
                acceptance_cfg=acc_cfg,
                enable_memory=enable_memory,
                enable_auditor=enable_auditor,
            )
            res = pipe.run(
                agent_version=variant_id,
                n_rounds=inner_rounds,
                n_hypotheses_per_round=inner_hyps,
                budget=inner_budget,
                seed=sd,
                run_id=f"rsi_{variant_id}_s{sd}",
                sft_export_path=sft_export_path,
            )
            # Charge inner loop tokens to outer loop hard budget as well
            outer_budget.record_usage(inner_budget.tokens_used, 0, call_seconds=inner_budget.elapsed_wall_seconds)

            v_sr = res.validation_outcome.metrics["net_sharpe"]
            v_ic = res.validation_outcome.metrics["rank_ic"]
            comp = self.composite_validation_score(v_sr, v_ic)
            scores.append(comp)
            sharpes.append(v_sr)
            ics.append(v_ic)
            if not rep_specs:
                rep_specs = res.accepted_specs
                rep_weights = res.accepted_weights
            for fk, fv in res.family_ic_summary.items():
                agg_fam.setdefault(fk, []).append(fv)

        fam_mean = {k: round(float(np.mean(vs)), 5) for k, vs in agg_fam.items()}
        return ArchiveEntry(
            variant_id=variant_id,
            parent_id=parent_id,
            generation=generation,
            genome=genome,
            genome_hash=DGMArchive.hash_genome(genome),
            val_composite_mean=round(float(np.mean(scores)), 5),
            val_composite_std=round(float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0, 5),
            val_sharpe_mean=round(float(np.mean(sharpes)), 4),
            val_rank_ic_mean=round(float(np.mean(ics)), 5),
            seed_scores=[round(s, 5) for s in scores],
            representative_specs=rep_specs,
            representative_weights=rep_weights,
            family_ic_summary=fam_mean,
        )

    def _statistical_acceptance_gate(
        self, parent: ArchiveEntry, child: ArchiveEntry
    ) -> tuple[bool, float, float, float, str]:
        gate_cfg = self.rsi_cfg.get("selection_gate", {})
        min_imp = float(gate_cfg.get("min_mean_improvement", 0.003))
        min_win = float(gate_cfg.get("min_win_fraction_across_seeds", 0.66))
        max_p = float(gate_cfg.get("max_p_value_paired_t", 0.25))

        p_arr = np.asarray(parent.seed_scores, dtype=float)
        c_arr = np.asarray(child.seed_scores, dtype=float)
        diffs = c_arr - p_arr
        mean_imp = float(np.mean(diffs))
        win_frac = float(np.mean(diffs > 0.0))

        if len(diffs) >= 2 and float(np.std(diffs, ddof=1)) > 1e-10:
            t_stat, p_two = stats.ttest_rel(c_arr, p_arr)
            p_one = float(p_two / 2.0) if t_stat > 0 else 1.0 - float(p_two / 2.0)
        else:
            p_one = 0.05 if mean_imp > 0 else 0.95

        if mean_imp < min_imp:
            return False, mean_imp, win_frac, p_one, f"mean_imp {mean_imp:.4f} < {min_imp:.4f}"
        if win_frac < min_win:
            return False, mean_imp, win_frac, p_one, f"win_frac {win_frac:.2f} < {min_win:.2f}"
        if p_one > max_p:
            return False, mean_imp, win_frac, p_one, f"paired_p {p_one:.3f} > {max_p:.3f}"
        return True, mean_imp, win_frac, p_one, "accepted_statistically"

    def run_mode(
        self,
        mode_name: str = "full_rsi_8b",
        seed_override: int | None = None,
    ) -> RSIRunSummary:
        """Execute a complete bi-level RSI run or one of the required ablations."""
        base_seed = seed_override if seed_override is not None else int(self.rsi_cfg.get("seed", 2026))
        rng = np.random.default_rng(base_seed)

        b_cfg = self.llm_cfg.get("budgets", {})
        vram = float(self.llm_cfg.get("hardware", {}).get("configured_vram_gb", 0.0))
        outer_budget = BudgetTracker(
            BudgetConfig(
                max_tokens=int(b_cfg.get("outer_loop_max_tokens", 300_000)),
                max_wall_seconds=float(b_cfg.get("outer_loop_max_wall_seconds", 600.0)),
                max_gpu_hours=float(b_cfg.get("outer_loop_max_gpu_hours", 1.50)),
                configured_vram_gb=vram,
            ),
            scope_name=f"outer_{mode_name}",
        )

        arch_cfg = self.rsi_cfg.get("archive", {})
        archive = DGMArchive(
            max_variants=int(arch_cfg.get("max_variants", 12)),
            fitness_weight=float(arch_cfg.get("fitness_weight", 0.75)),
            novelty_weight=float(arch_cfg.get("novelty_weight", 0.25)),
        )

        enable_memory = mode_name != "no_memory"
        enable_auditor = mode_name != "no_leakage_auditor"
        outer_iters = 0 if mode_name == "no_outer_loop" else int(self.rsi_cfg.get("outer_iterations", 4))
        sel_seeds = [int(s) for s in self.rsi_cfg.get("selection_seeds", [101, 202, 303])]
        sft_path = (
            str(self.rsi_cfg.get("self_training", {}).get("sft_output_path"))
            if mode_name == "full_rsi_8b" and self.rsi_cfg.get("self_training", {}).get("export_sft_trajectories", False)
            else None
        )

        init_genome = dict(self.agent_cfg["genome"])
        seed_entry = self._evaluate_genome_multi_seed(
            variant_id=f"{mode_name}_v0",
            parent_id=None,
            generation=0,
            genome=init_genome,
            seeds=sel_seeds,
            outer_budget=outer_budget,
            enable_memory=enable_memory,
            enable_auditor=enable_auditor,
            sft_export_path=sft_path,
        )
        archive.add(seed_entry)

        iter_logs: list[OuterIterationLog] = []
        accepted_cnt = 0
        rejected_cnt = 0
        hacking_attempts = 0

        for it in range(outer_iters):
            outer_budget.check()
            parent = archive.sample_parent(rng)
            child_id = f"{mode_name}_v{it + 1}"

            if mode_name == "random_mutation":
                rand_fams = rng.dirichlet(np.ones(4))
                f_names = ["momentum_liquidity", "reversal_microstructure", "value_quality_pit", "foreign_flow_regime"]
                proposal = GenomePatchProposal(
                    rationale="Random undirected mutation baseline on declared mutable surface.",
                    updates={
                        "planner_focus_weights": {f: round(float(w), 4) for f, w in zip(f_names, rand_fams, strict=True)},
                        "preferred_windows": [int(x) for x in rng.choice([3, 5, 8, 12, 20, 30], size=3, replace=False)],
                        "exploration_temperature": round(float(rng.uniform(0.15, 0.75)), 3),
                        "best_of_n": int(rng.choice([1, 2, 3, 4])),
                    },
                )
            else:
                prop_ctx = {
                    "iteration": it,
                    "parent_id": parent.variant_id,
                    "parent_genome": parent.genome,
                    "parent_val_score": parent.val_composite_mean,
                    "family_ic_summary": parent.family_ic_summary,
                }
                prop_obj, _ = self.outer_proposer_llm.generate_structured(
                    system_prompt=(
                        "You are the outer-loop RSI meta-architect. Propose bounded edits strictly inside "
                        f"mutable_surface={sorted(self.surface_guard.allowed_keys)}. Never touch evaluator or splits."
                    ),
                    user_prompt=f"Improve research agent genome.\nCONTEXT_JSON:{json.dumps(prop_ctx)}",
                    schema_cls=GenomePatchProposal,
                    budget=outer_budget,
                    temperature=float(parent.genome.get("exploration_temperature", 0.35)),
                    seed=base_seed + it * 97,
                )
                assert isinstance(prop_obj, GenomePatchProposal)
                proposal = prop_obj

            try:
                child_genome = self.surface_guard.apply_patch(parent.genome, proposal)
            except ImmutableSurfaceViolationError as exc:
                hacking_attempts += 1
                rejected_cnt += 1
                iter_logs.append(
                    OuterIterationLog(
                        iteration=it,
                        parent_id=parent.variant_id,
                        child_id=child_id,
                        rationale=proposal.rationale,
                        parent_mean_score=parent.val_composite_mean,
                        child_mean_score=0.0,
                        mean_improvement=-999.0,
                        win_fraction=0.0,
                        p_value_one_sided=1.0,
                        accepted=False,
                        rejection_reason=f"immutable_surface_violation:{exc}",
                        tokens_cumulative=outer_budget.tokens_used,
                    )
                )
                continue

            child_entry = self._evaluate_genome_multi_seed(
                variant_id=child_id,
                parent_id=parent.variant_id,
                generation=parent.generation + 1,
                genome=child_genome,
                seeds=sel_seeds,
                outer_budget=outer_budget,
                enable_memory=enable_memory,
                enable_auditor=enable_auditor,
                sft_export_path=sft_path,
            )

            accepted, mean_imp, win_frac, p_val, reason = self._statistical_acceptance_gate(parent, child_entry)
            if accepted:
                accepted_cnt += 1
                archive.add(child_entry)
            else:
                rejected_cnt += 1

            iter_logs.append(
                OuterIterationLog(
                    iteration=it,
                    parent_id=parent.variant_id,
                    child_id=child_id,
                    rationale=proposal.rationale,
                    parent_mean_score=parent.val_composite_mean,
                    child_mean_score=child_entry.val_composite_mean,
                    mean_improvement=round(mean_imp, 5),
                    win_fraction=round(win_frac, 4),
                    p_value_one_sided=round(p_val, 4),
                    accepted=accepted,
                    rejection_reason=reason,
                    tokens_cumulative=outer_budget.tokens_used,
                )
            )

        best = archive.best_entry()
        return RSIRunSummary(
            mode_name=mode_name,
            seed_entry=seed_entry,
            best_entry=best,
            iteration_logs=iter_logs,
            accepted_rewrites=accepted_cnt,
            rejected_rewrites=rejected_cnt,
            reward_hacking_attempts=hacking_attempts,
            outer_budget_snapshot=outer_budget.snapshot(),
            archive_size=len(archive.entries),
        )

    def run_ignition_check(
        self,
        v0_genome: dict[str, Any],
        evolved_genome: dict[str, Any],
        ignition_seeds: list[int],
    ) -> dict[str, Any]:
        """Ignition test (Section 6): Does evolved genome v* drive further outer-loop gains >= v0 across fresh seeds?"""
        b_cfg = self.llm_cfg.get("budgets", {})
        budget = BudgetTracker(
            BudgetConfig(
                max_tokens=int(b_cfg.get("outer_loop_max_tokens", 300_000)),
                max_wall_seconds=float(b_cfg.get("outer_loop_max_wall_seconds", 600.0)),
                max_gpu_hours=float(b_cfg.get("outer_loop_max_gpu_hours", 1.50)),
            ),
            scope_name="ignition_check",
        )
        v0_entry = self._evaluate_genome_multi_seed("ign_v0", None, 0, v0_genome, ignition_seeds, budget)
        vstar_entry = self._evaluate_genome_multi_seed("ign_vstar", None, 1, evolved_genome, ignition_seeds, budget)

        diffs = np.asarray(vstar_entry.seed_scores) - np.asarray(v0_entry.seed_scores)
        mean_diff = float(np.mean(diffs))
        if len(diffs) >= 2 and float(np.std(diffs, ddof=1)) > 1e-10:
            t_stat, p_two = stats.ttest_rel(vstar_entry.seed_scores, v0_entry.seed_scores)
            p_one = float(p_two / 2.0) if t_stat > 0 else 1.0 - float(p_two / 2.0)
        else:
            p_one = 0.5

        status = "CONCLUSIVE_POSITIVE" if (mean_diff > 0.0 and p_one < 0.10) else "INCONCLUSIVE"
        return {
            "status": status,
            "ignition_seeds": ignition_seeds,
            "v0_mean_val_composite": v0_entry.val_composite_mean,
            "vstar_mean_val_composite": vstar_entry.val_composite_mean,
            "mean_gain_on_fresh_seeds": round(mean_diff, 5),
            "paired_p_one_sided": round(p_one, 4),
            "note": (
                "Evaluated across independent fresh seeds; labeled INCONCLUSIVE whenever p >= 0.10 "
                "per honest scientific reporting rule."
            ),
        }
