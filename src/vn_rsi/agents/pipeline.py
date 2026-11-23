"""Constrained 7-Role Multi-Agent Research Pipeline on <=8B LLMs (Phase 3).

Roles:
1. Planner: Allocates round search across mechanism families using genome weights + belief memory.
2. DataSteward: Summarizes ONLY `search`-zone microstructure regimes via DataZoneGuard.
3. IdeaMiner: Emits falsifiable hypotheses (statement, mechanism, sign, refutation criteria) BEFORE building.
4. Implementer: Emits typed FactorSpec ASTs via JSON-schema constrained decoding + best-of-n verifier.
5. LeakageAuditor: Enforces AST causality, shift-test invariance on search subset, and novelty filtering.
6. EvaluatorProxy: Tests refutation criteria on `search` zone, scores ensemble on `validation` scalar view.
7. Librarian: Updates EpistemicBeliefMemory with run-ID links, decay, contradiction handling, & SFT trajectories.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

from vn_rsi.agents.schemas import AuditVerdict, FalsifiableHypothesis
from vn_rsi.data.qa import DataQASuite
from vn_rsi.dsl.ast_spec import ASTValidationError, ASTVerifier, FactorSpec, evaluate_factor_spec
from vn_rsi.eval.evaluator import EvaluationOutcome, SealedEvaluator
from vn_rsi.llm.backend import LLMBackend
from vn_rsi.memory.belief_store import EpistemicBeliefMemory
from vn_rsi.utils.budget import BudgetTracker


@dataclass
class CandidateRecord:
    round_idx: int
    hypothesis: FalsifiableHypothesis
    spec: FactorSpec
    audit: AuditVerdict
    search_rank_ic: float
    search_net_sharpe: float
    accepted_on_search: bool


@dataclass
class AgentPipelineResult:
    agent_version: str
    seed: int
    accepted_specs: list[FactorSpec]
    accepted_weights: list[float]
    candidate_records: list[CandidateRecord]
    validation_outcome: EvaluationOutcome
    audit_rejection_count: int
    refuted_on_search_count: int
    memory_snapshot: list[dict[str, Any]]
    family_ic_summary: dict[str, float]
    budget_snapshot: dict[str, Any]
    verified_sft_trajectories: list[dict[str, Any]] = field(default_factory=list)


class MultiAgentResearchPipeline:
    """Executes inner-loop multi-agent quantitative factor research under a fixed budget."""

    FAMILIES = (
        "momentum_liquidity",
        "reversal_microstructure",
        "value_quality_pit",
        "foreign_flow_regime",
    )

    def __init__(
        self,
        evaluator: SealedEvaluator,
        llm: LLMBackend,
        genome: dict[str, Any],
        acceptance_cfg: dict[str, float],
        enable_memory: bool = True,
        enable_auditor: bool = True,
    ) -> None:
        self.evaluator = evaluator
        self.llm = llm
        self.genome = genome
        self.acceptance_cfg = acceptance_cfg
        self.enable_memory = enable_memory
        self.enable_auditor = enable_auditor

        self.memory = EpistemicBeliefMemory(
            decay_rate=float(genome.get("memory_decay_rate", 0.90)),
            contradiction_penalty=float(genome.get("contradiction_penalty", 0.35)),
        )
        self.ast_verifier = ASTVerifier(max_depth=int(genome.get("implementer_max_ast_depth", 3)))

    def _data_steward_search_summary(self) -> dict[str, float]:
        """Role 2 (DataSteward): Reads ONLY `search` zone via DataZoneGuard."""
        search_bars = self.evaluator.zone_guard.get_agent_visible_bars(self.evaluator.snapshot.bars, "search")
        return {
            "search_rows": float(len(search_bars)),
            "ceiling_hit_rate": round(float(search_bars["hit_ceiling"].mean()), 5),
            "floor_hit_rate": round(float(search_bars["hit_floor"].mean()), 5),
            "halt_rate": round(float(search_bars["is_halted"].mean()), 5),
        }

    def _planner_select_families(self, n_hypotheses: int, rng: np.random.Generator) -> list[str]:
        """Role 1 (Planner): Combines genome focus weights with belief confidence."""
        raw_w = self.genome.get("planner_focus_weights", {})
        probs = np.array([max(0.05, float(raw_w.get(f, 0.25))) for f in self.FAMILIES], dtype=float)
        if self.enable_memory and self.memory.beliefs:
            for idx, fam in enumerate(self.FAMILIES):
                if fam in self.memory.beliefs:
                    b = self.memory.beliefs[fam]
                    probs[idx] *= 0.6 + 0.8 * b.confidence
        probs = probs / probs.sum()
        chosen_idx = rng.choice(len(self.FAMILIES), size=n_hypotheses, replace=True, p=probs)
        return [self.FAMILIES[int(i)] for i in chosen_idx]

    def _leakage_auditor_check(self, spec: FactorSpec, sample_bars: pd.DataFrame) -> AuditVerdict:
        """Role 5 (LeakageAuditor): Structural causality + mini shift-test + duplicate AST check."""
        fp = spec.root.structural_fingerprint()
        if not self.enable_auditor:
            return AuditVerdict(
                passed=True,
                ast_fingerprint=fp,
                is_novel=True,
                shift_test_passed=True,
                reason="auditor_disabled_ablation",
            )

        try:
            self.ast_verifier.verify(spec)
        except ASTValidationError as exc:
            return AuditVerdict(
                passed=False,
                ast_fingerprint=fp,
                is_novel=False,
                shift_test_passed=False,
                reason=f"ast_invalid:{exc}",
            )

        is_novel = fp not in self.memory.seen_ast_fingerprints
        if not is_novel and float(self.genome.get("novelty_similarity_threshold", 0.85)) < 0.99:
            return AuditVerdict(
                passed=False,
                ast_fingerprint=fp,
                is_novel=False,
                shift_test_passed=True,
                reason="duplicate_ast_rejected_by_novelty_filter",
            )

        # Dynamic shift-test on a compact 2-symbol search-zone subset to guarantee zero future look-ahead
        mini_syms = sorted(sample_bars["symbol"].unique().tolist())[:2]
        mini_bars = sample_bars[sample_bars["symbol"].isin(mini_syms)].copy()
        mini_fund = self.evaluator.snapshot.fundamentals_pit[
            self.evaluator.snapshot.fundamentals_pit["symbol"].isin(mini_syms)
        ].copy()

        def _fn(b_df: pd.DataFrame) -> pd.Series:
            from vn_rsi.dsl.ast_spec import compute_feature_panel

            p = compute_feature_panel(b_df, mini_fund)
            mat = evaluate_factor_spec(spec, p, verifier=self.ast_verifier)
            stacked = mat.stack(future_stack=True)
            return b_df.set_index(["date", "symbol"]).index.map(stacked).to_series().fillna(0.0)

        shift_ok = DataQASuite.verify_feature_causality_shift_test(mini_bars, _fn)
        return AuditVerdict(
            passed=shift_ok,
            ast_fingerprint=fp,
            is_novel=is_novel,
            shift_test_passed=shift_ok,
            reason="passed_all_checks" if shift_ok else "failed_dynamic_shift_test",
        )

    def run(
        self,
        agent_version: str,
        n_rounds: int,
        n_hypotheses_per_round: int,
        budget: BudgetTracker,
        seed: int = 101,
        run_id: str = "agent_v0",
        sft_export_path: str | Path | None = None,
    ) -> AgentPipelineResult:
        rng = np.random.default_rng(seed)
        steward_stats = self._data_steward_search_summary()
        search_bars = self.evaluator.zone_guard.get_agent_visible_bars(self.evaluator.snapshot.bars, "search")

        accepted_specs: list[FactorSpec] = []
        accepted_weights: list[float] = []
        records: list[CandidateRecord] = []
        sft_trajectories: list[dict[str, Any]] = []
        audit_rejections = 0
        search_refutations = 0

        min_search_ic = float(self.acceptance_cfg.get("min_search_rank_ic", 0.012))
        temp = float(self.genome.get("exploration_temperature", 0.35))
        best_n = int(self.genome.get("best_of_n", 3))
        max_depth = int(self.genome.get("implementer_max_ast_depth", 3))
        pref_windows = list(self.genome.get("preferred_windows", [5, 10, 20]))
        miner_sys = str(self.genome.get("idea_miner_system_prompt", "Propose falsifiable VN equity hypotheses."))

        for r_idx in range(n_rounds):
            budget.check()
            if self.enable_memory and r_idx > 0:
                self.memory.step_decay()

            families = self._planner_select_families(n_hypotheses_per_round, rng)
            mem_ctx = self.memory.retrieve_compact_context(top_k=4) if self.enable_memory else []

            for h_idx, fam in enumerate(families):
                budget.check()
                hyp_id = f"{agent_version}_r{r_idx}_h{h_idx}"
                ctx_payload = {
                    "family": fam,
                    "hypothesis_id": hyp_id,
                    "steward": steward_stats,
                    "memory": mem_ctx,
                }
                # Role 3: IdeaMiner emits FalsifiableHypothesis BEFORE implementation
                hyp_obj, _ = self.llm.generate_structured(
                    system_prompt=miner_sys,
                    user_prompt=f"Formulate falsifiable hypothesis.\nCONTEXT_JSON:{json.dumps(ctx_payload)}",
                    schema_cls=FalsifiableHypothesis,
                    budget=budget,
                    temperature=temp,
                    seed=seed + r_idx * 100 + h_idx * 13,
                    best_of_n=1,
                )
                assert isinstance(hyp_obj, FalsifiableHypothesis)

                # Role 4: Implementer emits typed FactorSpec with deterministic AST verifier best-of-n
                imp_ctx = {
                    "family": fam,
                    "hypothesis_id": hyp_id,
                    "factor_id": f"fac_{hyp_id}",
                    "preferred_windows": pref_windows,
                    "max_depth": max_depth,
                }

                def _ast_verifier_score(model_inst: BaseModel) -> float:
                    assert isinstance(model_inst, FactorSpec)
                    self.ast_verifier.verify(model_inst)
                    fp_cand = model_inst.root.structural_fingerprint()
                    novel_bonus = 1.0 if fp_cand not in self.memory.seen_ast_fingerprints else 0.0
                    # Slight preference for expressive depth 2-3 causal combinations over trivial depth-1
                    depth_score = 0.4 if model_inst.root.depth() >= 2 else 0.1
                    return novel_bonus + depth_score

                spec_obj, _ = self.llm.generate_structured(
                    system_prompt="Emit strictly causal FactorSpec AST matching hypothesis.",
                    user_prompt=f"Implement hypothesis {hyp_obj.statement}\nCONTEXT_JSON:{json.dumps(imp_ctx)}",
                    schema_cls=FactorSpec,
                    budget=budget,
                    temperature=temp,
                    seed=seed + r_idx * 100 + h_idx * 13 + 7,
                    best_of_n=best_n,
                    verifier_fn=_ast_verifier_score,
                )
                assert isinstance(spec_obj, FactorSpec)

                # Role 5: LeakageAuditor
                audit = self._leakage_auditor_check(spec_obj, search_bars)
                if not audit.passed:
                    audit_rejections += 1
                    records.append(
                        CandidateRecord(
                            round_idx=r_idx,
                            hypothesis=hyp_obj,
                            spec=spec_obj,
                            audit=audit,
                            search_rank_ic=0.0,
                            search_net_sharpe=0.0,
                            accepted_on_search=False,
                        )
                    )
                    continue

                # Role 6: EvaluatorProxy tests falsifiable prediction on `search` zone ONLY
                # If expected_sign == -1, negate via causal `neg` wrapper so positive score = long candidate
                effective_spec = spec_obj
                if hyp_obj.expected_sign == -1 and spec_obj.root.name != "neg":
                    from vn_rsi.dsl.ast_spec import ASTNode

                    effective_spec = FactorSpec(
                        factor_id=spec_obj.factor_id,
                        hypothesis_id=spec_obj.hypothesis_id,
                        description=spec_obj.description,
                        root=ASTNode(node_type="op", name="neg", args=[spec_obj.root]),
                        normalize_cross_section=spec_obj.normalize_cross_section,
                    )

                search_out = self.evaluator.evaluate_factor(effective_spec, zone="search", run_id=run_id)
                s_ic = search_out.metrics["rank_ic"]
                s_sr = search_out.metrics["net_sharpe"]
                accepted = (s_ic >= min_search_ic) and (s_sr > 0.0)
                if not accepted:
                    search_refutations += 1
                else:
                    accepted_specs.append(effective_spec)
                    accepted_weights.append(max(0.01, s_ic))
                    sft_trajectories.append(
                        {
                            "run_id": run_id,
                            "agent_version": agent_version,
                            "family": fam,
                            "hypothesis": hyp_obj.model_dump(),
                            "factor_spec": effective_spec.model_dump(),
                            "search_rank_ic": s_ic,
                            "search_net_sharpe": s_sr,
                        }
                    )

                # Role 7: Librarian updates persistent belief memory
                if self.enable_memory:
                    self.memory.record_outcome(
                        mechanism_family=fam,
                        statement=hyp_obj.statement,
                        expected_sign=1,  # effective_spec is oriented positively
                        observed_rank_ic=s_ic,
                        accepted=accepted,
                        run_id=f"{run_id}_{hyp_id}",
                        round_idx=r_idx,
                        ast_fingerprint=audit.ast_fingerprint,
                    )
                else:
                    self.memory.seen_ast_fingerprints.add(audit.ast_fingerprint)

                records.append(
                    CandidateRecord(
                        round_idx=r_idx,
                        hypothesis=hyp_obj,
                        spec=effective_spec,
                        audit=audit,
                        search_rank_ic=s_ic,
                        search_net_sharpe=s_sr,
                        accepted_on_search=accepted,
                    )
                )

        # Score accepted ensemble on `validation` scalar view (fallback to best search candidate if 0 accepted)
        if accepted_specs:
            val_out = self.evaluator.evaluate_factor_ensemble(
                accepted_specs,
                weights=accepted_weights,
                zone="validation",
                candidate_id=f"{agent_version}_ensemble",
                run_id=run_id,
            )
        elif records:
            best_rec = max(records, key=lambda r: r.search_rank_ic)
            accepted_specs = [best_rec.spec]
            accepted_weights = [1.0]
            val_out = self.evaluator.evaluate_factor(best_rec.spec, zone="validation", run_id=run_id)
        else:
            raise RuntimeError("Agent pipeline generated zero valid candidates.")

        if sft_export_path and sft_trajectories:
            p = Path(sft_export_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as f:
                for row in sft_trajectories:
                    f.write(json.dumps(row, sort_keys=True) + "\n")

        fam_ic: dict[str, list[float]] = {}
        for rec in records:
            if rec.audit.passed:
                fam_ic.setdefault(rec.hypothesis.mechanism_family, []).append(rec.search_rank_ic)
        fam_summary = {k: round(float(np.mean(v)), 5) for k, v in fam_ic.items()}

        return AgentPipelineResult(
            agent_version=agent_version,
            seed=seed,
            accepted_specs=accepted_specs,
            accepted_weights=accepted_weights,
            candidate_records=records,
            validation_outcome=val_out,
            audit_rejection_count=audit_rejections,
            refuted_on_search_count=search_refutations,
            memory_snapshot=self.memory.snapshot_list(),
            family_ic_summary=fam_summary,
            budget_snapshot=budget.snapshot(),
            verified_sft_trajectories=sft_trajectories,
        )
