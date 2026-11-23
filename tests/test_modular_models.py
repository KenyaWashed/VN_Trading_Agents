"""Tests for Modular Micro-RSI Models: ASTVocab, DiscreteASTDiffusion, HRMReasoner, and ModularMicroRSIBackend."""

from __future__ import annotations

import numpy as np

from vn_rsi.agents.schemas import FalsifiableHypothesis
from vn_rsi.dsl import ASTVerifier, FactorSpec
from vn_rsi.models import (
    ASTDiffusionConfig,
    ASTVocab,
    DiscreteASTDiffusion,
    HRMConfig,
    HRMReasoner,
    ModularMicroRSIBackend,
)
from vn_rsi.rsi.mutable_surface import GenomePatchProposal
from vn_rsi.utils import BudgetConfig, BudgetTracker


def test_ast_diffusion_guarantees_100pct_valid_causal_asts_and_subtree_infilling() -> None:
    vocab = ASTVocab(max_seq_len=16)
    assert 50 <= vocab.vocab_size <= 90

    diff = DiscreteASTDiffusion(ASTDiffusionConfig.from_preset("cpu_fast", seed=42))
    verifier = ASTVerifier(max_depth=4)

    sampled_specs: list[FactorSpec] = []
    for i, fam in enumerate(["momentum_liquidity", "reversal_microstructure", "value_quality_pit", "foreign_flow_regime"]):
        sp = diff.sample_factor(fam, factor_id=f"f_{i}", hypothesis_id=f"h_{i}", max_depth=3, seed=100 + i)
        verifier.verify(sp)
        sampled_specs.append(sp)

        # Subtree infilling mutation MUST also produce a strictly causal valid AST
        mut = diff.infill_mutate_factor(sp, fam, factor_id=f"m_{i}", hypothesis_id=f"h_{i}", max_depth=3, seed=200 + i)
        verifier.verify(mut)

    # Reward-weighted ReST online training step
    trajs = [
        {"family": "momentum_liquidity", "factor_spec": sp.model_dump(), "search_rank_ic": 0.045, "search_net_sharpe": 0.7}
        for sp in sampled_specs
    ]
    stats = diff.fit_on_trajectories(trajs, epochs=3)
    assert stats["steps"] == 3
    assert stats["samples"] == 4


def test_hrm_reasoner_multi_timescale_planning_and_surrogate_training() -> None:
    hrm = HRMReasoner(HRMConfig.from_preset("cpu_fast", seed=77))
    rv = hrm.encode_regime_state(
        {"ceiling_hit_rate": 0.04, "floor_hit_rate": 0.02, "halt_rate": 0.01, "search_rows": 30000.0},
        {"momentum_liquidity": 0.04, "reversal_microstructure": -0.01, "value_quality_pit": 0.05, "foreign_flow_regime": 0.06},
    )
    assert rv.shape == (12,)

    toks = [2, 5, 55, 3] + [0] * 12
    ic_pred = hrm.score_candidate_ast(rv, toks)
    assert -0.3 <= ic_pred <= 0.3

    policy = hrm.propose_outer_policy(rv, {"foreign_flow_regime": 0.06, "value_quality_pit": 0.05}, iteration=1)
    weights = policy["planner_focus_weights"]
    assert abs(sum(weights.values()) - 1.0) < 1e-3
    assert policy["implementer_max_ast_depth"] in (3, 4)

    obs = [
        {"regime_vec": rv, "ast_tokens": toks, "search_rank_ic": 0.05, "family": "foreign_flow_regime"},
        {"regime_vec": rv + np.float32(0.01), "ast_tokens": toks, "search_rank_ic": -0.02, "family": "reversal_microstructure"},
    ]
    tr = hrm.train_on_search_observations(obs, epochs=3)
    assert tr["steps"] == 3


def test_modular_backend_integrates_with_llm_interface() -> None:
    backend = ModularMicroRSIBackend(preset="cpu_fast", seed=99)
    assert backend.parameter_count_billions <= 8.0
    budget = BudgetTracker(BudgetConfig(max_tokens=20_000, max_wall_seconds=30.0))

    hyp, _ = backend.generate_structured(
        system_prompt="Propose hypothesis",
        user_prompt='CONTEXT_JSON:{"family":"foreign_flow_regime","hypothesis_id":"h1"}',
        schema_cls=FalsifiableHypothesis,
        budget=budget,
        seed=11,
    )
    assert isinstance(hyp, FalsifiableHypothesis)

    spec, _ = backend.generate_structured(
        system_prompt="Emit factor",
        user_prompt='CONTEXT_JSON:{"family":"foreign_flow_regime","hypothesis_id":"h1","factor_id":"f1","max_depth":3}',
        schema_cls=FactorSpec,
        budget=budget,
        seed=12,
    )
    assert isinstance(spec, FactorSpec)
    ASTVerifier(max_depth=3).verify(spec)

    patch, _ = backend.generate_structured(
        system_prompt="Propose outer patch",
        user_prompt='CONTEXT_JSON:{"iteration":1,"family_ic_summary":{"foreign_flow_regime":0.06}}',
        schema_cls=GenomePatchProposal,
        budget=budget,
        seed=13,
    )
    assert isinstance(patch, GenomePatchProposal)
    assert "planner_focus_weights" in patch.updates
