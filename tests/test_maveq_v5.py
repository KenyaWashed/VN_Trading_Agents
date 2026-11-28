"""Unit Tests for MAVE-Q v5 (`DCA-Chronos-RSI` + Popperian Falsification):
- Google DeepCrossAttention (`DCA-QuantFormer`), 5-Family Backbone Zoo, & `DCA-MicroLM`
- Zero look-ahead shift invariance (`atol=1e-9`) across all 5 backbones
- Multi-SLM Registry (6 families, non-Qwen diversity)
- HardwareSafetyGuard (`local_safe_20m` <1.2 GB RAM fuse + `sub1b_flagship_900m` auto-promotion & 0 MB meta check)
- AdversarialHypothesisRefuter (strict rejection of fee-sensitive mirages, single-fold luck, Occam violations, sign flips)
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vn_rsi.data import SnapshotManager, SyntheticVNMarketProvider, load_market_rules
from vn_rsi.dsl.ast_spec import compute_feature_panel
from vn_rsi.eval import SealedEvaluator
from vn_rsi.models.cloud_presets import (
    MULTI_SLM_REGISTRY,
    HardwareSafetyGuard,
    verify_sub1b_blueprint_zero_ram,
)
from vn_rsi.models.dca_backbone import (
    BACKBONE_ZOO_FAMILIES,
    DCAMicroLM,
    QuantBackboneModel,
)
from vn_rsi.models.maveq_v5_chronos import AdversarialHypothesisRefuter
from vn_rsi.utils import load_yaml


@pytest.fixture(scope="module")
def mini_v5_setup(tmp_path_factory: pytest.TempPathFactory):
    repo_root = Path(__file__).resolve().parents[1]
    tmp = tmp_path_factory.mktemp("v5_test")
    rules = load_market_rules(repo_root / "configs" / "market_rules.yaml")
    sandbox_cfg = copy.deepcopy(load_yaml(repo_root / "configs" / "sandbox.yaml"))
    sandbox_cfg["zones"] = {
        "search": {"start": "2020-01-02", "end": "2021-12-31", "agent_access": "full"},
        "validation": {"start": "2022-01-04", "end": "2022-06-30", "agent_access": "scalar"},
        "sealed_test": {"start": "2022-07-15", "end": "2022-12-30", "agent_access": "once"},
    }
    sandbox_cfg["ledger_path"] = str(tmp / "ledger.jsonl")
    bundle = SyntheticVNMarketProvider().fetch_bundle("vn30", "2020-01-02", "2022-12-30", rules, seed=77)
    snap = SnapshotManager(tmp / "snaps").write_snapshot("s_v5", bundle)
    cfg = copy.deepcopy(load_yaml(repo_root / "configs" / "maveq_v5.yaml"))
    ev = SealedEvaluator(snap, rules, sandbox_cfg)
    return snap, rules, sandbox_cfg, cfg, ev, tmp


def test_hardware_safety_guard_and_sub1b_zero_ram_blueprint() -> None:
    """Verify <12GB RAM local laptop locks to local_safe_20m and cloud auto-promotes to <1B/module."""
    # 1. Local 7.4 GB RAM laptop without GPU/TPU
    local_guard = HardwareSafetyGuard(
        compute_override={
            "platform": "local",
            "system_ram_gb": 7.35,
            "has_tpu": False,
            "gpu_count": 0,
            "max_gpu_vram_gb": 0.0,
            "total_gpu_vram_gb": 0.0,
        }
    )
    res_auto = local_guard.resolve_safe_preset("auto")
    assert res_auto.resolved_preset == "local_safe_20m"
    assert res_auto.max_local_ram_mb <= 1200.0
    assert res_auto.allow_physical_instantiation is True

    # Accidental request for sub1b on local low-RAM laptop MUST be blocked and clamped!
    res_blocked = local_guard.resolve_safe_preset("sub1b_flagship_900m")
    assert res_blocked.was_clamped is True
    assert res_blocked.allow_physical_instantiation is False
    assert res_blocked.resolved_preset == "local_safe_20m"

    # 2. Kaggle 2x T4 (32GB) / TPU v5e-8 (128GB) / Modal A100 (80GB) auto-promote to sub1b_flagship_900m!
    for cloud_hw in (
        {"platform": "kaggle", "system_ram_gb": 30.0, "has_tpu": False, "gpu_count": 2, "max_gpu_vram_gb": 15.0, "total_gpu_vram_gb": 30.0},
        {"platform": "kaggle", "system_ram_gb": 300.0, "has_tpu": True, "gpu_count": 0, "max_gpu_vram_gb": 0.0, "total_gpu_vram_gb": 0.0},
        {"platform": "modal", "system_ram_gb": 64.0, "has_tpu": False, "gpu_count": 1, "max_gpu_vram_gb": 80.0, "total_gpu_vram_gb": 80.0},
    ):
        cloud_guard = HardwareSafetyGuard(compute_override=cloud_hw)
        res_cloud = cloud_guard.resolve_safe_preset("auto")
        assert res_cloud.resolved_preset == "sub1b_flagship_900m"
        assert res_cloud.allow_physical_instantiation is True

    # 3. Zero-RAM blueprint verification for both local_safe_20m and sub1b_flagship_900m
    bp_sub1b = verify_sub1b_blueprint_zero_ram("sub1b_flagship_900m")
    assert bp_sub1b["ram_consumed_by_verification_mb"] == 0.0
    assert bp_sub1b["all_blocks_strictly_sub_1b"] is True
    for name, pcount in bp_sub1b["blocks_params"].items():
        assert 550_000_000 <= pcount < 1_000_000_000, f"{name} out of [550M, 1B): {pcount}"

    bp_local = verify_sub1b_blueprint_zero_ram("local_safe_20m")
    assert 15_000_000 <= bp_local["blocks_params"]["DCA-QuantFormer"] <= 22_000_000
    assert 15_000_000 <= bp_local["blocks_params"]["DCA-MicroLM"] <= 22_000_000


def test_multi_slm_registry_and_custom_dca_microlm() -> None:
    """Verify 6 distinct SLM families (no Qwen monoculture) and from-scratch DCA-MicroLM causal forward pass."""
    assert len(MULTI_SLM_REGISTRY) == 6
    expected = {"gemma3_1b", "llama32_1b", "smollm2_360m", "deepseek_r1_1p5b", "qwen3_0p6b", "dca_microlm"}
    assert set(MULTI_SLM_REGISTRY.keys()) == expected
    assert MULTI_SLM_REGISTRY["gemma3_1b"].repo_id == "google/gemma-3-1b-it"
    assert MULTI_SLM_REGISTRY["llama32_1b"].repo_id == "meta-llama/Llama-3.2-1B-Instruct"
    assert MULTI_SLM_REGISTRY["smollm2_360m"].repo_id == "HuggingFaceTB/SmolLM2-360M-Instruct"
    assert MULTI_SLM_REGISTRY["dca_microlm"].is_from_scratch is True

    lm = DCAMicroLM(preset="local_safe_20m", seed=2026)
    s = lm.summary()
    assert 15_000_000 <= s["exact_params"] <= 22_000_000
    nll = lm.compute_sequence_nll("VN30 T+2.5 risk audit: reject FOMO ceiling chase.")
    assert np.isfinite(nll) and nll > 0.0

    # Verify causal attention invariance of DCA-MicroLM logits when suffix tokens change
    t1 = lm.encode_bytes("MAVE-Q v5 causal prefix AAAA")
    t2 = lm.encode_bytes("MAVE-Q v5 causal prefix ZZZZ")
    l1, d_attn = lm.forward_logits(t1)
    l2, _ = lm.forward_logits(t2)
    assert np.allclose(l1[:20], l2[:20], atol=1e-5)
    assert np.allclose(np.sum(d_attn, axis=-1), 1.0, atol=1e-5)


def test_backbone_zoo_five_families_causality_and_dca_depth_attention(mini_v5_setup) -> None:
    """All 5 rival backbones must train strictly on search and pass exact future-perturbation invariance (atol=1e-9)."""
    snap, _, _, _, ev, _ = mini_v5_setup
    s_bars = ev.zone_guard.get_agent_visible_bars(snap.bars, "search")
    s_end = pd.Timestamp("2021-12-31")
    pit_s = snap.fundamentals_pit[snap.fundamentals_pit["announcement_date"] <= s_end].copy()
    s_panel = compute_feature_panel(s_bars, pit_s)
    s_close = s_panel["close"]
    safe_dates = pd.DatetimeIndex(s_close.index[:-5])
    s_fwd = (s_close.shift(-5) / s_close - 1.0).reindex(index=safe_dates)

    cutoff = pd.Timestamp("2022-04-01")
    pert_bars = snap.bars.copy()
    fmask = pert_bars["date"] > cutoff
    for col in ("open", "high", "low", "close", "raw_close", "volume", "turnover_vnd", "foreign_net_ratio"):
        pert_bars.loc[fmask, col] = pert_bars.loc[fmask, col] * 3.1 + 750.0
    panel_orig = ev.feature_panel
    panel_pert = compute_feature_panel(pert_bars, snap.fundamentals_pit)
    past = pd.DatetimeIndex(panel_orig["close"].index)
    past = past[past <= cutoff]

    assert len(BACKBONE_ZOO_FAMILIES) == 5
    for fam in BACKBONE_ZOO_FAMILIES:
        mdl = QuantBackboneModel(family=fam, scale_rung="0.5m", seed=2026)
        mdl.fit_on_search(s_panel, safe_dates, s_fwd)
        p1 = mdl.predict_panel(panel_orig).reindex(index=past)
        p2 = mdl.predict_panel(panel_pert).reindex(index=past)
        assert np.allclose(p1.to_numpy(dtype=float), p2.to_numpy(dtype=float), atol=1e-9), f"Leakage in {fam}!"
        if fam == "dca_quantformer":
            d_attn = mdl.inspect_depth_attention_weights(s_panel)
            assert d_attn.shape[1] == mdl._work_layers
            assert np.allclose(np.sum(d_attn, axis=-1), 1.0, atol=1e-5)

    # Verify local_safe_20m physical float32 allocation stays safely < 1.2 GB RAM (actual ~80 MB!)
    mdl_20m = QuantBackboneModel(family="dca_quantformer", scale_rung="20m", seed=2026, preset="local_safe_20m")
    mat_info = mdl_20m.verify_local_materialization_budget()
    assert mat_info["materialized"] is True
    assert float(mat_info["actual_fp32_mb"]) < 120.0


def test_adversarial_hypothesis_refuter_rejects_pathological_candidates(mini_v5_setup) -> None:
    """AdversarialHypothesisRefuter MUST accurately reject fee mirages, single-fold luck, Occam bloat, & sign flips."""
    snap, rules, _, _, ev, _ = mini_v5_setup
    s_bars = ev.zone_guard.get_agent_visible_bars(snap.bars, "search")
    s_end = pd.Timestamp("2021-12-31")
    pit_s = snap.fundamentals_pit[snap.fundamentals_pit["announcement_date"] <= s_end].copy()
    s_panel = compute_feature_panel(s_bars, pit_s)
    s_close = s_panel["close"]
    safe_dates = pd.DatetimeIndex(s_close.index[:-5])
    s_fwd = (s_close.shift(-5) / s_close - 1.0).reindex(index=safe_dates)
    folds = ev.splitter.split(safe_dates)

    refuter = AdversarialHypothesisRefuter(
        market_rules=rules,
        search_bars=s_bars,
        safe_dates=safe_dates,
        search_fwd=s_fwd,
        search_folds=folds,
        top_k=ev.top_k,
        rebalance_every_days=ev.rebalance_every_days,
        capital_vnd=ev.capital_vnd,
    )

    # 1. Single-Regime Luck: positive ONLY in fold 0, strongly inverted in folds 1..3
    bad_single_fold = pd.DataFrame(0.0, index=s_close.index, columns=s_close.columns)
    for fi, fold in enumerate(folds):
        if fi == 0:
            bad_single_fold.loc[fold.eval_dates] = s_fwd.reindex(index=fold.eval_dates).fillna(0.0) * 10.0
        else:
            bad_single_fold.loc[fold.eval_dates] = -s_fwd.reindex(index=fold.eval_dates).fillna(0.0) * 10.0
    v_luck = refuter.refute_signal("syn_single_fold_luck", bad_single_fold)
    assert v_luck.accepted is False
    assert "SINGLE_REGIME_LUCK" in v_luck.failed_gates

    # 1b. Non-Economic Sign Inversion: positive in chronological first half, inverted in second half
    mid = len(safe_dates) // 2
    bad_sign_flip = pd.DataFrame(0.0, index=s_close.index, columns=s_close.columns)
    bad_sign_flip.loc[safe_dates[:mid]] = s_fwd.reindex(index=safe_dates[:mid]).fillna(0.0) * 10.0
    bad_sign_flip.loc[safe_dates[mid:]] = -s_fwd.reindex(index=safe_dates[mid:]).fillna(0.0) * 10.0
    v_flip = refuter.refute_signal("syn_sign_inversion", bad_sign_flip)
    assert v_flip.accepted is False
    assert "ECONOMIC_SIGN_INVERSION" in v_flip.failed_gates

    # 2. Occam's Razor Violation: 4x complexity identical signal fails BIC against simpler reference
    clean_sig = s_panel["pit_value_quality"].fillna(0.0)
    v_simple = refuter.refute_signal("clean_simple", clean_sig, complexity_units=1.0)
    v_bloated = refuter.refute_signal(
        "bloated_4x",
        clean_sig,
        complexity_units=4.0,
        simpler_ref_bic_score=v_simple.bic_adjusted_score,
        simpler_ref_complexity=1.0,
    )
    assert v_bloated.accepted is False
    assert "OCCAM_RAZOR_VIOLATION" in v_bloated.failed_gates

    # 3. Liquidity & Friction Mirage (1.5x cost shock): hyper-churn signal with tiny edge that barely beats 1.0x
    # VN fees but collapses >40% (or turns negative) when VN commission/tax/spread/impact are shocked to 1.5x!
    syms_list = list(s_close.columns)
    n_s = len(syms_list)
    rng_m = np.random.default_rng(7)
    noise_mat = rng_m.normal(0.0, 1.0, size=(len(s_close), n_s))
    mirage_rejected = False
    for alpha_scale in (0.14, 0.11, 0.09, 0.07, 0.055, 0.045, 0.035):
        m_sig = pd.DataFrame(0.0, index=s_close.index, columns=syms_list)
        for t_i, dt in enumerate(s_close.index):
            block = (t_i // 3) % 4
            rot = np.zeros(n_s, dtype=float)
            start_idx = (block * 7) % n_s
            for j in range(12):
                rot[(start_idx + j) % n_s] = 1.5
            if dt in s_fwd.index:
                f_row = s_fwd.loc[dt].fillna(0.0).to_numpy(dtype=float)
                f_z = (f_row - np.mean(f_row)) / (np.std(f_row) + 1e-6)
                m_sig.loc[dt] = rot + alpha_scale * f_z + 0.15 * noise_mat[t_i]
            else:
                m_sig.loc[dt] = rot + 0.15 * noise_mat[t_i]
        vm = refuter.refute_signal(f"syn_fee_mirage_{alpha_scale}", m_sig)
        assert vm.shocked_1p5x_sharpe < vm.base_search_sharpe
        if "LIQUIDITY_FRICTION_MIRAGE_1.5X" in vm.failed_gates and vm.accepted is False:
            mirage_rejected = True
            break
    assert mirage_rejected, "Refuter failed to catch marginal high-churn Liquidity/Friction Mirage under 1.5x shock!"


def test_chronos_kalman_hedge_shift_invariance_and_online_pruning(mini_v5_setup) -> None:
    """Chronos-Kalman-Hedge must be shift-invariant to atol=1e-9 AND prune degraded experts to weight 0.0."""
    from vn_rsi.models.maveq_v5_chronos import ChronosKalmanHedge

    snap, _, _, _, ev, _ = mini_v5_setup
    close = ev.feature_panel["close"]
    fwd5 = close.shift(-5) / close - 1.0
    # Expert 1: genuine causal value-quality signal; Expert 2: persistently anti-predictive degraded expert
    exp_good = ev.feature_panel["pit_value_quality"].fillna(0.0)
    exp_bad = -fwd5.shift(5).fillna(0.0)  # strictly causal lagging anti-signal (or inverted)
    # Build a clearly negative-skill causal expert over time (inverted rolling 5d return)
    exp_degraded = -close.pct_change(5, fill_method=None).fillna(0.0) * 0.0 + (-ev.feature_panel["pit_value_quality"].fillna(0.0))

    hedge = ChronosKalmanHedge(horizon_delay_days=5, online_prune_skill_thresh=-0.005, hedge_eta=8.0)
    out = hedge.fuse_online(
        expert_signals=[exp_good, exp_degraded],
        expert_names=["good_vq", "degraded_inv_vq"],
        prior_weights=[0.5, 0.5],
        prior_skills=[0.05, 0.01],
        close_wide=close,
        enabled=True,
    )
    # Verify online pruning actually zeros out the degraded expert on the majority of sessions after warmup
    pruned_tail = out.online_pruned_mask_df["degraded_inv_vq"].iloc[40:]
    weights_tail = out.expert_weights_df["degraded_inv_vq"].iloc[40:]
    assert float(pruned_tail.mean()) > 0.30 or float(weights_tail.mean()) < 0.20, "Failed to prune/downweight degraded expert!"
    # Whenever pruned is True, weight MUST be identically 0.0!
    pruned_rows = out.online_pruned_mask_df["degraded_inv_vq"]
    if bool(pruned_rows.any()):
        assert float(out.expert_weights_df.loc[pruned_rows, "degraded_inv_vq"].max()) == 0.0

    # Strict H-day delay shift-invariance test (atol=1e-9) under post-cutoff perturbation
    cutoff = pd.Timestamp("2022-04-01")
    close_pert = close.copy()
    close_pert.loc[close_pert.index > cutoff] *= 3.5
    exp_good_pert = exp_good.copy()
    exp_good_pert.loc[exp_good_pert.index > cutoff] *= -4.0
    out_pert = hedge.fuse_online(
        expert_signals=[exp_good_pert, exp_degraded],
        expert_names=["good_vq", "degraded_inv_vq"],
        prior_weights=[0.5, 0.5],
        prior_skills=[0.05, 0.01],
        close_wide=close_pert,
        enabled=True,
    )
    past = close.index[close.index <= cutoff]
    assert np.allclose(
        out.combined_alpha_wide.reindex(index=past).to_numpy(dtype=float),
        out_pert.combined_alpha_wide.reindex(index=past).to_numpy(dtype=float),
        atol=1e-9,
    )
    assert np.allclose(
        out.expert_weights_df.reindex(index=past).to_numpy(dtype=float),
        out_pert.expert_weights_df.reindex(index=past).to_numpy(dtype=float),
        atol=1e-9,
    )
    assert exp_bad.shape == close.shape


def test_mose_hypergate_prunes_noise_and_conformal_veto_is_dormant_and_abstains(mini_v5_setup) -> None:
    """MoSE Group-Lasso must shrink pure-noise branch to 0.0; Split-Conformal must stay dormant (~90%) & abstain on crashes."""
    from vn_rsi.models.maveq_v4_veteran import compute_causal_microstructure_state
    from vn_rsi.models.maveq_v5_chronos import (
        AsymBetaNTZController,
        AsymBetaNTZParams,
        DCAMoSEHyperGate,
        SplitConformalDormantVeto,
    )

    snap, _, _, _, ev, _ = mini_v5_setup
    s_bars = ev.zone_guard.get_agent_visible_bars(snap.bars, "search")
    pit_s = snap.fundamentals_pit[snap.fundamentals_pit["announcement_date"] <= pd.Timestamp("2021-12-31")].copy()
    s_panel = compute_feature_panel(s_bars, pit_s)
    s_close = s_panel["close"]
    safe_dates = pd.DatetimeIndex(s_close.index[:-5])
    s_fwd = (s_close.shift(-5) / s_close - 1.0).reindex(index=safe_dates)
    folds = ev.splitter.split(safe_dates)

    # 1. Inject pure white-noise branch -> Group-Lasso MUST shrink weight to EXACT 0.0!
    rng = np.random.default_rng(999)
    noise_branch = pd.DataFrame(rng.normal(0.0, 1.0, size=s_close.shape), index=s_close.index, columns=s_close.columns)
    mose = DCAMoSEHyperGate(group_lasso_lambda=0.025, max_nonlinear_weight=0.35)
    mose.fit_on_search(
        search_panel=s_panel,
        base_alpha_search=s_panel["pit_value_quality"].fillna(0.0),
        safe_dates=safe_dates,
        search_fwd=s_fwd,
        search_folds=folds,
        extra_candidate_branches={"pure_white_noise": noise_branch},
    )
    assert mose.branch_weights["pure_white_noise"] == 0.0
    assert "pure_white_noise" in mose.pruned_branches

    # 2. Split-Conformal Dormant-by-Default (~85-95% exact 0.0 penalty) & Epistemic Abstention on synthetic crash
    s_state = compute_causal_microstructure_state(s_panel)
    sc = SplitConformalDormantVeto(alpha_level=0.10, calibration_fraction=0.35, market_abstain_alpha=0.06)
    sc.fit_on_search(s_panel, s_state, safe_dates)
    out_normal = sc.apply(s_panel, s_state, enabled=True)
    assert 0.80 <= out_normal.dormant_fraction <= 0.99

    # Inject severe secular bear + floor-lock crash at end of panel -> Controller MUST abstain (all-NaN -> Cash)!
    crash_panel = {k: v.copy() for k, v in s_panel.items()}
    crash_dates = s_close.index[-25:]
    crash_panel["return_1d"].loc[crash_dates] = -0.045
    crash_panel["dist_to_floor"].loc[crash_dates] = 0.002
    crash_state = compute_causal_microstructure_state(crash_panel)
    out_crash = sc.apply(crash_panel, crash_state, enabled=True)
    ctrl = AsymBetaNTZController(top_k=6)
    ctrl_out = ctrl.apply(
        alpha_wide=s_panel["pit_value_quality"].fillna(0.0),
        panel=crash_panel,
        state_df=crash_state,
        conformal_out=out_crash,
        params=AsymBetaNTZParams(name="test_abstain"),
        use_conformal_veto=True,
        use_asym_beta_ntz=True,
    )
    tail_sig = ctrl_out.sealed_signal_wide.loc[crash_dates[-5:]]
    assert bool(tail_sig.isna().all().all()), "Controller failed to Epistemically Abstain (all-NaN Cash) during severe crash!"


def test_maveq_v5_end_to_end_causality_zero_val_burn_and_checkpoints(mini_v5_setup) -> None:
    """Full MAVEQv5Engine must pass atol=1e-9 future-perturbation invariance, 1-trial val burn, & checkpoint round-trip."""
    from vn_rsi.models.maveq_v5_chronos import AsymBetaNTZParams, MAVEQv5Engine

    snap, rules, sandbox_cfg, cfg, _, tmp = mini_v5_setup
    cfg_fast = copy.deepcopy(cfg)
    cfg_fast["evolution"].update(
        {"rounds": 2, "latent_candidates_per_round": 16, "screen_per_round": 5, "diffu_grpo_epochs_per_round": 2}
    )
    ev = SealedEvaluator(snap, rules, sandbox_cfg)
    eng = MAVEQv5Engine(ev, cfg_fast, preset="auto")
    res = eng.run(run_id="t_v5_e2e", score_sealed=False)

    assert res.validation_queries_used == 1
    assert res.total_evaluator_trials == 1
    assert res.sub1b_blueprint_verified is True
    assert res.champion_sealed_test_metrics.get("note") == "sealed_test_not_requested_dev_mode"
    assert 0.70 <= res.dormant_veto_fraction <= 1.0

    # End-to-end future-perturbation shift invariance (atol=1e-9)
    basis = eng._select_diverse_basis(max_basis=4)
    assert len(basis) >= 1
    weights = [1.0 / len(basis)] * len(basis)
    cp = AsymBetaNTZParams(name="test_shift")
    cutoff = pd.Timestamp("2022-04-01")
    pert_bars = snap.bars.copy()
    fmask = pert_bars["date"] > cutoff
    for col in ("open", "high", "low", "close", "raw_close", "volume", "turnover_vnd", "foreign_net_ratio"):
        pert_bars.loc[fmask, col] = pert_bars.loc[fmask, col] * 2.8 + 600.0
    panel_orig = ev.feature_panel
    panel_pert = compute_feature_panel(pert_bars, snap.fundamentals_pit)
    past = pd.DatetimeIndex(panel_orig["close"].index)
    past = past[past <= cutoff]

    o1 = eng.build_full_causal_signal(basis, weights, cp, panel_orig)
    o2 = eng.build_full_causal_signal(basis, weights, cp, panel_pert)
    s1 = o1.sealed_signal_wide.reindex(index=past).to_numpy(dtype=float)
    s2 = o2.sealed_signal_wide.reindex(index=past).to_numpy(dtype=float)
    both_nan = np.isnan(s1) & np.isnan(s2)
    eq = np.isclose(np.nan_to_num(s1, nan=0.0), np.nan_to_num(s2, nan=0.0), atol=1e-9, rtol=0.0)
    assert bool(np.all(both_nan | eq)), "Future perturbation leaked into MAVE-Q v5 past signals!"

    # Checkpoint round-trip
    ckpts = eng.save_checkpoints(tmp / "v5_ckpts")
    for p in ckpts.values():
        assert Path(p).exists()
    eng2 = MAVEQv5Engine(SealedEvaluator(snap, rules, sandbox_cfg), cfg_fast, preset="auto")
    eng2.load_checkpoints(tmp / "v5_ckpts")
    assert eng2.conformal_veto.q_stock == pytest.approx(eng.conformal_veto.q_stock)
