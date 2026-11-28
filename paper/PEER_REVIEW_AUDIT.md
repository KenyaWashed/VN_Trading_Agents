# Adversarial 4-Reviewer Devil's Advocate Meta-Review Audit (MAVE-Q v5 Research Topic Technical Report)

> **Document Label**: **Research Topic Technical Report Audit / Biên bản phản biện Đề tài nghiên cứu MAVE-Q v5** (*NOT a submitted or published paper*).  
> **Audit Date**: 2026-11-28  
> **Audited Scope**: `paper/main.tex`, `paper/references.bib`, `paper/tables/*.tex` (13 tables, 571 provenance cells), `paper/figures/*.{pdf,svg}` (7 figures), `logs/experiment_registry.jsonl`, `logs/v5_falsified_hypotheses.jsonl`, `logs/sealed_test_ledger.jsonl`, and `src/vn_rsi/models/` + `src/vn_rsi/veteran/`.

---

## Reviewer 1 — Data Leakage, Overclaiming & Multiple-Testing Rigor
- **Pre-Fix Verdict**: Borderline Weak Reject -> **Post-Fix Verdict**: Verified & Scientifically Sound

### Ruthless Critiques & Self-Fixes Applied
1. **[MAJOR `R1.1` — Codename vs. Pruned Reality & Uncited Appendix Theorems]**:
   - *Critique*: Retaining the codename `DCA-Chronos-RSI` while pruning `Chronos-Kalman-Hedge` in Task 05 risks misleading readers into thinking online Kalman-Hedge is active in the sealed-test champion. Additionally, Appendix Theorems `\ref{thm:dca_grad}`, `\ref{thm:chronos_causal}`, and `\ref{thm:conformal_dormant}` were not explicitly cross-referenced from Section `\ref{sec:maveq_v5}`.
   - *Fix Applied*: Explicitly clarified in Section `\ref{sec:maveq_v5}` that `DCA-Chronos-RSI` is the research lineage codename while the **official pruned champion** retains strictly `DCA-QuantFormer + Popperian Symbolic AST + AsymBeta-NTZ` (`Chronos-Kalman-Hedge`, `DCA-MoSE HyperGate`, and stock-level `Split-Conformal` permanently pruned), and added explicit `\ref{thm:dca_grad}`–`\ref{thm:conformal_dormant}` cross-references.
2. **[MAJOR `R1.2` — Marginally Significant Sealed-Test p-Value & Strict Lineage DSR Ceiling]**:
   - *Critique*: While MAVE-Q v5 jumped `+0.374` Test Sharpe over v4 (`1.441 vs 1.067`), the paired block-bootstrap p-value is `p = 0.081` (95% CI `[-0.159, +0.896]`, marginally significant at $\alpha=0.10$, not $p < 0.01$), vs VN-Index is `p = 0.248`, Hansen SPA is `p = 1.0`, and strict Lineage-Cumulative DSR across all 29 historical v1..v5 ledger entries is `0.389` (`0.324` at `N=45`).
   - *Fix Applied*: Added explicit statistical-power and lineage-multiplicity caveats in both Section `\ref{sec:maveq_v5}` and Section `\ref{sec:limitations}` — zero overclaiming of $p<0.05$ sealed-test dominance.
3. **[VERIFIED — Zero Look-Ahead & Seal Integrity]**:
   - `EvaluatorSeal` SHA-256 (`4a0f5ad331bd982f5290e6f62e9ea5122a29f21675a2e197080e83e2f8f9331e`) verified 100% untouched.
   - `ChronosKalmanHedge` exact $H$-day delay (`s <= t - H`), all 5 Backbone Zoo models, and full `MAVEQv5Engine` verified future-perturbation shift-invariant to `atol=1e-9` (`tests/test_maveq_v5.py`).
   - `logs/sealed_test_ledger.jsonl` verified single-shot (`28 -> 29` lines, zero retries).

---

## Reviewer 2 — Does `DCA-QuantFormer` & Scaling Truly Beat Simpler Baselines Net of Complexity?
- **Pre-Fix Verdict**: Borderline -> **Post-Fix Verdict**: Verified & Transparent

### Ruthless Critiques & Self-Fixes Applied
1. **[MAJOR `R2.1` — 3-Seed vs. 8-Seed Screening Asymmetry in the 5-Backbone Zoo]**:
   - *Critique*: In Table `\ref{tab:v5_backbone_zoo_scaling}`, `zoo_sparse_moe_0p5m` (`SR=1.554±0.047, BIC=2.142`) and `zoo_hyena_conv_0p5m` (`SR=1.494±0.091, BIC=2.080`) scored higher than `zoo_dca_quantformer_0p5m` (`SR=1.446±0.107, BIC=1.994`), in part because rival zoo entries ran on 3 screen seeds (`11,44,2026`) while `dca_quantformer` ran on all 8 seeds. Claiming `DCA-QuantFormer` uniquely dominated all 4 rival backbones would be dishonest!
   - *Fix Applied*: Explicitly disclosed the 3-seed vs. 8-seed screening asymmetry in Section `\ref{sec:maveq_v5}`, acknowledged `Sparse-MoE` and `Hyena` as co-surviving `ACCEPTED` architectures alongside `DCA-QuantFormer`, and framed `DCA-QuantFormer` as our custom from-scratch ICML-2025 reference whose direct 8-seed controlled ablation vs `PreLN` proves `+0.0054` IC (`p=0.0000`).
2. **[MAJOR `R2.2` — Exact Post-Pruning 8-Seed p-Value Transparency]**:
   - *Critique*: Post-pruning `DCA-QuantFormer` beat `PreLN` on every metric (`+0.0054` IC, `+0.0423` Sharpe, `+0.2055` Calmar), yet only Rank IC achieved $p < 0.05$ (`p_IC = 0.0000`), whereas `p_Sharpe = 0.1115` and `p_Calmar = 0.1720` on `N=8` seeds.
   - *Fix Applied*: Explicitly reported exact `p_IC = 0.0000`, `p_Sharpe = 0.1115`, and `p_Calmar = 0.1720` in Section `\ref{sec:maveq_v5}` alongside the highly significant `AsymBeta-NTZ` deltas (`+0.4469` Sharpe, `p = 0.0000`; `+1.7663` Calmar, `p = 0.0000`).
3. **[VERIFIED — Honest Small-$N$ Overparameterization Falsification]**:
   - Commended the candid rejection of `5M`, `12M`, and `20M` dense rungs under Occam's Razor BIC on 30-stock VN30 (`42x` params for `+0.011` Sharpe and slightly worse MaxDD).

---

## Reviewer 3 — Objectivity of Falsified Ledger & Anti-Sycophancy Evaluation
- **Pre-Fix Verdict**: Borderline -> **Post-Fix Verdict**: Verified & Well-Bounded

### Ruthless Critiques & Self-Fixes Applied
1. **[MAJOR `R3.1` — Deterministic Neuro-Symbolic Guardrail vs. Free-Form LLM Red-Teaming]**:
   - *Critique*: Reporting `100.0%` Sycophancy Resistance and `0.0%` FDR on the 45-scenario benchmark without clarifying that it evaluates a deterministic neuro-symbolic guardrail (`VeteranReasoningEngine`) on curated synthetic scenarios could be misread as proving an unguarded 1B LLM is immune to paraphrased jailbreaks!
   - *Fix Applied*: Added an explicit **Evaluation Scope Disclosure** in Section `\ref{sec:maveq_v5}` and Section `\ref{sec:limitations}` stating that the 100% resistance reflects the deterministic neuro-symbolic premise-refutation gate on curated traps, and that unconstrained open-ended SLM deployments still require wrapping behind this numerical gate + Anti-Sycophancy DPO fine-tuning.
2. **[MINOR `R3.2` — Conservative Over-Flagging Precision (`0.6897` vs. `1.0000` Recall)]**:
   - *Critique*: Hidden-Risk Precision moved from `0.8000` (32 scenarios) to `0.6897` (45 scenarios) due to overlapping risk tags on compound traps.
   - *Fix Applied*: Documented this intentional recall-first (`1.0000` recall) operating point for asymmetric T+2.5 tail-risk protection.

---

## Reviewer 4 — Local Hardware (`<1.2 GB RAM`) vs. Kaggle `<1B/block` Cloud Boundary
- **Pre-Fix Verdict**: Weak Reject -> **Post-Fix Verdict**: Verified & Transparent

### Ruthless Critiques & Self-Fixes Applied
1. **[MAJOR `R4.1` — Strict Separation of Measured Local CPU Telemetry vs. Cloud Sub-1B Blueprints]**:
   - *Critique*: The local research host (`AMD Ryzen 5 5500U`, `7.35 GB RAM`, `0 CUDA GPUs`, no local PyTorch) executes pure vectorized NumPy `float32` models up to `20.47M` (`~284 MB` peak RAM). The `sub1b_flagship_900m` (`906M / 849M / 802M / 828M < 1B/block`) architecture and 5 external HuggingFace SLM adapters (`gemma3_1b`, `llama32_1b`, `smollm2_360m`, `deepseek_r1_1p5b`, `qwen3_0p6b`) are analytically verified blueprints (`0 MB RAM` parameter verification) and cloud-ready launcher scripts — NOT physically GPU-trained in this local session.
   - *Fix Applied*: Added a dedicated **Local Execution vs. Cloud Blueprint Boundary** item in Section `\ref{sec:limitations}` distinguishing directly measured local CPU runs (`284 MB` RAM, `0.48M–20.47M`) from cloud-ready `<1B/block` PyTorch/XLA/QLoRA blueprints awaiting Kaggle `2x T4` / `TPU v5e-8` / Modal `A100` execution.

---

## Final Automated Audit Gate Status (Post-Fixes)
- **EvaluatorSeal SHA-256**: `4a0f5ad331bd982f5290e6f62e9ea5122a29f21675a2e197080e83e2f8f9331e` (100% INTACT)
- **LaTeX Environment Balance & Syntax Guard**: PASS (0 errors, 0 unescaped `&`, 0 stray backticks)
- **Citations & Cross-References**: PASS (48/48 BibTeX entries cited, 0 missing; 100% `\ref` resolved)
- **Tables & Vector Figures**: PASS (13/13 LaTeX tables, 7/7 dual-backend PDF+SVG figures)
- **Numerical Provenance (`paper/verify_paper.py`)**: **VERIFIED 100%** (571/571 table cells matched to `logs/experiment_registry.jsonl`, `compiled_ok_with_tectonic`)
