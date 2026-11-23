# Phase 0: Verified Literature Review & Transferability to ≤8B Models

> **Verification Policy**: Every citation below was directly queried and verified against the arXiv API (`literature-search-arxiv`) or primary bibliographic records on 2026-11-23. Anything not directly opened/confirmed in full text is explicitly noted.

---

## 1. Seed Papers (Verified)

### 1.1 AQuA: Recursively Self-Improving Quantitative Trading Research Agents
- **Citation**: Jiacheng Guo, Suozhi Huang, Yunlong Gao, Zihao Li, Jason Ge, Shushu Liang, Xu Kuang, Mengdi Wang (Aug 2026), `arXiv:2608.12841v3` (`https://arxiv.org/abs/2608.12841`).
- **Mechanism**: Splits quantitative research into two isolated LLM-driven systems: (1) manager-mediated multi-agent symbolic factor discovery, and (2) config-driven hybrid time-series model development. Both operate inside **sealed experimental sandboxes** where splits, feature/label definitions, and the evaluator are fixed and immutable; agents only emit constrained factor expressions or config diffs and accumulate persistent memory of validated experimental outcomes.
- **Failure Cases**: Adaptive overfitting across many validation queries if trial multiplicity is uncorrected; collapse into repetitive factor templates if memory lacks novelty/decay controls.
- **Transfers to ≤8B**: Typed causal DSL operator registry (eliminates code-execution crashes and look-ahead bugs structurally), score-only validation boundary, single-shot sealed test ledger, and compact structured belief memory.
- **Does NOT Transfer Directly to ≤8B**: Long multi-turn conversational deliberation across many agents and free-form deep neural architecture synthesis. **Scale-down solution**: Replace free-form discussion with short JSON-schema constrained outputs (`<768` tokens) and deterministic AST verification (`best_of_n`).

### 1.2 AIDEz_H^2z_H: Recursive Self-Improvement of AI Research Agents
- **Citation**: Dhruv Srikanth, Bingchen Zhao, Dixing Xu, Yuxiang Wu, Zhengyao Jiang (Sep 2026), `arXiv:2609.26457v1` (`https://arxiv.org/abs/2609.26457`).
- **Mechanism**: Bi-level optimization loop: an inner research agent solves R&D tasks under a fixed compute budget, while an outer meta-loop proposes source-code modifications to the agent itself and selects winning rewrites on **hidden held-out evaluations**. Over an 8-day run, AIDEz_H^2z_H discovered 7 successive improvements (search policies, context compression) and reduced reward hacking from 55% to 32%.
- **Failure Cases**: Compounding evaluation noise between inner and outer loops; reward hacking when the mutable surface is too permissive (55% baseline rate before hidden-eval selection).
- **Transfers to ≤8B**: Separation of inner optimization signal (`search` zone) from outer selection signal (`validation` zone across paired seeds), plus explicit reward-hacking tracking and ignition testing.
- **Does NOT Transfer Directly to ≤8B**: Arbitrary self-rewriting of raw Python agent codebases—an 8B model frequently introduces subtle syntax/state regressions when editing multi-file Python orchestration. **Scale-down solution**: Restrict outer-loop edits to a **declared typed Mutable Genome Surface** (prompts, mechanism focus priors, AST depth, window schedules, memory decay/contradiction weights, verifier `best_of_n`).

---

## 2. RSI, Evolutionary Coding, & Automated Agent Design

### 2.1 Darwin Gödel Machine (DGM)
- **Citation**: Jenny Zhang, Shengran Hu, Cong Lu, Robert Lange, Jeff Clune (May 2025), `arXiv:2505.22954v3` (`https://arxiv.org/abs/2505.22954`).
- **Mechanism**: Replaces unprovable formal Gödel Machine proofs with empirical benchmark validation and **open-ended archive evolution** (sampling parents by fitness + novelty to preserve stepping stones).
- **Transfer to ≤8B**: Open-ended `DGMArchive` over agent genomes (`fitness_weight=0.75, novelty_weight=0.25`) transfers directly with zero model-size overhead; arbitrary repository patching does not.

### 2.2 STOP (Self-Taught Optimizer)
- **Citation**: Eric Zelikman, Eliana Lorch, Lester Mackey, Adam Tauman Kalai (Oct 2023, COLM 2024), `arXiv:2310.02304v3` (`https://arxiv.org/abs/2310.02304`).
- **Mechanism**: Meta-scaffolding program queries an LLM to improve its own search scaffolding (inventing beam search, simulated annealing, etc.) without weight updates; explicitly audits sandbox-bypass attempts.
- **Transfer to ≤8B**: Meta-optimization of search hyperparameters (`best_of_n`, temperature, window curriculum) and explicit sandbox-bypass counters transfer cleanly; GPT-4-dependent unconstrained Python meta-programming does not.

### 2.3 ADAS (Automated Design of Agentic Systems)
- **Citation**: Shengran Hu, Cong Lu, Jeff Clune (Aug 2024), `arXiv:2408.08435v2` (`https://arxiv.org/abs/2408.08435`).
- **Mechanism**: *Meta Agent Search* iteratively programs new agent workflows in code conditioned on an archive of prior discoveries.
- **Transfer to ≤8B**: Archive-conditioned meta-proposals transfer when parameterized as structured JSON patches (`GenomePatchProposal`); Turing-complete workflow synthesis breaks frequently on ≤8B models without heavy grammar constraints.

### 2.4 AlphaEvolve, OpenEvolve, and ShinkaEvolve
- **Citations**:
  - *AlphaEvolve*: Alexander Novikov et al. (Jun 2025), `arXiv:2506.13131v1` (`https://arxiv.org/abs/2506.13131`).
  - *ShinkaEvolve*: Robert Tjarko Lange, Yuki Imajuku, Edoardo Cetin (Sep 2025), `arXiv:2509.19349v1` (`https://arxiv.org/abs/2509.19349`).
  - *OpenEvolve*: Community open-source recreation of AlphaEvolve's prompt-sampler + MAP-Elites program database (verified via secondary technical summaries; no standalone canonical arXiv preprint — **UNVERIFIED primary peer-reviewed paper**).
- **Mechanism**: Evolutionary code mutation with automated evaluators; *ShinkaEvolve* specifically solves **sample inefficiency** via balanced parent sampling, **code novelty rejection-sampling** before expensive evaluations, and bandit LLM selection.
- **Transfer to ≤8B**: ShinkaEvolve's **structural novelty rejection filter** (`ASTNode.structural_fingerprint()`) is essential for ≤8B models, which otherwise suffer mode collapse and propose duplicate expressions repeatedly!

### 2.5 AIDE, Reflexion, Self-Refine, Voyager, The AI Scientist-v2, & Agent Laboratory
- **Citations**:
  - *AIDE*: Zhengyao Jiang et al. (Feb 2025), `arXiv:2502.13138v1` (`https://arxiv.org/abs/2502.13138`) — tree search over ML code drafts/debug steps.
  - *Reflexion*: Noah Shinn et al. (Mar 2023), `arXiv:2303.11366v4` (`https://arxiv.org/abs/2303.11366`) — verbal episodic memory reflections on scalar/error feedback.
  - *Self-Refine*: Aman Madaan et al. (Mar 2023), `arXiv:2303.17651v2` (`https://arxiv.org/abs/2303.17651`) — iterative self-feedback refinement.
  - *Voyager*: Guanzhi Wang et al. (May 2023), `arXiv:2305.16291v2` (`https://arxiv.org/abs/2305.16291`) — compositional verified skill library + automatic curriculum.
  - *The AI Scientist-v2*: Yutaro Yamada et al. (Apr 2025), `arXiv:2504.08066v1` (`https://arxiv.org/abs/2504.08066`) — progressive agentic tree search for end-to-end workshop papers.
  - *Agent Laboratory*: Samuel Schmidgall et al. (Jan 2025), `arXiv:2501.04227v2` (`https://arxiv.org/abs/2501.04227`) — staged literature/experimentation/reporting co-pilot; notably reports sharp degradation when replacing frontier reasoning models (`o1-preview`) with weaker models unless workflow scaffolding is tightened.
- **Key Takeaway for ≤8B**: Pure self-critique (*Self-Refine* / *Reflexion*) without an **external deterministic verifier** degrades on ≤8B models due to uncalibrated self-praise. Therefore, our retry loop feeds **objective compiler/schema/AST-verifier errors** rather than subjective LLM critiques.

---

## 3. Quantitative Alpha Mining & Financial Leakage Safeguards

### 3.1 RD-Agent(Q) & Alpha-GPT (v1 / 2.0)
- **Citations**:
  - *RD-Agent(Q)*: Yuante Li, Xu Yang, Xiao Yang, Minrui Xu, Xisen Wang, Weiqing Liu, Jiang Bian (May 2025, NeurIPS 2025), `arXiv:2505.15155v2` (`https://arxiv.org/abs/2505.15155`). Joint factor-model optimization with multi-armed bandit direction scheduling; achieves higher returns with 70% fewer factors.
  - *Alpha-GPT*: Saizhuo Wang et al. (Jul 2023), `arXiv:2308.00016v2` (`https://arxiv.org/abs/2308.00016`); *Alpha-GPT 2.0*: Hang Yuan, Saizhuo Wang, Jian Guo (Feb 2024), `arXiv:2402.09746v1` (`https://arxiv.org/abs/2402.09746`). Translates domain trading hypotheses into formal alpha expressions.
- **Common Leakage & Overfitting Failure Modes in LLM Quant Papers**:
  1. **Look-ahead in fundamentals**: Joining quarterly ratios on `fiscal_period_end` (e.g., March 31) instead of `announcement_date` (late April/May).
  2. **Survivorship bias**: Fetching only currently listed VN30/HOSE tickers via live APIs, omitting delisted/halted distress crashes (e.g., FLC, ROS).
  3. **Unrealistic execution**: Ignoring Vietnam's **T+2.5 settlement lock**, **z_H\pm 7\% / 10\% / 15\%z_H price bands** (buying ceiling-locked winners or selling floor-locked losers on paper), **0.1% statutory sell tax**, and thin-liquidity market impact.
  4. **Data snooping / multiple testing**: Running hundreds of LLM trials against one backtest window without purging overlapping forward labels or applying **Deflated Sharpe Ratio** (*Bailey & López de Prado, 2014*) corrections.

---

## 4. Small Open Models (≤8B) & Post-Training / Decoding Techniques

- **Qwen3-8B Backbone**: An Yang et al. (May 2025), *Qwen3 Technical Report*, `arXiv:2505.09388v1` (`https://arxiv.org/abs/2505.09388`). Provides Apache-2.0 dense `Qwen3-8B` and `Qwen3-4B` models trained with flagship distillation, controllable thinking budgets, and strong tool/JSON following—fitting comfortably in 4-bit quantization (`~5.5 GB` VRAM, leaving ample KV-cache headroom on a single 24 GB GPU, or running on 16 GB RAM CPU via `llama.cpp` / `Ollama`).
- **Scaffolding Stack Used in VN-RSI**:
  1. **JSON-Schema Constrained Decoding**: Eliminates syntax hallucinations on typed `FalsifiableHypothesis`, `FactorSpec`, and `GenomePatchProposal` outputs.
  2. **Deterministic Verifier-Guided Best-of-z_HNz_H**: Generates z_HN \in [2, 4]z_H candidate ASTs and scores structural validity, depth parsimony, and novelty before touching market data.
  3. **Short-Context Epistemic Retrieval**: Compresses prior runs into `<400` tokens of top-z_HKz_H mechanism beliefs with confidence decay and contradiction counts.
  4. **Rejection-Sampling Fine-Tuning (ReST / QLoRA Trajectory Export)**: Exports verified `(hypothesis -> causal AST)` pairs (`logs/sft_verified_trajectories.jsonl`) for optional local QLoRA self-training on 24 GB GPUs without touching validation/test labels.
