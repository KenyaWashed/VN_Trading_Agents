# Phase 0 Deliverable: Architecture Design Decisions & Candidate Paper Contributions

## 1. Selected System Architecture: **VN-RSI**

We synthesize **AQuA** (`arXiv:2608.12841`), **AIDE$^2$** (`arXiv:2609.26457`), **Darwin Gödel Machine** (`arXiv:2505.22954`), and **ShinkaEvolve** (`arXiv:2509.19349`) into a leak-free bi-level multi-agent architecture specifically engineered for **$\le$8B open models** and **Vietnamese equity microstructure (HOSE / HNX / UPCoM)**:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ OUTER LOOP (Bi-Level Meta-Optimizer — Restricted to Declared Mutable Genome) │
│  • DGM Open-Ended Archive (fitness + genome novelty parent sampling)         │
│  • <=8B Meta-Proposer emits JSON GenomePatchProposal                         │
│  • MutableSurfaceGuard blocks edits to evaluator, splits, costs, or DSL      │
│  • Statistical Acceptance Gate across N paired seeds on Validation scalars   │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │ Genome v_k (prompts, focus priors, windows,
                                │ max_ast_depth, best_of_n, memory decay)
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ INNER LOOP (7-Role Constrained Multi-Agent Pipeline — <=8B LLMs)             │
│  1. Planner       -> Allocates round budget across mechanism families        │
│  2. DataSteward   -> Reads `search` zone summary ONLY via DataZoneGuard      │
│  3. IdeaMiner     -> Writes FalsifiableHypothesis (sign + refutation rule)   │
│  4. Implementer   -> Constrained JSON FactorSpec AST + best-of-n verifier    │
│  5. LeakageAuditor-> AST causality + mini dynamic shift-test + novelty check │
│  6. EvaluatorProxy-> Tests refutation on `search`, scores ensemble on `val`  │
│  7. Librarian     -> Updates EpistemicBeliefMemory + exports SFT trajectories│
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │ Typed Causal FactorSpec ASTs ONLY
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ HUMAN-OWNED SEALED SANDBOX (SHA-256 Tamper-Locked, Read-Only in Container)   │
│  • Immutable Parquet Snapshots (manifest SHA-256 verified, offline-only)     │
│  • 3 Zones: `search` (open) | `validation` (scalar only) | `sealed_test` (1x)│
│  • VN Simulator: T+2.5 lock, +/-7/10/15% ceiling/floor blocks, 0.1% sell tax │
│  • Metrics: Rank IC, ICIR, Net Sharpe, Deflated Sharpe Ratio, Bootstrap CIs  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Explicit Architectural Trade-Offs (Scaling Frontier Ideas Down to $\le$8B)

| Design Dimension | Frontier LLM Approach (70B–600B+) | VN-RSI $\le$8B Design | Trade-Off Rationale |
| :--- | :--- | :--- | :--- |
| **Inner Search Space** | Free-form Python feature/model code (*AIDE*, *RD-Agent*) | **Typed Causal Operator AST (`FactorSpec`)** | Loses exotic custom stateful indicators, but achieves **0% runtime syntax crashes** and **structural immunity to look-ahead leakage** on 8B models. |
| **Outer Self-Modification** | Raw git diffs on Python agent source (*DGM*, *AIDE$^2$*) | **Declared Mutable Genome Whitelist (`MutableSurfaceGuard`)** | Cannot invent brand-new Python classes mid-run, but eliminates silent evaluator tampering and catastrophic self-breakage common in $\le$8B code edits. |
| **Self-Correction** | Unstructured verbal self-critique (*Self-Refine*) | **Compiler/Schema Error Retries + Deterministic AST Verifier Best-of-$N$** | Replaces noisy 8B self-opinion with exact mathematical & novelty verification. |
| **Long-Horizon Memory** | 32k–128k raw trajectory logs in prompt | **`<400`-token `EpistemicBeliefMemory` with decay & contradiction penalty** | Keeps prompts well inside 4k tokens (fast CPU/24GB-GPU inference, high instruction adherence on 8B models). |
| **Sample Efficiency** | Thousands of evolutionary rollouts (*AlphaEvolve*) | **ShinkaEvolve-style AST Fingerprint Novelty Filter + Paired Multi-Seed Gate** | Prevents 8B mode-collapse duplicates before running costly portfolio simulations. |

---

## 3. Five Candidate Paper Contributions (Ranked by Novelty & Feasibility)

1. **[Rank 1 — Highest Novelty + High Feasibility, Primary Paper Claim]**
   **Trustworthy Bi-Level RSI Under Extreme Microstructure & Parameter Constraints ($\le$8B)**:
   Demonstrating when and how a $\le$8B open-model multi-agent research system can recursively improve its own search policy, prompt priors, and memory dynamics inside a SHA-256 sealed sandbox—and quantifying the exact failure modes (reward hacking attempts, mode collapse without novelty filtering, multiple-testing inflation via Deflated Sharpe Ratio) in an illiquid, rule-constrained emerging market (Vietnam's T+2.5 settlement and $\pm 7\%/10\%/15\%$ price limits).
2. **[Rank 2 — High Feasibility, Strong Empirical Workshop Contribution]**
   **Microstructure-Aware Leakage & Execution Benchmark for Emerging Equities**:
   Quantifying the "illusion gap" between naive frictionless backtests (static universe, period-end fundamentals, unconstrained daily rebalancing) vs. leak-free Vietnamese market simulation (delisted stocks, announcement-date PIT joins, ceiling/floor unfillable locks, T+2.5 holding locks, 0.1% statutory sell tax + square-root illiquidity impact).
3. **[Rank 3 — Strong Systems/Methodology Angle]**
   **Typed Mutable Genomes vs. Unconstrained Code Rewriting for Small-Model RSI**:
   Showing empirically via ablations (`full_rsi_8b` vs. `random_mutation` vs. `no_outer_loop` vs. `teacher_proposer_sim`) that bounding outer-loop self-modification to a typed hyperparameter/prompt surface captures most of a larger teacher's gains without destabilizing an 8B agent.
4. **[Rank 4 — Medium-High Novelty, Requires 24 GB GPU Compute Run]**
   **Closed-Loop ReST / QLoRA Self-Training on Sealed Search Trajectories**:
   Using verified search-zone `(FalsifiableHypothesis -> FactorSpec)` trajectories exported by our Librarian (`logs/sft_verified_trajectories.jsonl`) to LoRA-adapt `Qwen3-8B` across outer iterations without ever touching validation/test data.
5. **[Rank 5 — Focused Domain Study]**
   **Cross-Regime & Cross-Exchange OOD Transfer (`VN30` HOSE $\to$ `VN100`/HNX/UPCoM)**:
   Testing whether research policies evolved on liquid large-cap HOSE names generalize zero-shot to wider price bands ($\pm 10\%, \pm 15\%$) and thinner liquidity tiers, or degrade predictably under higher spread/impact drag.
