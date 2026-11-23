# MASTER PROMPT: Self-Improving Multi-Agent Research System for Vietnamese Equities

> Paste this whole file to the coding agent (or save as `AGENTS.md` / `PROJECT_BRIEF.md` in the repo root).
> Work in PHASES. Finish a phase, run its acceptance checks, write a report, then STOP and wait for my "continue" before the next phase.

---

## 0. Role and goal

You are a senior ML research engineer. Build, from scratch and end to end, a **recursively self-improving (RSI) multi-agent research system for Vietnamese equities (HOSE, HNX, UPCoM)**, suitable as the basis of an academic paper (target: arXiv cs.AI / cs.LG; first venue is likely a workshop).

The core scientific question:
> Can an LLM-driven research agent improve its own research process (prompts, search policy, memory, tool use) from validated experimental evidence, while evaluation stays leak-free and trustworthy, in a small, illiquid, rule-constrained emerging market, using only small open models (<= 8B parameters)?

Starting inspirations (read them first, do not copy):
- AQuA: Recursively Self-Improving Quantitative Trading Research Agents, arXiv 2608.12841 (two-system design, sealed sandbox, fixed causal operator registry, falsifiable hypotheses, persistent belief memory, test window scored once).
- Recursive self-improvement of AI research agents (AIDE²), arXiv 2609.26457 (bi-level loop: inner agent optimizes tasks under a budget; outer loop rewrites the agent; selection on private held-out scores).

This is a research codebase, **not a trading bot**. See Section 9 (hard prohibitions).

## 1. Hard constraints

1. **Model size: every LLM used in the main loop must be <= 8B parameters** (e.g. Qwen3-8B class, Llama-3.1-8B class, smaller distilled/finetuned variants). Serve locally (vLLM / llama.cpp / Ollama; 4-bit or 8-bit quantization allowed). Hardware is unknown: make VRAM/RAM budgets config parameters and document the minimum (target: one 24 GB GPU; degrade gracefully to CPU/quantized).
2. **Larger models (e.g. ~70B-180B+) may be used only as (a) reading/reference for architecture ideas and (b) an OPTIONAL, clearly flagged "teacher" backend** (distillation data, ablation upper bound). Never required to run the system. For every idea borrowed from a big-model system, document how you scaled it down to <= 8B (smaller context, structured outputs, constrained decoding, more tool scaffolding, distillation, best-of-n with verifier, etc.).
3. **Scalable by configuration, not rewrite**: a `LLMBackend` interface with interchangeable local / API / larger-model backends, selected in YAML. Same for compute (single GPU -> multi-GPU / cluster via a job-queue abstraction), data scale (VN30 -> VN100 -> all listed), and agent count.
4. **Reproducibility**: fixed seeds, pinned dependencies, versioned data snapshots, every run logged with config hash + git commit. A reviewer must be able to re-run any number in the paper from a log file.
5. Budgets everywhere: token budget, wall-clock budget, GPU-hour budget per inner loop and per outer iteration. Hard stops, not warnings.

## 2. Phase 0: Research before coding (no code yet)

Use web search. For each item produce a short note in `docs/related_work.md`: what it does, architecture, what transfers to <= 8B, what does not, and why.
- Verify the two seed papers above by reading them; list their exact mechanisms and failure cases.
- Survey current RSI / self-improving-agent architectures. Starting points to verify and extend (do not trust this list, check it and find the newest top performers): Darwin Goedel Machine, STOP (Self-Taught Optimizer), ADAS / automated agent design, AlphaEvolve and open re-implementations (OpenEvolve, ShinkaEvolve), AIDE / MLE-bench agents, Reflexion, Self-Refine, Voyager-style skill libraries, The AI Scientist (v1/v2), Agent Laboratory.
- Identify which currently top-ranked systems/models are relevant, using up-to-date leaderboards and papers. **Cite only sources you actually opened; mark anything unverified as UNVERIFIED.**
- Survey LLM-driven alpha/factor mining and leakage failure modes (e.g. Alpha-GPT-style work, RD-Agent(Q)-style work, and others you find).
- Survey small-model techniques relevant here: distillation, constrained/structured decoding, verifier-guided search, LoRA/QLoRA self-training, rejection-sampling fine-tuning.
- Deliverable: `docs/design_decision.md` choosing the architecture with explicit trade-offs, plus a list of 3-5 candidate paper contributions ranked by novelty and feasibility.

STOP and report after Phase 0.

## 3. Phase 1: Data layer (leak-free by construction)

- Sources: use vnstock (or equivalent) behind a `DataProvider` interface with multiple providers + fallback and cross-source consistency checks. Free public endpoints are unofficial and unstable: **download once, store immutable versioned snapshots (Parquet) with manifest + checksums; experiments read snapshots only, never live APIs.**
- Data licensing: do not redistribute raw data. Write `docs/data_terms.md` listing each source, what the terms allow (mark UNVERIFIED where you could not confirm), and a reproduction script so others fetch their own copy.
- Universe: start VN30 -> VN100 -> all HOSE/HNX/UPCoM liquid names, controlled by config. Daily (EOD) bars plus 1h/15m where available. Intraday history is limited in free sources: detect and document coverage honestly.
- Must handle and test: delisted symbols (survivorship bias), adjusted vs raw prices (corporate actions, stock dividends, splits) with cross-source reconciliation, trading halts, price limits, missing bars, point-in-time fundamentals (use announcement dates, never period-end dates), index membership as-of date.
- Market rules as CONFIG with sources and a "verify" flag (do not hardcode silently): price-limit bands per exchange, lot size, settlement cycle, no retail short selling (long-only or VN30 futures path), transaction fee, sell-side tax, slippage/impact model that grows with illiquidity, foreign ownership limits.
- Data QA suite: automated leakage tests (shift-test every feature; future-information detector; label-vs-feature timestamp assertions) that run in CI.

## 4. Phase 2: Sealed evaluation sandbox

Follow the AQuA principle: **agents emit constrained specifications; humans own the evaluator.**
- Splits, labels, cost model, evaluator are human-authored, fixed, hash-pinned, and **not writable by any agent**. Agents run in a container/sandbox with no network, no write access to the evaluator or test data.
- Three data zones: `search` (agent sees), `validation` (agent sees scores only), `sealed test` (scored once per frozen config; every access is logged to an append-only ledger; refuse a second look at the same config).
- Walk-forward / purged + embargoed cross-validation to respect overlapping labels. Report mean +/- std across seeds and across time folds.
- Metrics: rank IC, ICIR, long/short or long-only return, turnover, max drawdown, Sharpe/Sortino **after** costs; always with baselines: buy-and-hold VN-Index, equal-weight, simple momentum/reversal, ridge, LightGBM, GRU/LSTM. A claimed improvement must beat baselines under identical splits and costs, with confidence intervals or a multiple-testing correction (deflated Sharpe / reality-check style) because the agent performs many trials.
- Anti-reward-hacking checks: metric tamper detection, held-out "canary" tasks, checks that improvements do not come from evaluation-code changes, selection signal separated from optimization signal (bi-level, as in AIDE²).

## 5. Phase 3: Agent system v0 (before any self-improvement)

Implement a constrained hypothesis-to-experiment multi-agent pipeline, all on <= 8B models:
- Roles (merge or split as the design decision says): Manager/Planner, Data Steward, Idea Miner, Implementer (emits specs, not free code), Evaluator, Critic/Leakage-Auditor, Librarian (memory).
- Hypotheses must be **falsifiable**: statement, economic mechanism, expected sign, refutation criteria, written before building.
- Search space is a **typed DSL / operator registry** (causal operators only; config-diff for model variants), so leakage is excluded structurally. Free-form code generation is allowed only inside the sandbox and only for pre-approved templates.
- Small-model scaffolding: JSON-schema constrained decoding, short context + retrieval from memory, verifier-guided best-of-n, retry with error feedback, deterministic tool calls instead of reasoning for arithmetic.
- Persistent memory: beliefs about mechanisms with confidence, evidence links to run IDs, decay, and contradiction handling.
- Baseline result: v0 agent vs the human-designed baselines on the sealed protocol. Record cost (tokens, GPU-hours) per accepted finding.

## 6. Phase 4: Recursive self-improvement loop

Bi-level design, scaled to <= 8B:
- **Inner loop**: agent v_k does research tasks under a fixed budget -> candidate findings.
- **Outer loop**: a proposer rewrites parts of the agent (prompts, tool policy, search/forking strategy, memory policy, context management, hypothesis templates; optionally LoRA self-training on its own verified successful trajectories). Edits are restricted to a declared **mutable surface**; the evaluator, splits and operator-safety rules are immutable.
- Selection: accept a rewrite only if it improves the private held-out score **with statistical evidence** (multiple seeds, paired comparison, noise estimate), because noise compounds across the two loops. Keep an archive of agent variants (DGM-style open-ended archive) rather than a single incumbent.
- Track and report: accepted/rejected rewrites, per-iteration cost, regression checks, reward-hacking rate, held-out and out-of-distribution generalization (e.g. other universe, other period, other horizon), and an "ignition" check: does the improved agent drive the outer loop at least as well as the agent that produced it (use enough seeds; say plainly if inconclusive).
- Required ablations: no-memory, no-outer-loop, random-mutation, larger-model-proposer (teacher) vs 8B-proposer, with/without leakage auditor.

## 7. Phase 5: Experiments and paper pipeline

- Config-driven experiment runner, result tables auto-generated from logs, each table cell linked to a run ID.
- Honest reporting: failures, negative results, cost, variance. No cherry-picked seeds or windows. Report what did NOT transfer.
- Produce a paper skeleton (LaTeX) with: problem, method, sealed protocol, results, ablations, limitations (single market, simulated costs, no live trading, human-owned sandbox). **All citations must come from a bibliography file built from verified sources; a script must check every citation key resolves and every number in the paper matches a log.** I will read and edit the paper myself; do not submit anything anywhere.

## 8. Engineering requirements

- Python 3.11+, `uv` or `pip-tools` pinned deps; `src/` layout; typed (mypy), linted (ruff), tests (pytest) with leakage tests mandatory; CI script; Dockerfile for the sandbox; Makefile targets: `data`, `test`, `baseline`, `agent`, `rsi`, `report`.
- Suggested layout: `configs/ data/ (gitignored) docs/ src/{data,sandbox,agents,dsl,memory,rsi,eval,llm,utils}/ experiments/ tests/ paper/`.
- Every module has a README line stating its trust level (agent-writable vs human-owned).
- Structured logging (JSONL), experiment tracking (local MLflow/W&B-offline or plain files).
- Document minimum and recommended hardware, and a "scale-up" guide: bigger backend, more GPUs, wider universe, longer loops.
- Keep a `PROGRESS.md` updated at the end of each phase: what is done, what is verified, what is UNVERIFIED, next decisions for me.

## 9. Hard prohibitions

- No live trading, no brokerage/order-placement code, no API keys for trading accounts, no connection to real accounts. Simulation and research only. No investment advice in outputs.
- No scraping that bypasses CAPTCHAs, authentication, rate limits or robots.txt. Respect source terms; rate-limit and cache.
- Never let an agent modify the evaluator, splits, cost model, sealed test data, or the test ledger.
- Never invent citations, numbers, datasets, or results. If something cannot be verified or run, label it UNVERIFIED / NOT RUN.
- Do not install or run code from untrusted sources outside the sandbox.

## 10. How to work with me

- Decide on small details yourself and note the decision in `docs/decisions.md`. Ask me only about choices that change scope, cost, or the scientific claim (e.g. hardware available, which paper contribution to pursue, data licensing).
- After each phase: summary (what was built, test results, known issues), the acceptance checklist with pass/fail, and the proposed next step. Then wait.
- Phase acceptance (minimum): Phase 0 notes verified and cited; Phase 1 data QA + leakage tests green on a snapshot; Phase 2 evaluator hash-locked and baselines reproduce; Phase 3 v0 beats or honestly fails to beat baselines with CIs; Phase 4 at least one full RSI run with ablations and logged costs; Phase 5 every paper number traceable to a log.

Begin with Phase 0.
