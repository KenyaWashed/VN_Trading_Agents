# VN-RSI: Recursively Self-Improving Multi-Agent Research System for Vietnamese Equities ($\le$8B & Modular Micro-RSI)

**Author**: **Tran Duy Khang** — *AI Software Engineer / Freelance Vibe Researcher*

> **Research Simulation Codebase ONLY (Section 9 Hard Prohibition)**: This system contains **zero** live trading, brokerage connectivity, account credentials, or investment advice. All experiments run offline against immutable versioned Parquet snapshots inside a SHA-256 sealed evaluation sandbox.

---

## Core Scientific Question
> *Can an LLM-driven research agent improve its own research process (prompts, search policy, memory, tool use) from validated experimental evidence, while evaluation stays leak-free and trustworthy, in a small, illiquid, rule-constrained emerging market (HOSE, HNX, UPCoM), using only small open models ($\le$8B parameters)?*

---

## Repository Structure & Trust Levels

| Path | Trust Level | Description |
| :--- | :--- | :--- |
| `configs/` | **HUMAN-OWNED IMMUTABLE** | Market rules (`HOSE ±7%`, `HNX ±10%`, `UPCoM ±15%`, `T+2.5`, `0.1%` sell tax), splits, LLM budgets, mutable surface whitelist |
| `src/vn_rsi/data/` | **HUMAN-OWNED IMMUTABLE** | `DataProvider` (`vnstock` + deterministic synthetic VN fixture), immutable Parquet `SnapshotManager`, `DataQASuite` shift-leakage tests |
| `src/vn_rsi/dsl/` | **HUMAN-OWNED IMMUTABLE** | Fixed causal operator registry, `FactorSpec` AST JSON schema for $\le$8B constrained decoding, `ASTVerifier` |
| `src/vn_rsi/sandbox/` | **HUMAN-OWNED IMMUTABLE** | `EvaluatorSeal` (SHA-256 tamper lock), `DataZoneGuard` (`search`/`val`/`sealed_test`), `VNPortfolioSimulator`, single-shot `SealedTestLedger` |
| `src/vn_rsi/eval/` | **HUMAN-OWNED IMMUTABLE** | Rank IC, ICIR, Net Sharpe/Sortino, Deflated Sharpe Ratio, block-bootstrap CIs, 7 human baselines, Canary null checks |
| `src/vn_rsi/llm/` | **HUMAN-OWNED IMMUTABLE** | `LLMBackend` enforcing $\le$8B main-loop size limit (`Ollama`/`vLLM`/`llama.cpp` + CPU deterministic scaffolding fallback + flagged optional teacher) |
| `src/vn_rsi/memory/` | **SCHEMA HUMAN-OWNED / POLICY MUTABLE** | `EpistemicBeliefMemory` with confidence decay, run-ID links, and contradiction penalties |
| `src/vn_rsi/agents/` | **AGENT-MUTABLE VIA GENOME** | 7-role constrained pipeline (`Planner`, `DataSteward`, `IdeaMiner`, `Implementer`, `LeakageAuditor`, `EvaluatorProxy`, `Librarian`) |
| `src/vn_rsi/rsi/` | **HUMAN-OWNED META-CONTROLLER** | `BiLevelRSIRunner`, `MutableSurfaceGuard`, `DGMArchive`, paired multi-seed statistical gate, Ignition & OOD checks |
| `experiments/` | **HUMAN-OWNED CLI** | Reproducible runners for Phases 1–5 (`run_data.py`, `run_baselines.py`, `run_agent_v0.py`, `run_rsi.py`, `build_report.py`) |
| `tests/` | **HUMAN-OWNED CI** | Mandatory leakage shift tests, AST causality tests, sandbox tamper/ledger tests, $\le$8B & budget hard-stop tests |
| `paper/` | **AUTO-VERIFIED MANUSCRIPT** | LaTeX skeleton (`main.tex`), verified BibTeX (`references.bib`), auto-generated tables (`tables/`), and `verify_paper.py` auditor |

---

## Quickstart (End-to-End Reproducible Pipeline)

```bash
cd D:\Research\VN_Trading_Agents

# 1. Install package & dev tools into local .venv via uv
uv sync --extra dev

# 2. Run mandatory unit, leakage, causality, and sandbox tamper tests + linters
uv run pytest
uv run ruff check src tests experiments paper
uv run mypy src/vn_rsi

# 3. Execute Phases 1 -> 5 end-to-end and verify every paper citation & table number
uv run python experiments/run_data.py
uv run python experiments/run_baselines.py --reset-ledger
uv run python experiments/run_agent_v0.py
uv run python experiments/run_rsi.py
uv run python experiments/build_report.py
uv run python paper/verify_paper.py
```

See [`docs/related_work.md`](docs/related_work.md), [`docs/design_decision.md`](docs/design_decision.md), [`docs/data_terms.md`](docs/data_terms.md), and [`docs/scale_up_guide.md`](docs/scale_up_guide.md) for full documentation.
