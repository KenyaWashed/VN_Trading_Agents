# Data Sources, Licensing Terms, and Non-Redistribution Policy

> **Hard Rule (Section 3 & Section 9)**: Raw market data MUST NEVER be committed to git or redistributed. `data/` is gitignored. Every researcher or reviewer reproduces local immutable Parquet snapshots via `make data`.

---

## 1. Source Inventory & Terms Assessment

| Source / Adapter | Underlying Endpoints | Data Types | Redistribution Allowed? | Terms Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **`SyntheticVNMarketProvider`** (`provider_mode: synthetic_fixture`) | 100% locally generated stochastic microstructure fixture | EOD OHLCV, corporate splits, halts/delistings, PIT quarterly fundamentals, foreign flows, index membership | **Yes** (code is MIT; deterministic from seed `20261123`) | **VERIFIED** (human-authored mathematical generator in `src/vn_rsi/data/providers.py`) |
| **`VnstockDataProvider`** (`provider_mode: live_vnstock`) | Open-source Python wrapper (`thinh-vu/vnstock`) querying public web endpoints of Vietnamese brokers (VCI/Vietcap, TCBS) | EOD OHLCV, company overview, basic financial ratios | **NO — Do NOT redistribute raw data** | **UNVERIFIED** for commercial/redistribution use. Broker public chart endpoints are provided for personal viewing; automated bulk redistribution is generally restricted by broker ToS. |
| **HOSE / HNX / VSDC / SSC Regulatory Rules** (`configs/market_rules.yaml`) | Public statutory circulars & exchange trading rules | Price bands (z_H\pm 7\%/10\%/15\%z_H), board lot (`100`), settlement (`T+2.5`), sell tax (`0.1%`), FOL (`30%/49%`) | Yes (public regulatory facts) | **VERIFIED** against SSC / HNX / Circular 111/2013/TT-BTC summaries (exact proprietary broker slippage curve marked `verified: false` in YAML). |

---

## 2. Responsible Fetching Rules Enforced in Code

1. **Zero Bypass / Zero Scraping Abuse**: `VnstockDataProvider` never bypasses CAPTCHAs, authentication, `robots.txt`, or rate limits. It enforces `rate_limit_seconds >= 0.5` (default `1.0s` between requests).
2. **Download Once, Read Offline Forever**: Once `experiments/run_data.py` writes `data/snapshots/<snapshot_id>/*.parquet` and `manifest.json` (with SHA-256 checksums), Phases 2–5 read **exclusively** from local Parquet files and run inside `--network=none` sandbox containers.
3. **Honest Synthetic Flagging**: Whenever `SyntheticVNMarketProvider` is used (e.g., in CI, offline replication, or when `vnstock` is not installed), `manifest.json` records `"is_synthetic": true` and every downstream experiment log (`logs/experiment_registry.jsonl`) and auto-generated LaTeX table explicitly states `is_synthetic=true`.
4. **Honest Intraday Coverage Disclosure**: Free Vietnamese public APIs typically retain only ~6–12 months of 1H/15m intraday history and omit delisted tickers. `manifest.json` logs `intraday_coverage_report` with `available: false` for multi-year walk-forward splits rather than silently truncating backtest years.

---

## 3. Reproduction Commands for Reviewers

```bash
# Option A: Fully deterministic offline microstructure benchmark (default CI / zero network)
uv run python experiments/run_data.py --config configs/data.yaml --market-rules configs/market_rules.yaml

# Option B: Fetch your own local copy via vnstock (requires internet + optional vnstock package)
uv pip install -e ".[vnstock]"
# Edit configs/data.yaml -> set `provider_mode: "live_vnstock"`, then run once:
uv run python experiments/run_data.py --config configs/data.yaml --market-rules configs/market_rules.yaml --overwrite
```
