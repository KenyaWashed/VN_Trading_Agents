"""MAVE-Q v2: Friction-Aware Quality-Diversity Evolution with Portfolio-Level Credit Assignment.

Author: Tran Duy Khang â€” AI Software Engineer / Freelance Vibe Researcher

What changed vs. MAVE-Q v1 (Orthogonal diffu-GRPO arena), and why:

1. Portfolio-Level Credit Assignment (PLCA). v1 rewarded each candidate with a hand-mixed IC proxy, so the
   generator never saw the quantity we actually care about: net-of-friction Sharpe of the *ensemble*
   under T+2.5, price bands, lots, 0.1% sell tax and impact. v2 credits each candidate with its marginal
   contribution to a distributionally-robust search-zone portfolio objective
   (full-period net Sharpe + 0.5 * worst-fold Sharpe - 0.25 * fold dispersion + 3 * Rank IC),
   computed with the human-owned simulator on the `search` zone only.
2. Friction-Aware Temporal Smoothing (FATS). Every admitted factor gets a causal `ts_mean` hysteresis
   window chosen by the same robust objective, letting high-IC fast signals enter the book at low turnover.
3. Quality-Diversity elite archive (MAP-Elites). Elites are kept per (dominant information source x
   turnover bucket) cell so the ensemble can draw from fundamentals, flows, liquidity, price and
   microstructure niches instead of collapsing onto a single family.
4. Sign-aware repair. Strongly negative-IC structures are kept as negated elites instead of discarded.
5. Elite-replay diffu-GRPO + Thompson family bandit + replay-trained HRM surrogate with epsilon-exploration
   pruning (v1 retrained HRM only on the current round's 7 points and pruned greedily from round 1).
6. Signed simplex weight search (multiplicative coordinate moves) on the robust objective.

Champion selection still uses the scalar validation view only, and `sealed_test` is touched once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from vn_rsi.dsl.ast_spec import ASTNode, ASTVerifier, FactorSpec, evaluate_factor_spec
from vn_rsi.eval.evaluator import EvaluationOutcome, SealedEvaluator
from vn_rsi.models.alpha_mining_baselines import GrammarTreeSampler, validation_selection_score
from vn_rsi.models.ast_diffusion import ASTDiffusionConfig, DiscreteASTDiffusion
from vn_rsi.models.ast_vocab import MECHANISM_FAMILIES
from vn_rsi.models.hrm_reasoner import HRMConfig, HRMReasoner
from vn_rsi.sandbox.seal import SealedTestReuseError

TERMINAL_GROUPS: dict[str, tuple[str, ...]] = {
    "fundamental": ("pit_roe", "pit_pb_ratio", "pit_revenue_yoy", "pit_value_quality"),
    "flow": ("foreign_net_ratio", "fol_room_pct"),
    "liquidity": ("volume", "turnover_vnd", "turnover_shock_20d"),
    "price": ("close", "open", "high", "low", "return_1d"),
    "micro": ("intraday_range", "close_to_open", "dist_to_ceiling", "dist_to_floor"),
}
_TERM_TO_GROUP = {t: g for g, ts in TERMINAL_GROUPS.items() for t in ts}
TURNOVER_EDGES: tuple[float, ...] = (0.03, 0.07, 0.14)

# Dimensional-Consistency / Stationarity Guard (DCG): physical units of raw terminals.
#   P = price level (VND/share), V = share volume, M = traded value (VND), U = dimensionless / stationary.
TERMINAL_UNITS: dict[str, str] = {"close": "P", "open": "P", "high": "P", "low": "P", "volume": "V",
                                  "turnover_vnd": "M"}
_SCALE_FREE_TS_OPS = {"ts_return", "ts_zscore", "ts_rank"}
_UNIT_PRESERVING_OPS = {"ts_mean", "ts_min", "ts_max", "lag", "ts_std", "ts_delta", "neg", "abs_val",
                        "tanh_squash", "cs_rank", "cs_zscore"}


def dimension_of(node: ASTNode) -> str:
    """Propagate physical units through the AST; 'U' means scale-free and time-stationary.

    A cross-sectional rank of a raw price/volume *level* (e.g. cs_rank(high)) is a static stock ranking that
    encodes which names happened to trend in-sample (look-back selection bias), not a repeatable signal.
    DCG therefore admits only factors whose final unit is 'U'. Mixed-unit add/sub is invalid ('!').
    """
    if node.node_type == "terminal":
        return TERMINAL_UNITS.get(node.name, "U")
    dims = [dimension_of(ch) for ch in node.args]
    if "!" in dims:
        return "!"
    name = node.name
    if name in _SCALE_FREE_TS_OPS:
        return "U"
    if name in ("cs_rank", "cs_zscore", "tanh_squash") and dims[0] != "U":
        return "!"  # cross-sectional ranking / squashing of a raw level is a static, non-stationary bet
    if name in _UNIT_PRESERVING_OPS:
        return dims[0]
    a, b = dims[0], dims[1]
    if name in ("add", "sub"):
        return a if a == b else "!"
    if name == "mul":
        if a == "U":
            return b
        if b == "U":
            return a
        return f"({a}*{b})"
    if name == "safe_div":
        if a == b:
            return "U"
        if b == "U":
            return a
        return f"({a}/{b})"
    return "!"


def is_dimensionally_stationary(node: ASTNode) -> bool:
    return dimension_of(node) == "U"


def _terminals(node: ASTNode) -> list[str]:
    if node.node_type == "terminal":
        return [node.name]
    out: list[str] = []
    for ch in node.args:
        out.extend(_terminals(ch))
    return out


def dominant_information_source(node: ASTNode) -> str:
    counts: dict[str, int] = {g: 0 for g in TERMINAL_GROUPS}
    for t in _terminals(node):
        counts[_TERM_TO_GROUP.get(t, "price")] += 1
    return max(TERMINAL_GROUPS, key=lambda g: (counts[g], -list(TERMINAL_GROUPS).index(g)))


def turnover_bucket(turnover: float) -> int:
    for i, edge in enumerate(TURNOVER_EDGES):
        if turnover < edge:
            return i
    return len(TURNOVER_EDGES)


def smoothed_spec(spec: FactorSpec, window: int) -> FactorSpec:
    """Causal hysteresis wrapper ts_mean(expr, w); w <= 1 returns the raw factor."""
    if window <= 1:
        return spec
    return FactorSpec(
        factor_id=f"{spec.factor_id}_sm{window}",
        hypothesis_id=spec.hypothesis_id,
        description=f"FATS ts_mean(w={window}) of {spec.factor_id}",
        root=ASTNode(node_type="op", name="ts_mean", window=int(window), args=[spec.root]),
        normalize_cross_section=spec.normalize_cross_section,
    )


def negated_spec(spec: FactorSpec) -> FactorSpec:
    return FactorSpec(
        factor_id=f"{spec.factor_id}_neg",
        hypothesis_id=spec.hypothesis_id,
        description=f"sign-repaired neg of {spec.factor_id}",
        root=ASTNode(node_type="op", name="neg", args=[spec.root]),
        normalize_cross_section=spec.normalize_cross_section,
    )


@dataclass
class ScreenedFactor:
    spec: FactorSpec          # sign-correct spec (possibly neg-wrapped)
    family: str
    fitness: float
    rank_ic: float
    icir: float
    net_sharpe: float
    turnover: float
    fold_pos_frac: float
    cell: tuple[str, int]
    signal_vec: np.ndarray = field(repr=False)
    fold_metrics: list[dict[str, float]] = field(default_factory=list, repr=False)

    @property
    def fingerprint(self) -> str:
        return self.spec.root.structural_fingerprint()


class QualityDiversityArchive:
    """MAP-Elites archive over (dominant information source, turnover bucket)."""

    def __init__(self) -> None:
        self.cells: dict[tuple[str, int], ScreenedFactor] = {}

    def try_insert(self, sf: ScreenedFactor) -> bool:
        cur = self.cells.get(sf.cell)
        if cur is None or sf.fitness > cur.fitness:
            self.cells[sf.cell] = sf
            return True
        return False

    def elites(self) -> list[ScreenedFactor]:
        return sorted(self.cells.values(), key=lambda s: s.fitness, reverse=True)

    @property
    def coverage(self) -> int:
        return len(self.cells)


class FamilyThompsonBandit:
    """Gaussian Thompson sampling over mechanism families on search-zone screen fitness."""

    def __init__(self, families: tuple[str, ...], rng: np.random.Generator) -> None:
        self.families = families
        self.rng = rng
        self.sum = {f: 0.0 for f in families}
        self.n = {f: 0 for f in families}

    def update(self, family: str, reward: float) -> None:
        self.sum[family] += float(reward)
        self.n[family] += 1

    def probs(self, n_draws: int = 256) -> np.ndarray:
        prior_mu, prior_prec, obs_prec = 0.0, 1.0 / 0.004, 1.0 / 0.003
        mus, sds = [], []
        for f in self.families:
            post_prec = prior_prec + obs_prec * self.n[f]
            mus.append((prior_mu * prior_prec + obs_prec * self.sum[f]) / post_prec)
            sds.append(math.sqrt(1.0 / post_prec))
        draws = self.rng.normal(np.asarray(mus)[None, :], np.asarray(sds)[None, :], size=(n_draws, len(self.families)))
        wins = np.bincount(np.argmax(draws, axis=1), minlength=len(self.families)).astype(float)
        p = (wins + 1.0) / (wins.sum() + len(self.families))
        return p / p.sum()


@dataclass
class EnsembleMember:
    factor: ScreenedFactor
    smooth: int = 1
    weight: float = 1.0

    def _core(self) -> tuple[FactorSpec, float]:
        """Strip a sign-repair `neg` so FATS smoothing stays within depth 5; sign moves to the weight.

        Exact: cs-normalize(ts_mean(-x)) == -cs-normalize(ts_mean(x)).
        """
        sp = self.factor.spec
        if sp.root.name == "neg" and sp.root.args:
            core = FactorSpec(factor_id=sp.factor_id, hypothesis_id=sp.hypothesis_id, description=sp.description,
                              root=sp.root.args[0], normalize_cross_section=sp.normalize_cross_section)
            return core, -1.0
        return sp, 1.0

    def eval_spec(self) -> FactorSpec:
        core, _ = self._core()
        return smoothed_spec(core, self.smooth)

    def signed_weight(self) -> float:
        _, sign = self._core()
        return sign * self.weight


class CrossFoldTransferCalibrator:
    """Cross-Fold Transfer Calibration (CFTC): a search-zone-only meta-learner.

    Raw search Sharpe of a top-6 book is a very noisy estimator of out-of-sample Sharpe. CFTC learns, purely
    inside the `search` zone, which statistics of a factor/ensemble over chronological folds [j, j+w) predict
    its net Sharpe on the *next* unseen fold j+w (forward-chaining). The fitted ridge model is then applied to
    the most recent w folds to forecast next-period (i.e. post-search) net Sharpe, and used as the
    friction-aware fitness for screening and for portfolio-level credit. No validation data is ever used.
    """

    def __init__(self, window: int = 2, ridge: float = 3.0, min_rows: int = 30) -> None:
        self.window = window
        self.ridge = ridge
        self.min_rows = min_rows
        self.records: dict[str, list[dict[str, float]]] = {}
        self.coef: np.ndarray | None = None
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None
        self.n_rows = 0

    @staticmethod
    def features(folds: list[dict[str, float]]) -> np.ndarray:
        ic = np.asarray([float(f["rank_ic"]) for f in folds])
        sr = np.clip(np.asarray([float(f["net_sharpe"]) for f in folds]), -4.0, 4.0)
        tu = np.asarray([float(f["daily_turnover"]) for f in folds])
        return np.asarray([
            float(np.mean(ic)), float(np.min(ic)), float(ic[-1] - ic[0]),
            float(np.mean(sr)), float(sr[-1]), float(np.mean(tu)), math.sqrt(max(0.0, float(np.mean(tu)))),
        ])

    def add(self, key: str, folds: list[dict[str, float]]) -> None:
        if len(folds) >= self.window + 1:
            self.records[key] = folds

    def fit(self) -> None:
        xs, ys = [], []
        for folds in self.records.values():
            for j in range(0, len(folds) - self.window):
                xs.append(self.features(folds[j: j + self.window]))
                ys.append(float(np.clip(folds[j + self.window]["net_sharpe"], -4.0, 4.0)))
        self.n_rows = len(ys)
        if self.n_rows < self.min_rows:
            self.coef = None
            return
        x = np.vstack(xs)
        y = np.asarray(ys)
        self.mu, self.sd = x.mean(axis=0), x.std(axis=0) + 1e-8
        xz = np.column_stack([np.ones(len(y)), (x - self.mu) / self.sd])
        reg = self.ridge * np.eye(xz.shape[1])
        reg[0, 0] = 0.0
        self.coef = np.linalg.solve(xz.T @ xz + reg, xz.T @ y)

    def predict(self, folds: list[dict[str, float]]) -> float:
        if not folds:
            return 0.0
        recent = folds[-self.window:]
        if self.coef is None or self.mu is None or self.sd is None:
            f = self.features(recent)
            return float(0.5 * f[3] + 3.0 * f[0])  # conservative shrinkage prior before enough data
        z = (self.features(recent) - self.mu) / self.sd
        return float(self.coef[0] + z @ self.coef[1:])


class RobustPortfolioObjective:
    """Friction-aware search-zone ensemble objective driven by CFTC next-period Sharpe forecasts (cached)."""

    def __init__(self, evaluator: SealedEvaluator, run_id: str, cfg: dict[str, Any],
                 calibrator: CrossFoldTransferCalibrator) -> None:
        self.evaluator = evaluator
        self.run_id = run_id
        self.calibrator = calibrator
        self.transfer_w = float(cfg.get("transfer_weight", 1.0))
        self.full_w = float(cfg.get("full_sharpe_weight", 0.25))
        self.min_fold_w = float(cfg.get("min_fold_weight", 0.0))
        self.ic_w = float(cfg.get("ic_weight", 3.0))
        self.size_pen = float(cfg.get("size_penalty", 0.01))
        self.cache: dict[str, EvaluationOutcome] = {}
        self.n_calls = 0

    @staticmethod
    def key(members: list[EnsembleMember]) -> str:
        parts = sorted(
            f"{m.factor.fingerprint}:{m.smooth}:{round(m.weight, 4)}" for m in members
        )
        return "|".join(parts)

    def score(self, out: EvaluationOutcome, n_members: int) -> float:
        m = out.metrics
        fold_sr = [float(fm["net_sharpe"]) for fm in out.fold_metrics] or [float(m["net_sharpe"])]
        return (
            self.transfer_w * self.calibrator.predict(out.fold_metrics)
            + self.full_w * float(m["net_sharpe"])
            + self.min_fold_w * float(np.min(fold_sr))
            + self.ic_w * float(m["rank_ic"])
            - self.size_pen * n_members
        )

    def __call__(self, members: list[EnsembleMember]) -> tuple[float, EvaluationOutcome | None]:
        if not members:
            return -1e9, None
        k = self.key(members)
        if k not in self.cache:
            self.n_calls += 1
            out = self.evaluator.evaluate_factor_ensemble(
                specs=[m.eval_spec() for m in members],
                weights=[m.signed_weight() for m in members],
                zone="search",
                candidate_id=f"maveq_v2_search_ens_{self.n_calls}",
                run_id=self.run_id,
            )
            self.cache[k] = out
            self.calibrator.add(f"ens::{k}", out.fold_metrics)
        out = self.cache[k]
        return float(self.score(out, len(members))), out


@dataclass
class V2RoundLog:
    round_idx: int
    latent_pool: int
    screened: int
    archive_coverage: int
    archive_inserts: int
    accepted_adds: int
    removals: int
    ensemble_size: int
    search_objective: float
    mean_grpo_reward: float
    grpo_loss: float
    val_rank_ic: float
    val_net_sharpe: float
    val_selection_score: float
    evaluator_trials: int


@dataclass
class MAVEQv2Result:
    preset: str
    backend_runtime: str
    specialist_total_params: int
    round_logs: list[V2RoundLog]
    champion_round: int
    champion_expressions: list[str]
    champion_weights: list[float]
    champion_smoothing: list[int]
    champion_search_metrics: dict[str, float]
    champion_validation_metrics: dict[str, float]
    champion_validation_fold_std: dict[str, float]
    champion_sealed_test_metrics: dict[str, Any]
    search_evaluations: int
    total_evaluator_trials: int
    archive_coverage: int
    validation_daily_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float), repr=False)


class MAVEQv2Engine:
    """Friction-Aware QD Evolution with Portfolio-Level Credit Assignment (search-only learning)."""

    def __init__(
        self,
        evaluator: SealedEvaluator,
        cfg: dict[str, Any],
        preset: str = "cpu_fast",
        device: str | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.cfg = cfg
        self.preset = preset
        self.seed = int(cfg.get("seed", 2026))
        self.rng = np.random.default_rng(self.seed)
        self.diffusion = DiscreteASTDiffusion(ASTDiffusionConfig.from_preset(preset, seed=self.seed), device=device)
        self.hrm = HRMReasoner(HRMConfig.from_preset(preset, seed=self.seed + 31), device=device)
        self.verifier = ASTVerifier(max_depth=5)
        self.sampler = GrammarTreeSampler(max_depth=4)
        self.archive = QualityDiversityArchive()
        self.bandit = FamilyThompsonBandit(MECHANISM_FAMILIES, self.rng)

        bars = evaluator.zone_guard.get_agent_visible_bars(evaluator.snapshot.bars, "search")
        dates = pd.DatetimeIndex(np.sort(bars["date"].unique()))
        self.safe_dates = dates[: max(1, len(dates) - evaluator.horizon_days)]
        self.steward_stats = {
            "search_rows": float(len(bars)),
            "ceiling_hit_rate": float(bars["hit_ceiling"].mean()),
            "floor_hit_rate": float(bars["hit_floor"].mean()),
            "halt_rate": float(bars["is_halted"].mean()),
        }
        self.screen_cache: dict[str, ScreenedFactor | None] = {}
        self.n_screens = 0
        self.hrm_replay: list[dict[str, Any]] = []
        self.dcg_rejections = 0
        cftc = cfg.get("transfer_calibration", {})
        self.calibrator = CrossFoldTransferCalibrator(
            window=int(cftc.get("window", 2)), ridge=float(cftc.get("ridge", 3.0)),
            min_rows=int(cftc.get("min_rows", 30)),
        )
        self.transfer_fitness_w = float(cftc.get("fitness_weight", 0.04))

    def _fitness(self, rank_ic: float, fold_metrics: list[dict[str, float]]) -> float:
        """Screen fitness = search Rank IC + w * CFTC-forecast next-period net Sharpe."""
        return float(rank_ic + self.transfer_fitness_w * self.calibrator.predict(fold_metrics))

    def _refresh_archive(self) -> None:
        """Re-score every screened factor with the refitted calibrator and rebuild the QD archive."""
        self.archive = QualityDiversityArchive()
        for sf in self.screen_cache.values():
            if sf is None:
                continue
            if sf.fold_metrics:
                sf.fitness = self._fitness(sf.rank_ic, sf.fold_metrics)
            self.archive.try_insert(sf)

    # ------------------------------------------------------------------ screening
    def _signal_vec(self, spec: FactorSpec) -> np.ndarray:
        mat = evaluate_factor_spec(spec, self.evaluator.feature_panel, verifier=self.verifier)
        return mat.reindex(index=self.safe_dates).fillna(0.0).to_numpy(dtype=float).ravel()

    def _screen_once(self, spec: FactorSpec, family: str, run_id: str) -> ScreenedFactor | None:
        fp = spec.root.structural_fingerprint()
        if fp in self.screen_cache:
            return self.screen_cache[fp]
        try:
            self.verifier.verify(spec)
            vec = self._signal_vec(spec)
        except Exception:
            self.screen_cache[fp] = None
            return None
        if float(np.std(vec)) < 1e-8:
            self.screen_cache[fp] = None
            return None
        out = self.evaluator.evaluate_factor(spec, zone="search", run_id=run_id)
        self.n_screens += 1
        m = out.metrics
        fold_ics = [fm["rank_ic"] for fm in out.fold_metrics] or [m["rank_ic"]]
        fold_pos = float(np.mean([1.0 if x > 0 else 0.0 for x in fold_ics]))
        self.calibrator.add(f"fac::{fp}", out.fold_metrics)
        fit = self._fitness(float(m["rank_ic"]), out.fold_metrics) if out.fold_metrics else float(m["rank_ic"])
        sf = ScreenedFactor(
            spec=spec,
            family=family,
            fitness=float(fit),
            rank_ic=float(m["rank_ic"]),
            icir=float(m["icir"]),
            net_sharpe=float(m["net_sharpe"]),
            turnover=float(m["daily_turnover"]),
            fold_pos_frac=fold_pos,
            cell=(dominant_information_source(spec.root), turnover_bucket(float(m["daily_turnover"]))),
            signal_vec=vec,
            fold_metrics=list(out.fold_metrics),
        )
        self.screen_cache[fp] = sf
        return sf

    def screen(self, spec: FactorSpec, family: str, run_id: str) -> ScreenedFactor | None:
        sf = self._screen_once(spec, family, run_id)
        repair_thr = float(self.cfg.get("sign_repair_ic_threshold", -0.015))
        if sf is not None and sf.rank_ic < repair_thr and spec.root.depth() <= 4:
            neg = self._screen_once(negated_spec(spec), family, run_id)
            if neg is not None and neg.fitness > sf.fitness:
                return neg
        return sf

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        return float(np.dot(a, b)) / denom

    def _generate_pool(self, r_idx: int, n_latent: int, fam_probs: np.ndarray, max_depth: int) -> list[tuple[str, FactorSpec]]:
        elites = self.archive.elites()
        pool: list[tuple[str, FactorSpec]] = []
        seen: set[str] = set()
        explore_frac = float(self.cfg.get("grammar_exploration_frac", 0.20))
        use_dcg = bool(self.cfg.get("dimensional_guard", True))
        for k in range(n_latent * 4):
            if len(pool) >= n_latent:
                break
            fam = MECHANISM_FAMILIES[int(self.rng.choice(len(MECHANISM_FAMILIES), p=fam_probs))]
            fid, hid = f"v2_r{r_idx}_c{k}", f"v2h_r{r_idx}_c{k}"
            u = self.rng.random()
            seed = self.seed + r_idx * 7919 + k * 31
            if u < explore_frac:
                cand = FactorSpec(factor_id=fid, hypothesis_id=hid, description="grammar exploration",
                                  root=self.sampler.random_tree(self.rng))
            elif elites and u < explore_frac + 0.45:
                # tournament parent from QD elites -> subtree infilling mutation
                idx = self.rng.integers(0, len(elites), size=min(3, len(elites)))
                parent = max((elites[int(i)] for i in idx), key=lambda s: s.fitness)
                base = parent.spec
                if base.root.name == "neg" and base.root.args:
                    base = FactorSpec(factor_id=base.factor_id, hypothesis_id=base.hypothesis_id,
                                      description=base.description, root=base.root.args[0])
                cand = self.diffusion.infill_mutate_factor(
                    parent_spec=base, family=fam, factor_id=fid, hypothesis_id=hid, max_depth=max_depth,
                    temperature=max(0.45, 0.85 - 0.04 * r_idx), seed=seed,
                )
            else:
                cand = self.diffusion.sample_factor(
                    family=fam, factor_id=fid, hypothesis_id=hid, max_depth=max_depth,
                    temperature=max(0.50, 0.90 - 0.04 * r_idx), seed=seed,
                )
            fp = cand.root.structural_fingerprint()
            if fp in seen or fp in self.screen_cache:
                continue
            try:
                self.verifier.verify(cand)
            except Exception:
                continue
            seen.add(fp)
            if use_dcg and not is_dimensionally_stationary(cand.root):
                self.dcg_rejections += 1
                continue
            pool.append((fam, cand))
        return pool

    # ------------------------------------------------------------------ main loop
    def run(self, run_id: str = "phase6_maveq_v2", score_sealed: bool = False) -> MAVEQv2Result:
        loop = self.cfg.get("loop", {})
        rounds = int(loop.get("rounds", 8))
        n_latent = int(loop.get("latent_candidates_per_round", 48))
        n_eval = int(loop.get("screen_per_round", 10))
        eps_explore = float(loop.get("surrogate_epsilon", 0.30))
        max_ens = int(loop.get("max_ensemble_factors", 6))
        max_corr = float(loop.get("max_member_corr", 0.75))
        add_cands = int(loop.get("add_candidates_per_round", 6))
        smooth_grid = [int(w) for w in loop.get("smoothing_grid", [1, 3, 5, 10])]
        credit_w = float(loop.get("credit_reward_weight", 1.5))
        grpo_epochs = int(loop.get("diffu_grpo_epochs_per_round", 12))
        hrm_epochs = int(loop.get("hrm_epochs_per_round", 12))
        min_gain = float(loop.get("min_objective_gain", 0.01))

        objective = RobustPortfolioObjective(
            self.evaluator, run_id, self.cfg.get("objective", {}), self.calibrator
        )
        members: list[EnsembleMember] = []
        cur_obj = -1e9
        logs: list[V2RoundLog] = []
        best: tuple[float, int, list[EnsembleMember], EvaluationOutcome, EvaluationOutcome] | None = None
        last_val_key: str | None = None
        last_val: EvaluationOutcome | None = None

        for r_idx in range(1, rounds + 1):
            fam_ic = {f: (self.bandit.sum[f] / self.bandit.n[f]) if self.bandit.n[f] else 0.02 for f in MECHANISM_FAMILIES}
            regime_vec = self.hrm.encode_regime_state(self.steward_stats, fam_ic)
            policy = self.hrm.propose_outer_policy(regime_vec, fam_ic, iteration=r_idx)
            hrm_p = np.array([policy["planner_focus_weights"][f] for f in MECHANISM_FAMILIES], dtype=float)
            fam_probs = 0.5 * hrm_p / hrm_p.sum() + 0.5 * self.bandit.probs()
            fam_probs /= fam_probs.sum()
            max_depth = int(policy["implementer_max_ast_depth"])

            # (1) latent generation: QD-elite infilling + fresh diffusion samples + grammar exploration
            latent = self._generate_pool(r_idx, n_latent, fam_probs, max_depth)

            # (2) HRM surrogate pruning with epsilon-exploration (pure exploration in round 1)
            scored = []
            for fam, cand in latent:
                toks = self.diffusion.vocab.encode_ast(cand.root)
                scored.append((self.hrm.score_candidate_ast(regime_vec, toks), fam, cand, toks))
            scored.sort(key=lambda x: x[0], reverse=True)
            n_greedy = 0 if r_idx == 1 else int(round(n_eval * (1.0 - eps_explore)))
            chosen = scored[:n_greedy]
            rest = scored[n_greedy:]
            if rest:
                pick = self.rng.choice(len(rest), size=min(len(rest), n_eval - len(chosen)), replace=False)
                chosen += [rest[int(i)] for i in pick]

            # (3) exact search-zone screening + QD archive insertion
            screened: list[tuple[ScreenedFactor, list[int]]] = []
            for _, fam, cand, toks in chosen:
                sf = self.screen(cand, fam, run_id=f"{run_id}_r{r_idx}")
                if sf is None:
                    continue
                screened.append((sf, toks))
                self.bandit.update(fam, sf.fitness)
            # CFTC refit on all search-zone fold records, then re-score + rebuild the QD archive
            self.calibrator.fit()
            self._refresh_archive()
            round_fps = {sf.fingerprint for sf, _ in screened}
            new_elites = [e for e in self.archive.elites() if e.fingerprint in round_fps]
            inserts = len(new_elites)
            for sf, toks in screened:
                fam = sf.family
                self.hrm_replay.append({
                    "regime_vec": regime_vec,
                    "ast_tokens": toks,
                    "search_rank_ic": float(np.clip(1.5 * sf.fitness, -0.25, 0.25)),
                    "family": fam,
                })

            # (4) Portfolio-Level Credit Assignment: greedy add / remove / FATS / weight moves
            credit: dict[str, float] = {}
            if members:
                cur_obj, _ = objective(members)  # re-score cached outcome under the refitted calibrator
            member_fps = {m.factor.fingerprint for m in members}
            pool_c = [e for e in new_elites if e.fingerprint not in member_fps]
            for e in self.archive.elites():
                if len(pool_c) >= add_cands:
                    break
                if e.fingerprint not in member_fps and all(e.fingerprint != p.fingerprint for p in pool_c):
                    pool_c.append(e)
            pool_c = [
                c for c in pool_c[:add_cands]
                if c.fitness > 0.0 and all(abs(self._corr(c.signal_vec, m.factor.signal_vec)) < max_corr for m in members)
            ]

            accepted = 0
            if not members and pool_c:
                seed_c = max(pool_c, key=lambda c: c.fitness)
                members = [EnsembleMember(seed_c, 1, 1.0)]
                cur_obj, _ = objective(members)
                pool_c = [c for c in pool_c if c.fingerprint != seed_c.fingerprint]
                accepted += 1

            # Multi-proposal move: decorrelated QD-elite "pool" books of size k (equal risk budget)
            pool_sizes = [int(k) for k in loop.get("pool_proposal_sizes", [3, 4, 5, 6])]
            ranked_elites = [e for e in self.archive.elites() if e.fitness > 0.0]
            book: list[ScreenedFactor] = []
            for e in ranked_elites:
                if all(abs(self._corr(e.signal_vec, b.signal_vec)) < max_corr for b in book):
                    book.append(e)
                if len(book) >= max(pool_sizes, default=0):
                    break
            for k in pool_sizes:
                if len(book) < k:
                    break
                trial = [EnsembleMember(e, 1, 1.0) for e in book[:k]]
                obj, _ = objective(trial)
                if obj > cur_obj + min_gain:
                    members, cur_obj = trial, obj
                    accepted += 1

            best_add: tuple[float, EnsembleMember] | None = None
            mean_w = float(np.mean([abs(m.weight) for m in members])) if members else 1.0
            member_fps = {m.factor.fingerprint for m in members}
            pool_c = [c for c in pool_c if c.fingerprint not in member_fps
                      and all(abs(self._corr(c.signal_vec, m.factor.signal_vec)) < max_corr for m in members)]
            for c in pool_c:
                trial = list(members)
                if len(trial) >= max_ens:
                    weakest = min(range(len(trial)), key=lambda i: abs(trial[i].weight))
                    trial.pop(weakest)
                mem = EnsembleMember(c, 1, mean_w)
                obj, _ = objective(trial + [mem])
                credit[c.fingerprint] = obj - cur_obj
                if obj - cur_obj > min_gain and (best_add is None or obj > best_add[0]):
                    best_add = (obj, mem)
            if best_add is not None:
                if len(members) >= max_ens:
                    weakest = min(range(len(members)), key=lambda i: abs(members[i].weight))
                    members.pop(weakest)
                members = members + [best_add[1]]
                cur_obj = best_add[0]
                accepted += 1

            # FATS: tune hysteresis window of the newest member + one rotating member
            tune_idx = {len(members) - 1, (r_idx - 1) % max(1, len(members))} if members else set()
            for ti in sorted(tune_idx):
                for w in smooth_grid:
                    if w == members[ti].smooth:
                        continue
                    trial = [EnsembleMember(m.factor, m.smooth, m.weight) for m in members]
                    trial[ti].smooth = w
                    obj, _ = objective(trial)
                    if obj > cur_obj + 1e-4:
                        members, cur_obj = trial, obj

            # Removal (leave-one-out credit)
            removals = 0
            if len(members) >= 3:
                loo = []
                for i in range(len(members)):
                    obj, _ = objective(members[:i] + members[i + 1:])
                    loo.append((obj, i))
                obj_best, i_best = max(loo)
                if obj_best > cur_obj + 0.005:
                    members = members[:i_best] + members[i_best + 1:]
                    cur_obj = obj_best
                    removals += 1

            # Signed simplex weight search (multiplicative coordinate moves)
            if len(members) >= 2:
                for i in range(len(members)):
                    for mult in (1.6, 0.6):
                        trial = [EnsembleMember(m.factor, m.smooth, m.weight) for m in members]
                        trial[i].weight = trial[i].weight * mult
                        obj, _ = objective(trial)
                        if obj > cur_obj + 1e-4:
                            members, cur_obj = trial, obj
                            break

            # (5) Elite-replay diffu-GRPO on credit-augmented rewards + HRM replay training
            rollouts = []
            for sf, _ in screened:
                rew = sf.fitness + credit_w * max(-0.05, min(0.25, credit.get(sf.fingerprint, 0.0) * 0.1))
                rollouts.append({"family": sf.family, "factor_spec": sf.spec.model_dump(), "orthogonal_reward": rew})
            for e in self.archive.elites()[:3]:
                rollouts.append({"family": e.family, "factor_spec": e.spec.model_dump(),
                                 "orthogonal_reward": e.fitness + 0.02})
            grpo = self.diffusion.fit_diffu_grpo(rollouts, epochs=grpo_epochs)
            self.hrm.train_on_search_observations(self.hrm_replay[-256:], epochs=hrm_epochs)

            # (6) Scalar validation scoreboard (only when the frozen ensemble changed)
            val_rank_ic = val_sr = val_score = float("nan")
            if members:
                vkey = RobustPortfolioObjective.key(members)
                if vkey != last_val_key:
                    last_val = self.evaluator.evaluate_factor_ensemble(
                        specs=[m.eval_spec() for m in members],
                        weights=[m.signed_weight() for m in members],
                        zone="validation",
                        candidate_id=f"maveq_v2_r{r_idx}",
                        run_id=run_id,
                    )
                    last_val_key = vkey
                assert last_val is not None
                val_score = validation_selection_score(last_val)
                val_rank_ic = float(last_val.metrics["rank_ic"])
                val_sr = float(last_val.metrics["net_sharpe"])
                if best is None or val_score > best[0]:
                    _, search_out = objective(members)
                    assert search_out is not None
                    best = (val_score, r_idx, [EnsembleMember(m.factor, m.smooth, m.weight) for m in members],
                            last_val, search_out)

            logs.append(V2RoundLog(
                round_idx=r_idx,
                latent_pool=len(latent),
                screened=len(screened),
                archive_coverage=self.archive.coverage,
                archive_inserts=inserts,
                accepted_adds=accepted,
                removals=removals,
                ensemble_size=len(members),
                search_objective=round(float(cur_obj), 4),
                mean_grpo_reward=float(grpo.get("mean_reward", 0.0)),
                grpo_loss=float(grpo.get("loss", 0.0)),
                val_rank_ic=round(val_rank_ic, 5),
                val_net_sharpe=round(val_sr, 4),
                val_selection_score=round(val_score, 4),
                evaluator_trials=int(self.evaluator.n_trials_evaluated),
            ))

        if best is None:
            raise RuntimeError("MAVE-Q v2 produced no ensemble")
        _, champ_round, champ, champ_val, champ_search = best

        test_metrics: dict[str, Any] = {"note": "sealed_test_not_requested_dev_mode"}
        if score_sealed:
            try:
                test_out = self.evaluator.evaluate_factor_ensemble(
                    specs=[m.eval_spec() for m in champ],
                    weights=[m.signed_weight() for m in champ],
                    zone="sealed_test",
                    candidate_id="maveq_v2_champion",
                    run_id=run_id,
                    allow_sealed=True,
                )
                test_metrics = dict(test_out.metrics)
            except SealedTestReuseError:
                test_metrics = {"note": "champion_hash_already_scored_in_ledger"}

        tot_w = sum(abs(m.weight) for m in champ) or 1.0
        return MAVEQv2Result(
            preset=self.preset,
            backend_runtime=f"diffusion={self.diffusion.backend_runtime}; hrm={self.hrm.backend_runtime}",
            specialist_total_params=self.diffusion.parameter_count() + self.hrm.parameter_count(),
            round_logs=logs,
            champion_round=champ_round,
            champion_expressions=[m.eval_spec().root.canonical_expr() for m in champ],
            champion_weights=[round(m.signed_weight() / tot_w, 5) for m in champ],
            champion_smoothing=[m.smooth for m in champ],
            champion_search_metrics=dict(champ_search.metrics),
            champion_validation_metrics=dict(champ_val.metrics),
            champion_validation_fold_std=dict(champ_val.fold_std),
            champion_sealed_test_metrics=test_metrics,
            search_evaluations=self.n_screens + objective.n_calls,
            total_evaluator_trials=int(self.evaluator.n_trials_evaluated),
            archive_coverage=self.archive.coverage,
            validation_daily_returns=champ_val.daily_net_returns,
        )
