"""Same-topic automated alpha-mining SOTA competitors evaluated inside the identical sealed sandbox.

Author: Tran Duy Khang — AI Software Engineer / Freelance Vibe Researcher

Implements 7 representative automated alpha-mining paradigms from the quantitative AI literature,
all sharing the exact closed causal DSL, search-zone-only fitness, independent trial-deflated
SealedEvaluator instances, validation-only checkpoint selection, and single-shot sealed_test protocol:

1. ``Alpha101StyleLibrary``   — Handcrafted institutional formula library (Kakushadze, 2016).
2. ``RandomDSLSearch``        — Budget-matched uniform causal grammar sampling (sanity floor).
3. ``GPAlphaMiner``           — Tournament Genetic Programming w/ crossover & mutation (AutoAlpha / gplearn).
4. ``PPOAlphaMiner``          — Actor-Critic Clipped PPO optimizing synergistic pool IC (AlphaGen, KDD 2023).
5. ``MCTSAlphaMiner``         — UCT Monte Carlo Tree Search over prefix grammar rules (Alpha-MCTS / RiskMiner).
6. ``MAPElitesAlphaMiner``    — Behavioral Quality-Diversity evolutionary archive (MAP-Elites / OpenELM).
7. ``AlphaGPTIterativeMiner`` — Hypothesis-guided iterative refinement agent (Alpha-GPT / R&D-Agent-Quant).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from vn_rsi.dsl.ast_spec import ASTNode, ASTVerifier, FactorSpec, evaluate_factor_spec
from vn_rsi.dsl.operators import ALLOWED_TERMINALS
from vn_rsi.eval.evaluator import EvaluationOutcome, SealedEvaluator
from vn_rsi.eval.metrics import compute_rank_ic_series
from vn_rsi.llm.backend import RuleGuidedSmallLMBackend
from vn_rsi.models.ast_vocab import (
    BINARY_OPS,
    CANONICAL_WINDOWS,
    MECHANISM_FAMILIES,
    UNARY_NON_WINDOW_OPS,
    WINDOWED_DSL_OPS,
    ASTVocab,
)
from vn_rsi.sandbox.seal import SealedTestReuseError
from vn_rsi.utils.budget import BudgetConfig, BudgetTracker

TERMINAL_NAMES: tuple[str, ...] = tuple(sorted(ALLOWED_TERMINALS.keys()))

_NICHE_GROUPS: dict[str, tuple[str, ...]] = {
    "fundamental": ("pit_roe", "pit_pb_ratio", "pit_revenue_yoy", "pit_value_quality"),
    "flow": ("foreign_net_ratio", "fol_room_pct"),
    "liquidity": ("volume", "turnover_vnd", "turnover_shock_20d"),
    "price": ("close", "open", "high", "low", "return_1d"),
    "micro": ("intraday_range", "close_to_open", "dist_to_ceiling", "dist_to_floor"),
}
_TERM_TO_NICHE = {t: g for g, ts in _NICHE_GROUPS.items() for t in ts}


def validation_selection_score(outcome: EvaluationOutcome) -> float:
    """Shared scalar-validation champion selection objective (identical across all methods)."""
    m = outcome.metrics
    return (
        float(m["net_sharpe"])
        + 3.0 * float(m["rank_ic"])
        - 0.12 * float(outcome.fold_std.get("net_sharpe", 0.0))
        - 1.5 * float(m["daily_turnover"])
    )


class GrammarTreeSampler:
    """Uniform-ish random causal AST generator over the closed DSL (canonical windows only)."""

    def __init__(self, max_depth: int = 4) -> None:
        self.max_depth = max_depth

    def random_tree(
        self, rng: np.random.Generator, max_depth: int | None = None, depth: int = 1
    ) -> ASTNode:
        limit = max_depth if max_depth is not None else self.max_depth
        if depth >= limit or (depth > 1 and rng.random() < 0.30):
            return ASTNode(node_type="terminal", name=str(rng.choice(TERMINAL_NAMES)))
        u = rng.random()
        if u < 0.45:
            op = str(rng.choice(WINDOWED_DSL_OPS))
            w = int(rng.choice(CANONICAL_WINDOWS))
            return ASTNode(
                node_type="op", name=op, window=w, args=[self.random_tree(rng, limit, depth + 1)]
            )
        if u < 0.65:
            op = str(rng.choice(UNARY_NON_WINDOW_OPS))
            return ASTNode(node_type="op", name=op, args=[self.random_tree(rng, limit, depth + 1)])
        op = str(rng.choice(BINARY_OPS))
        return ASTNode(
            node_type="op",
            name=op,
            args=[
                self.random_tree(rng, limit, depth + 1),
                self.random_tree(rng, limit, depth + 1),
            ],
        )


def _subtree_paths(node: ASTNode, prefix: tuple[int, ...] = ()) -> list[tuple[int, ...]]:
    paths = [prefix]
    for i, ch in enumerate(node.args):
        paths.extend(_subtree_paths(ch, prefix + (i,)))
    return paths


def _get_subtree(node: ASTNode, path: tuple[int, ...]) -> ASTNode:
    cur = node
    for i in path:
        cur = cur.args[i]
    return cur


def _replace_subtree(node: ASTNode, path: tuple[int, ...], new_sub: ASTNode) -> ASTNode:
    if not path:
        return new_sub.model_copy(deep=True)
    out = node.model_copy(deep=True)
    parent = out
    for i in path[:-1]:
        parent = parent.args[i]
    parent.args[path[-1]] = new_sub.model_copy(deep=True)
    return out


def _node_count(node: ASTNode) -> int:
    return 1 + sum(_node_count(c) for c in node.args)


def _dominant_niche(node: ASTNode) -> str:
    counts = {g: 0 for g in _NICHE_GROUPS}

    def _walk(n: ASTNode) -> None:
        if n.node_type == "terminal":
            counts[_TERM_TO_NICHE.get(n.name, "price")] += 1
        for ch in n.args:
            _walk(ch)

    _walk(node)
    return max(_NICHE_GROUPS, key=lambda g: counts[g])


@dataclass
class MinedFactor:
    spec: FactorSpec
    fitness: float
    rank_ic: float
    net_sharpe: float
    turnover: float
    sign: float
    signal_vec: np.ndarray = field(repr=False)
    signal_df: pd.DataFrame | None = field(default=None, repr=False)


class _SearchZoneScorer:
    """Search-zone-only factor scoring + signal vectors for correlation-aware alpha pools."""

    def __init__(self, evaluator: SealedEvaluator, run_id: str, keep_df: bool = False) -> None:
        self.evaluator = evaluator
        self.run_id = run_id
        self.keep_df = keep_df
        self.verifier = ASTVerifier(max_depth=5)
        bars = evaluator.zone_guard.get_agent_visible_bars(evaluator.snapshot.bars, "search")
        dates = pd.DatetimeIndex(np.sort(bars["date"].unique()))
        self.safe_dates = dates[: max(1, len(dates) - evaluator.horizon_days)]
        self.search_fwd = evaluator.forward_return_wide.reindex(index=self.safe_dates)
        self.cache: dict[str, MinedFactor | None] = {}
        self.n_evaluated = 0

    def signal_matrix(self, spec: FactorSpec) -> pd.DataFrame:
        mat = evaluate_factor_spec(spec, self.evaluator.feature_panel, verifier=self.verifier)
        return mat.reindex(index=self.safe_dates).fillna(0.0)

    def score(self, spec: FactorSpec, fitness_mode: str = "ic") -> MinedFactor | None:
        fp = spec.root.structural_fingerprint()
        if fp in self.cache:
            return self.cache[fp]
        try:
            self.verifier.verify(spec)
            s_df = self.signal_matrix(spec)
            vec = s_df.to_numpy(dtype=float).ravel()
        except Exception:
            self.cache[fp] = None
            return None
        if float(np.std(vec)) < 1e-8:
            self.cache[fp] = None
            return None
        out = self.evaluator.evaluate_factor(spec, zone="search", run_id=self.run_id)
        self.n_evaluated += 1
        ic = float(out.metrics["rank_ic"])
        sr = float(out.metrics["net_sharpe"])
        turn = float(out.metrics["daily_turnover"])
        size_pen = 0.0005 * _node_count(spec.root)
        if fitness_mode == "ic":
            fit = abs(ic) - size_pen
        elif fitness_mode == "friction_ic":
            fit = abs(ic) + 0.04 * sr - 0.18 * turn - size_pen
        else:
            fit = ic + 0.06 * sr - 0.30 * turn - size_pen
        mf = MinedFactor(
            spec=spec,
            fitness=float(fit),
            rank_ic=ic,
            net_sharpe=sr,
            turnover=turn,
            sign=1.0 if ic >= 0.0 else -1.0,
            signal_vec=vec,
            signal_df=s_df if self.keep_df else None,
        )
        self.cache[fp] = mf
        return mf


def build_decorrelated_pool(
    mined: list[MinedFactor], pool_size: int = 5, max_corr: float = 0.70
) -> list[MinedFactor]:
    """Greedy top-k alpha pool by fitness subject to a pairwise signal-correlation cap (search zone)."""
    pool: list[MinedFactor] = []
    for mf in sorted(mined, key=lambda m: m.fitness, reverse=True):
        ok = True
        for p in pool:
            denom = float(np.linalg.norm(mf.signal_vec) * np.linalg.norm(p.signal_vec)) + 1e-12
            if abs(float(np.dot(mf.signal_vec, p.signal_vec)) / denom) > max_corr:
                ok = False
                break
        if ok:
            pool.append(mf)
        if len(pool) >= pool_size:
            break
    return pool


@dataclass
class CompetitorResult:
    name: str
    family: str
    search_evaluations: int
    total_evaluator_trials: int
    validation_metrics: dict[str, float]
    validation_fold_std: dict[str, float]
    sealed_test_metrics: dict[str, Any]
    champion_expressions: list[str]
    champion_weights: list[float]
    checkpoints: list[dict[str, float]]
    validation_daily_returns: pd.Series = field(
        default_factory=lambda: pd.Series(dtype=float), repr=False
    )


def _finalize_competitor(
    name: str,
    family: str,
    evaluator: SealedEvaluator,
    scorer_evals: int,
    pools: list[tuple[list[MinedFactor], list[float] | None]] | list[list[MinedFactor]],
    run_id: str,
    score_sealed: bool,
) -> CompetitorResult:
    best: tuple[float, EvaluationOutcome, list[MinedFactor], list[float]] | None = None
    checkpoints: list[dict[str, float]] = []
    seen_hashes: set[str] = set()

    for ci, item in enumerate(pools):
        if isinstance(item, tuple):
            pool, custom_w = item
        else:
            pool, custom_w = item, None
        if not pool:
            continue
        weights = (
            custom_w
            if custom_w is not None and len(custom_w) == len(pool)
            else [p.sign / len(pool) for p in pool]
        )
        key = "|".join(
            f"{p.spec.root.canonical_expr()}:{round(w, 4)}"
            for p, w in sorted(
                zip(pool, weights, strict=True), key=lambda x: x[0].spec.root.canonical_expr()
            )
        )
        if key in seen_hashes:
            continue
        seen_hashes.add(key)
        specs = [p.spec for p in pool]
        val = evaluator.evaluate_factor_ensemble(
            specs, weights, zone="validation", candidate_id=f"{name}_ckpt{ci}", run_id=run_id
        )
        score = validation_selection_score(val)
        checkpoints.append(
            {
                "checkpoint": float(ci),
                "val_net_sharpe": float(val.metrics["net_sharpe"]),
                "val_rank_ic": float(val.metrics["rank_ic"]),
                "val_selection_score": round(score, 5),
            }
        )
        if best is None or score > best[0]:
            best = (score, val, pool, list(weights))

    if best is None:
        raise RuntimeError(f"{name}: no valid checkpoint pool was produced")

    _, best_val, best_pool, best_weights = best
    specs = [p.spec for p in best_pool]
    test_metrics: dict[str, Any] = {"note": "sealed_test_not_requested_dev_mode"}
    if score_sealed:
        try:
            test_out = evaluator.evaluate_factor_ensemble(
                specs,
                best_weights,
                zone="sealed_test",
                candidate_id=f"{name}_champion",
                run_id=run_id,
                allow_sealed=True,
            )
            test_metrics = dict(test_out.metrics)
        except SealedTestReuseError:
            test_metrics = {"note": "champion_hash_already_scored_in_ledger"}
    return CompetitorResult(
        name=name,
        family=family,
        search_evaluations=scorer_evals,
        total_evaluator_trials=int(evaluator.n_trials_evaluated),
        validation_metrics=dict(best_val.metrics),
        validation_fold_std=dict(best_val.fold_std),
        sealed_test_metrics=test_metrics,
        champion_expressions=[s.root.canonical_expr() for s in specs],
        champion_weights=[round(float(w), 5) for w in best_weights],
        checkpoints=checkpoints,
        validation_daily_returns=best_val.daily_net_returns,
    )


# ---------------------------------------------------------------------------
# 1. Genetic Programming (AutoAlpha / gplearn style)
# ---------------------------------------------------------------------------
class GPAlphaMiner:
    """gplearn / AutoAlpha-style genetic programming formulaic alpha miner."""

    def __init__(
        self,
        evaluator: SealedEvaluator,
        population: int = 24,
        generations: int = 8,
        tournament: int = 4,
        p_crossover: float = 0.65,
        p_subtree_mutation: float = 0.20,
        max_depth: int = 4,
        pool_size: int = 5,
        seed: int = 2026,
    ) -> None:
        self.evaluator = evaluator
        self.population = population
        self.generations = generations
        self.tournament = tournament
        self.p_crossover = p_crossover
        self.p_subtree_mutation = p_subtree_mutation
        self.max_depth = max_depth
        self.pool_size = pool_size
        self.rng = np.random.default_rng(seed)
        self.sampler = GrammarTreeSampler(max_depth=max_depth)

    def _spec(self, node: ASTNode, tag: str) -> FactorSpec:
        return FactorSpec(factor_id=f"gp_{tag}", hypothesis_id="gp", description="GP mined", root=node)

    def _tournament(self, scored: list[MinedFactor]) -> MinedFactor:
        idx = self.rng.integers(0, len(scored), size=min(self.tournament, len(scored)))
        return max((scored[int(i)] for i in idx), key=lambda m: m.fitness)

    def _offspring(self, scored: list[MinedFactor], tag: str) -> FactorSpec:
        parent = self._tournament(scored).spec.root
        u = self.rng.random()
        child: ASTNode
        if u < self.p_crossover:
            donor = self._tournament(scored).spec.root
            p_paths = _subtree_paths(parent)
            p_path = p_paths[int(self.rng.integers(0, len(p_paths)))]
            d_paths = _subtree_paths(donor)
            d_sub = _get_subtree(donor, d_paths[int(self.rng.integers(0, len(d_paths)))])
            child = _replace_subtree(parent, p_path, d_sub)
        elif u < self.p_crossover + self.p_subtree_mutation:
            paths = _subtree_paths(parent)
            path = paths[int(self.rng.integers(0, len(paths)))]
            sub_depth = max(1, self.max_depth - len(path))
            child = _replace_subtree(parent, path, self.sampler.random_tree(self.rng, max_depth=sub_depth))
        else:
            child = parent.model_copy(deep=True)
            paths = _subtree_paths(child)
            node = _get_subtree(child, paths[int(self.rng.integers(0, len(paths)))])
            if node.node_type == "terminal":
                node.name = str(self.rng.choice(TERMINAL_NAMES))
            elif node.window is not None:
                node.window = int(self.rng.choice(CANONICAL_WINDOWS))
            else:
                child = self.sampler.random_tree(self.rng)
        if child.depth() > self.max_depth:
            child = self.sampler.random_tree(self.rng)
        return self._spec(child, tag)

    def run(self, run_id: str = "gp_alpha_miner", score_sealed: bool = False) -> CompetitorResult:
        scorer = _SearchZoneScorer(self.evaluator, run_id)
        pop_specs = [
            self._spec(self.sampler.random_tree(self.rng), f"g0_{i}") for i in range(self.population)
        ]
        all_scored: dict[str, MinedFactor] = {}
        pools: list[list[MinedFactor]] = []
        for gen in range(self.generations):
            scored: list[MinedFactor] = []
            for sp in pop_specs:
                mf = scorer.score(sp, fitness_mode="ic")
                if mf is not None:
                    scored.append(mf)
                    all_scored[sp.root.structural_fingerprint()] = mf
            if not scored:
                pop_specs = [
                    self._spec(self.sampler.random_tree(self.rng), f"g{gen}_r{i}")
                    for i in range(self.population)
                ]
                continue
            pools.append(build_decorrelated_pool(list(all_scored.values()), pool_size=self.pool_size))
            elite = sorted(scored, key=lambda m: m.fitness, reverse=True)[:2]
            nxt = [e.spec for e in elite]
            k = 0
            while len(nxt) < self.population and k < self.population * 6:
                child = self._offspring(scored, f"g{gen + 1}_{k}")
                k += 1
                if child.root.structural_fingerprint() in scorer.cache:
                    continue
                nxt.append(child)
            pop_specs = nxt
        return _finalize_competitor(
            "gp_alpha_miner",
            "genetic_programming",
            self.evaluator,
            scorer.n_evaluated,
            pools,
            run_id,
            score_sealed,
        )


# ---------------------------------------------------------------------------
# 2. Uniform Random Grammar Search
# ---------------------------------------------------------------------------
class RandomDSLSearch:
    """Budget-matched uniform random search over the closed causal DSL."""

    def __init__(
        self,
        evaluator: SealedEvaluator,
        budget: int = 192,
        checkpoint_every: int = 24,
        pool_size: int = 5,
        seed: int = 2026,
    ) -> None:
        self.evaluator = evaluator
        self.budget = budget
        self.checkpoint_every = checkpoint_every
        self.pool_size = pool_size
        self.rng = np.random.default_rng(seed + 7)
        self.sampler = GrammarTreeSampler(max_depth=4)

    def run(self, run_id: str = "random_dsl_search", score_sealed: bool = False) -> CompetitorResult:
        scorer = _SearchZoneScorer(self.evaluator, run_id)
        mined: list[MinedFactor] = []
        pools: list[list[MinedFactor]] = []
        attempts = 0
        while scorer.n_evaluated < self.budget and attempts < self.budget * 8:
            attempts += 1
            sp = FactorSpec(
                factor_id=f"rnd_{attempts}",
                hypothesis_id="rnd",
                description="random",
                root=self.sampler.random_tree(self.rng),
            )
            before = scorer.n_evaluated
            mf = scorer.score(sp, fitness_mode="ic")
            if mf is not None and scorer.n_evaluated > before:
                mined.append(mf)
                if scorer.n_evaluated % self.checkpoint_every == 0:
                    pools.append(build_decorrelated_pool(mined, pool_size=self.pool_size))
        if mined:
            pools.append(build_decorrelated_pool(mined, pool_size=self.pool_size))
        return _finalize_competitor(
            "random_dsl_search",
            "random_search",
            self.evaluator,
            scorer.n_evaluated,
            pools,
            run_id,
            score_sealed,
        )


# ---------------------------------------------------------------------------
# 3. AlphaGen-style Synergistic PPO Reinforcement Learning Miner (Yu et al., KDD 2023)
# ---------------------------------------------------------------------------
class PPOAlphaMiner:
    """AlphaGen-style Actor-Critic PPO over prefix AST grammar optimizing synergistic pool IC."""

    def __init__(
        self,
        evaluator: SealedEvaluator,
        episodes: int = 8,
        batch_size: int = 24,
        ppo_epochs: int = 4,
        clip_eps: float = 0.20,
        pool_size: int = 5,
        max_depth: int = 4,
        seed: int = 2026,
    ) -> None:
        self.evaluator = evaluator
        self.episodes = episodes
        self.batch_size = batch_size
        self.ppo_epochs = ppo_epochs
        self.clip_eps = clip_eps
        self.pool_size = pool_size
        self.max_depth = max_depth
        self.rng = np.random.default_rng(seed + 19)
        self.vocab = ASTVocab(max_seq_len=16)
        v = self.vocab.vocab_size
        # Contextual autoregressive policy conditioned on (position, prev_token) -> next_token
        self.actor_pos = np.zeros((16, v), dtype=float)
        self.actor_bi = np.zeros((v, v), dtype=float)
        self.critic_v = np.zeros(16, dtype=float)

    def _step_probs(self, pos: int, prev_tok: int, valid_mask: np.ndarray) -> np.ndarray:
        logits = self.actor_pos[pos] + self.actor_bi[prev_tok]
        logits = np.where(valid_mask, logits, -1e9)
        shifted = logits - float(np.max(logits))
        p = np.exp(shifted)
        return p / (float(np.sum(p)) + 1e-12)

    def _sample_trajectory(self) -> tuple[list[int], list[np.ndarray], list[float], FactorSpec]:
        seq = [self.vocab.bos_id]
        masks: list[np.ndarray] = []
        old_lps: list[float] = []
        max_body = 10
        for pos in range(1, 15):
            vm = self.vocab.valid_next_token_mask(seq, max_depth=self.max_depth, max_body_tokens=max_body)
            probs = self._step_probs(pos, seq[-1], vm)
            tok = int(self.rng.choice(self.vocab.vocab_size, p=probs))
            seq.append(tok)
            masks.append(vm)
            old_lps.append(float(math.log(max(1e-12, probs[tok]))))
            if tok == self.vocab.eos_id:
                break
        while len(seq) < 16:
            seq.append(self.vocab.pad_id)
        spec = self.vocab.to_factor_spec(seq[:16], factor_id="ppo", hypothesis_id="ppo", family="ppo")
        return seq[:16], masks, old_lps, spec

    @staticmethod
    def _fit_synergistic_pool(
        candidates: list[MinedFactor], fwd_wide: pd.DataFrame, pool_size: int
    ) -> tuple[list[MinedFactor], list[float], float]:
        """AlphaGen linear combination pool optimization via search-zone ridge IC regression."""
        top = build_decorrelated_pool(candidates, pool_size=pool_size, max_corr=0.72)
        if not top:
            return [], [], 0.0
        ic_cols = []
        for mf in top:
            assert mf.signal_df is not None
            ic_cols.append(compute_rank_ic_series(mf.signal_df * mf.sign, fwd_wide))
        panel = pd.concat(ic_cols, axis=1, sort=False).dropna()
        if len(panel) < 20 or len(top) == 1:
            w = [mf.sign / len(top) for mf in top]
            mean_ic = float(panel.mean().mean()) if not panel.empty else abs(top[0].rank_ic)
            return top, w, mean_ic
        mu = panel.mean(axis=0).to_numpy(dtype=float)
        cov = np.cov(panel.to_numpy(dtype=float), rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])
        reg = 0.35 * (float(np.mean(np.diag(cov))) + 1e-6) * np.eye(len(top))
        raw_w = np.clip(np.linalg.solve(cov + reg, np.maximum(0.001, mu)), 0.05, None)
        raw_w /= float(np.sum(raw_w)) + 1e-12
        comb_df: pd.DataFrame | None = None
        for mf, wt in zip(top, raw_w, strict=True):
            assert mf.signal_df is not None
            part = mf.signal_df * (mf.sign * float(wt))
            comb_df = part if comb_df is None else (comb_df + part)
        assert comb_df is not None
        comb_ic = float(compute_rank_ic_series(comb_df, fwd_wide).mean())
        signed_w = [mf.sign * float(wt) for mf, wt in zip(top, raw_w, strict=True)]
        return top, signed_w, comb_ic

    def run(self, run_id: str = "ppo_alpha_miner", score_sealed: bool = False) -> CompetitorResult:
        scorer = _SearchZoneScorer(self.evaluator, run_id, keep_df=True)
        discovered: dict[str, MinedFactor] = {}
        pools: list[tuple[list[MinedFactor], list[float] | None]] = []
        prev_pool_ic = 0.0

        for ep in range(self.episodes):
            trajs: list[tuple[list[int], list[np.ndarray], list[float], float]] = []
            tries = 0
            while len(trajs) < self.batch_size and tries < self.batch_size * 5:
                tries += 1
                seq, masks, old_lps, sp = self._sample_trajectory()
                sp.factor_id = f"ppo_e{ep}_{tries}"
                fp = sp.root.structural_fingerprint()
                if fp in scorer.cache:
                    continue
                mf = scorer.score(sp, fitness_mode="ic")
                if mf is None:
                    continue
                discovered[fp] = mf
                # Synergistic reward: standalone |IC| + marginal gain to current synergistic pool IC
                trial_pool = list(discovered.values())
                _, _, new_pool_ic = self._fit_synergistic_pool(
                    trial_pool, scorer.search_fwd, self.pool_size
                )
                marginal = max(0.0, new_pool_ic - prev_pool_ic)
                rew = abs(mf.rank_ic) + 2.5 * marginal
                trajs.append((seq, masks, old_lps, float(rew)))

            if discovered:
                pool, weights, prev_pool_ic = self._fit_synergistic_pool(
                    list(discovered.values()), scorer.search_fwd, self.pool_size
                )
                pools.append((pool, weights))

            if len(trajs) < 2:
                continue
            rews = np.asarray([t[3] for t in trajs], dtype=float)
            advs = (rews - float(rews.mean())) / (float(rews.std()) + 1e-6)
            lr = 0.12
            for _ in range(self.ppo_epochs):
                for (seq, masks, old_lps, rew), adv in zip(trajs, advs, strict=True):
                    for step_i, (vm, old_lp) in enumerate(zip(masks, old_lps, strict=True)):
                        pos = step_i + 1
                        prev_tok = seq[pos - 1]
                        act = seq[pos]
                        probs = self._step_probs(pos, prev_tok, vm)
                        new_lp = float(math.log(max(1e-12, probs[act])))
                        ratio = math.exp(max(-4.0, min(4.0, new_lp - old_lp)))
                        clipped = max(1.0 - self.clip_eps, min(1.0 + self.clip_eps, ratio))
                        eff_w = ratio if (adv >= 0 and ratio <= clipped) or (adv < 0 and ratio >= clipped) else 0.0
                        grad = -probs
                        grad[act] += 1.0
                        # Policy gradient + small entropy bonus
                        ent_grad = -(np.log(np.maximum(probs, 1e-12)) + 1.0) * probs * vm
                        ent_grad -= probs * float(np.sum(ent_grad))
                        step_delta = (eff_w * float(adv) * grad + 0.01 * ent_grad) / len(trajs)
                        self.actor_pos[pos] += lr * step_delta
                        self.actor_bi[prev_tok] += lr * step_delta
                        self.critic_v[pos] += 0.05 * (rew - self.critic_v[pos]) / len(trajs)

        return _finalize_competitor(
            "ppo_alpha_miner",
            "reinforcement_learning_ppo",
            self.evaluator,
            scorer.n_evaluated,
            pools,
            run_id,
            score_sealed,
        )


# ---------------------------------------------------------------------------
# 4. Monte Carlo Tree Search over Causal Grammar (Alpha-MCTS / RiskMiner style)
# ---------------------------------------------------------------------------
@dataclass
class _MCTSNode:
    prefix: tuple[int, ...]
    visits: int = 0
    value_sum: float = 0.0
    children: dict[int, _MCTSNode] = field(default_factory=dict)
    valid_actions: list[int] | None = None

    @property
    def q_value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0


class MCTSAlphaMiner:
    """UCT Monte Carlo Tree Search over prefix causal AST productions."""

    def __init__(
        self,
        evaluator: SealedEvaluator,
        budget: int = 192,
        c_puct: float = 1.25,
        checkpoint_every: int = 24,
        pool_size: int = 5,
        max_depth: int = 4,
        seed: int = 2026,
    ) -> None:
        self.evaluator = evaluator
        self.budget = budget
        self.c_puct = c_puct
        self.checkpoint_every = checkpoint_every
        self.pool_size = pool_size
        self.max_depth = max_depth
        self.rng = np.random.default_rng(seed + 41)
        self.vocab = ASTVocab(max_seq_len=16)

    def _valid_actions(self, prefix: tuple[int, ...]) -> list[int]:
        vm = self.vocab.valid_next_token_mask(list(prefix), max_depth=self.max_depth, max_body_tokens=10)
        return [int(i) for i in np.where(vm)[0]]

    def _rollout_from(self, prefix: list[int]) -> list[int]:
        seq = list(prefix)
        while len(seq) < 15 and seq[-1] != self.vocab.eos_id:
            vm = self.vocab.valid_next_token_mask(seq, max_depth=self.max_depth, max_body_tokens=10)
            ids = np.where(vm)[0]
            if len(ids) == 0:
                break
            seq.append(int(self.rng.choice(ids)))
        while len(seq) < 16:
            seq.append(self.vocab.pad_id)
        return seq[:16]

    def run(self, run_id: str = "mcts_alpha_miner", score_sealed: bool = False) -> CompetitorResult:
        scorer = _SearchZoneScorer(self.evaluator, run_id)
        root = _MCTSNode(prefix=(self.vocab.bos_id,))
        mined: list[MinedFactor] = []
        pools: list[list[MinedFactor]] = []
        iters = 0

        while scorer.n_evaluated < self.budget and iters < self.budget * 8:
            iters += 1
            node = root
            path = [node]

            # Selection + Progressive Expansion
            while len(node.prefix) < 15 and node.prefix[-1] != self.vocab.eos_id:
                if node.valid_actions is None:
                    node.valid_actions = self._valid_actions(node.prefix)
                if not node.valid_actions:
                    break
                # Progressive widening: expand up to ceil(2 * N^0.5) children
                max_ch = max(2, int(math.ceil(2.2 * math.sqrt(node.visits + 1))))
                unexpanded = [a for a in node.valid_actions if a not in node.children]
                if unexpanded and len(node.children) < max_ch:
                    act = int(self.rng.choice(unexpanded))
                    child = _MCTSNode(prefix=node.prefix + (act,))
                    node.children[act] = child
                    node = child
                    path.append(node)
                    break
                if not node.children:
                    break
                log_parent = math.sqrt(max(1, node.visits))
                best_act, best_child = max(
                    node.children.items(),
                    key=lambda kv: kv[1].q_value + self.c_puct * 0.12 * log_parent / (1 + kv[1].visits),
                )
                _ = best_act
                node = best_child
                path.append(node)

            full_seq = self._rollout_from(list(node.prefix))
            sp = self.vocab.to_factor_spec(
                full_seq, factor_id=f"mcts_{iters}", hypothesis_id="mcts", family="mcts"
            )
            before = scorer.n_evaluated
            mf = scorer.score(sp, fitness_mode="ic")
            reward = mf.fitness if mf is not None else 0.0
            for nd in path:
                nd.visits += 1
                nd.value_sum += reward
            if mf is not None and scorer.n_evaluated > before:
                mined.append(mf)
                if scorer.n_evaluated % self.checkpoint_every == 0:
                    pools.append(build_decorrelated_pool(mined, pool_size=self.pool_size))

        if mined:
            pools.append(build_decorrelated_pool(mined, pool_size=self.pool_size))
        return _finalize_competitor(
            "mcts_alpha_miner",
            "monte_carlo_tree_search",
            self.evaluator,
            scorer.n_evaluated,
            pools,
            run_id,
            score_sealed,
        )


# ---------------------------------------------------------------------------
# 5. Quality-Diversity MAP-Elites Evolutionary Miner
# ---------------------------------------------------------------------------
class MAPElitesAlphaMiner:
    """Behavioral Quality-Diversity MAP-Elites miner over (information_niche x turnover_bucket)."""

    TURN_EDGES = (0.04, 0.10, 0.20)

    def __init__(
        self,
        evaluator: SealedEvaluator,
        budget: int = 192,
        init_random: int = 36,
        checkpoint_every: int = 24,
        pool_size: int = 5,
        max_depth: int = 4,
        seed: int = 2026,
    ) -> None:
        self.evaluator = evaluator
        self.budget = budget
        self.init_random = init_random
        self.checkpoint_every = checkpoint_every
        self.pool_size = pool_size
        self.max_depth = max_depth
        self.rng = np.random.default_rng(seed + 67)
        self.sampler = GrammarTreeSampler(max_depth=max_depth)

    def _cell(self, mf: MinedFactor) -> tuple[str, int]:
        b = len(self.TURN_EDGES)
        for i, e in enumerate(self.TURN_EDGES):
            if mf.turnover < e:
                b = i
                break
        return (_dominant_niche(mf.spec.root), b)

    def _mutate(self, root: ASTNode) -> ASTNode:
        paths = _subtree_paths(root)
        path = paths[int(self.rng.integers(0, len(paths)))]
        if self.rng.random() < 0.65:
            sub_d = max(1, self.max_depth - len(path))
            cand = _replace_subtree(root, path, self.sampler.random_tree(self.rng, max_depth=sub_d))
        else:
            cand = root.model_copy(deep=True)
            target = _get_subtree(cand, path)
            if target.node_type == "terminal":
                target.name = str(self.rng.choice(TERMINAL_NAMES))
            elif target.window is not None:
                target.window = int(self.rng.choice(CANONICAL_WINDOWS))
        return cand if cand.depth() <= self.max_depth else self.sampler.random_tree(self.rng)

    def run(
        self, run_id: str = "map_elites_alpha_miner", score_sealed: bool = False
    ) -> CompetitorResult:
        scorer = _SearchZoneScorer(self.evaluator, run_id)
        archive: dict[tuple[str, int], MinedFactor] = {}
        pools: list[list[MinedFactor]] = []
        attempts = 0

        while scorer.n_evaluated < self.budget and attempts < self.budget * 8:
            attempts += 1
            if scorer.n_evaluated < self.init_random or not archive:
                tree = self.sampler.random_tree(self.rng)
            else:
                elites = list(archive.values())
                parent = elites[int(self.rng.integers(0, len(elites)))]
                tree = self._mutate(parent.spec.root)
            sp = FactorSpec(
                factor_id=f"qd_{attempts}", hypothesis_id="qd", description="map_elites", root=tree
            )
            before = scorer.n_evaluated
            mf = scorer.score(sp, fitness_mode="friction_ic")
            if mf is not None and scorer.n_evaluated > before:
                cell = self._cell(mf)
                cur = archive.get(cell)
                if cur is None or mf.fitness > cur.fitness:
                    archive[cell] = mf
                if scorer.n_evaluated % self.checkpoint_every == 0:
                    pools.append(
                        build_decorrelated_pool(list(archive.values()), pool_size=self.pool_size)
                    )

        if archive:
            pools.append(build_decorrelated_pool(list(archive.values()), pool_size=self.pool_size))
        return _finalize_competitor(
            "map_elites_alpha_miner",
            "quality_diversity_evolution",
            self.evaluator,
            scorer.n_evaluated,
            pools,
            run_id,
            score_sealed,
        )


# ---------------------------------------------------------------------------
# 6. Alpha-GPT / R&D-Agent-Quant Hypothesis-Guided Iterative Refinement Miner
# ---------------------------------------------------------------------------
class AlphaGPTIterativeMiner:
    """Idea-driven hypothesis-to-formula iterative refinement agent (Alpha-GPT / R&D-Agent style)."""

    def __init__(
        self,
        evaluator: SealedEvaluator,
        rounds: int = 8,
        ideas_per_round: int = 12,
        pool_size: int = 5,
        seed: int = 2026,
    ) -> None:
        self.evaluator = evaluator
        self.rounds = rounds
        self.ideas_per_round = ideas_per_round
        self.pool_size = pool_size
        self.seed = seed
        self.rng = np.random.default_rng(seed + 97)
        self.llm = RuleGuidedSmallLMBackend()
        self.sampler = GrammarTreeSampler(max_depth=4)

    def _refine_ast(self, base: ASTNode) -> ASTNode:
        """Local operator/window/combination refinement mimicking verbal self-refine edits."""
        u = self.rng.random()
        if u < 0.35 and base.depth() <= 3:
            w = int(self.rng.choice((3, 5, 10)))
            return ASTNode(node_type="op", name="ts_mean", window=w, args=[base.model_copy(deep=True)])
        if u < 0.70 and base.depth() <= 3:
            aux = ASTNode(node_type="terminal", name=str(self.rng.choice(TERMINAL_NAMES)))
            op = str(self.rng.choice(("add", "sub", "mul", "safe_div")))
            return ASTNode(node_type="op", name=op, args=[base.model_copy(deep=True), aux])
        paths = _subtree_paths(base)
        p = paths[int(self.rng.integers(0, len(paths)))]
        sub_d = max(1, 4 - len(p))
        return _replace_subtree(base, p, self.sampler.random_tree(self.rng, max_depth=sub_d))

    def run(
        self, run_id: str = "alphagpt_iterative_miner", score_sealed: bool = False
    ) -> CompetitorResult:
        scorer = _SearchZoneScorer(self.evaluator, run_id)
        budget = BudgetTracker(
            BudgetConfig(max_tokens=200_000, max_wall_seconds=300.0, max_gpu_hours=0.5), run_id
        )
        fam_scores = {f: 0.03 for f in MECHANISM_FAMILIES}
        elites: list[MinedFactor] = []
        pools: list[tuple[list[MinedFactor], list[float] | None]] = []

        for r_idx in range(self.rounds):
            weights_f = np.asarray([max(0.05, fam_scores[f]) for f in MECHANISM_FAMILIES])
            weights_f /= weights_f.sum()
            for i in range(self.ideas_per_round):
                fam = MECHANISM_FAMILIES[int(self.rng.choice(len(MECHANISM_FAMILIES), p=weights_f))]
                if elites and i % 2 == 1:
                    parent = elites[int(self.rng.integers(0, min(6, len(elites))))]
                    cand = FactorSpec(
                        factor_id=f"agpt_r{r_idx}_{i}",
                        hypothesis_id=f"h_{r_idx}_{i}",
                        description=f"refined {fam}",
                        root=self._refine_ast(parent.spec.root),
                    )
                else:
                    ctx = f'{{"family":"{fam}","hypothesis_id":"h_{r_idx}_{i}","factor_id":"agpt_r{r_idx}_{i}","preferred_windows":[3,5,10,15,20],"max_depth":4}}'
                    cand_m, _ = self.llm.generate_structured(
                        system_prompt="Emit causal FactorSpec.",
                        user_prompt=f"Implement {fam}\nCONTEXT_JSON:{ctx}",
                        schema_cls=FactorSpec,
                        budget=budget,
                        temperature=0.5,
                        seed=self.seed + r_idx * 313 + i * 17,
                    )
                    assert isinstance(cand_m, FactorSpec)
                    cand = cand_m
                    if cand.root.structural_fingerprint() in scorer.cache:
                        cand = FactorSpec(
                            factor_id=cand.factor_id,
                            hypothesis_id=cand.hypothesis_id,
                            description=cand.description,
                            root=self._refine_ast(cand.root),
                        )
                mf = scorer.score(cand, fitness_mode="friction_ic")
                if mf is not None:
                    elites.append(mf)
                    fam_scores[fam] = 0.8 * fam_scores[fam] + 0.2 * abs(mf.rank_ic)

            elites.sort(key=lambda m: m.fitness, reverse=True)
            pool = build_decorrelated_pool(elites, pool_size=self.pool_size, max_corr=0.68)
            if pool:
                ic_w = [mf.sign * max(0.02, abs(mf.rank_ic)) for mf in pool]
                pools.append((pool, ic_w))

        return _finalize_competitor(
            "alphagpt_iterative_miner",
            "agentic_hypothesis_refinement",
            self.evaluator,
            scorer.n_evaluated,
            pools,
            run_id,
            score_sealed,
        )


# ---------------------------------------------------------------------------
# 7. Handcrafted Institutional Alpha101-Style Formula Library
# ---------------------------------------------------------------------------
def _t(name: str) -> ASTNode:
    return ASTNode(node_type="terminal", name=name)


def _op(name: str, *args: ASTNode, window: int | None = None) -> ASTNode:
    return ASTNode(node_type="op", name=name, window=window, args=list(args))


def alpha101_style_library() -> list[tuple[str, ASTNode]]:
    """WorldQuant-101-inspired formulas expressible in the closed causal DSL."""
    return [
        ("a101_body_to_range", _op("safe_div", _op("sub", _t("close"), _t("open")), _op("sub", _t("high"), _t("low")))),
        ("a033_open_close", _op("cs_rank", _t("close_to_open"))),
        ("a004_low_rank_rev", _op("neg", _op("ts_rank", _op("cs_rank", _t("low")), window=10))),
        (
            "a012_vol_price_rev",
            _op(
                "neg",
                _op(
                    "mul",
                    _op("tanh_squash", _op("ts_delta", _t("volume"), window=3)),
                    _op("ts_delta", _t("close"), window=3),
                ),
            ),
        ),
        ("a009_short_rev", _op("neg", _op("ts_return", _t("close"), window=5))),
        ("lowvol_anomaly", _op("neg", _op("ts_std", _t("return_1d"), window=20))),
        ("a019_long_mom", _op("ts_return", _t("close"), window=20)),
        ("a041_vwap_proxy_rev", _op("neg", _op("ts_zscore", _t("close"), window=10))),
    ]


class Alpha101StyleLibrary:
    """Fixed handcrafted library; only formula signs are set from search-zone IC (no structural search)."""

    def __init__(self, evaluator: SealedEvaluator) -> None:
        self.evaluator = evaluator

    def run(
        self, run_id: str = "alpha101_style_library", score_sealed: bool = False
    ) -> CompetitorResult:
        scorer = _SearchZoneScorer(self.evaluator, run_id)
        mined: list[MinedFactor] = []
        for fid, root in alpha101_style_library():
            mf = scorer.score(
                FactorSpec(factor_id=fid, hypothesis_id="a101", description=fid, root=root), "ic"
            )
            if mf is not None:
                mined.append(mf)
        return _finalize_competitor(
            "alpha101_style_library",
            "handcrafted_library",
            self.evaluator,
            scorer.n_evaluated,
            [mined],
            run_id,
            score_sealed,
        )
