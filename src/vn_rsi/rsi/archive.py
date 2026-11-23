"""Darwin Godel Machine (DGM) + ShinkaEvolve inspired Open-Ended Agent Variant Archive."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vn_rsi.dsl.ast_spec import FactorSpec
from vn_rsi.utils.repro import compute_config_hash


@dataclass
class ArchiveEntry:
    variant_id: str
    parent_id: str | None
    generation: int
    genome: dict[str, Any]
    genome_hash: str
    val_composite_mean: float
    val_composite_std: float
    val_sharpe_mean: float
    val_rank_ic_mean: float
    seed_scores: list[float]
    representative_specs: list[FactorSpec] = field(default_factory=list)
    representative_weights: list[float] = field(default_factory=list)
    family_ic_summary: dict[str, float] = field(default_factory=dict)


class DGMArchive:
    """Maintains diverse high-fitness agent genomes rather than collapsing to a single greedy point."""

    def __init__(self, max_variants: int = 12, fitness_weight: float = 0.75, novelty_weight: float = 0.25) -> None:
        self.max_variants = max_variants
        self.fitness_weight = fitness_weight
        self.novelty_weight = novelty_weight
        self.entries: list[ArchiveEntry] = []

    def add(self, entry: ArchiveEntry) -> bool:
        if any(e.genome_hash == entry.genome_hash for e in self.entries):
            return False
        self.entries.append(entry)
        if len(self.entries) > self.max_variants:
            self.entries.sort(key=lambda x: x.val_composite_mean, reverse=True)
            self.entries = self.entries[: self.max_variants]
        return True

    def best_entry(self) -> ArchiveEntry:
        if not self.entries:
            raise RuntimeError("Archive is empty")
        return max(self.entries, key=lambda x: x.val_composite_mean)

    def _genome_distance(self, g1: dict[str, Any], g2: dict[str, Any]) -> float:
        s1 = json.dumps(g1, sort_keys=True)
        s2 = json.dumps(g2, sort_keys=True)
        if s1 == s2:
            return 0.0
        w1 = g1.get("planner_focus_weights", {})
        w2 = g2.get("planner_focus_weights", {})
        keys = sorted(set(w1.keys()) | set(w2.keys()))
        l1 = sum(abs(float(w1.get(k, 0.25)) - float(w2.get(k, 0.25))) for k in keys)
        win_diff = len(set(g1.get("preferred_windows", [])) ^ set(g2.get("preferred_windows", []))) * 0.15
        depth_diff = abs(int(g1.get("implementer_max_ast_depth", 3)) - int(g2.get("implementer_max_ast_depth", 3))) * 0.2
        return float(min(2.0, l1 + win_diff + depth_diff + 0.1))

    def sample_parent(self, rng: np.random.Generator) -> ArchiveEntry:
        if len(self.entries) == 1:
            return self.entries[0]
        fitness = np.array([e.val_composite_mean for e in self.entries], dtype=float)
        f_norm = (fitness - fitness.min()) / (fitness.max() - fitness.min() + 1e-8)

        novelties = np.zeros(len(self.entries), dtype=float)
        for i, ei in enumerate(self.entries):
            dists = [self._genome_distance(ei.genome, ej.genome) for j, ej in enumerate(self.entries) if i != j]
            novelties[i] = float(np.mean(dists)) if dists else 1.0
        n_norm = (novelties - novelties.min()) / (novelties.max() - novelties.min() + 1e-8)

        combined = self.fitness_weight * f_norm + self.novelty_weight * n_norm + 0.05
        probs = combined / combined.sum()
        idx = int(rng.choice(len(self.entries), p=probs))
        return self.entries[idx]

    @staticmethod
    def hash_genome(genome: dict[str, Any]) -> str:
        return compute_config_hash(genome)[:16]
