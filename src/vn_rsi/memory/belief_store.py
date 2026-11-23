"""Persistent Epistemic Belief Memory with confidence decay, run-ID provenance, and contradiction handling."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EpistemicBelief:
    belief_id: str
    mechanism_family: str
    statement: str
    expected_sign: int
    confidence: float
    mean_rank_ic: float
    supporting_run_ids: list[str] = field(default_factory=list)
    refuting_run_ids: list[str] = field(default_factory=list)
    contradictions_count: int = 0
    last_updated_round: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EpistemicBeliefMemory:
    """Compact mechanism memory tailored for <=8B short-context retrieval."""

    def __init__(self, decay_rate: float = 0.90, contradiction_penalty: float = 0.35) -> None:
        self.decay_rate = float(decay_rate)
        self.contradiction_penalty = float(contradiction_penalty)
        self.beliefs: dict[str, EpistemicBelief] = {}
        self.seen_ast_fingerprints: set[str] = set()

    def step_decay(self) -> None:
        for b in self.beliefs.values():
            b.confidence = round(max(0.05, b.confidence * self.decay_rate), 4)

    def record_outcome(
        self,
        mechanism_family: str,
        statement: str,
        expected_sign: int,
        observed_rank_ic: float,
        accepted: bool,
        run_id: str,
        round_idx: int,
        ast_fingerprint: str | None = None,
    ) -> EpistemicBelief:
        if ast_fingerprint:
            self.seen_ast_fingerprints.add(ast_fingerprint)

        key = mechanism_family
        sign_match = (expected_sign * observed_rank_ic) > 0.0

        if key not in self.beliefs:
            init_conf = 0.65 if (accepted and sign_match) else 0.25
            belief = EpistemicBelief(
                belief_id=f"belief_{key}",
                mechanism_family=key,
                statement=statement,
                expected_sign=expected_sign,
                confidence=init_conf,
                mean_rank_ic=round(observed_rank_ic, 5),
                supporting_run_ids=[run_id] if accepted else [],
                refuting_run_ids=[] if accepted else [run_id],
                contradictions_count=0 if sign_match else 1,
                last_updated_round=round_idx,
            )
            self.beliefs[key] = belief
            return belief

        b = self.beliefs[key]
        n_obs = len(b.supporting_run_ids) + len(b.refuting_run_ids)
        b.mean_rank_ic = round((b.mean_rank_ic * n_obs + observed_rank_ic) / max(1, n_obs + 1), 5)
        b.last_updated_round = round_idx

        if accepted and sign_match:
            b.supporting_run_ids.append(run_id)
            b.confidence = round(min(0.98, b.confidence + 0.14), 4)
        else:
            b.refuting_run_ids.append(run_id)
            if not sign_match:
                b.contradictions_count += 1
                b.confidence = round(max(0.05, b.confidence * (1.0 - self.contradiction_penalty)), 4)
            else:
                b.confidence = round(max(0.05, b.confidence - 0.08), 4)
        return b

    def retrieve_compact_context(self, top_k: int = 5) -> list[dict[str, Any]]:
        """Return short structured summary (<400 tokens) for <=8B prompts."""
        ranked = sorted(self.beliefs.values(), key=lambda x: x.confidence, reverse=True)[:top_k]
        return [
            {
                "family": b.mechanism_family,
                "conf": b.confidence,
                "mean_ic": b.mean_rank_ic,
                "support_n": len(b.supporting_run_ids),
                "contradictions": b.contradictions_count,
            }
            for b in ranked
        ]

    def family_ic_summary(self) -> dict[str, float]:
        return {k: b.mean_rank_ic for k, b in self.beliefs.items()}

    def snapshot_list(self) -> list[dict[str, Any]]:
        return [b.to_dict() for b in sorted(self.beliefs.values(), key=lambda x: x.belief_id)]
