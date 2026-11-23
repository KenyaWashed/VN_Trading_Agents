"""Mutable Surface Guard restricting outer-loop agent rewrites to an explicit whitelist (Phase 4)."""

from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel, Field


class ImmutableSurfaceViolationError(PermissionError):
    """Raised if an outer-loop proposer attempts to edit outside the declared mutable surface."""


class GenomePatchProposal(BaseModel):
    rationale: str = Field(min_length=10)
    updates: dict[str, Any]


class MutableSurfaceGuard:
    """Enforces that outer-loop rewrites touch ONLY whitelisted agent genome keys."""

    FORBIDDEN_IMMUTABLE_KEYS = {
        "evaluator",
        "splits",
        "zones",
        "sealed_test",
        "market_rules",
        "transaction_costs",
        "ledger_path",
        "allowed_operators",
        "allowed_terminals",
        "min_holding_bars_eod",
    }

    def __init__(self, allowed_keys: list[str]) -> None:
        self.allowed_keys = set(allowed_keys)
        overlap = self.allowed_keys.intersection(self.FORBIDDEN_IMMUTABLE_KEYS)
        if overlap:
            raise ValueError(f"Mutable surface config illegally includes immutable keys: {overlap}")

    def apply_patch(self, parent_genome: dict[str, Any], proposal: GenomePatchProposal) -> dict[str, Any]:
        for key in proposal.updates:
            if key in self.FORBIDDEN_IMMUTABLE_KEYS or key not in self.allowed_keys:
                raise ImmutableSurfaceViolationError(
                    f"Outer-loop attempted unauthorized edit to '{key}'. "
                    f"Declared mutable surface: {sorted(self.allowed_keys)}"
                )

        child = copy.deepcopy(parent_genome)
        for key, val in proposal.updates.items():
            if key == "implementer_max_ast_depth":
                val = int(max(1, min(4, int(val))))
            elif key == "best_of_n":
                val = int(max(1, min(6, int(val))))
            elif key == "exploration_temperature":
                val = float(max(0.05, min(1.0, float(val))))
            elif key == "preferred_windows":
                val = [int(max(2, min(60, int(w)))) for w in val][:5] or [5, 15]
            child[key] = val
        return child
