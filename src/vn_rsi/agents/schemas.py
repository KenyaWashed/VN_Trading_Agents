"""Schemas for falsifiable hypotheses and leakage-auditor verdicts (Phase 3)."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class FalsifiableHypothesis(BaseModel):
    hypothesis_id: str
    mechanism_family: str
    statement: str = Field(min_length=15)
    economic_mechanism: str = Field(min_length=15)
    expected_sign: int = Field(description="+1 or -1 expected correlation sign with forward excess return")
    refutation_criteria: str = Field(min_length=10)
    suggested_terminals: list[str] = Field(default_factory=list)
    suggested_operators: list[str] = Field(default_factory=list)

    @field_validator("expected_sign")
    @classmethod
    def _check_sign(cls, v: int) -> int:
        if v not in (-1, 1):
            raise ValueError(f"expected_sign must be -1 or +1, got {v}")
        return v


class AuditVerdict(BaseModel):
    passed: bool
    ast_fingerprint: str
    is_novel: bool
    shift_test_passed: bool
    reason: str
