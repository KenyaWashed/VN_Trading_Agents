"""Hard budget enforcement for tokens, wall-clock seconds, and GPU-hours.

Hard constraint (Section 1.5): Budgets everywhere — hard stops, NOT warnings.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


class BudgetExceededError(RuntimeError):
    """Raised immediately when token, wall-clock, or GPU-hour budget is exceeded."""


@dataclass(frozen=True)
class BudgetConfig:
    max_tokens: int = 60_000
    max_wall_seconds: float = 180.0
    max_gpu_hours: float = 0.25
    configured_vram_gb: float = 0.0  # 0.0 denotes CPU fallback mode


@dataclass
class BudgetTracker:
    config: BudgetConfig
    scope_name: str = "inner_loop"
    tokens_used: int = 0
    calls_made: int = 0
    gpu_seconds_accumulated: float = 0.0
    _start_monotonic: float = field(default_factory=time.monotonic)

    @property
    def elapsed_wall_seconds(self) -> float:
        return max(0.0, time.monotonic() - self._start_monotonic)

    @property
    def gpu_hours_used(self) -> float:
        return self.gpu_seconds_accumulated / 3600.0

    def check(self) -> None:
        """Verify all budgets are within limits; raise BudgetExceededError otherwise."""
        if self.tokens_used > self.config.max_tokens:
            raise BudgetExceededError(
                f"[{self.scope_name}] Token budget exceeded: {self.tokens_used} > {self.config.max_tokens}"
            )
        wall = self.elapsed_wall_seconds
        if wall > self.config.max_wall_seconds:
            raise BudgetExceededError(
                f"[{self.scope_name}] Wall-clock budget exceeded: {wall:.2f}s > {self.config.max_wall_seconds:.2f}s"
            )
        if self.gpu_hours_used > self.config.max_gpu_hours:
            raise BudgetExceededError(
                f"[{self.scope_name}] GPU-hour budget exceeded: {self.gpu_hours_used:.5f}h > {self.config.max_gpu_hours:.5f}h"
            )

    def record_usage(self, prompt_tokens: int, completion_tokens: int, call_seconds: float = 0.0) -> None:
        """Record token and compute consumption, then enforce hard stop."""
        if prompt_tokens < 0 or completion_tokens < 0:
            raise ValueError("Token counts must be non-negative")
        self.tokens_used += int(prompt_tokens + completion_tokens)
        self.calls_made += 1
        if self.config.configured_vram_gb > 0.0:
            self.gpu_seconds_accumulated += max(0.0, call_seconds)
        self.check()

    def add_gpu_seconds(self, seconds: float) -> None:
        """Explicitly charge GPU seconds (e.g., for GPU training or simulation tests)."""
        self.gpu_seconds_accumulated += max(0.0, seconds)
        self.check()

    def snapshot(self) -> dict[str, float | int | str]:
        return {
            "scope": self.scope_name,
            "tokens_used": self.tokens_used,
            "max_tokens": self.config.max_tokens,
            "calls_made": self.calls_made,
            "wall_seconds": round(self.elapsed_wall_seconds, 4),
            "max_wall_seconds": self.config.max_wall_seconds,
            "gpu_hours_used": round(self.gpu_hours_used, 6),
            "max_gpu_hours": self.config.max_gpu_hours,
        }
