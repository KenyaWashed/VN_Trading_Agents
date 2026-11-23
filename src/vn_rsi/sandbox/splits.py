"""Purged and Embargoed Walk-Forward Cross-Validation Splitter (Lopez de Prado, 2018)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: int
    train_dates: pd.DatetimeIndex
    eval_dates: pd.DatetimeIndex
    purge_days: int
    embargo_days: int


class PurgedEmbargoWalkForwardSplitter:
    """Expanding-window walk-forward splitter with label-overlap purging and post-fold embargo."""

    def __init__(self, n_folds: int = 4, purge_days: int = 6, embargo_days: int = 5) -> None:
        if n_folds < 2:
            raise ValueError("n_folds must be >= 2")
        if purge_days < 1:
            raise ValueError("purge_days must be >= 1")
        self.n_folds = n_folds
        self.purge_days = purge_days
        self.embargo_days = embargo_days

    def split(self, unique_dates: pd.DatetimeIndex | np.ndarray) -> list[WalkForwardFold]:
        dates = pd.DatetimeIndex(np.sort(pd.unique(pd.to_datetime(unique_dates))))
        n_total = len(dates)
        chunk = n_total // (self.n_folds + 1)
        if chunk <= self.purge_days + 5:
            raise ValueError(
                f"Insufficient trading dates ({n_total}) for {self.n_folds} purged folds (purge={self.purge_days})."
            )

        folds: list[WalkForwardFold] = []
        for k in range(self.n_folds):
            train_end_raw = chunk * (k + 1)
            train_end_purged = max(5, train_end_raw - self.purge_days)
            eval_start = train_end_raw + self.embargo_days
            eval_end = chunk * (k + 2) if k < self.n_folds - 1 else n_total
            if eval_start >= eval_end:
                continue
            folds.append(
                WalkForwardFold(
                    fold_id=k,
                    train_dates=dates[:train_end_purged],
                    eval_dates=dates[eval_start:eval_end],
                    purge_days=self.purge_days,
                    embargo_days=self.embargo_days,
                )
            )
        return folds
