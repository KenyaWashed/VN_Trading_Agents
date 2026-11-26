"""10-Year Veteran CIO Cognitive Reasoning, Preference Dataset Builder, and Expert Benchmark Suite."""

from vn_rsi.veteran.expert_benchmark import (
    ExpertBenchmarkReport,
    VeteranScenario,
    build_30_plus_veteran_scenarios,
    compute_brier_and_ece,
    evaluate_veteran_benchmark,
)
from vn_rsi.veteran.preference_builder import (
    PreferenceDatasetReport,
    build_veteran_sft_and_dpo_datasets,
    train_veteran_two_stage_qlora_dpo,
)
from vn_rsi.veteran.reasoning_engine import (
    MarketContextSnapshot,
    VeteranReasoningEngine,
    VeteranReasoningResponse,
)

__all__ = [
    "ExpertBenchmarkReport",
    "MarketContextSnapshot",
    "PreferenceDatasetReport",
    "VeteranReasoningEngine",
    "VeteranReasoningResponse",
    "VeteranScenario",
    "build_30_plus_veteran_scenarios",
    "build_veteran_sft_and_dpo_datasets",
    "compute_brier_and_ece",
    "evaluate_veteran_benchmark",
    "train_veteran_two_stage_qlora_dpo",
]
