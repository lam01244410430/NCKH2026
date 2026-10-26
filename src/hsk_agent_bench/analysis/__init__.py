"""Prompt-paired analysis and reproducible benchmark figures."""

from .aggregate import aggregate_group_means, join_analysis_tables
from .hypothesis_tests import (
    ModelReport, TestResult, fit_mixed_effects, holm_adjust,
    paired_origin_test, paired_wilcoxon,
)

__all__ = [
    "aggregate_group_means", "join_analysis_tables", "TestResult", "ModelReport",
    "paired_origin_test", "paired_wilcoxon", "holm_adjust", "fit_mixed_effects",
]
