"""
Green Love — PyTorch training cost & carbon analysis.

Uses adaptive multi-sample benchmarking to estimate training time,
cost, and CO₂ emissions. Model and optimizer states are preserved.
"""

from .estimator import GreenLoveEstimator, BenchmarkResults
from .data_utils import sample_dataset, sample_dataloader
from .benchmarks import CrusoeGPUEstimate, GPUSpec, BENCHMARK_TASKS
from .co2_equivalences import (
    CO2Equivalence,
    format_time,
    format_cost,
    format_co2,
)

# Backward compat alias
CrusoeEstimator = GreenLoveEstimator

__all__ = [
    "GreenLoveEstimator",
    "CrusoeEstimator",
    "BenchmarkResults",
    "CrusoeGPUEstimate",
    "CO2Equivalence",
    "BENCHMARK_TASKS",
    "sample_dataset",
    "sample_dataloader",
    "format_time",
    "format_cost",
    "format_co2",
]

__version__ = "0.1.0"
