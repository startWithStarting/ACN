"""ACN Benchmark module.

Provides benchmarking framework:
- BenchmarkRunner: orchestrates benchmark execution
- calculate_metrics: compute common metrics
"""

from .runner import BenchmarkRunner, run_benchmark, Scenario, AlgorithmResult
from .metrics import (
    calculate_metrics,
    avg_episode_reward,
    detection_rate,
    red_score_rate,
    communication_utility,
    convergence_speed,
)

__all__ = [
    "BenchmarkRunner",
    "run_benchmark",
    "Scenario",
    "AlgorithmResult",
    "calculate_metrics",
    "avg_episode_reward",
    "detection_rate",
    "red_score_rate",
    "communication_utility",
    "convergence_speed",
]