"""Benchmark runner for ACN experiments.

This module provides a framework for running and comparing different algorithms:
- BenchmarkRunner: orchestrates benchmark execution
- Scenario: defines a single test scenario
- Algorithm: defines an algorithm to run

Usage:
    runner = BenchmarkRunner("config/benchmark.yaml")
    results = runner.run()
    runner.save_comparison("benchmark_results/comparison.csv")
"""

import os
import csv
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any


from src.utils.logger import get_logger

logger = get_logger("acn.benchmark")


@dataclass
class Scenario:
    """A benchmark scenario definition."""
    name: str
    env_config: str
    algorithms: List[Dict[str, Any]]
    episodes: int = 10
    metrics: List[str] = field(default_factory=lambda: ["avg_reward", "detection_rate"])

    @property
    def metric_calc(self):
        """Lazy import metrics calculator."""
        from .metrics import calculate_metrics
        return calculate_metrics


@dataclass
class AlgorithmResult:
    """Results from running an algorithm on a scenario."""
    label: str
    scenario_name: str
    episodes: List[Dict[str, Any]]
    aggregated_metrics: Dict[str, float]


class BenchmarkRunner:
    """Runs benchmark scenarios and compares algorithms."""

    def __init__(self, config_path: str, results_dir: str = "benchmark_results"):
        self.config_path = config_path
        self.results_dir = results_dir
        self.scenarios: List[Scenario] = []
        self.results: List[AlgorithmResult] = []

        self._load_config()

    def _load_config(self) -> None:
        """Load benchmark configuration from YAML."""
        from src.utils.config_loader import load_config

        config = load_config(self.config_path)
        if not config:
            logger.error("Failed to load benchmark config: {}", self.config_path)
            return

        self.benchmark_name = config.get("benchmark_name", "benchmark")
        scenario_configs = config.get("scenarios", [])

        for sc in scenario_configs:
            scenario = Scenario(
                name=sc["name"],
                env_config=sc["env_config"],
                algorithms=sc.get("algorithms", []),
                episodes=sc.get("episodes", 10),
                metrics=sc.get("metrics", ["avg_reward", "detection_rate"]),
            )
            self.scenarios.append(scenario)

        logger.info("Loaded {} scenarios", len(self.scenarios))

    def run(self) -> List[AlgorithmResult]:
        """Run all scenarios and algorithms."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.results_dir, f"{self.benchmark_name}_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        for scenario in self.scenarios:
            logger.info("Running scenario: {}", scenario.name)

            for algo in scenario.algorithms:
                result = self._run_algorithm(scenario, algo)
                self.results.append(result)

        self._save_results()
        return self.results

    def _run_algorithm(self, scenario: Scenario, algo: Dict[str, Any]) -> AlgorithmResult:
        """Run a single algorithm on a scenario."""
        algo_type = algo.get("type", "heuristic")
        label = algo.get("label", algo_type)
        total_timesteps = algo.get("total_timesteps", 0)

        logger.info("  Running algorithm: {} ({})", label, algo_type)

        episodes = []

        for ep in range(scenario.episodes):
            logger.debug("    Episode {}/{}", ep + 1, scenario.episodes)

            if algo_type == "sb3_ppo":
                ep_data = self._run_ppo_episode(scenario, total_timesteps)
            else:
                ep_data = self._run_heuristic_episode(scenario)

            episodes.append(ep_data)

        # Aggregate metrics
        from .metrics import calculate_metrics
        aggregated = calculate_metrics(episodes, scenario.metrics)

        return AlgorithmResult(
            label=label,
            scenario_name=scenario.name,
            episodes=episodes,
            aggregated_metrics=aggregated,
        )

    def _run_heuristic_episode(self, scenario: Scenario) -> Dict[str, Any]:
        """Run a heuristic (non-RL) episode."""
        from src.utils.config_loader import load_config
        from src.agents.factory import create_agents_from_config
        from src.env.parallel_env import ParallelGameEnv

        config = load_config(scenario.env_config)
        if not config:
            return {"error": "failed to load config"}

        env_config = config.get("environment", {})
        agents_config = config.get("agents", {})

        agents = create_agents_from_config(agents_config, env_config)
        env = ParallelGameEnv(agents=agents, **env_config)

        obs, infos = env.reset()
        max_cycles = env_config.get("max_cycles", 100)

        total_reward = 0.0
        red_detections = 0

        for _ in range(max_cycles):
            actions = {}
            for agent_name in env.agents:
                if agent_name in obs:
                    agent_obj = env.agent_objects[agent_name]
                    action = agent_obj.choose_action(obs[agent_name])
                    actions[agent_name] = action

            obs, rewards, terminations, truncations, infos = env.step(actions)
            total_reward += sum(rewards.values())

            if all(terminations.values()) or all(truncations.values()):
                break

        env.close()

        return {
            "total_reward": total_reward,
            "red_detections": red_detections,
            "steps": max_cycles,
        }

    def _run_ppo_episode(self, scenario: Scenario, total_timesteps: int) -> Dict[str, Any]:
        """Run a PPO-trained episode (placeholder - would load model)."""
        # This is a placeholder - full implementation would load a trained model
        logger.warning("PPO episodes not fully implemented - falling back to heuristic")
        return self._run_heuristic_episode(scenario)

    def _save_results(self) -> None:
        """Save results to the run directory."""
        # Save detailed results as JSON
        results_data = []
        for result in self.results:
            results_data.append({
                "label": result.label,
                "scenario": result.scenario_name,
                "metrics": result.aggregated_metrics,
            })

        with open(os.path.join(self.run_dir, "results.json"), "w") as f:
            json.dump(results_data, f, indent=2)

        logger.info("Results saved to: {}", self.run_dir)

    def save_comparison(self, output_path: str) -> None:
        """Save a comparison CSV of all results."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        rows = []
        for result in self.results:
            row = {
                "scenario": result.scenario_name,
                "algorithm": result.label,
            }
            row.update(result.aggregated_metrics)
            rows.append(row)

        if not rows:
            logger.warning("No results to save")
            return

        # Get all metric keys
        metric_keys = set()
        for row in rows:
            for key in row:
                if key not in ("scenario", "algorithm"):
                    metric_keys.add(key)

        fieldnames = ["scenario", "algorithm"] + sorted(metric_keys)

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        logger.info("Comparison saved to: {}", output_path)


def run_benchmark(config_path: str, results_dir: str = "benchmark_results") -> List[AlgorithmResult]:
    """Convenience function to run a benchmark."""
    runner = BenchmarkRunner(config_path, results_dir)
    results = runner.run()
    runner.save_comparison(os.path.join(results_dir, "comparison.csv"))
    return results
