"""Tests for benchmark module."""

import unittest
import tempfile
import shutil

from src.benchmark.metrics import (
    avg_episode_reward,
    detection_rate,
    calculate_metrics,
    convergence_speed,
)
from src.benchmark.runner import Scenario


class TestMetrics(unittest.TestCase):
    """Test cases for metric calculators."""

    def test_avg_episode_reward(self):
        """Test average reward calculation."""
        episodes = [
            {"total_reward": 10.0},
            {"total_reward": 20.0},
            {"total_reward": 30.0},
        ]
        self.assertEqual(avg_episode_reward(episodes), 20.0)

    def test_detection_rate(self):
        """Test detection rate calculation."""
        episodes = [
            {"red_detections": 5, "steps": 100},
            {"red_detections": 10, "steps": 100},
        ]
        self.assertEqual(detection_rate(episodes), 0.075)

    def test_calculate_metrics(self):
        """Test multi-metric calculation."""
        episodes = [
            {"total_reward": 10.0, "red_detections": 5, "steps": 100},
            {"total_reward": 20.0, "red_detections": 10, "steps": 100},
        ]
        metrics = calculate_metrics(episodes, ["avg_reward", "detection_rate"])

        self.assertAlmostEqual(metrics["avg_reward"], 15.0)
        self.assertAlmostEqual(metrics["detection_rate"], 0.075)

    def test_convergence_speed(self):
        """Test convergence speed calculation."""
        episodes = [
            {"total_reward": 1.0},
            {"total_reward": 2.0},
            {"total_reward": 5.0},
            {"total_reward": 10.0},
        ]
        # Threshold 0.95 of final avg (10/4=2.5 * 0.95 = 2.375)
        step = convergence_speed(episodes, threshold=0.95)
        self.assertEqual(step, 3)  # 3rd episode reaches 5.0


class TestBenchmarkRunner(unittest.TestCase):
    """Test cases for BenchmarkRunner."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_scenario_creation(self):
        """Test Scenario creation."""
        scenario = Scenario(
            name="test_scenario",
            env_config="config/test.yaml",
            algorithms=[
                {"type": "heuristic", "label": "baseline"}
            ],
            episodes=5,
            metrics=["avg_reward"],
        )
        self.assertEqual(scenario.name, "test_scenario")
        self.assertEqual(scenario.episodes, 5)


if __name__ == "__main__":
    unittest.main()