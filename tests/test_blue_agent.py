"""Tests for BlueAgent."""

import unittest
import numpy as np
from gymnasium.spaces import Box, Dict as DictSpace

from src.agents.blue_agent import BlueAgent


class TestBlueAgent(unittest.TestCase):
    """Test cases for BlueAgent."""

    def setUp(self):
        """Set up test fixtures."""
        self.agent = BlueAgent(
            name="test_blue",
            communication_bandwidth=10,
            processing_capability=50,
            detection_radius=20.0,
            strategy_type="pursuit",
            prediction_timeout=50,
            observation_window_size=5,
            prediction_interval=1,
        )

    def test_agent_creation(self):
        """Test that BlueAgent is created correctly."""
        self.assertEqual(self.agent.name, "test_blue")
        self.assertEqual(self.agent.detection_radius, 20.0)

    def test_action_space(self):
        """Test that action space is correctly defined."""
        self.assertIsInstance(self.agent.action_space, DictSpace)

        # Check direction space
        self.assertIn("direction", self.agent.action_space.spaces)
        dir_space = self.agent.action_space["direction"]
        self.assertIsInstance(dir_space, Box)
        self.assertEqual(dir_space.shape, (2,))
        np.testing.assert_array_equal(dir_space.low, [-1.0, -1.0])
        np.testing.assert_array_equal(dir_space.high, [1.0, 1.0])

        # Check speed space
        self.assertIn("speed", self.agent.action_space.spaces)
        speed_space = self.agent.action_space["speed"]
        self.assertIsInstance(speed_space, Box)
        self.assertEqual(speed_space.shape, (1,))
        np.testing.assert_array_equal(speed_space.low, [0.0])
        np.testing.assert_array_equal(speed_space.high, [10.0])

    def test_history_cap(self):
        """Test that history is capped at max_history_length."""
        max_hist = self.agent.max_history_length

        # Add more entries than the cap
        for i in range(max_hist + 100):
            self.agent.prediction_history[f"red_{i % 5}"].append(
                (np.array([float(i), float(i)]), float(i))
            )

        # Check that history is capped
        for key, history in self.agent.prediction_history.items():
            self.assertLessEqual(len(history), max_hist)

    def test_choose_action(self):
        """Test choose_action returns valid action."""
        # Create a mock observation
        obs = {
            "position": np.array([50.0, 50.0], dtype=np.float32),
            "grid_center": np.array([50.0, 50.0], dtype=np.float32),
            "timestamp": 0.0,
            "red_agents": {},
        }

        action = self.agent.choose_action(obs)

        # Action should be a dict with direction and speed
        self.assertIsInstance(action, dict)
        self.assertIn("direction", action)
        self.assertIn("speed", action)

        # Check shapes
        self.assertEqual(action["direction"].shape, (2,))
        self.assertEqual(action["speed"].shape, (1,))


if __name__ == "__main__":
    unittest.main()