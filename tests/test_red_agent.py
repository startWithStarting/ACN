import unittest
import numpy as np
from unittest.mock import patch
from gymnasium.spaces import Dict, Box, Discrete

from src.agents.red_agent import RedAgent, AgentType, CommsType

class TestRedAgent(unittest.TestCase):
    def setUp(self):
        self.agent = RedAgent(
            name="Red1", communication_bandwidth=10, processing_capability=5
        )

    def test_initialization(self):
        self.assertEqual(self.agent.name, "Red1")
        self.assertEqual(self.agent.agent_type, AgentType.RED)
        self.assertEqual(self.agent.comms_type, CommsType.DIST)  # Assuming DIST is the default
        self.assertEqual(self.agent.communication_bandwidth, 10)
        self.assertEqual(self.agent.processing_capability, 5)
        self.assertIsInstance(self.agent.action_space, Dict)
        self.assertIsInstance(self.agent.action_space["direction"], Box)
        self.assertIsInstance(self.agent.action_space["speed"], Box)

        # Check action space boundaries
        direction_space = self.agent.action_space["direction"]
        self.assertTrue(np.array_equal(direction_space.low, np.array([-1.0, -1.0])))
        self.assertTrue(np.array_equal(direction_space.high, np.array([1.0, 1.0])))
        speed_space = self.agent.action_space["speed"]
        self.assertEqual(speed_space.shape, (1,))
        self.assertEqual(speed_space.low, np.array([0.0]))
        self.assertEqual(speed_space.high, np.array([10.0]))

    def test_choose_action_no_observation(self):
        action = self.agent.choose_action()
        self.assertTrue(np.array_equal(action["direction"], np.array([0.0, 0.0])))
        self.assertEqual(action["speed"], 0)

    def test_choose_action_missing_info(self):
        # Test with missing 'position'
        action = self.agent.choose_action({"grid_center": (50, 50)})
        self.assertTrue(np.array_equal(action["direction"], np.array([0.0, 0.0])))
        self.assertEqual(action["speed"], 0)

        # Test with missing 'grid_center'
        action = self.agent.choose_action({"position": (25, 25)})
        self.assertTrue(np.array_equal(action["direction"], np.array([0.0, 0.0])))
        self.assertEqual(action["speed"], 0)

    def test_choose_action_move_towards_center(self):
        # Agent at (20, 20), center at (50, 50) - should move towards
        observation = {"position": (20, 20), "grid_center": (50, 50)}
        action = self.agent.choose_action(observation)
        direction = action["direction"]
        speed = action["speed"]
        expected_direction = np.array([30, 30]) / np.linalg.norm([30, 30])

        self.assertTrue(np.allclose(direction, expected_direction))
        self.assertGreater(speed, 0)  # Should have some speed
        self.assertLessEqual(speed, 5)  # Speed should be capped

    def test_choose_action_maintain_min_distance(self):
        # Agent close to center, should move away
        observation = {"position": (48, 48), "grid_center": (50, 50)}
        action = self.agent.choose_action(observation)
        direction = action["direction"]
        speed = action["speed"]

        # Direction should be away from center (roughly -2, -2, normalized)
        expected_direction = np.array([-2, -2]) / np.linalg.norm([-2, -2])
        self.assertTrue(np.allclose(direction, expected_direction))
        self.assertGreater(speed, 0)

    def test_choose_action_at_center(self):
        # Unlikely scenario, but should handle - move in a random direction
        observation = {"position": (50, 50), "grid_center": (50, 50)}
        with patch("numpy.random.uniform") as mock_uniform:
            mock_uniform.return_value = np.array([0.5, -0.5])  # Fixed random values
            action = self.agent.choose_action(observation)
            direction = action["direction"]
            speed = action["speed"]

            # Normalize the mocked random direction
            expected_direction = np.array([0.5, -0.5]) / np.linalg.norm([0.5, -0.5])
            self.assertTrue(np.allclose(direction, expected_direction))
            self.assertEqual(speed, 5)  # Should move away at moderate speed

    def test_str_representation(self):
        self.agent.x, self.agent.y = 25.5, 30.2
        self.agent.speed, self.agent.direction = 2.0, np.array([0.5, 0.5])
        agent_str = str(self.agent)
        self.assertIn("RedAgent(Name: Red1", agent_str)
        self.assertIn("Type: red", agent_str)
        self.assertIn("Comms: dist", agent_str)
        self.assertIn("CommBW: 10", agent_str)
        self.assertIn("ProcCap: 5", agent_str)
        self.assertIn("Pos: (25.50, 30.20)", agent_str)
        self.assertIn("Speed: 2.00, Dir: [0.5 0.5]", agent_str)

    def test_repr_representation(self):
        agent_repr = repr(self.agent)
        self.assertIn("RedAgent(name='Red1'", agent_repr)
        self.assertIn("communication_bandwidth=10", agent_repr)
        self.assertIn("processing_capability=5", agent_repr)

if __name__ == "__main__":
    unittest.main()