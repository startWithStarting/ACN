import unittest
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pettingzoo.test import parallel_api_test

from src.env.aec_env import AECGameEnv
from src.env.parallel_env import ParallelGameEnv
from src.agents.blue_agent import BlueAgent
from src.agents.red_agent import RedAgent

class TestEnvironments(unittest.TestCase):
    def setUp(self):
        self.blue_agent = BlueAgent("blue_0", 10, 10, 20, "pursuit")
        self.red_agent = RedAgent("red_0", 10, 10, 15, "center")
        self.agents = [self.blue_agent, self.red_agent]
        self.env_config = {
            "width": 100,
            "height": 80,
            "max_cycles": 10,
            "save_episode_gifs": False
        }

    def test_aec_env_run(self):
        """Test that the refactored AEC environment still runs."""
        print("\nTesting AEC Environment...")
        env = AECGameEnv(self.agents, **self.env_config)
        env.reset()
        
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            if termination or truncation:
                action = None
            else:
                # Simple action
                action = {'direction': (1,0), 'speed': 1.0}
            
            env.step(action)
        
        env.close()
        print("AEC Environment run successful.")

    def test_parallel_env_run(self):
        """Test that the Parallel environment runs."""
        print("\nTesting Parallel Environment...")
        env = ParallelGameEnv(self.agents, **self.env_config)
        observations, infos = env.reset()
        
        self.assertEqual(len(observations), 2)
        
        # valid steps
        for _ in range(5):
            actions = {
                "blue_0": {'direction': (1,0), 'speed': 1.0},
                "red_0": {'direction': (-1,0), 'speed': 1.0}
            }
            obs, rewards, terms, truncs, infos = env.step(actions)
            
            self.assertEqual(len(obs), 2)
            self.assertEqual(len(rewards), 2)
        
        env.close()
        print("Parallel Environment run successful.")

    def test_parallel_env_uses_physics_engine(self):
        """Test that parallel movement is synchronized through PhysicsEngine."""
        env_config = dict(self.env_config)
        env_config["physics"] = {
            "enabled": True,
            "control_mode": "velocity",
            "default_radius": 0.0,
            "enable_collisions": False,
        }
        env = ParallelGameEnv(self.agents, **env_config)
        env.reset(seed=1)

        env.agent_objects["red_0"].x = 10.0
        env.agent_objects["red_0"].y = 10.0
        env.agent_objects["blue_0"].x = 50.0
        env.agent_objects["blue_0"].y = 50.0
        env._init_physics_engine()

        actions = {
            "blue_0": {"direction": (0, 0), "speed": 0.0},
            "red_0": {"direction": (1, 0), "speed": 2.0},
        }
        env.step(actions)

        self.assertIsNotNone(env.physics_engine)
        self.assertAlmostEqual(env.agent_objects["red_0"].x, 12.0)
        np.testing.assert_array_almost_equal(env.physics_engine.get_position("red_0"), [12.0, 10.0])

        env.close()

    def test_parallel_env_force_mode_preserves_inertia(self):
        """Test force-mode controls preserve velocity between physics steps."""
        env_config = dict(self.env_config)
        env_config["physics"] = {
            "enabled": True,
            "control_mode": "force",
            "default_radius": 0.0,
            "default_max_force": 10.0,
            "enable_collisions": False,
        }
        env = ParallelGameEnv(self.agents, **env_config)
        env.reset(seed=1)

        env.agent_objects["red_0"].x = 10.0
        env.agent_objects["red_0"].y = 10.0
        env.agent_objects["blue_0"].x = 50.0
        env.agent_objects["blue_0"].y = 50.0
        env._init_physics_engine()

        env.step({
            "blue_0": {"direction": (0, 0), "speed": 0.0},
            "red_0": {"direction": (1, 0), "speed": 1.0},
        })
        self.assertAlmostEqual(env.agent_objects["red_0"].x, 11.0)

        env.step({
            "blue_0": {"direction": (0, 0), "speed": 0.0},
            "red_0": {"direction": (0, 0), "speed": 0.0},
        })
        self.assertAlmostEqual(env.agent_objects["red_0"].x, 12.0)

        env.close()

    def test_observations_are_detection_limited(self):
        """Test that observations expose only agents within detection radius."""
        blue = BlueAgent("blue_0", 10, 10, detection_radius=10.0, strategy_type="pursuit")
        red_near = RedAgent("red_0", 10, 10, detection_radius=10.0, strategy_type="center")
        red_far = RedAgent("red_1", 10, 10, detection_radius=10.0, strategy_type="center")
        env = ParallelGameEnv([blue, red_near, red_far], **self.env_config)
        env.reset(seed=1)

        env.agent_objects["blue_0"].x = 0.0
        env.agent_objects["blue_0"].y = 0.0
        env.agent_objects["red_0"].x = 3.0
        env.agent_objects["red_0"].y = 4.0
        env.agent_objects["red_1"].x = 50.0
        env.agent_objects["red_1"].y = 50.0
        env._update_active_agents()

        observation = env._get_observation("blue_0", step_count=0)
        self.assertIn("red_0", observation["red_agents"])
        self.assertNotIn("red_1", observation["red_agents"])
        self.assertAlmostEqual(observation["red_agents"]["red_0"]["distance"], 5.0)

        env.close()

    def test_parallel_api_conformance(self):
        """Test PettingZoo Parallel API conformance."""
        print("\nTesting Parallel API Conformance...")
        env = ParallelGameEnv(self.agents, **self.env_config)
        parallel_api_test(env, num_cycles=100)
        print("Parallel API conformance test passed.")

if __name__ == "__main__":
    unittest.main()
