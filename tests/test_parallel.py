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

    def test_parallel_api_conformance(self):
        """Test PettingZoo Parallel API conformance."""
        print("\nTesting Parallel API Conformance...")
        env = ParallelGameEnv(self.agents, **self.env_config)
        parallel_api_test(env, num_cycles=100)
        print("Parallel API conformance test passed.")

if __name__ == "__main__":
    unittest.main()
