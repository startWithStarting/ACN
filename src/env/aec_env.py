import functools
import random
import os
import io
import imageio.v2 as imageio 

import pygame
import gymnasium
import numpy as np
from gymnasium.spaces import Discrete, Box, Dict as GymDict

from pettingzoo.utils.agent_selector import AgentSelector
from pettingzoo.utils import wrappers 
from pettingzoo.utils.env import AECEnv

from src.agents.base_agent import BaseAgent, AgentType 
from src.env.common_env_logic import ACNEnvironmentLogic, BLUE_AGENT_PASSIVE_REWARD

def env(**kwargs):
    """
    The env function wraps the environment in wrappers PZ provides.
    """
    env = AECGameEnv(**kwargs)
    return env

class AECGameEnv(AECEnv, ACNEnvironmentLogic):
    """
    A basic PettingZoo AEC environment for the communicating agents game.
    """
    metadata = {
        "render_modes": ["human", "human_matplotlib", "human_matplotlib_pred", "human_pygame"],
        "name": "communicating_agents_v0",
        "is_parallelizable": False, 
        "render_fps": 120,
    }

    def __init__(self, agents: list[BaseAgent], render_mode=None, **env_config):
        """
        Args:
            agents (list[BaseAgent]): List of agent objects participating.
            render_mode (str, optional): Rendering mode. Defaults to None.
            max_cycles (int): Max number of cycles (steps per agent) before truncation.
            env_config (dict): Additional environment configuration parameters.
        """
        super().__init__() # Initialize AECEnv base class

        if not agents:
            raise ValueError("Environment must be initialized with at least one agent.")

        self.env_config = env_config
        self.render_mode = render_mode
        
        # Initialize common logic (grid, gifs, lists)
        self._init_common()
        
        self.max_cycles = env_config.get("max_cycles", 10)
        
        # Initialize agent positions randomly
        for agent in agents:
            agent.x = random.uniform(0, self.grid_width)
            agent.y = random.uniform(0, self.grid_height)

        # Store agent objects and create mapping from name to object
        self.agent_objects = {agent.name: agent for agent in agents}
        self.possible_agents = [agent.name for agent in agents]
        self.agent_name_mapping = {i: name for i, name in enumerate(self.possible_agents)}
        
        # Distribute agents into red/blue lists
        self._distribute_agents()

        # PettingZoo API requirements
        self.agents = self.possible_agents[:] 
        self._agent_selector = AgentSelector(self.agents)

        # Define spaces
        self._action_spaces = {
            name: self.agent_objects[name].action_space
            for name in self.possible_agents
        }
        
        self._observation_spaces = self._create_observation_spaces()

        # Internal state
        self.steps = 0
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.rewards = {agent: 0 for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0 for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}

        # Start the agent selection cycle
        self.agent_selection = self._agent_selector.reset()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]
    
    def observe(self, agent):
        """
        Return the observation dictionary for the given agent.
        """
        return self._get_observation(agent, self.steps)

    def reset(self, seed=None, options=None):
        """
        Reset the environment to a starting state.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.agents = self.possible_agents[:] 

        # Save GIF and reset tracking
        if self.save_episode_gifs and self.episode_gif_frames:
            self._save_current_episode_gif()
            
            # Save score plot
            if hasattr(self, 'red_team_scores') and len(self.red_team_scores) > 0:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 4))
                plt.plot(self.step_counts, self.red_team_scores, 'r-', label='Red Team')
                plt.plot(self.step_counts, self.blue_team_scores, 'b-', label='Blue Team')
                plt.xlabel('Time Step')
                plt.ylabel('Average Score')
                plt.title('Team Scores Over Time')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                
                # Save the score plot
                plot_path = os.path.join(self.gif_dir, f'scores_episode_{self.current_episode_number}.png')
                plt.savefig(plot_path, dpi=100, bbox_inches='tight')
                plt.close()

        self.episode_gif_frames = []
        self.red_team_scores = []
        self.blue_team_scores = []
        self.step_counts = []
            
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.steps = 0
        self.current_episode_number += 1
        self._red_agent_scored_this_step = False 

        # Reset agent object states & positions
        self._reset_agent_positions()
            
        self._update_active_agents()
        
        return {agent: self.observe(agent) for agent in self.agents}

    def step(self, action):
        """
        Apply the action for the current agent_selection.
        """
        agent_name = self.agent_selection

        self._update_active_agents()
        self._red_agent_scored_this_step = False
        
        # Track scores at beginning of cycle
        if agent_name == self.agents[0]:
            # Use dictionary values directly since active lists might be stale
            red_scores = [self._cumulative_rewards[name] for name, a in self.agent_objects.items()
                         if getattr(a.agent_type, 'value', None) == 'red' and name in self._cumulative_rewards]
            blue_scores = [self._cumulative_rewards[name] for name, a in self.agent_objects.items()
                          if getattr(a.agent_type, 'value', None) == 'blue' and name in self._cumulative_rewards]
            
            avg_red = sum(red_scores) / len(red_scores) if red_scores else 0
            avg_blue = sum(blue_scores) / len(blue_scores) if blue_scores else 0
            
            self.red_team_scores.append(avg_red)
            self.blue_team_scores.append(avg_blue)
            self.step_counts.append(self.steps)

        if self.terminations[agent_name] or self.truncations[agent_name]:
            self._was_dead_step(action)
            return

        # --- Action Processing ---
        current_agent_obj = self.agent_objects[agent_name]
        
        valid_move = self._apply_action(current_agent_obj, action)
        
        reward = 0
        if valid_move:
            reward, scored = self._calculate_reward(agent_name, current_agent_obj)
            if scored:
                self._red_agent_scored_this_step = True
        else:
             # Basic action fallback (e.g. discrete 1/-1)
            if not isinstance(action, dict): # Only applies if not a dict
                if action == 1:
                    reward = 1
                elif action == 0:
                    reward = -1
            else:
                reward = -1 # Penalty for invalid dict action? Or just ignore
        
        self.rewards[agent_name] = reward
        self._cumulative_rewards[agent_name] += reward

        # --- Termination/Truncation Logic ---
        self.steps += 1
        agent_cycles = self.steps / len(self.possible_agents)
        if agent_cycles >= self.max_cycles:
            self.truncations[agent_name] = True

        self.infos[agent_name] = {"cumulative_reward": self._cumulative_rewards[agent_name]}
        self._accumulate_rewards()
        
        self.agent_selection = self._agent_selector.next()
        
        # while (self.terminations.get(self.agent_selection, False) or 
        #        self.truncations.get(self.agent_selection, False)) and \
        #       len(self.agents) > 0:
        #     self.agent_selection = self._agent_selector.next()
            
        if self.agent_selection == self.agents[0] and not self._red_agent_scored_this_step:
            for blue_agent in self.active_blue_agents:
                if blue_agent.name in self.rewards:
                    self.rewards[blue_agent.name] += BLUE_AGENT_PASSIVE_REWARD

        if self.render_mode in ["human", "human_matplotlib", "human_matplotlib_pred", "human_pygame"]:
            self.render()

    def render(self):
        return ACNEnvironmentLogic.render(self)
    
    def close(self):
        """
        Close the environment, release resources.
        """
        if self.save_episode_gifs and self.episode_gif_frames:
             self._save_current_episode_gif()
        
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            
        if hasattr(self, 'fig') and self.fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self.fig)
            self.fig = None
