import functools
import random
import os
import numpy as np
import pygame
from pettingzoo.utils.env import ParallelEnv

from src.agents.base_agent import BaseAgent
from src.env.common_env_logic import ACNEnvironmentLogic, BLUE_AGENT_PASSIVE_REWARD

def env(**kwargs):
    """
    The env function wraps the environment in wrappers PZ provides.
    """
    env = ParallelGameEnv(**kwargs)
    return env

class ParallelGameEnv(ParallelEnv, ACNEnvironmentLogic):
    """
    Parallel API version of the communicating agents game.
    Agents step simultaneously.
    """
    metadata = {
        "render_modes": ["human", "human_matplotlib", "human_matplotlib_pred", "human_pygame"],
        "name": "communicating_agents_parallel_v0", 
        "render_fps": 120,
    }

    def __init__(self, agents: list[BaseAgent], render_mode=None, **env_config):
        super().__init__()
        
        if not agents:
            raise ValueError("Environment must be initialized with at least one agent.")

        self.env_config = env_config
        self.render_mode = render_mode
        self.max_cycles = env_config.get("max_cycles", 100) # Steps in parallel env usually mean "time steps"
        
        # Initialize common logic
        self._init_common()

        # Initialize agent positions randomly
        for agent in agents:
            agent.x = random.uniform(0, self.grid_width)
            agent.y = random.uniform(0, self.grid_height)

        self.agent_objects = {agent.name: agent for agent in agents}
        self.possible_agents = [agent.name for agent in agents] # List of strings
        self.agent_name_mapping = {name: i for i, name in enumerate(self.possible_agents)}

        self._distribute_agents()
        
        self.agents = self.possible_agents[:]
        
        self._action_spaces = {
            name: self.agent_objects[name].action_space
            for name in self.possible_agents
        }
        
        self._observation_spaces = self._create_observation_spaces()

        # State tracking
        self.steps = 0
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.agents = self.possible_agents[:]
        
        # Save GIF from previous episode
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
                
                plot_path = os.path.join(self.gif_dir, f'scores_episode_{self.current_episode_number}.png')
                plt.savefig(plot_path, dpi=100, bbox_inches='tight')
                plt.close()

        self.episode_gif_frames = []
        self.red_team_scores = []
        self.blue_team_scores = []
        self.step_counts = []
        
        self.steps = 0
        self.current_episode_number += 1
        
        # Reset agents
        self._reset_agent_positions()
            
        self._update_active_agents()
        
        observations = {agent: self._get_observation(agent, self.steps) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos

    def step(self, actions):
        """
        Execute simultaneous steps for all agents.
        actions: dict {agent_name: action}
        """
        # Determine current active agents based on keys in actions? 
        # Or standard PZ parallel API says we must return info for all agents in self.agents?
        
        self._update_active_agents()
        
        rewards = {agent: 0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        red_scored_this_step = False
        
        # 1. Apply movement controls for all agents, then advance physics once.
        if self.use_physics:
            self._begin_physics_step()
        for agent_name, action in actions.items():
            if agent_name in self.agent_objects:
                agent_obj = self.agent_objects[agent_name]
                self._apply_action(agent_obj, action, advance_physics=False)
        if self.use_physics:
            self._advance_physics()
        
        # 2. Calculate rewards (after all have moved)
        for agent_name in self.agents:
            agent_obj = self.agent_objects[agent_name]
            # Use shared reward calculation
            # Note: _calculate_reward logic assumes 'red' scores if near center and unseen
            # Does this logic hold if all moved at once? Yes, we check detection based on new positions.
            r, scored = self._calculate_reward(agent_name, agent_obj)
            rewards[agent_name] = r
            if scored:
                red_scored_this_step = True
                
        # 3. Apply team penalty/bonus (if no red scored, blue gets reward)
        if not red_scored_this_step:
            for blue_agent in self.active_blue_agents:
                if blue_agent.name in rewards:
                    rewards[blue_agent.name] += BLUE_AGENT_PASSIVE_REWARD
        
        # 4. Check termination/truncation
        self.steps += 1
        if self.steps >= self.max_cycles:
            for agent in self.agents:
                truncations[agent] = True

        # Update internal state (needed by common logic)
        self.terminations.update(terminations)
        self.truncations.update(truncations)
                
        # Update scores for plotting
        # Note: 'rewards' assumes instantaneous reward. _cumulative_rewards update is handled by PZ mostly?
        # ParallelEnv doesn't auto-sum rewards in a property like AECEnv does. We track it manually for plotting if needed.
        # But we need self.red_team_scores for the renderer.
        
        # Calculate accumulating average score is tricky in ParallelStep if we just use instantaneous. 
        # Let's assume we plot instantaneous or windowed? 
        # The AEC implementation sums cumulative rewards. Let's assume we want to track cumulative.
        
        # We need to maintain cumulative rewards if we want to match AEC logic for plotting
        if not hasattr(self, '_cumulative_rewards_map'):
             self._cumulative_rewards_map = {agent: 0.0 for agent in self.possible_agents}
             
        for agent, r in rewards.items():
            self._cumulative_rewards_map[agent] += r
            
        # Calculate stats for plotting
        current_red_scores = [self._cumulative_rewards_map[name] for name in self.agents 
                             if name in [a.name for a in self.red_agents]]
        current_blue_scores = [self._cumulative_rewards_map[name] for name in self.agents 
                              if name in [a.name for a in self.blue_agents]]
                              
        avg_red = sum(current_red_scores) / len(current_red_scores) if current_red_scores else 0
        avg_blue = sum(current_blue_scores) / len(current_blue_scores) if current_blue_scores else 0
        
        self.red_team_scores.append(avg_red)
        self.blue_team_scores.append(avg_blue)
        self.step_counts.append(self.steps)

        # 5. Observations
        observations = {agent: self._get_observation(agent, self.steps) for agent in self.agents}
        
        # 6. Render
        if self.render_mode in ["human", "human_matplotlib", "human_matplotlib_pred", "human_pygame"]:
            self.render()
            
        # 7. Prune dead agents (if agents die/terminate, remove from self.agents)
        self.agents = [
            agent for agent in self.agents 
            if not (terminations[agent] or truncations[agent])
        ]
        
        return observations, rewards, terminations, truncations, infos

    # --- Rendering ---
    def render(self):
        return ACNEnvironmentLogic.render(self)

    def close(self):
        if self.save_episode_gifs and self.episode_gif_frames:
             self._save_current_episode_gif()
        
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            
        if hasattr(self, 'fig') and self.fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self.fig)
            self.fig = None
