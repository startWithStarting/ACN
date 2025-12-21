import functools
import random
import os
import io
# Ensure imageio is installed: pip install imageio
import imageio.v2 as imageio # Using v2 for modern API

import pygame

import gymnasium
import numpy as np
from gymnasium.spaces import Discrete, Box, Dict as GymDict # Import Dict space

from pettingzoo.utils.agent_selector import AgentSelector
from pettingzoo.utils import wrappers # Keep wrappers import separate
from pettingzoo.utils.env import AECEnv

# Import agent types (adjust path if needed)
from src.agents.base_agent import BaseAgent, AgentType # Need AgentType for checking agent types in step

def env(**kwargs):
    """
    The env function wraps the environment in wrappers PZ provides.
    """
    env = AECGameEnv(**kwargs)
    # Example wrapper: env = wrappers.AssertOutOfBoundsWrapper(env)
    # Example wrapper: env = wrappers.OrderEnforcingWrapper(env)
    return env

class AECGameEnv(AECEnv):
    """
    A basic PettingZoo AEC environment for the communicating agents game.

    In this simple version:
    - Observation: A dummy value (e.g., agent's index).
    - Action: A dummy discrete action (e.g., 0 or 1).
    - Reward: Simple reward (e.g., +1 for action 1, -1 for action 0).
    - Termination/Truncation: After a fixed number of steps per agent.
    """
    metadata = {
        "render_modes": ["human", "human_matplotlib", "human_matplotlib_pred", "human_pygame"],
        "name": "communicating_agents_v0",
        "is_parallelizable": False, # Usually True if step() doesn't depend on agent order
        "render_fps": 120,
    }

    def __init__(self, agents: list[BaseAgent], render_mode=None, **env_config):
        """
        Args:
            agents (list[BaseAgent]): List of agent objects participating.
            render_mode (str, optional): Rendering mode. Defaults to None.
            max_cycles (int): Max number of cycles (steps per agent) before truncation.
            env_config (dict): Additional environment configuration parameters,
                               should include 'width' and 'height' for the grid.
        """
        super().__init__() # Initialize AECEnv base class

        if not agents:
            raise ValueError("Environment must be initialized with at least one agent.")

        self.max_cycles = env_config.get("max_cycles", 10) # Prioritize env_config, then param, then hardcoded default if needed
        
        self.env_config = env_config

        # --- GIF Saving Configuration ---
        self.save_episode_gifs = self.env_config.get("save_episode_gifs", True) # Default to True
        self.gif_dir = None # Initialize gif_dir
        self.gif_figsize = self.env_config.get("gif_figsize", (10, 8)) # (width, height) in inches
        self.episode_gif_frames = []
        self.current_episode_number = 0 # To number GIF filenames
        self.should_quit = False # Flag to signal simulation should stop

        if self.save_episode_gifs:
            # The specific directory for this experiment's run, passed from main.py
            # This path will be like "results/experiment_name_timestamp"
            experiment_run_dir = self.env_config.get("experiment_results_dir")

            if experiment_run_dir:
                self.gif_dir = os.path.join(experiment_run_dir, "gifs")
                os.makedirs(self.gif_dir, exist_ok=True)
            else:
                # print("Warning: 'save_episode_gifs' is True, but 'experiment_results_dir' was not provided in env_config. Disabling GIF saving.")
                self.save_episode_gifs = False

        # --- Grid Initialization ---
        # User requested H (height) and W (width)
        # These should be passed in env_config from main.py, sourced from experiment_config.yaml
        self.grid_height = self.env_config.get("height", 80)  # H (Height)
        self.grid_width = self.env_config.get("width", 100)   # W (Width)
        
        # Initialize agent positions randomly
        for agent in agents:
            agent.x = random.uniform(0, self.grid_width)
            agent.y = random.uniform(0, self.grid_height)

        # Store agent objects and create mapping from name to object
        self.agent_objects = {agent.name: agent for agent in agents}
        self.possible_agents = [agent.name for agent in agents]  # All possible agent names
        self.agent_name_mapping = {i: name for i, name in enumerate(self.possible_agents)}
        
        # Separate agent lists for more efficient lookups
        self.red_agents = []
        self.blue_agents = []
        self.active_red_agents = []
        self.active_blue_agents = []
        
        for agent in agents:
            if hasattr(agent, 'agent_type'):
                agent_type = getattr(agent.agent_type, 'value', None)
                if agent_type == 'red':
                    self.red_agents.append(agent)
                    self.active_red_agents.append(agent)
                elif agent_type == 'blue':
                    self.blue_agents.append(agent)
                    self.active_blue_agents.append(agent)

        # PettingZoo API requirements
        self.agents = self.possible_agents[:] # Current list of active agents (names)
        self._agent_selector = AgentSelector(self.agents) # Cycles through agent names

        # Define spaces (MUST be defined after agents are known)
        # Action spaces are now defined in each agent class and accessed here
        self._action_spaces = {
            name: self.agent_objects[name].action_space
            for name in self.possible_agents
        }

        # Define observation space using Dict
        # Position: Box from (0,0) to (width, height)
        # Grid Center: Box representing a single point (width/2, height/2)
        # Note: Using float32 for consistency, even if center is fixed.
        self._observation_spaces = {
            name: GymDict({
                'position': Box(low=np.array([0.0, 0.0], dtype=np.float32),
                                high=np.array([self.grid_width, self.grid_height], dtype=np.float32),
                                shape=(2,), dtype=np.float32),
                'grid_center': Box(low=np.array([self.grid_width / 2, self.grid_height / 2], dtype=np.float32),
                                   high=np.array([self.grid_width / 2, self.grid_height / 2], dtype=np.float32),
                                   shape=(2,), dtype=np.float32)
            })
            for name in self.possible_agents
        }

        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        # Internal state
        self.steps = 0
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.rewards = {agent: 0 for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0 for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}

        # Start the agent selection cycle
        self.agent_selection = self._agent_selector.reset() # Get the first agent

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]
    

    def _update_active_agents(self):
        """Update the lists of active red and blue agents."""
        self.active_red_agents = [
            agent for agent in self.red_agents 
            if (hasattr(agent, 'is_active') and agent.is_active and 
                agent.name in self.agents and 
                not self.terminations.get(agent.name, False) and 
                not self.truncations.get(agent.name, False))
        ]
        
        self.active_blue_agents = [
            agent for agent in self.blue_agents 
            if (hasattr(agent, 'is_active') and agent.is_active and 
                agent.name in self.agents and 
                not self.terminations.get(agent.name, False) and 
                not self.truncations.get(agent.name, False))
        ]

    def observe(self, agent):
        """
        Return the observation dictionary for the given agent.
        Contains the agent's current position and the grid center coordinates.
        """
        agent_obj = self.agent_objects[agent]
        agent_pos = np.array([agent_obj.x, agent_obj.y], dtype=np.float32)
        grid_center = np.array([self.grid_width / 2, self.grid_height / 2], dtype=np.float32)

        observation = {
            'position': agent_pos,
            'grid_center': grid_center
        }

        # Add timestamp to all observations
        observation['timestamp'] = self.steps
        
        # Add agent-specific information
        agent_obj = self.agent_objects[agent]
        
        # If agent is a BlueAgent, add red_agents info
        if hasattr(agent_obj, 'agent_type') and getattr(agent_obj.agent_type, 'value', None) == 'blue':
            # Gather active red agent positions
            red_agents_info = {}
            for red_agent in self.active_red_agents:
                if (hasattr(red_agent, 'x') and hasattr(red_agent, 'y') and 
                    red_agent.x is not None and red_agent.y is not None):
                    red_agents_info[red_agent.name] = {'position': (red_agent.x, red_agent.y)}
            observation['red_agents'] = red_agents_info
        
        # If agent is a RedAgent, add blue_agents info and red_teammates info
        elif hasattr(agent_obj, 'agent_type') and getattr(agent_obj.agent_type, 'value', None) == 'red':
            # Gather active blue agent positions
            blue_agents_info = {}
            for blue_agent in self.active_blue_agents:
                if (hasattr(blue_agent, 'x') and hasattr(blue_agent, 'y') and 
                    blue_agent.x is not None and blue_agent.y is not None):
                    blue_agents_info[blue_agent.name] = {'position': (blue_agent.x, blue_agent.y)}
            observation['blue_agents'] = blue_agents_info
            
            # Gather active red teammate positions (excluding self)
            red_teammates_info = {}
            for red_agent in self.active_red_agents:
                if (red_agent.name != agent and  # not self
                    hasattr(red_agent, 'x') and hasattr(red_agent, 'y') and
                    red_agent.x is not None and red_agent.y is not None):
                    red_teammates_info[red_agent.name] = {'position': (red_agent.x, red_agent.y)}
            observation['red_teammates'] = red_teammates_info
        
        # If using flocking strategy, pass the flocking-specific parameters if they exist in env_config
        if hasattr(agent_obj, 'strategy_type') and agent_obj.strategy_type == 'flocking':
            # Get parameters from the environment config if available
            # These would have been specified in the configuration file
            agent_config = self._get_agent_config(agent)
            if agent_config:
                # Add flocking parameters to the observation if they exist in the config
                for param in ['cohesion_weight', 'alignment_weight', 'separation_weight', 'separation_radius']:
                    if param in agent_config:
                        observation[param] = agent_config[param]

        return observation

    def reset(self, seed=None, options=None):
        """
        Reset the environment to a starting state.
        """
        if seed is not None:
            # Seed the random number generator if needed (e.g., for procedural generation)
            random.seed(seed)
            np.random.seed(seed)
            # Note: PettingZoo recommends seeding action/observation spaces separately if needed

        self.agents = self.possible_agents[:] # Reset active agents

        # Save GIF of the completed episode (if any frames were collected)
        if self.save_episode_gifs and self.episode_gif_frames:
            self._save_current_episode_gif()
            
            # Save score plot if we have score data
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
                plot_path = os.path.join(os.path.dirname(self.episode_gif_path), 'scores.png')
                plt.savefig(plot_path, dpi=100, bbox_inches='tight')
                plt.close()

        # Reset episode tracking
        self.episode_gif_frames = [] # Clear frames for the new episode
        self.red_team_scores = []
        self.blue_team_scores = []
        self.step_counts = []
            
        self._agent_selector.reinit(self.agents) # Reinitialize selector
        self.agent_selection = self._agent_selector.reset() # Get first agent

        # Reset internal state
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.steps = 0
        self.current_episode_number += 1 # Increment for the new episode
        self._red_agent_scored_this_step = False  # Reset scoring tracker

        # Reset agent object states
        for agent_obj in self.agent_objects.values():
            agent_obj.is_active = True
            # Re-initialize positions on reset
            agent_obj.x = random.uniform(0, self.grid_width)
            agent_obj.y = random.uniform(0, self.grid_height)
            
        # Reset active agent lists
        self._update_active_agents()
        
        return {agent: self.observe(agent) for agent in self.agents}
        
        # Reset active agent lists
        self._update_active_agents()

    def step(self, action):
        """
        Apply the action for the current agent_selection.
        Updates rewards, terminations, truncations, and selects the next agent.
        """
        agent_name = self.agent_selection

        # Update active agents before processing the step
        self._update_active_agents()
        
        # Track if any red agent scores this step
        self._red_agent_scored_this_step = False
        
        # Track scores at the beginning of each full cycle (when we're back to the first agent)
        if agent_name == self.agents[0]:
            # Calculate average scores for each team
            red_scores = [self._cumulative_rewards[name] for name, agent in self.agent_objects.items() 
                         if hasattr(agent, 'agent_type') and getattr(agent.agent_type, 'value', None) == 'red']
            blue_scores = [self._cumulative_rewards[name] for name, agent in self.agent_objects.items() 
                          if hasattr(agent, 'agent_type') and getattr(agent.agent_type, 'value', None) == 'blue']
            
            avg_red = sum(red_scores) / len(red_scores) if red_scores else 0
            avg_blue = sum(blue_scores) / len(blue_scores) if blue_scores else 0
            
            self.red_team_scores.append(avg_red)
            self.blue_team_scores.append(avg_blue)
            self.step_counts.append(self.steps)

        if self.terminations[agent_name] or self.truncations[agent_name]:
            # If agent is done, handle potential cleanup and select next agent
            self._was_dead_step(action) # PZ utility - using _was_dead_step (newer API name)
            return

        # --- Action Processing ---
        # Get the agent object
        current_agent_obj = self.agent_objects[agent_name]

        # Process the action based on agent type
        if current_agent_obj.agent_type in [AgentType.RED, AgentType.BLUE]:
            # For RED and BLUE agents, handle the composite action (direction + speed)
            if isinstance(action, dict) and 'direction' in action and 'speed' in action:
                direction = action['direction']
                speed = action['speed']
                
                # Calculate movement vector (direction * speed)
                # Speed can be negative or positive (negative means move in opposite direction)
                movement_x = direction[0] * speed
                movement_y = direction[1] * speed
                
                # Update agent position
                current_agent_obj.x += movement_x
                current_agent_obj.y += movement_y
                
                # Ensure agent stays within grid boundaries
                current_agent_obj.x = max(0, min(self.grid_width, current_agent_obj.x))
                current_agent_obj.y = max(0, min(self.grid_height, current_agent_obj.y))
                
                # Update agent's internal direction and speed attributes
                current_agent_obj.direction = tuple(direction)
                current_agent_obj.speed = float(speed)  # Convert to float for consistency
                
                # Calculate distance to center
                center = (self.grid_width / 2, self.grid_height / 2)
                distance_to_center = np.sqrt((current_agent_obj.x - center[0])**2 + 
                                           (current_agent_obj.y - center[1])**2)
                
                # Initialize reward
                reward = 0
                
                # Check if agent is a red agent
                if hasattr(current_agent_obj, 'agent_type') and getattr(current_agent_obj.agent_type, 'value', None) == 'red':
                    # Check if within 1 unit of the attractor circle (radius 50)
                    if abs(distance_to_center - 50) <= 1.0:
                        # Check if not detected by any blue agent
                        if not self._is_red_agent_detected(current_agent_obj):
                            reward = 1  # Give reward for being in the right place undetected
                            self.rewards[agent_name] = 1
                            self._red_agent_scored_this_step = True
            else:
                # Invalid action format for agent
                reward = -1  # Penalty for invalid action
        else:
            # For other agent types, use the existing placeholder logic
            reward = 0
            if action == 1:
                reward = 1
            elif action == 0:
                reward = -1
        
        self.rewards[agent_name] = reward # Store reward for this step
        self._cumulative_rewards[agent_name] += reward # Update cumulative reward

        # --- Termination/Truncation Logic ---
        self.steps += 1
        # Truncate based on max_cycles (total steps / num_agents)
        agent_cycles = self.steps / len(self.possible_agents)
        if agent_cycles >= self.max_cycles:
            self.truncations[agent_name] = True

        # Update info dictionary for the agent
        self.infos[agent_name] = {"cumulative_reward": self._cumulative_rewards[agent_name]}

        # PZ utility: Accumulate rewards correctly for the next call to last()
        self._accumulate_rewards()
        
        # Select the next agent using PettingZoo's agent selector
        self.agent_selection = self._agent_selector.next()
        
        # Skip over any terminated/truncated agents
        while (self.terminations.get(self.agent_selection, False) or 
               self.truncations.get(self.agent_selection, False)) and \
              len(self.agents) > 0:
            self.agent_selection = self._agent_selector.next()
            
        # If we've gone through all agents for this step and no red agent scored,
        # give a small reward to all active blue agents
        if self.agent_selection == self.agents[0] and not self._red_agent_scored_this_step:
            for blue_agent in self.active_blue_agents:
                if blue_agent.name in self.rewards:
                    self.rewards[blue_agent.name] += 0.1  # Small reward for preventing red agents from scoring

        # Call render if a human-viewable mode is active
        if self.render_mode in ["human", "human_matplotlib", "human_matplotlib_pred", "human_pygame"]:
            self.render()

    def render(self):
        """
        Renders the environment.
        """
        if self.render_mode is None:
            # gymnasium.logger.warn(
            #     "You are calling render method without specifying any render mode."
            # )
            return

        # Add custom rendering modes as needed
        if self.render_mode == "ansi" or self.render_mode == "human_text":
            return self._render_text()
        elif self.render_mode == "human" or self.render_mode == "human_matplotlib":
            return self._render_matplotlib()
        elif self.render_mode == "human_matplotlib_pred":
            return self._render_matplotlib(show_predictions=True)
        elif self.render_mode == "human_pygame":
            return self._render_pygame()
        else:
            raise NotImplementedError(f"Render mode '{self.render_mode}' not supported.")

    def _render_pygame(self):
        """
        Renders the environment using PyGame.
        """
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            # Scale up the window for better visibility
            self.window_scale = 8 
            self.window_width = int(self.grid_width * self.window_scale)
            self.window_height = int(self.grid_height * self.window_scale)
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))
            pygame.display.set_caption("ACN Simulation")
            self.clock = pygame.time.Clock()

        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255)) # White background

        # Draw attractor zone
        center_x = int((self.grid_width / 2) * self.window_scale)
        center_y = int((self.grid_height / 2) * self.window_scale)
        # In PyGame y is down, but our grid might be bottom-up. 
        # Usually standard is top-left (0,0). Let's assume standard PyGame coords match our grid logic 
        # (0,0 at top-left) or we might need to flip Y.
        # For now, assuming direct mapping is fine.
        
        attractor_radius = int(10.0 * self.window_scale)
        pygame.draw.circle(self.screen, (255, 165, 0), (center_x, center_y), attractor_radius, 2) # Orange outline

        # Draw agents
        for name, agent_obj in self.agent_objects.items():
            is_done = self.terminations.get(name, True) or self.truncations.get(name, True)
            if not is_done and hasattr(agent_obj, 'x') and hasattr(agent_obj, 'y') \
               and agent_obj.x is not None and agent_obj.y is not None:
                
                x = int(agent_obj.x * self.window_scale)
                y = int(agent_obj.y * self.window_scale)
                
                color = (128, 128, 128) # Grey
                if hasattr(agent_obj, 'agent_type') and hasattr(agent_obj.agent_type, 'value'):
                    if agent_obj.agent_type.value == 'blue':
                        color = (0, 0, 255) # Blue
                    elif agent_obj.agent_type.value == 'red':
                        color = (255, 0, 0) # Red
                
                pygame.draw.circle(self.screen, color, (x, y), 5)
                
                # Draw predictions for Blue agents
                if agent_obj.agent_type.value == 'blue' and hasattr(agent_obj, 'predicted_positions'):
                     for red_name, predictions in agent_obj.predicted_positions.items():
                        for i, pred_pos in enumerate(predictions):
                            px = int(pred_pos[0] * self.window_scale)
                            py = int(pred_pos[1] * self.window_scale)
                            
                            # Clamp coordinates to prevent PyGame overflow/crash with huge numbers
                            # PyGame uses C ints, so keep it within safe 16-bit range or reasonable screen bounds
                            max_coord = 30000
                            px = max(-max_coord, min(max_coord, px))
                            py = max(-max_coord, min(max_coord, py))

                            # Fade to green
                            alpha = max(50, 255 - i * 40)
                            # PyGame doesn't support alpha on direct draw calls easily without a surface
                            # So we just use a lighter green
                            pred_color = (0, 255, 0)
                            try:
                                pygame.draw.circle(self.screen, pred_color, (px, py), 3)
                            except TypeError as e:
                                print(f"[RENDER ERROR] Failed to draw prediction at ({px}, {py}) type: {type(px)}, {type(py)}. Error: {e}")

        # Update display
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
        
        # Capture frame for GIF
        if self.save_episode_gifs:
            # pygame.surfarray.array3d returns (width, height, 3)
            # imageio expects (height, width, 3)
            frame = np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))
            self.episode_gif_frames.append(frame)
        
        # Handle events to prevent freezing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.should_quit = True
                pygame.quit()
                self.screen = None

    def _is_red_agent_detected(self, red_agent):
        """Check if a red agent is detected by any active blue agent.
        
        Args:
            red_agent: The red agent object to check
            
        Returns:
            bool: True if detected by any blue agent, False otherwise
        """
        if not hasattr(red_agent, 'x') or not hasattr(red_agent, 'y') or red_agent.x is None or red_agent.y is None:
            return False
            
        red_pos = (red_agent.x, red_agent.y)
        
        # Only iterate through active blue agents
        for blue_agent in self.active_blue_agents:
            # Skip if blue agent is terminated
            if (not hasattr(blue_agent, 'is_active') or not blue_agent.is_active or
                not hasattr(blue_agent, 'is_within_detection_radius') or 
                not hasattr(blue_agent, 'x') or not hasattr(blue_agent, 'y') or 
                blue_agent.x is None or blue_agent.y is None):
                continue
                
            # Call is_within_detection_radius with just the red agent's position
            # The blue agent already knows its own position
            if blue_agent.is_within_detection_radius(red_pos):
                return True
        return False

    def _render_text(self):
        """Text-based rendering for debugging."""
        # print("\n--- Rendering Frame ---")
        # print(f"Step: {self.steps}")
        # print(f"Grid Dimensions: W={self.grid_width}, H={self.grid_height}")
        # print(f"Current Agent: {self.agent_selection}")
        # print("Agent States:")
        for name in self.possible_agents:  # Iterate in defined order
            agent_obj = self.agent_objects[name]
            state = "Done" if (self.terminations[name] or self.truncations[name]) else "Active"
            position_str = f", Pos: ({agent_obj.x:.2f}, {agent_obj.y:.2f})" if agent_obj.x is not None and agent_obj.y is not None else ""
            # print(f"  - {name}: Reward={self._cumulative_rewards[name]:.2f}, State={state}{position_str}")
        # print("-----------------------\n")

    def _render_matplotlib(self, show_predictions=False):
        """Matplotlib rendering for visualization."""
        import matplotlib.pyplot as plt
        
        # Initialize figure if it doesn't exist
        if not hasattr(self, 'fig') or self.fig is None:
            # Create a figure with two subplots: one for the environment and one for scores
            self.fig = plt.figure(figsize=(self.gif_figsize[0], self.gif_figsize[1] + 2))
            gs = self.fig.add_gridspec(2, 1, height_ratios=[3, 1])
            self.ax_env = self.fig.add_subplot(gs[0])
            self.ax_score = self.fig.add_subplot(gs[1])
            plt.ion() # Turn on interactive mode
            self.fig.show()
        
        # Clear axes for new frame
        self.ax_env.clear()
        self.ax_score.clear()
        
        # Set up environment plot
        self.ax_env.set_xlim(0, self.grid_width)
        self.ax_env.set_ylim(0, self.grid_height)
        self.ax_env.set_aspect('equal', adjustable='box')
        
        # Plot score history if we have data
        if len(self.step_counts) > 0:
            self.ax_score.plot(self.step_counts, self.red_team_scores, 'r-', label='Red Team')
            self.ax_score.plot(self.step_counts, self.blue_team_scores, 'b-', label='Blue Team')
            self.ax_score.set_xlabel('Time Step')
            self.ax_score.set_ylabel('Avg Score')
            self.ax_score.legend()
            self.ax_score.grid(True)
            
            # Add current scores as text
            current_red = self.red_team_scores[-1] if self.red_team_scores else 0
            current_blue = self.blue_team_scores[-1] if self.blue_team_scores else 0
            score_text = f'Red: {current_red:.2f}  |  Blue: {current_blue:.2f}'
            self.ax_score.set_title(f'Team Scores (Current: {score_text})')
            
        self.fig.tight_layout()
        
        # The rest of the environment plotting will use ax_env
        ax = self.ax_env  # For backward compatibility with existing code
        
        # Draw a ring around the center of the grid to indicate the position of attractors
        center_x, center_y = self.grid_width / 2, self.grid_height / 2  # Center at (50, 50) for 100x80 grid
        attractor_radius = 10.0  # Radius of the attractor ring
        attractor_circle = plt.Circle((center_x, center_y), attractor_radius, fill=False, color='orange', linestyle='--', linewidth=2)
        ax.add_patch(attractor_circle)
        # Add a small text label explaining the circle
        ax.text(center_x, center_y - attractor_radius - 2, 'Attractor Zone', color='orange', ha='center', fontsize=8)
        
        something_plotted = False
        for name, agent_obj in self.agent_objects.items():
            is_done = self.terminations.get(name, True) or self.truncations.get(name, True)
            if not is_done and hasattr(agent_obj, 'x') and hasattr(agent_obj, 'y') \
               and agent_obj.x is not None and agent_obj.y is not None:
                color = 'grey' # Default color
                agent_number = ''
                if hasattr(agent_obj, 'agent_type') and hasattr(agent_obj.agent_type, 'value'):
                    agent_type_val = agent_obj.agent_type.value
                    if agent_type_val == 'blue':
                        color = 'blue'
                    elif agent_type_val == 'red':
                        color = 'red'
                    # Extract number from agent name (e.g., blue_2 -> 2)
                    if '_' in name:
                        agent_number = name.split('_')[-1]
                    else:
                        agent_number = name
                # Plot agent number as text at its position
                ax.text(agent_obj.x, agent_obj.y, agent_number, color=color, fontsize=14, fontweight='bold', ha='center', va='center')
                something_plotted = True

                # --- Plot scatter of detected red agent paths for BlueAgents ---
                if agent_type_val == 'blue' and hasattr(agent_obj, 'get_observed_paths'):
                    observed_paths = agent_obj.get_observed_paths()
                    # Use a different marker for each blue agent
                    blue_marker = 'x'
                    for red_name, path in observed_paths.items():
                        if path:  # path is a list of (position, timestamp)
                            # For regular mode, show all observations
                            if not show_predictions:
                                xs = [pos[0][0] for pos in path]
                                ys = [pos[0][1] for pos in path]
                                # Use alpha for visibility if many points
                                ax.scatter(xs, ys, marker=blue_marker, color=color, alpha=0.6, label=f"{name} sees {red_name}")
                            # For prediction mode, show only the latest two observations
                            else:
                                # Get the latest two observations if available
                                latest_obs = path[-2:] if len(path) >= 2 else path
                                if latest_obs:
                                    latest_xs = [pos[0][0] for pos in latest_obs]
                                    latest_ys = [pos[0][1] for pos in latest_obs]
                                    ax.scatter(latest_xs, latest_ys, marker=blue_marker, color=color, s=50, label=f"{name} latest {red_name} obs")
                                    
                                    # If we have stored predictions for this red agent, visualize them
                                    if hasattr(agent_obj, 'predicted_positions') and red_name in agent_obj.predicted_positions:
                                        # Get all the predicted positions
                                        future_positions = agent_obj.predicted_positions[red_name]
                                        
                                        # Plot all predicted positions with decreasing opacity for further steps
                                        for step_idx, future_pos in enumerate(future_positions):
                                            steps_ahead = step_idx + 1  # 1-indexed for display
                                            # Draw the prediction with increasing transparency for steps further in future
                                            alpha = 1.0 - (steps_ahead-1) * 0.15  # decreasing alpha for further predictions
                                            ax.scatter(future_pos[0], future_pos[1], marker='*', s=100, 
                                                      color='lime' if steps_ahead == 1 else 'green', alpha=max(0.3, alpha),
                                                      label=f"{name} pred {red_name} +{steps_ahead}" if steps_ahead == 1 else "")
                                            
                                            # If it's the first prediction, draw a line from last observation to first prediction
                                            if steps_ahead == 1 and latest_obs:
                                                ax.plot([latest_obs[-1][0][0], future_pos[0]], 
                                                       [latest_obs[-1][0][1], future_pos[1]], 
                                                       'g--', alpha=0.7)

        ax.set_title(f"Episode: {self.current_episode_number}, Step: {self.steps}, Agent: {self.agent_selection}")
        
        # Only create a legend if there are labeled artists
        # This prevents 'No artists with labels found to put in legend' warnings
        if something_plotted:
            # Check if there are any labeled artists before creating legend
            handles, labels = ax.get_legend_handles_labels()
            if handles and labels:
                ax.legend(loc='upper right', fontsize='small')

        # Update the display
        plt.draw()
        plt.pause(0.001)

        if self.save_episode_gifs:
            try:
                buf = io.BytesIO()
                self.fig.savefig(buf, format='png')
                buf.seek(0)
                frame = imageio.imread(buf)
                self.episode_gif_frames.append(frame)
                # print(f"[GIF DEBUG] Frame added. Total frames: {len(self.episode_gif_frames)}")
                buf.close()
            except Exception as e:
                import traceback
                # print(f"[GIF ERROR] Error capturing frame for GIF: {e}")
                traceback.print_exc()

    def _save_current_episode_gif(self):
        """Saves the collected frames as a GIF for the current episode."""
        if not self.save_episode_gifs or not self.episode_gif_frames:
            # print(f"[GIF SAVE] Skipping GIF save: save_episode_gifs={self.save_episode_gifs}, frames={len(self.episode_gif_frames)}")
            return

        gif_filename = os.path.join(self.gif_dir, f"episode_{self.current_episode_number}.gif")
        # print(f"[GIF SAVE] Attempting to save GIF to: {gif_filename}")
        try:
            frame_duration = 1.0 / self.metadata.get("render_fps", 10) # seconds per frame
            imageio.mimsave(gif_filename, self.episode_gif_frames, duration=frame_duration*1000, loop=0) # duration in ms for mimsave
            # print(f"[GIF SAVE] GIF saved successfully: {gif_filename}")
        except Exception as e:
            import traceback
            # print(f"[GIF ERROR] Error saving GIF {gif_filename}: {e}")
            traceback.print_exc()

    def _get_agent_config(self, agent_name):
        """
        Get the configuration for a specific agent from the environment config.
        
        Args:
            agent_name (str): The name of the agent (e.g., 'red_0', 'blue_1')
            
        Returns:
            dict or None: The configuration dictionary for the agent if found, None otherwise
        """
        agent_type = None
        agent_index = None
        
        # Parse agent name to get type and index
        if '_' in agent_name:
            parts = agent_name.split('_')
            agent_type = parts[0]  # 'red' or 'blue'
            try:
                agent_index = int(parts[1])  # The numeric part (0, 1, 2, ...)
            except ValueError:
                return None
        
        if not agent_type or agent_index is None:
            return None
            
        # Look up in environment config
        agent_configs = self.env_config.get('agents', {}).get(f"{agent_type}_agents", [])
        
        # Find which agent group this agent belongs to
        current_count = 0
        for group_config in agent_configs:
            group_count = group_config.get('count', 0)
            if agent_index < current_count + group_count:
                # This agent belongs to this group
                return group_config
            current_count += group_count
            
        return None
    
    def close(self):
        """
        Close the environment, release resources.
        """
        # Save GIF of the last episode if frames were collected
        if self.save_episode_gifs and self.episode_gif_frames:
            self._save_current_episode_gif()
            self.episode_gif_frames = [] # Clear frames
            
        if self.screen is not None:
            pygame.quit()
            self.screen = None
