import functools
import random
import os
import io
# Ensure imageio is installed: pip install imageio
import imageio.v2 as imageio # Using v2 for modern API

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
        "render_modes": ["human", "human_matplotlib", "human_matplotlib_pred"],
        "name": "communicating_agents_v0",
        "is_parallelizable": False, # Usually True if step() doesn't depend on agent order
        "render_fps": 100,
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

        if self.save_episode_gifs:
            # The specific directory for this experiment's run, passed from main.py
            # This path will be like "results/experiment_name_timestamp"
            experiment_run_dir = self.env_config.get("experiment_results_dir")

            if experiment_run_dir:
                self.gif_dir = os.path.join(experiment_run_dir, "gifs")
                os.makedirs(self.gif_dir, exist_ok=True)
            else:
                print("Warning: 'save_episode_gifs' is True, but 'experiment_results_dir' was not provided in env_config. Disabling GIF saving.")
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
        self.possible_agents = [agent.name for agent in agents] # List of agent names
        self.agent_name_mapping = {i: name for i, name in enumerate(self.possible_agents)}

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
    

    def observe(self, agent):
        """
        Return the observation dictionary for the given agent.
        Contains the agent's current position and the grid center coordinates.
        """
        agent_obj = self.agent_objects[agent]
        agent_pos = np.array([agent_obj.x, agent_obj.y], dtype=np.float32)
        grid_center = np.array([self.grid_width / 2, self.grid_height / 2], dtype=np.float32)

        # Build base observation
        observation = {
            'position': agent_pos,
            'grid_center': grid_center
        }

        # If agent is a BlueAgent, add red_agents info and timestamp
        agent_obj = self.agent_objects[agent]
        if hasattr(agent_obj, 'agent_type') and getattr(agent_obj.agent_type, 'value', None) == 'blue':
            # Gather red agent positions
            red_agents_info = {}
            for red_name, red_obj in self.agent_objects.items():
                if hasattr(red_obj, 'agent_type') and getattr(red_obj.agent_type, 'value', None) == 'red':
                    if hasattr(red_obj, 'x') and hasattr(red_obj, 'y') and red_obj.x is not None and red_obj.y is not None:
                        red_agents_info[red_name] = {'position': (red_obj.x, red_obj.y)}
            observation['red_agents'] = red_agents_info
            observation['timestamp'] = self.steps

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

        self.episode_gif_frames = [] # Clear frames for the new episode
        self._agent_selector.reinit(self.agents) # Reinitialize selector
        self.agent_selection = self._agent_selector.reset() # Get first agent

        # Reset internal state
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.steps = 0
        self.current_episode_number +=1 # Increment for the new episode

        # Reset agent object states if necessary
        for agent_obj in self.agent_objects.values():
            agent_obj.is_active = True # Example reset
            # Re-initialize positions on reset
            agent_obj.x = random.uniform(0, self.grid_width)
            agent_obj.y = random.uniform(0, self.grid_height)

    def step(self, action):
        """
        Apply the action for the current agent_selection.
        Updates rewards, terminations, truncations, and selects the next agent.
        """
        agent_name = self.agent_selection

        if self.terminations[agent_name] or self.truncations[agent_name]:
            # If agent is done, handle potential cleanup and select next agent
            self._was_done_step(action) # PZ utility
            return

        # --- Action Processing ---
        # Get the agent object
        current_agent_obj = self.agent_objects[agent_name]

        # Process the action based on agent type
        if current_agent_obj.agent_type == AgentType.RED:
            # For RED agents, handle the composite action (direction + speed)
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
                
                # Simple reward for now (can be customized based on game mechanics)
                reward = 0
            else:
                # Invalid action format for RED agent
                reward = -1  # Penalty for invalid action
        else:
            # For non-RED agents, use the existing placeholder logic
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

        # Call render if a human-viewable mode is active
        if self.render_mode in ["human", "human_matplotlib", "human_matplotlib_pred"]:
            self.render()

    def render(self):
        """
        Renders the environment.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        # Add custom rendering modes as needed
        if self.render_mode == "ansi" or self.render_mode == "human_text":
            return self._render_text()
        elif self.render_mode == "human" or self.render_mode == "human_matplotlib":
            return self._render_matplotlib()
        elif self.render_mode == "human_matplotlib_pred":
            return self._render_matplotlib(show_predictions=True)
        else:
            raise NotImplementedError(f"Render mode '{self.render_mode}' not supported.")

    def _render_text(self):
        """Text-based rendering for debugging."""
        print("\n--- Rendering Frame ---")
        print(f"Step: {self.steps}")
        print(f"Grid Dimensions: W={self.grid_width}, H={self.grid_height}")
        print(f"Current Agent: {self.agent_selection}")
        print("Agent States:")
        for name in self.possible_agents:  # Iterate in defined order
            agent_obj = self.agent_objects[name]
            state = "Done" if (self.terminations[name] or self.truncations[name]) else "Active"
            position_str = f", Pos: ({agent_obj.x:.2f}, {agent_obj.y:.2f})" if agent_obj.x is not None and agent_obj.y is not None else ""
            print(f"  - {name}: Reward={self._cumulative_rewards[name]:.2f}, State={state}{position_str}")
        print("-----------------------\n")

    def _render_matplotlib(self, show_predictions=False):
        """Matplotlib rendering for visualization."""
        # This function will now handle both displaying and/or saving frames for a GIF.
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=self.gif_figsize)
        ax.set_xlim(0, self.grid_width)
        ax.set_ylim(0, self.grid_height)
        ax.set_aspect('equal', adjustable='box')

        something_plotted = False
        for name, agent_obj in self.agent_objects.items():
            is_done = self.terminations.get(name, True) or self.truncations.get(name, True)
            if not is_done and hasattr(agent_obj, 'x') and hasattr(agent_obj, 'y') \
               and agent_obj.x is not None and agent_obj.y is not None:
                
                color = 'grey' # Default color
                if hasattr(agent_obj, 'agent_type') and hasattr(agent_obj.agent_type, 'value'):
                    agent_type_val = agent_obj.agent_type.value
                    if agent_type_val == 'blue':
                        color = 'blue'
                    elif agent_type_val == 'red':
                        color = 'red'

                ax.plot(agent_obj.x, agent_obj.y, marker='o', markersize=8, color=color, label=name)
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
        if something_plotted:
            ax.legend(loc='upper right', fontsize='small')

        if self.save_episode_gifs:
            try:
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                frame = imageio.imread(buf)
                self.episode_gif_frames.append(frame)
                print(f"[GIF DEBUG] Frame added. Total frames: {len(self.episode_gif_frames)}")
                buf.close()
            except Exception as e:
                import traceback
                print(f"[GIF ERROR] Error capturing frame for GIF: {e}")
                traceback.print_exc()
            finally:
                plt.close(fig) # Close the figure to free memory
        else:
            # Original behavior if not saving GIFs: pause and then close the figure
            plt.pause(0.01) # Reduced pause time for faster human rendering
            plt.close(fig) # Close the figure

    def _save_current_episode_gif(self):
        """Saves the collected frames as a GIF for the current episode."""
        if not self.save_episode_gifs or not self.episode_gif_frames:
            print(f"[GIF SAVE] Skipping GIF save: save_episode_gifs={self.save_episode_gifs}, frames={len(self.episode_gif_frames)}")
            return

        gif_filename = os.path.join(self.gif_dir, f"episode_{self.current_episode_number}.gif")
        print(f"[GIF SAVE] Attempting to save GIF to: {gif_filename}")
        try:
            frame_duration = 1.0 / self.metadata.get("render_fps", 10) # seconds per frame
            imageio.mimsave(gif_filename, self.episode_gif_frames, duration=frame_duration*1000, loop=0) # duration in ms for mimsave
            print(f"[GIF SAVE] GIF saved successfully: {gif_filename}")
        except Exception as e:
            import traceback
            print(f"[GIF ERROR] Error saving GIF {gif_filename}: {e}")
            traceback.print_exc()

    def close(self):
        """
        Close the environment, release resources.
        """
        # Save GIF of the last episode if frames were collected
        if self.save_episode_gifs and self.episode_gif_frames:
            self._save_current_episode_gif()
            self.episode_gif_frames = [] # Clear frames
