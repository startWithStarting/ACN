import os
import random
import io
import numpy as np
import pygame
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from gymnasium.spaces import Box, Dict as GymDict

from src.physics.engine import PhysicsEngine
from src.physics.fields import create_force_field
from src.physics.obstacles import create_obstacle
from src.utils.geometry import calculate_distance

# Constants
DEFAULT_GRID_HEIGHT = 80
DEFAULT_GRID_WIDTH = 100
DEFAULT_GIF_FIGSIZE = (10, 8)
RENDER_FPS = 10

# Reward parameters
REWARD_ATTRACTOR_DISTANCE = 50.0  # Distance from center to receive reward
REWARD_ATTRACTOR_TOLERANCE = 1.0  # Tolerance band width
BLUE_AGENT_PASSIVE_REWARD = 0.1   # Reward for blue agents when red doesn't score

# Rendering constants
RENDER_ATTRACTOR_RADIUS = 10.0
PYGAME_WINDOW_SCALE = 8
PYGAME_MAX_COORD = 30000
PYGAME_CIRCLE_RADIUS = 5
MATPLOTLIB_CIRCLE_RADIUS = 10.0 # For attractor visual
MATPLOTLIB_TEXT_OFFSET = 12

class ACNEnvironmentLogic:
    """
    Mixin class containing shared logic for both AEC and Parallel ACN environments.
    Expected to be mixed in with an Environment class that provides:
    - self.env_config
    - self.agent_objects
    - self.possible_agents
    - self.agents
    - self.terminations
    - self.truncations
    - self.metadata
    """

    def _init_common(self):
        """Initialize common attributes. Call this from the child class __init__."""
        # --- Grid Initialization ---
        self.grid_height = self.env_config.get("height", DEFAULT_GRID_HEIGHT)
        self.grid_width = self.env_config.get("width", DEFAULT_GRID_WIDTH)

        # --- GIF Saving Configuration ---
        self.save_episode_gifs = self.env_config.get("save_episode_gifs", True)
        self.gif_dir = None
        self.gif_figsize = self.env_config.get("gif_figsize", DEFAULT_GIF_FIGSIZE)
        self.episode_gif_frames = []
        self.current_episode_number = 0
        self.should_quit = False
        self.screen = None
        self.clock = None
        self.fig = None

        if self.save_episode_gifs:
            experiment_run_dir = self.env_config.get("experiment_results_dir")
            if experiment_run_dir:
                self.gif_dir = os.path.join(experiment_run_dir, "gifs")
                os.makedirs(self.gif_dir, exist_ok=True)
            else:
                self.save_episode_gifs = False

        # Initialize lists
        self.red_agents = []
        self.blue_agents = []
        self.active_red_agents = []
        self.active_blue_agents = []
        self.red_team_scores = []
        self.blue_team_scores = []
        self.step_counts = []

        # --- Physics Configuration ---
        raw_physics_config = self.env_config.get("physics", {})
        if isinstance(raw_physics_config, bool):
            raw_physics_config = {"enabled": raw_physics_config}
        elif raw_physics_config is None:
            raw_physics_config = {}
        self.physics_config = raw_physics_config
        self.use_physics = self.physics_config.get("enabled", True)
        self.physics_control_mode = self.physics_config.get("control_mode", "velocity")
        self.physics_dt = float(self.physics_config.get("dt", 1.0))
        self.physics_engine = None

    def _distribute_agents(self):
        """Categorize agents into red and blue lists."""
        self.red_agents = []
        self.blue_agents = []
        
        for agent in self.agent_objects.values():
            if hasattr(agent, 'agent_type'):
                agent_type = getattr(agent.agent_type, 'value', None)
                if agent_type == 'red':
                    self.red_agents.append(agent)
                elif agent_type == 'blue':
                    self.blue_agents.append(agent)
        
        self.active_red_agents = self.red_agents[:]
        self.active_blue_agents = self.blue_agents[:]

    def _create_observation_spaces(self):
        """Define observation spaces for all agents."""
        base = {
            'position': Box(
                low=np.array([0.0, 0.0], dtype=np.float32),
                high=np.array([self.grid_width, self.grid_height], dtype=np.float32),
                shape=(2,), dtype=np.float32),
            'grid_center': Box(
                low=np.array([self.grid_width / 2, self.grid_height / 2], dtype=np.float32),
                high=np.array([self.grid_width / 2, self.grid_height / 2], dtype=np.float32),
                shape=(2,), dtype=np.float32),
            'timestamp': Box(
                low=0.0, high=np.finfo(np.float32).max, shape=(), dtype=np.float32),
        }
        return {
            name: GymDict(base)
            for name in self.possible_agents
        }

    def _reset_agent_positions(self):
        """
        Reset agent positions based on their type and strategy.
        - Flocking Red Agents: Start in top-left corner.
        - Others: Random distribution.
        """
        for agent_obj in self.agent_objects.values():
            agent_obj.is_active = True
            
            # Default to full random distribution
            x = random.uniform(0, self.grid_width)
            y = random.uniform(0, self.grid_height)
            
            # Special initialization for Flocking strategy
            # Check if it's a Red Agent using Flocking
            is_red = getattr(agent_obj.agent_type, 'value', None) == 'red'
            strategy = getattr(agent_obj, 'strategy_type', None)
            
            if is_red and strategy == 'flocking':
                # Start in a cluster in the top-left corner (10% of grid size)
                # This ensures they start together to facilitate flocking
                x = random.uniform(0, self.grid_width * 0.1)
                y = random.uniform(0, self.grid_height * 0.1)
            
            agent_obj.x = x
            agent_obj.y = y
            agent_obj.speed = 0.0
            agent_obj.direction = (0.0, 0.0)

        self._init_physics_engine()

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

    def _get_observation(self, agent_name, step_count):
        """
        Return the observation dictionary for the given agent.
        """
        agent_obj = self.agent_objects[agent_name]
        agent_pos = np.array([agent_obj.x, agent_obj.y], dtype=np.float32)
        grid_center = np.array([self.grid_width / 2, self.grid_height / 2], dtype=np.float32)

        observation = {
            'position': agent_pos,
            'grid_center': grid_center,
            'timestamp': step_count
        }
        
        # If agent is a BlueAgent, add red_agents info
        if hasattr(agent_obj, 'agent_type') and getattr(agent_obj.agent_type, 'value', None) == 'blue':
            red_agents_info = {}
            for red_agent in self.active_red_agents:
                visible_info = self._visible_agent_info(agent_obj, red_agent)
                if visible_info is not None:
                    red_agents_info[red_agent.name] = visible_info
            observation['red_agents'] = red_agents_info
        
        # If agent is a RedAgent, add blue_agents info and red_teammates info
        elif hasattr(agent_obj, 'agent_type') and getattr(agent_obj.agent_type, 'value', None) == 'red':
            blue_agents_info = {}
            for blue_agent in self.active_blue_agents:
                visible_info = self._visible_agent_info(agent_obj, blue_agent)
                if visible_info is not None:
                    blue_agents_info[blue_agent.name] = visible_info
            observation['blue_agents'] = blue_agents_info
            
            red_teammates_info = {}
            for red_agent in self.active_red_agents:
                if red_agent.name == agent_name:
                    continue
                visible_info = self._visible_agent_info(agent_obj, red_agent)
                if visible_info is not None:
                    red_teammates_info[red_agent.name] = visible_info
            observation['red_teammates'] = red_teammates_info
        
        # Flocking parameters
        if hasattr(agent_obj, 'strategy_type') and agent_obj.strategy_type == 'flocking':
            agent_config = self._get_agent_config(agent_name)
            if agent_config:
                for param in ['cohesion_weight', 'alignment_weight', 'separation_weight', 
                             'separation_radius', 'inertia_weight', 'max_speed', 'max_force',
                             'wall_avoidance_weight', 'wall_detection_radius']:
                    if param in agent_config:
                        observation[param] = agent_config[param]
            
            # Pass current direction and speed for smoothing/physics
            if hasattr(agent_obj, 'direction'):
                observation['current_direction'] = agent_obj.direction
            if hasattr(agent_obj, 'speed'):
                observation['current_speed'] = agent_obj.speed

        return observation

    def _visible_agent_info(self, observer_obj, target_obj):
        """Return target info if it is visible to observer under detection radius."""
        if (
            not hasattr(observer_obj, "x")
            or not hasattr(observer_obj, "y")
            or observer_obj.x is None
            or observer_obj.y is None
            or not hasattr(target_obj, "x")
            or not hasattr(target_obj, "y")
            or target_obj.x is None
            or target_obj.y is None
        ):
            return None

        observer_pos = (observer_obj.x, observer_obj.y)
        target_pos = (target_obj.x, target_obj.y)
        distance = float(calculate_distance(observer_pos, target_pos))
        detection_radius = getattr(observer_obj, "detection_radius", None)
        if detection_radius is not None and distance > float(detection_radius):
            return None

        return {
            "position": target_pos,
            "distance": distance,
        }

    def _is_red_agent_detected(self, red_agent):
        """Check if a red agent is detected by any active blue agent."""
        if not hasattr(red_agent, 'x') or not hasattr(red_agent, 'y') or red_agent.x is None or red_agent.y is None:
            return False
            
        red_pos = (red_agent.x, red_agent.y)
        
        for blue_agent in self.active_blue_agents:
            if (not hasattr(blue_agent, 'is_active') or not blue_agent.is_active or
                not hasattr(blue_agent, 'is_within_detection_radius') or 
                not hasattr(blue_agent, 'x') or not hasattr(blue_agent, 'y') or 
                blue_agent.x is None or blue_agent.y is None):
                continue
                
            if blue_agent.is_within_detection_radius(red_pos):
                return True
        return False

    def _get_agent_config(self, agent_name):
        """Get the configuration for a specific agent from the environment config."""
        agent_type = None
        agent_index = None
        
        if '_' in agent_name:
            parts = agent_name.split('_')
            agent_type = parts[0]
            try:
                agent_index = int(parts[1])
            except ValueError:
                return None
        
        if not agent_type or agent_index is None:
            return None
            
        agent_configs = self.env_config.get('agents', {}).get(f"{agent_type}_agents", [])
        
        current_count = 0
        for group_config in agent_configs:
            group_count = group_config.get('count', 0)
            if agent_index < current_count + group_count:
                return group_config
            current_count += group_count
            
        return None

    def _init_physics_engine(self):
        """Create and populate the physics engine from environment config."""
        if not self.use_physics:
            self.physics_engine = None
            return

        self.physics_engine = PhysicsEngine(
            grid_width=float(self.grid_width),
            grid_height=float(self.grid_height),
            boundary_mode=self.physics_config.get("boundary_mode", "clamp"),
            default_drag=float(self.physics_config.get("default_drag", 0.0)),
            default_mass=float(self.physics_config.get("default_mass", 1.0)),
            default_max_speed=float(self.physics_config.get("default_max_speed", 10.0)),
            default_max_force=float(self.physics_config.get("default_max_force", 10.0)),
            default_turning_rate=self.physics_config.get("default_turning_rate", float("inf")),
            default_radius=float(self.physics_config.get("default_radius", 0.5)),
            default_restitution=float(self.physics_config.get("default_restitution", 0.8)),
            enable_collisions=bool(self.physics_config.get("enable_collisions", True)),
            enable_obstacles=bool(self.physics_config.get("enable_obstacles", True)),
        )

        for obstacle_config in self.physics_config.get("obstacles", []):
            self.physics_engine.add_obstacle(create_obstacle(obstacle_config))

        field_configs = self.physics_config.get(
            "force_fields",
            self.physics_config.get("fields", []),
        )
        for field_config in field_configs:
            self.physics_engine.add_force_field(create_force_field(field_config))

        for agent_obj in self.agent_objects.values():
            self._register_physics_body(agent_obj)

    def _register_physics_body(self, agent_obj):
        """Register a single agent as a physics body."""
        if self.physics_engine is None:
            return

        position = np.array([agent_obj.x, agent_obj.y], dtype=np.float32)
        velocity = self._agent_velocity(agent_obj)
        self.physics_engine.register_body(
            agent_obj.name,
            position=position,
            velocity=velocity,
            **self._physics_body_kwargs(agent_obj),
        )

    def _physics_body_kwargs(self, agent_obj):
        """Build per-body physics options from global, type, and group config."""
        agent_type = getattr(agent_obj.agent_type, "value", None)
        body_config = {
            "mass": self.physics_config.get("agent_mass", self.physics_config.get("default_mass", 1.0)),
            "max_speed": self.physics_config.get(
                "agent_max_speed",
                self.physics_config.get("default_max_speed", 10.0),
            ),
            "max_force": self.physics_config.get(
                "agent_max_force",
                self.physics_config.get("default_max_force", 10.0),
            ),
            "drag": self.physics_config.get("agent_drag", self.physics_config.get("default_drag", 0.0)),
            "turning_rate": self.physics_config.get(
                "agent_turning_rate",
                self.physics_config.get("default_turning_rate", float("inf")),
            ),
            "radius": self.physics_config.get(
                "agent_radius",
                self.physics_config.get("default_radius", 0.5),
            ),
            "restitution": self.physics_config.get(
                "agent_restitution",
                self.physics_config.get("default_restitution", 0.8),
            ),
        }

        type_config = self.physics_config.get(agent_type, {})
        if isinstance(type_config, dict):
            body_config.update(type_config)

        group_config = self._get_agent_config(agent_obj.name) or {}
        group_physics = group_config.get("physics", {})
        if isinstance(group_physics, dict):
            body_config.update(group_physics)

        return body_config

    def _agent_velocity(self, agent_obj):
        """Return an agent's current velocity vector from direction and speed."""
        direction = getattr(agent_obj, "direction", None)
        speed = getattr(agent_obj, "speed", 0.0)
        if direction is None:
            return np.zeros(2, dtype=np.float32)
        try:
            direction_arr = np.asarray(direction, dtype=np.float32).reshape(-1)
            speed_value = float(np.asarray(speed, dtype=np.float32).reshape(-1)[0])
        except (TypeError, ValueError, IndexError):
            return np.zeros(2, dtype=np.float32)
        if direction_arr.shape[0] < 2:
            return np.zeros(2, dtype=np.float32)
        return direction_arr[:2] * speed_value

    def _parse_movement_action(self, action):
        """Parse the common direction/speed action format into a movement vector."""
        if not isinstance(action, dict) or "direction" not in action or "speed" not in action:
            return None

        try:
            direction = np.asarray(action["direction"], dtype=np.float32).reshape(-1)
            speed = float(np.asarray(action["speed"], dtype=np.float32).reshape(-1)[0])
        except (TypeError, ValueError, IndexError):
            return None

        if direction.shape[0] < 2 or not np.isfinite(speed):
            return None

        movement = direction[:2] * speed
        if not np.all(np.isfinite(movement)):
            return None

        return direction[:2], speed, movement.astype(np.float32)

    def _begin_physics_step(self):
        """Prepare velocity-controlled bodies before actions are applied."""
        if (
            not self.use_physics
            or self.physics_engine is None
            or self.physics_control_mode != "velocity"
        ):
            return

        for agent_name in self.agents:
            body = self.physics_engine.get_body(agent_name)
            if body is not None:
                body.velocity[:] = 0.0

    def _advance_physics(self):
        """Step physics and copy body state back onto agent objects."""
        if not self.use_physics or self.physics_engine is None:
            return

        self.physics_engine.step(dt=self.physics_dt)
        self._sync_agents_from_physics()

    def _sync_agents_from_physics(self):
        """Copy physics positions and velocities back to agent objects."""
        if self.physics_engine is None:
            return

        for agent_name, agent_obj in self.agent_objects.items():
            position = self.physics_engine.get_position(agent_name)
            velocity = self.physics_engine.get_velocity(agent_name)
            if position is None or velocity is None:
                continue

            agent_obj.x = float(position[0])
            agent_obj.y = float(position[1])
            speed = float(np.linalg.norm(velocity))
            agent_obj.speed = speed
            if speed > 1e-6:
                agent_obj.direction = tuple((velocity / speed).astype(np.float32))
            else:
                agent_obj.direction = (0.0, 0.0)

    def _apply_action(self, agent_obj, action, advance_physics=True):
        """
        Apply movement action to an agent.
        Returns check_detection (bool) - True if we should check for detection/rewards.
        """
        # We need to import AgentType here or check strings if avoiding circular imports is tricky
        # Checking string value of enum is safer for mixins
        agent_type = getattr(agent_obj.agent_type, 'value', None)
        
        if agent_type in ['red', 'blue']:
            parsed_action = self._parse_movement_action(action)
            if parsed_action is None:
                return False

            direction, speed, movement = parsed_action

            if self.use_physics:
                if self.physics_engine is None:
                    self._init_physics_engine()
                body = self.physics_engine.get_body(agent_obj.name)
                if body is None:
                    self._register_physics_body(agent_obj)
                    body = self.physics_engine.get_body(agent_obj.name)
                if body is None:
                    return False

                if advance_physics:
                    self._begin_physics_step()

                if self.physics_control_mode == "force":
                    self.physics_engine.apply_force(agent_obj.name, movement)
                else:
                    body.velocity = movement

                if advance_physics:
                    self._advance_physics()
                else:
                    agent_obj.direction = tuple(direction)
                    agent_obj.speed = float(speed)

                return True

            # Calculate movement vector (using floats for sub-pixel precision)
            movement_x = float(movement[0])
            movement_y = float(movement[1])

            # Update position
            agent_obj.x += movement_x
            agent_obj.y += movement_y

            # Boundary check
            agent_obj.x = max(0, min(self.grid_width, agent_obj.x))
            agent_obj.y = max(0, min(self.grid_height, agent_obj.y))

            # Update internal state
            agent_obj.direction = tuple(direction)
            agent_obj.speed = float(speed)

            return True # Valid action processed
        return False # Not a movable agent type

    def _calculate_reward(self, agent_name, agent_obj):
        """
        Calculate reward for a single agent after they moved.
        Returns: (reward, red_scored)
        """
        reward = 0
        red_scored = False
        
        agent_type = getattr(agent_obj.agent_type, 'value', None)
        
        if agent_type == 'red':
            # Calculate distance to center
            center = (self.grid_width / 2, self.grid_height / 2)
            distance_to_center = np.sqrt((agent_obj.x - center[0])**2 + 
                                       (agent_obj.y - center[1])**2)
            
            # Check if within tolerance of the attractor distance
            if abs(distance_to_center - REWARD_ATTRACTOR_DISTANCE) <= REWARD_ATTRACTOR_TOLERANCE:
                # Check if not detected by any blue agent
                if not self._is_red_agent_detected(agent_obj):
                    reward = 1
                    red_scored = True
        
        return reward, red_scored

    # --- Rendering Methods ---
    def render(self):
        if self.render_mode is None:
            return

        if self.render_mode in ["ansi", "human_text"]:
            return self._render_text()
        elif self.render_mode in ["human", "human_matplotlib"]:
            return self._render_matplotlib()
        elif self.render_mode == "human_matplotlib_pred":
            return self._render_matplotlib(show_predictions=True)
        elif self.render_mode == "human_pygame":
            return self._render_pygame()
        else:
            raise NotImplementedError(f"Render mode '{self.render_mode}' not supported.")

    def _render_pygame(self):
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            # Dynamic scale: Target max window dimension ~800px
            max_dim = max(self.grid_width, self.grid_height)
            target_size = 800
            self.window_scale = max(1, target_size // max_dim)
            if self.window_scale < 1: self.window_scale = 1 # Safety
            
            self.window_width = int(self.grid_width * self.window_scale)
            self.window_height = int(self.grid_height * self.window_scale)
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))
            pygame.display.set_caption("ACN Simulation")
            self.clock = pygame.time.Clock()

        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))

        # Draw attractor zone
        center_x = int((self.grid_width / 2) * self.window_scale)
        center_y = int((self.grid_height / 2) * self.window_scale)
        attractor_radius = int(RENDER_ATTRACTOR_RADIUS * self.window_scale)
        pygame.draw.circle(self.screen, (255, 165, 0), (center_x, center_y), attractor_radius, 2)

        # Draw agents
        for name, agent_obj in self.agent_objects.items():
            is_done = self.terminations.get(name, True) or self.truncations.get(name, True)
            if not is_done and hasattr(agent_obj, 'x') and hasattr(agent_obj, 'y') \
               and agent_obj.x is not None and agent_obj.y is not None:
                
                x = int(agent_obj.x * self.window_scale)
                y = int(agent_obj.y * self.window_scale)
                
                color = (128, 128, 128)
                if hasattr(agent_obj, 'agent_type') and hasattr(agent_obj.agent_type, 'value'):
                    if agent_obj.agent_type.value == 'blue':
                        color = (0, 0, 255)
                    elif agent_obj.agent_type.value == 'red':
                        color = (255, 0, 0)
                
                pygame.draw.circle(self.screen, color, (x, y), PYGAME_CIRCLE_RADIUS)
                
                # Draw predictions for Blue agents
                if agent_obj.agent_type.value == 'blue' and hasattr(agent_obj, 'predicted_positions'):
                     for red_name, predictions in agent_obj.predicted_positions.items():
                        for i, pred_pos in enumerate(predictions):
                            px = int(pred_pos[0] * self.window_scale)
                            py = int(pred_pos[1] * self.window_scale)
                            
                            max_coord = PYGAME_MAX_COORD
                            px = max(-max_coord, min(max_coord, px))
                            py = max(-max_coord, min(max_coord, py))

                            try:
                                pygame.draw.circle(self.screen, (0, 255, 0), (px, py), 3)
                            except TypeError:
                                pass

        pygame.display.flip()
        self.clock.tick(self.metadata.get("render_fps", RENDER_FPS))
        
        if self.save_episode_gifs:
            frame = np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))
            self.episode_gif_frames.append(frame)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.should_quit = True
                pygame.quit()
                self.screen = None

    def _render_text(self):
        pass # Optional text rendering

    def _render_matplotlib(self, show_predictions=False):
        if not hasattr(self, 'fig') or self.fig is None:
            self.fig = plt.figure(figsize=(self.gif_figsize[0], self.gif_figsize[1] + 2))
            gs = self.fig.add_gridspec(2, 1, height_ratios=[3, 1])
            self.ax_env = self.fig.add_subplot(gs[0])
            self.ax_score = self.fig.add_subplot(gs[1])
            plt.ion()
            self.fig.show()
        
        self.ax_env.clear()
        self.ax_score.clear()
        
        self.ax_env.set_xlim(0, self.grid_width)
        self.ax_env.set_ylim(0, self.grid_height)
        self.ax_env.set_aspect('equal', adjustable='box')
        
        if len(self.step_counts) > 0:
            self.ax_score.plot(self.step_counts, self.red_team_scores, 'r-', label='Red Team')
            self.ax_score.plot(self.step_counts, self.blue_team_scores, 'b-', label='Blue Team')
            self.ax_score.set_xlabel('Time Step')
            self.ax_score.set_ylabel('Avg Score')
            self.ax_score.legend()
            self.ax_score.grid(True)
            
            current_red = self.red_team_scores[-1] if self.red_team_scores else 0
            current_blue = self.blue_team_scores[-1] if self.blue_team_scores else 0
            score_text = f'Red: {current_red:.2f}  |  Blue: {current_blue:.2f}'
            self.ax_score.set_title(f'Team Scores (Current: {score_text})')
            
        self.fig.tight_layout()
        ax = self.ax_env
        
        center_x, center_y = self.grid_width / 2, self.grid_height / 2
        attractor_circle = plt.Circle((center_x, center_y), MATPLOTLIB_CIRCLE_RADIUS, fill=False, color='orange', linestyle='--', linewidth=2)
        ax.add_patch(attractor_circle)
        ax.text(center_x, center_y - MATPLOTLIB_TEXT_OFFSET, 'Attractor Zone', color='orange', ha='center', fontsize=8)
        
        something_plotted = False
        for name, agent_obj in self.agent_objects.items():
            is_done = self.terminations.get(name, True) or self.truncations.get(name, True)
            if not is_done and hasattr(agent_obj, 'x') and hasattr(agent_obj, 'y') \
               and agent_obj.x is not None and agent_obj.y is not None:
                color = 'grey'
                agent_number = ''
                if hasattr(agent_obj, 'agent_type') and hasattr(agent_obj.agent_type, 'value'):
                    agent_type_val = agent_obj.agent_type.value
                    if agent_type_val == 'blue':
                        color = 'blue'
                    elif agent_type_val == 'red':
                        color = 'red'
                    if '_' in name:
                        agent_number = name.split('_')[-1]
                    else:
                        agent_number = name
                        
                ax.text(agent_obj.x, agent_obj.y, agent_number, color=color, fontsize=14, fontweight='bold', ha='center', va='center')
                something_plotted = True

                if agent_type_val == 'blue' and hasattr(agent_obj, 'get_observed_paths'):
                    observed_paths = agent_obj.get_observed_paths()
                    blue_marker = 'x'
                    for red_name, path in observed_paths.items():
                        if path:
                            if not show_predictions:
                                xs = [pos[0][0] for pos in path]
                                ys = [pos[0][1] for pos in path]
                                ax.scatter(xs, ys, marker=blue_marker, color=color, alpha=0.6, label=f"{name} sees {red_name}")
                            else:
                                latest_obs = path[-2:] if len(path) >= 2 else path
                                if latest_obs:
                                    latest_xs = [pos[0][0] for pos in latest_obs]
                                    latest_ys = [pos[0][1] for pos in latest_obs]
                                    ax.scatter(latest_xs, latest_ys, marker=blue_marker, color=color, s=50, label=f"{name} latest {red_name} obs")
                                    
                                    if hasattr(agent_obj, 'predicted_positions') and red_name in agent_obj.predicted_positions:
                                        future_positions = agent_obj.predicted_positions[red_name]
                                        for step_idx, future_pos in enumerate(future_positions):
                                            steps_ahead = step_idx + 1
                                            alpha = 1.0 - (steps_ahead-1) * 0.15
                                            ax.scatter(future_pos[0], future_pos[1], marker='*', s=100, 
                                                      color='lime' if steps_ahead == 1 else 'green', alpha=max(0.3, alpha),
                                                      label=f"{name} pred {red_name} +{steps_ahead}" if steps_ahead == 1 else "")
                                            
                                            if steps_ahead == 1 and latest_obs:
                                                ax.plot([latest_obs[-1][0][0], future_pos[0]], 
                                                       [latest_obs[-1][0][1], future_pos[1]], 
                                                       'g--', alpha=0.7)

        title_suffix = ""
        if hasattr(self, 'agent_selection'):
            title_suffix = f", Agent: {self.agent_selection}"
        ax.set_title(f"Episode: {self.current_episode_number}, Step: {self.steps}{title_suffix}")
        
        if something_plotted:
            handles, labels = ax.get_legend_handles_labels()
            if handles and labels:
                ax.legend(loc='upper right', fontsize='small')

        plt.draw()
        plt.pause(0.001)

        if self.save_episode_gifs:
            try:
                buf = io.BytesIO()
                self.fig.savefig(buf, format='png')
                buf.seek(0)
                frame = imageio.imread(buf)
                self.episode_gif_frames.append(frame)
                buf.close()
            except Exception:
                pass

    def _save_current_episode_gif(self):
        if not self.save_episode_gifs or not self.episode_gif_frames:
            return

        gif_filename = os.path.join(self.gif_dir, f"episode_{self.current_episode_number}.gif")
        try:
            frame_duration = 1.0 / self.metadata.get("render_fps", 10)
            imageio.mimsave(gif_filename, self.episode_gif_frames, duration=frame_duration*1000, loop=0)
        except Exception:
            pass
