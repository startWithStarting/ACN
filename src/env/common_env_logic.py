import os
import random
import io
import numpy as np
import pygame
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from gymnasium.spaces import Box, Dict as GymDict

from src.agents.action_spaces import (
    DEFAULT_MAX_SPEED,
    ActionSpaceConfig,
    build_discrete_movement_spec,
)
from src.communication.config import parse_communication_config
from src.communication.registry import create_communication_plan
from src.communication.runtime import CommunicationRuntime
from src.communication.sources import create_message_source
from src.communication.types import CommunicationResult
from src.env.sensors import ObservationConfig, build_contact_reports
from src.physics.engine import PhysicsEngine
from src.physics.fields import create_force_field
from src.physics.obstacles import create_obstacle
from src.env.rewards import (
    blue_benchmark_reward,
    build_detection_state,
    parse_reward_settings,
    red_benchmark_reward,
    ring_potential,
)
from src.utils.geometry import calculate_distance
from src.utils.logger import get_logger

logger = get_logger("acn.env")

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

        # --- Action-Space Configuration ---
        # Default (absent block) is the legacy continuous Dict spec.
        self.action_space_config = ActionSpaceConfig.from_dict(
            self.env_config.get("action_space")
        )
        self.discrete_actions = self.action_space_config.is_discrete
        # Per-agent discrete decode specs, built lazily from each agent's
        # resolved max_speed (per-team caps may differ).
        self._discrete_movement_specs = {}

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

        # --- Observation / Sensor Configuration ---
        # environment.observation gates the blue sensor model; the default
        # ("legacy") reproduces historical observations exactly.
        self.observation_config = ObservationConfig.from_dict(self.env_config.get("observation"))
        # Privileged per-report red identities from the latest bearing-only
        # blue observations: {blue_name: {report_index: red_name}}. Exposed
        # only through the step/reset infos dict, never through the
        # policy-visible observation.
        self._ground_truth_contacts = {}

        # --- Reward Configuration (benchmark modes are opt-in; default is legacy) ---
        self.reward_settings = parse_reward_settings(self.env_config.get("reward"))
        # Per-episode state owned by the env for the benchmark reward modes.
        self._red_pin_streaks = {}
        self._red_ring_potentials = {}
        self._initial_red_count = 0

        # --- Communication Configuration (config-gated; defaults to disabled) ---
        # An absent environment.communication block, enabled: false, or scheme
        # "none" all leave the runtime at None: the disabled path adds no
        # observation keys and changes no behavior.
        self.communication_config = parse_communication_config(
            self.env_config.get("communication")
        )
        self.communication_runtime = None
        # The compiled plan of an active scheme (None when disabled). For a
        # DIFFERENTIABLE scheme (e.g. multihop_gnn) this is the only
        # communication state the env keeps: trainers read the plan's
        # topology/round definitions off it, but the env itself never
        # executes the rounds.
        self.communication_plan = None
        self._communication_source = None
        # Trace records of the most recent communication step (plain dicts from
        # src.communication.tracing), for recorders and analysis code.
        self.last_communication_trace_records = []
        if self.communication_config.is_active:
            plan = create_communication_plan(self.communication_config)
            self.communication_plan = plan
            if plan.differentiable:
                # Differentiable communication MUST run in the policy/trainer
                # forward path, never inside env.step(): the env skips
                # communication execution entirely, so observations carry NO
                # "communication" key for this scheme and the policy computes
                # communication itself (docs/communication_implementation_plan
                # .md, "Risks: Environment And Policy Ownership Is Ambiguous").
                logger.info(
                    "Communication scheme '{}' is differentiable: env-side execution "
                    "is skipped; the policy/trainer computes communication.",
                    self.communication_config.scheme,
                )
            else:
                self.communication_runtime = CommunicationRuntime(
                    plan, scheme_name=self.communication_config.scheme
                )
                self._communication_source = create_message_source(
                    self.communication_config.payload
                )

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
        # Drop privileged sensor state from any previous episode.
        self._ground_truth_contacts = {}

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
        self._reset_benchmark_reward_state()

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
        
        # If agent is a BlueAgent, add red-contact info in the configured sensor mode
        if hasattr(agent_obj, 'agent_type') and getattr(agent_obj.agent_type, 'value', None) == 'blue':
            if self.observation_config.bearing_only:
                self._build_blue_bearing_observation(agent_name, agent_obj, observation, step_count)
            else:
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

    def _build_blue_bearing_observation(self, agent_name, agent_obj, observation, step_count):
        """Attach bearing-only contact reports to a blue agent's observation.

        Adds the policy-visible 'contact_reports' list (anonymous, sorted by
        bearing angle) and records the privileged report-index -> red-name
        mapping for the infos dict. When the environment grants scripted
        controllers privileged access, the legacy red dict is also attached
        as 'privileged_red_agents' (trace/scripted-only; learned policies
        must never consume it).
        """
        visible_reds = []
        privileged_info = {}
        for red_agent in self.active_red_agents:
            visible_info = self._visible_agent_info(agent_obj, red_agent)
            if visible_info is not None:
                visible_reds.append((red_agent.name, visible_info["position"]))
                privileged_info[red_agent.name] = visible_info

        reports, ground_truth = build_contact_reports(
            agent_name,
            (agent_obj.x, agent_obj.y),
            visible_reds,
            step_count,
        )
        observation['contact_reports'] = reports
        self._ground_truth_contacts[agent_name] = ground_truth
        if self.observation_config.scripted_blue_privileged:
            observation['privileged_red_agents'] = privileged_info

    def _attach_ground_truth_infos(self, infos):
        """Expose per-report red identities through infos in bearing-only mode.

        For every blue agent with a freshly built bearing-only observation,
        sets infos[blue_name]['ground_truth_contacts'] to the report-index ->
        red-name mapping aligned with that observation's report order. No-op
        in legacy mode.
        """
        if not self.observation_config.bearing_only:
            return
        for agent_name, contacts in self._ground_truth_contacts.items():
            agent_infos = infos.get(agent_name)
            if agent_infos is not None:
                agent_infos['ground_truth_contacts'] = contacts

    # --- Communication Phase ---
    def _run_communication_phase(self):
        """Run the R communication rounds on the frozen pre-move state.

        Implements the runtime model of docs/communication_implementation_plan.md
        ("Environment Integration"): rebuilds the current local observations
        (the ones the caller just acted on; positions have not moved yet),
        derives the source outboxes from them (``u_i(t) = g_comm(o_i(t))``),
        and runs the fixed-round scheduler over the same-team radius graph
        frozen at the current positions. Movement and physics are applied by
        the caller afterwards, so delivered messages become available to the
        NEXT decision. The step's trace records are kept on
        ``self.last_communication_trace_records``.

        Returns:
            The step's :class:`~src.communication.types.CommunicationResult`.
        """
        step = self.steps
        active_names = [
            name for name in self.agents
            if getattr(self.agent_objects[name], "is_active", True)
        ]
        local_observations = {
            name: self._get_observation(name, step) for name in active_names
        }
        outboxes = self._communication_source.build_outboxes(local_observations, step)
        result = self.communication_runtime.run_step(
            local_observations=local_observations,
            outboxes=outboxes,
            step=step,
            agents=[self.agent_objects[name] for name in active_names],
        )
        self.last_communication_trace_records = list(
            result.diagnostics.get("trace_records", [])
        )
        return result

    def _empty_communication_result(self):
        """Build the explicit empty communication result (reset baseline).

        Every current agent maps to an explicit empty view with the same shape
        as a delivering step and zero messages, per the decision log's
        "no-communication baseline receives an explicit empty view" rule.
        """
        return CommunicationResult.empty(
            agent_ids=tuple(self.agents),
            payload_dim=self.communication_config.payload.dimension,
        )

    def _communication_view(self, agent_name, result):
        """Build one agent's policy-facing communication view.

        Args:
            agent_name: The receiving agent's name.
            result: The step's communication result (or the explicit empty
                result at reset).

        Returns:
            Dict with ``scheme`` (the configured scheme name), ``inbox`` (the
            agent's entry of ``result.agent_output`` — for ``one_hop_direct`` a
            preserved-inbox ``EdgeMessageBatch``; an empty tuple when the agent
            was off the graph or at reset), ``agent_ids`` (node-index ->
            agent-id decode table for the inbox index fields), ``cache`` (the
            agent's persistent ``MessageCache`` handle), and ``cache_window``
            (its configured C window in rounds).
        """
        graph = result.diagnostics.get("graph")
        agent_ids = tuple(graph.agent_ids) if graph is not None else tuple(result.agent_output)
        return {
            "scheme": self.communication_config.scheme,
            "inbox": result.agent_output.get(agent_name, ()),
            "agent_ids": agent_ids,
            "cache": self.communication_runtime.cache_for(agent_name),
            "cache_window": self.communication_runtime.plan.cache_window,
        }

    def _attach_communication_views(self, observations, result):
        """Attach per-agent communication views under the 'communication' key.

        Only called when communication is enabled; the disabled path never
        adds the key, so legacy observation dicts stay structurally identical.
        """
        for agent_name, observation in observations.items():
            observation["communication"] = self._communication_view(agent_name, result)

    def _attach_communication_infos(self, infos, result):
        """Surface per-step delivery counts in ``infos[agent]['communication']``.

        Args:
            infos: Mutable per-agent infos dict for this step; updated in place.
            result: The step's communication result.
        """
        delivered = result.delivered_messages
        graph = result.diagnostics.get("graph")
        counts = {}
        if graph is not None and delivered.num_messages > 0:
            for receiver in delivered.receiver_index.tolist():
                receiver_name = graph.agent_ids[receiver]
                counts[receiver_name] = counts.get(receiver_name, 0) + 1
        for agent_name, agent_infos in infos.items():
            agent_infos["communication"] = {
                "messages_delivered": counts.get(agent_name, 0),
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

    def _movement_spec_for(self, agent_obj):
        """Return (building if needed) the discrete decode spec for one agent."""
        spec = self._discrete_movement_specs.get(agent_obj.name)
        if spec is None:
            max_speed = float(getattr(agent_obj, "max_speed", DEFAULT_MAX_SPEED))
            spec = build_discrete_movement_spec(self.action_space_config, max_speed)
            self._discrete_movement_specs[agent_obj.name] = spec
        return spec

    def _parse_movement_action(self, action, agent_obj=None):
        """Parse an action into (direction, speed, movement) or None if invalid.

        Dict actions ({'direction', 'speed'}, the scripted-controller format) are
        always accepted. Integer actions are decoded through the discrete movement
        spec when ``environment.action_space.type`` is ``discrete``; in continuous
        mode they are rejected with an actionable error.

        Args:
            action: The raw action (dict, or integer index in discrete mode).
            agent_obj: The acting agent; required to decode integer actions
                because the discrete spec depends on the agent's max_speed.

        Returns:
            Tuple of (direction ndarray, speed float, movement float32 ndarray),
            or None when the action is not a usable movement command.

        Raises:
            ValueError: If an integer action arrives while the environment is
                configured for continuous actions, or a discrete index is out
                of range.
        """
        if not isinstance(action, (bool, np.bool_)) and isinstance(action, (int, np.integer)):
            if not self.discrete_actions:
                raise ValueError(
                    "Received integer action {!r}, but environment.action_space.type is "
                    "'continuous'. Continuous mode expects dict actions with 'direction' "
                    "and 'speed'; set environment.action_space.type: 'discrete' to use "
                    "integer movement actions.".format(action)
                )
            if agent_obj is None:
                return None
            spec = self._movement_spec_for(agent_obj)
            direction, speed = spec.decode(int(action))
            movement = (direction * speed).astype(np.float32)
            return direction, speed, movement

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
            parsed_action = self._parse_movement_action(action, agent_obj)
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

    def _reset_benchmark_reward_state(self):
        """Reset the per-episode reward state the env owns for benchmark modes.

        Clears the pinned-streak counters, re-anchors the ring potentials to the
        agents' (freshly reset) positions, and records the initial red count
        used to normalize the blue scoring penalty.
        """
        self._red_pin_streaks = {}
        self._red_ring_potentials = {}
        self._initial_red_count = len(self.red_agents)
        if self.reward_settings.red_benchmark:
            center = (self.grid_width / 2, self.grid_height / 2)
            for red_agent in self.red_agents:
                if getattr(red_agent, "x", None) is None or getattr(red_agent, "y", None) is None:
                    continue
                self._red_ring_potentials[red_agent.name] = ring_potential(
                    (red_agent.x, red_agent.y), center, REWARD_ATTRACTOR_DISTANCE
                )

    def _apply_benchmark_rewards(self, rewards):
        """Compute config-gated benchmark rewards for one simultaneous step.

        Builds the detection state ONCE for the step and shares it between the
        blue and red benchmark rewards. Teams left in legacy mode keep their
        legacy per-agent reward; the legacy passive blue bonus is applied by the
        caller and only when blue mode is legacy (the benchmark blue reward
        replaces it).

        Args:
            rewards: Mutable dict of agent name -> reward for agents in
                ``self.agents``; updated in place.

        Returns:
            bool: True if any red newly scored this step (on the attractor ring
            and undetected), matching the legacy scoring event.
        """
        settings = self.reward_settings
        state = build_detection_state(
            self.active_blue_agents, self.active_red_agents, self._red_pin_streaks
        )
        self._red_pin_streaks = dict(state.streaks)

        center = (self.grid_width / 2, self.grid_height / 2)
        on_ring = {}
        new_potentials = {}
        for red_agent in self.active_red_agents:
            if getattr(red_agent, "x", None) is None or getattr(red_agent, "y", None) is None:
                continue
            distance_to_center = np.sqrt(
                (red_agent.x - center[0]) ** 2 + (red_agent.y - center[1]) ** 2
            )
            band_offset = abs(distance_to_center - REWARD_ATTRACTOR_DISTANCE)
            on_ring[red_agent.name] = band_offset <= REWARD_ATTRACTOR_TOLERANCE
            new_potentials[red_agent.name] = -band_offset

        # Newly scoring reds this step (per-step ring semantics: every scoring
        # step counts). k == 0 is exactly the legacy "undetected" test.
        scoring_reds = {
            name
            for name in state.red_names
            if on_ring.get(name, False) and state.k.get(name, 0) == 0
        }
        red_scored_this_step = bool(scoring_reds)
        red_score_fraction = (
            len(scoring_reds) / self._initial_red_count if self._initial_red_count else 0.0
        )

        for agent_name in self.agents:
            agent_obj = self.agent_objects[agent_name]
            agent_type = getattr(agent_obj.agent_type, "value", None)
            if agent_type == "red":
                if settings.red_benchmark:
                    if agent_name in new_potentials:
                        previous_phi = self._red_ring_potentials.get(
                            agent_name, new_potentials[agent_name]
                        )
                        rewards[agent_name] = red_benchmark_reward(
                            agent_name,
                            state,
                            on_ring.get(agent_name, False),
                            new_potentials[agent_name] - previous_phi,
                            settings,
                        )
                    else:
                        rewards[agent_name] = 0.0
                else:
                    reward, _scored = self._calculate_reward(agent_name, agent_obj)
                    rewards[agent_name] = reward
            elif agent_type == "blue" and settings.blue_benchmark:
                rewards[agent_name] = blue_benchmark_reward(
                    agent_name, state, red_score_fraction, settings
                )

        self._red_ring_potentials.update(new_potentials)
        return red_scored_this_step

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
            if self.window_scale < 1:
                self.window_scale = 1  # Safety
            
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
