import numpy as np
from typing import Tuple, Dict, Any
from collections import defaultdict
from .base_agent import BaseAgent, AgentType, CommsType # Import CommsType if used in __str__
import gymnasium.spaces as spaces # Import gymnasium spaces

# Import all available strategies
from .strategies.red_agent_strategy import center_based_movement_strategy
from .strategies.avoidant_red_strategy import avoidant_red_strategy
from .strategies.aggressive_red_strategy import aggressive_red_strategy
from .strategies.team_based_red_strategy import team_based_red_strategy
from .strategies.flocking_red_strategy import flocking_red_strategy

class RedAgent(BaseAgent):
    """
    Represents a Red agent in the simulation.
    """
    def __init__(self, name: str, communication_bandwidth: int, processing_capability: int, 
                 detection_radius: float = 15.0, strategy_type: str = "center"):
        """
        Initializes a Red agent.

        Args:
            name (str): A unique identifier for the agent.
            communication_bandwidth (int): Communication capacity.
            processing_capability (int): Computational power.
            detection_radius (float): Radius within which the agent can detect other agents.
            strategy_type (str): Movement strategy to use. Options are:
                                  "center" - Basic movement around center
                                  "avoidant" - Detect and avoid blue agents
                                  "aggressive" - Detect and pursue blue agents
                                  "team" - Move toward red teammates
                                  "flocking" - Flocking behavior with neighbors
                                  "trainable" - Controlled by external policy (RL)
        """
        super().__init__(
            name=name,
            agent_type=AgentType.RED,
            communication_bandwidth=communication_bandwidth,
            processing_capability=processing_capability
            # x, y, speed, direction will use defaults from BaseAgent's __init__
            # (x=None, y=None, speed=0.0, direction=None).
            # The environment (AECGameEnv) will set initial x and y.
            # CommsType will also default from BaseAgent (CommsType.DIST).
        )

        # Define the specific action space for Red Agents, overriding the default
        self.action_space = spaces.Dict({
            'direction': spaces.Box(low=np.array([-1.0, -1.0], dtype=np.float32),
                                    high=np.array([1.0, 1.0], dtype=np.float32),
                                    shape=(2,), dtype=np.float32), # Normalized direction vector
            'speed': spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32) # Continuous speed
        })

        # Add detection radius for finding other agents
        self.detection_radius = detection_radius
        self.observed_teammates = {}  # Stores history of teammate positions: {name: [(pos, timestamp), ...]}
        
        # Set the movement strategy type
        self.strategy_type = strategy_type
        
        self.observed_teammates = defaultdict(list)

    def record_teammate_movement(self, teammate_name: str, position: Tuple[float, float], timestamp: float):
        """
        Record the movement of a red teammate if it's within detection radius.
        Only record if the position is not (0,0) and is a real detection.
        """
        if position is not None and not (position[0] == 0 and position[1] == 0):
            self.observed_teammates[teammate_name].append((position, timestamp))
            # Keep only the last 3 observations (reduced lag)
            if len(self.observed_teammates[teammate_name]) > 3:
                self.observed_teammates[teammate_name].pop(0)
            
    def choose_action(self, observation=None):
        """
        Chooses an action for the Red agent by delegating to a strategy.

        Args:
            observation (Dict[str, Any], optional): The agent's observation, expected to contain
                                          'position' (Tuple[float, float]),
                                          'grid_center' (Tuple[float, float]),
                                          'blue_agents' (Dict[str, Dict]),
                                          and 'red_teammates' (Dict[str, Dict]).
                                          Defaults to None for backward compatibility.

        Returns:
            Dict[str, Any]: A dictionary containing the 'direction' (normalized numpy array)
                            and 'speed' (integer). Returns default action if info is missing.
        """
        if observation is None:
            return {'direction': np.array([0.0, 0.0], dtype=np.float32), 'speed': 0}

        current_pos = observation.get('position')
        grid_center = observation.get('grid_center')
        blue_agents = observation.get('blue_agents', {})
        red_teammates = observation.get('red_teammates', {})
        timestamp = observation.get('timestamp', 0.0)
        
        # Record positions of red teammates if within detection radius
        for teammate_name, teammate_data in red_teammates.items():
            if 'position' in teammate_data:
                position = teammate_data['position']
                # Only record if within detection radius
                if self.is_within_detection_radius(current_pos, position):
                    self.record_teammate_movement(teammate_name, position, timestamp)
        
        # Select strategy based on strategy_type
        if self.strategy_type == "avoidant":
            return avoidant_red_strategy(current_pos, grid_center, blue_agents, self.detection_radius)
        elif self.strategy_type == "aggressive":
            return aggressive_red_strategy(current_pos, grid_center, blue_agents, self.detection_radius)
        elif self.strategy_type == "team":
            return team_based_red_strategy(current_pos, grid_center, red_teammates, blue_agents, self.detection_radius)
        elif self.strategy_type == "flocking":
            # For flocking, we need to extract the previous positions of teammates for alignment behavior
            # For flocking, we need to pass the history list, not the raw observation dict
            flocking_neighbors = {}
            for name in red_teammates:
                if name in self.observed_teammates and self.observed_teammates[name]:
                     flocking_neighbors[name] = self.observed_teammates[name]
            
            # Get flocking-specific parameters from config if available
            cohesion_weight = observation.get('cohesion_weight', 0.33)
            alignment_weight = observation.get('alignment_weight', 0.33)
            separation_weight = observation.get('separation_weight', 0.33)
            separation_radius = observation.get('separation_radius', 5.0)
            
            # Get the current timestamp for initial time step behavior
            timestamp = observation.get('timestamp', 0.0)
            
            return flocking_red_strategy(
                current_pos, grid_center, flocking_neighbors, blue_agents, 
                self.detection_radius, cohesion_weight, alignment_weight, 
                separation_weight, separation_radius, timestamp,
                observation=observation
            )
        elif self.strategy_type == "trainable":
            # For trainable agents, the action is typically provided externally during training (e.g. via step(action)).
            # If choose_action is called (e.g. during inference without a model wrapper), we strictly rely on input.
            # Here we just return a no-op placeholder if forced, but usually the Trainer controls this.
            return {'direction': np.array([0.0, 0.0], dtype=np.float32), 'speed': 0}
        else:  # Default to center-based
            return center_based_movement_strategy(current_pos, grid_center)

    def is_within_detection_radius(self, current_pos: Tuple[float, float], other_pos: Tuple[float, float]) -> bool:
        """
        Check if another agent is within the detection radius.
        
        Args:
            current_pos (Tuple[float, float]): Current position of this agent
            other_pos (Tuple[float, float]): Position of the other agent
            
        Returns:
            bool: True if the other agent is within detection radius, False otherwise
        """
        if current_pos is None or other_pos is None:
            return False
        return self.calculate_distance(current_pos, other_pos) <= self.detection_radius
    
    def calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """
        Calculate Euclidean distance between two positions.

        Args:
            pos1 (Tuple[float, float]): First position (x, y)
            pos2 (Tuple[float, float]): Second position (x, y)

        Returns:
            float: Euclidean distance between the positions
        """
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        
    def __str__(self):
        position_str = f", Pos: ({self.x:.2f}, {self.y:.2f})" if self.x is not None and self.y is not None else ""
        movement_str = f", Speed: {self.speed:.2f}, Dir: {self.direction}" if self.direction is not None else ""
        strategy_str = f", Strategy: {self.strategy_type}"
        # Assuming self.agent_type and self.comms_type are set by BaseAgent.
        return f"RedAgent(Name: {self.name}, Type: {self.agent_type.value}, Comms: {self.comms_type.value}, CommBW: {self.communication_bandwidth}, ProcCap: {self.processing_capability}{position_str}{movement_str}{strategy_str})"

    def __repr__(self):
        position_repr = f", x={self.x}, y={self.y}" if self.x is not None and self.y is not None else ""
        movement_repr = f", speed={self.speed}, direction={self.direction}" if self.direction is not None else ""
        # RedAgent implies AgentType.RED. CommsType defaults in BaseAgent.
        return (f"RedAgent(name='{self.name}', "
                f"communication_bandwidth={self.communication_bandwidth}, "
                f"processing_capability={self.processing_capability}, "
                f"detection_radius={self.detection_radius}, "
                f"strategy_type='{self.strategy_type}'{position_repr}{movement_repr})")
