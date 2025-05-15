import numpy as np
from typing import Tuple, Dict, Any
from .base_agent import BaseAgent, AgentType, CommsType # Import CommsType if used in __str__
import gymnasium.spaces as spaces # Import gymnasium spaces

class RedAgent(BaseAgent):
    """
    Represents a Red agent in the simulation.
    """
    def __init__(self, name: str, communication_bandwidth: int, processing_capability: int):
        """
        Initializes a Red agent.

        Args:
            name (str): A unique identifier for the agent.
            communication_bandwidth (int): Communication capacity.
            processing_capability (int): Computational power.
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
            'speed': spaces.Discrete(11, start=-5)  # Represents integers from -5 to 5
        })

        # Add any Red-agent-specific attributes or methods here
        # For example:
        # self.special_red_ability_cooldown = 0

    def choose_action(self, observation=None):
        """
        Chooses an action based on a rule: move towards the center of the grid, but maintain a minimum
        distance of 10 units from the center. Speed is proportional to the distance from the center.
        When closer than 10 units, the agent is repelled away from the center.

        Args:
            observation (Dict[str, Any], optional): The agent's observation, expected to contain
                                          'position' (Tuple[float, float]) and
                                          'grid_center' (Tuple[float, float]).
                                          Defaults to None for backward compatibility.

        Returns:
            Dict[str, Any]: A dictionary containing the 'direction' (normalized numpy array)
                            and 'speed' (integer). Returns default action if info is missing.
        """
        if observation is None:
            return {'direction': np.array([0.0, 0.0], dtype=np.float32), 'speed': 0}

        current_pos = observation.get('position')
        grid_center = observation.get('grid_center')

        default_speed = 0  # Default is to stay still
        default_direction = np.array([0.0, 0.0], dtype=np.float32)
        min_distance = 10.0  # Minimum distance to maintain from the center

        if current_pos is None or grid_center is None:
            # Handle missing information, return a default action (stay still)
            return {'direction': default_direction, 'speed': 0}

        current_x, current_y = current_pos
        center_x, center_y = grid_center

        # Calculate vector towards the center
        direction_vector = np.array([center_x - current_x, center_y - current_y])
        
        # Calculate distance to center
        distance_to_center = np.linalg.norm(direction_vector)
        
        if distance_to_center < 1e-6:  # If exactly at center (very unlikely)
            # Choose a random direction to move away
            random_direction = np.random.uniform(-1, 1, 2)
            normalized_direction = (random_direction / np.linalg.norm(random_direction)).astype(np.float32)
            speed = 5  # Move away with a moderate speed
        elif distance_to_center < min_distance:
            # Too close to center, reverse direction to move away
            normalized_direction = (-direction_vector / distance_to_center).astype(np.float32)
            # Speed proportional to how much closer than min_distance
            speed = int(5 * (1 + (min_distance - distance_to_center) / min_distance))
        else:
            # Moving towards center with speed proportional to distance
            normalized_direction = (direction_vector / distance_to_center).astype(np.float32)
            # Cap the speed at 5 (the maximum for the action space)
            speed = min(5, int(distance_to_center / 10))
        
        # The action is a dictionary containing direction and speed
        action = {
            'direction': normalized_direction,
            'speed': speed
        }
        
        return action

    def __str__(self):
        position_str = f", Pos: ({self.x:.2f}, {self.y:.2f})" if self.x is not None and self.y is not None else ""
        movement_str = f", Speed: {self.speed:.2f}, Dir: {self.direction}" if self.direction is not None else ""
        # Assuming self.agent_type and self.comms_type are set by BaseAgent.
        return f"RedAgent(Name: {self.name}, Type: {self.agent_type.value}, Comms: {self.comms_type.value}, CommBW: {self.communication_bandwidth}, ProcCap: {self.processing_capability}{position_str}{movement_str})"

    def __repr__(self):
        position_repr = f", x={self.x}, y={self.y}" if self.x is not None and self.y is not None else ""
        movement_repr = f", speed={self.speed}, direction={self.direction}" if self.direction is not None else ""
        # RedAgent implies AgentType.RED. CommsType defaults in BaseAgent.
        return (f"RedAgent(name='{self.name}', "
                f"communication_bandwidth={self.communication_bandwidth}, "
                f"processing_capability={self.processing_capability}{position_repr}{movement_repr})")
