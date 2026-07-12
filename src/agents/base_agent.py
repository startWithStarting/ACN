from enum import Enum
from typing import Tuple, Optional # For type hinting

import gymnasium.spaces as spaces

from src.agents.action_spaces import DEFAULT_MAX_SPEED, build_continuous_movement_space

class AgentType(Enum):
    """Enumeration for different agent types."""
    BLUE = "blue"
    RED = "red"
class CommsType(Enum):
    CENT = 'centralised'
    DIST = 'distributed'

class BaseAgent:
    """
    Base class for all agents in the simulation.
    """
    def __init__(self, name: str, agent_type: AgentType, communication_bandwidth: int, processing_capability: int, comms_type: CommsType = CommsType.DIST, x: float = None, y: float = None, speed: float = 0.0, direction: Optional[Tuple[float, float]] = None, max_speed: float = DEFAULT_MAX_SPEED, action_space: Optional[spaces.Space] = None):
        """
        Initializes a base agent.

        Args:
            name (str): A unique identifier for the agent (e.g., "blue_0", "red_5").
            agent_type (AgentType): The type of the agent (BLUE or RED).
            communication_bandwidth (int): An abstract measure of communication capacity
                                           (e.g., max number of agents to communicate with).
            processing_capability (int): An abstract measure of computational power.
            comms_type (CommsType): The communication type of the agent.
            x (float, optional): The x-coordinate of the agent. Defaults to None.
            y (float, optional): The y-coordinate of the agent. Defaults to None.
            speed (float, optional): The movement speed of the agent. Defaults to 0.0.
            direction (Optional[Tuple[float, float]], optional): The movement direction (normalized vector) of the agent. Defaults to None.
            max_speed (float, optional): Resolved per-team speed cap written onto the
                agent by the factory. Defaults to 10.0 (the historic hardcoded cap).
            action_space (Optional[spaces.Space], optional): Movement action space built
                by src.agents.action_spaces (the factory passes it). When None, the
                default continuous Dict space is built from max_speed, reproducing the
                previously hardcoded space exactly.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("Agent name must be a non-empty string.")
        if not isinstance(agent_type, AgentType):
            raise ValueError("Agent type must be an instance of AgentType enum.")
        if not isinstance(comms_type, CommsType):
            raise ValueError("Comms type must be an instance of CommsType enum.")        
        if not isinstance(communication_bandwidth, int) or communication_bandwidth < 0:
            raise ValueError("Communication bandwidth must be a non-negative integer.")
        if not isinstance(processing_capability, int) or processing_capability < 0:
            raise ValueError("Processing capability must be a non-negative integer.")
        if not isinstance(speed, (int, float)) or speed < 0:
            raise ValueError("Speed must be a non-negative number.")
        if isinstance(max_speed, bool) or not isinstance(max_speed, (int, float)) or max_speed <= 0:
            raise ValueError("Max speed must be a positive number.")
        if direction is not None:
            if not isinstance(direction, (list, tuple)) or len(direction) != 2:
                 raise ValueError("Direction must be a tuple or list of two numbers.")
            # Optional: Add check for normalization if strictly required upon init
            # norm = np.linalg.norm(direction)
            # if not np.isclose(norm, 1.0):
            #     raise ValueError("Direction vector must be normalized.")


        self.name = name
        self.agent_type = agent_type
        self.comms_type = comms_type
        self.communication_bandwidth = communication_bandwidth
        self.processing_capability = processing_capability
        self.x = x
        self.y = y
        self.speed = speed
        self.direction = tuple(direction) if direction else None # Store as tuple if provided
        # Resolved per-team speed cap and builder-provided movement action space.
        # With the defaults this is exactly the Dict space agents used to hardcode.
        self.max_speed = float(max_speed)
        self.action_space = (
            action_space
            if action_space is not None
            else build_continuous_movement_space(self.max_speed)
        )
        # Add other common agent state variables here if needed
        # e.g., self.health, self.current_observation, etc.
        self.is_active = True # Flag to indicate if the agent is currently active in the env

    def get_observation(self, sub_env):
        """
        Returns the agent's current observation of the environment.
        This should be implemented by subclasses or the environment itself.
        """
        pass
        
    def choose_action(self, observation=None):
        """
        Determines the agent's next action based on the observation.
        This will typically involve the RL policy.

        Args:
            observation: The agent's observation of the environment state.
                        Defaults to None for backward compatibility.

        Returns:
            The chosen action.
        """
        # Placeholder: The actual policy network will go here (likely managed by the Trainer)
        raise NotImplementedError("choose_action() must be implemented, likely linking to an RL policy.")
    
    def receive_message(self, sender_name: str, message_content: dict):
        """Handles receiving a message from another agent."""
        # Placeholder for communication logic
        pass

    def send_message(self, recipient_name: str, message_content: dict):
        """Sends a message to another agent (likely via the environment)."""
        # Placeholder for communication logic
        pass

    def __str__(self):
        position_str = f", Pos: ({self.x:.2f}, {self.y:.2f})" if self.x is not None and self.y is not None else ""
        movement_str = f", Speed: {self.speed:.2f}, Dir: {self.direction}" if self.direction else ""
        return f"Agent(Name: {self.name}, Type: {self.agent_type.value}, Comms: {self.comms_type.value}, CommBW: {self.communication_bandwidth}, ProcCap: {self.processing_capability}{position_str}{movement_str})"

    def __repr__(self):
        position_repr = f", x={self.x}, y={self.y}" if self.x is not None and self.y is not None else ""
        movement_repr = f", speed={self.speed}, direction={self.direction}" if self.direction else "" # Simplified for repr
        return f"{self.__class__.__name__}(name='{self.name}', agent_type=AgentType.{self.agent_type.name}, communication_bandwidth={self.communication_bandwidth}, processing_capability={self.processing_capability}{position_repr}{movement_repr})"
