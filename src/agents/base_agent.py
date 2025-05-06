from enum import Enum

class AgentType(Enum):
    """Enumeration for different agent types."""
    BLUE = "blue"
    RED = "red"

class BaseAgent:
    """
    Base class for all agents in the simulation.
    """
    def __init__(self, name: str, agent_type: AgentType, communication_bandwidth: int, processing_capability: int):
        """
        Initializes a base agent.

        Args:
            name (str): A unique identifier for the agent (e.g., "blue_0", "red_5").
            agent_type (AgentType): The type of the agent (BLUE or RED).
            communication_bandwidth (int): An abstract measure of communication capacity
                                           (e.g., max number of agents to communicate with).
            processing_capability (int): An abstract measure of computational power.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("Agent name must be a non-empty string.")
        if not isinstance(agent_type, AgentType):
            raise ValueError("Agent type must be an instance of AgentType enum.")
        if not isinstance(communication_bandwidth, int) or communication_bandwidth < 0:
            raise ValueError("Communication bandwidth must be a non-negative integer.")
        if not isinstance(processing_capability, int) or processing_capability < 0:
            raise ValueError("Processing capability must be a non-negative integer.")

        self.name = name
        self.agent_type = agent_type
        self.communication_bandwidth = communication_bandwidth
        self.processing_capability = processing_capability
        # Add other common agent state variables here if needed
        # e.g., self.position, self.health, self.current_observation, etc.
        self.is_active = True # Flag to indicate if the agent is currently active in the env

    def get_observation(self):
        """
        Returns the agent's current observation of the environment.
        This should be implemented by subclasses or the environment itself.
        """
        # Placeholder: In a real scenario, this would return relevant state info
        raise NotImplementedError("get_observation() must be implemented by the environment or specific agent logic.")

    def choose_action(self, observation):
        """
        Determines the agent's next action based on the observation.
        This will typically involve the RL policy.
        """
        # Placeholder: The actual policy network will go here (likely managed by the Trainer)
        raise NotImplementedError("choose_action() must be implemented, likely linking to an RL policy.")

    def receive_message(self, sender_name: str, message_content: dict):
        """
        Handles receiving a message from another agent.
        """
        # Placeholder: Logic for processing incoming messages
        print(f"{self.name} received message from {sender_name}: {message_content}")
        # Communication logic (potentially using GNNs) would be triggered here.

    def send_message(self, recipient_name: str, message_content: dict):
        """
        Sends a message to another agent (likely via the environment).
        """
        # Placeholder: Logic for initiating message sending
        print(f"{self.name} sending message to {recipient_name}: {message_content}")
        # The environment will likely mediate the actual delivery.

    def __str__(self):
        return f"Agent(Name: {self.name}, Type: {self.agent_type.value}, CommBW: {self.communication_bandwidth}, ProcCap: {self.processing_capability})"

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', agent_type=AgentType.{self.agent_type.name}, communication_bandwidth={self.communication_bandwidth}, processing_capability={self.processing_capability})"
