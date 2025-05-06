from .base_agent import BaseAgent, AgentType

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
        )
        # Add any Red-agent-specific attributes or methods here
        # For example:
        # self.special_red_ability_cooldown = 0

    # You can override methods from BaseAgent if Red agents behave differently
    # For example:
    # def choose_action(self, observation):
    #     # Red agent specific action selection logic
    #     pass

    def __str__(self):
        return f"RedAgent(Name: {self.name}, CommBW: {self.communication_bandwidth}, ProcCap: {self.processing_capability})"

    def __repr__(self):
        return f"RedAgent(name='{self.name}', communication_bandwidth={self.communication_bandwidth}, processing_capability={self.processing_capability})"
