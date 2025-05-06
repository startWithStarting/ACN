from .base_agent import BaseAgent, AgentType

class BlueAgent(BaseAgent):
    """
    Represents a Blue agent in the simulation.
    """
    def __init__(self, name: str, communication_bandwidth: int, processing_capability: int):
        """
        Initializes a Blue agent.

        Args:
            name (str): A unique identifier for the agent.
            communication_bandwidth (int): Communication capacity.
            processing_capability (int): Computational power.
        """
        super().__init__(
            name=name,
            agent_type=AgentType.BLUE,
            communication_bandwidth=communication_bandwidth,
            processing_capability=processing_capability
        )
        # Add any Blue-agent-specific attributes or methods here
        # For example:
        # self.special_blue_skill_charge = 100

    # You can override methods from BaseAgent if Blue agents behave differently
    # For example:
    # def receive_message(self, sender_name: str, message_content: dict):
    #     # Blue agent specific message processing
    #     super().receive_message(sender_name, message_content) # Optionally call parent
    #     # ... additional blue logic

    def __str__(self):
        return f"BlueAgent(Name: {self.name}, CommBW: {self.communication_bandwidth}, ProcCap: {self.processing_capability})"

    def __repr__(self):
        return f"BlueAgent(name='{self.name}', communication_bandwidth={self.communication_bandwidth}, processing_capability={self.processing_capability})"
