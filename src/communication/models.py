"""Communication models for multi-agent message passing.

This module provides the foundational communication infrastructure. In the
current codebase it is optional: agents can operate without using it. The
intention is that a CommunicationChannel (see Phase 6) will mediate actual
message delivery.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

from src.utils.logger import get_logger

logger = get_logger("acn.comms")


class CommunicationModel(ABC):
    """Abstract base class for different communication strategies."""

    def __init__(self, config=None):
        self.config = config if config else {}
        logger.debug("Initialized CommunicationModel with config: {}", self.config)

    @abstractmethod
    def process_messages(self, agent, incoming_messages: List[Tuple[str, dict]]) -> dict:
        """Processes incoming messages for an agent."""
        ...

    @abstractmethod
    def select_communication_partners(self, agent, all_other_agents: list) -> list:
        """Determines which agents to communicate with."""
        ...


class GNNCommunicationModel(CommunicationModel):
    """Placeholder for GNN-based communication."""

    def __init__(self, config=None):
        super().__init__(config)
        logger.debug("Initialized GNNCommunicationModel.")

    def process_messages(self, agent, incoming_messages: List[Tuple[str, dict]]) -> dict:
        logger.debug("Agent {} processing messages using GNNCommunicationModel.", agent.name)
        raise NotImplementedError("GNN communication not yet implemented")

    def select_communication_partners(self, agent, all_other_agents: list) -> list:
        return []


class NoCommunicationModel(CommunicationModel):
    """No-op communication model — agents act purely on local observations."""

    def __init__(self, config=None):
        super().__init__(config)
        logger.debug("Initialized NoCommunicationModel.")

    def process_messages(self, agent, incoming_messages: List[Tuple[str, dict]]) -> dict:
        return {"communication_output": None}

    def select_communication_partners(self, agent, all_other_agents: list) -> list:
        return []
