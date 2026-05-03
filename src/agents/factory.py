"""Agent factory for creating agents from configuration."""

from typing import Optional, Tuple

from src.agents.red_agent import RedAgent
from src.agents.blue_agent import BlueAgent
from src.utils.logger import get_logger

logger = get_logger("acn.agents.factory")


def create_agents_from_config(
    agents_config: dict,
    env_config: Optional[dict] = None,
) -> list:
    """
    Create all agents from a configuration dict.

    Args:
        agents_config: The 'agents' section from the config file containing
                       'blue_agents' and 'red_agents' lists.
        env_config: Optional environment config to pass grid_size and debug_mode
                    to BlueAgent (used in parallel mode).

    Returns:
        List of agent instances (BlueAgent and RedAgent mixed).
    """
    if env_config is None:
        env_config = {}

    all_agents = []
    agent_id_counter = 0

    blue_specs = agents_config.get("blue_agents", [])
    for spec in blue_specs:
        count = spec.get("count", 0)
        for _ in range(count):
            agent_name = f"blue_{agent_id_counter}"

            # Extract blue agent parameters
            kwargs = {
                "name": agent_name,
                "communication_bandwidth": spec.get("communication_bandwidth", 0),
                "processing_capability": spec.get("processing_capability", 0),
                "detection_radius": spec.get("detection_radius", 20.0),
                "strategy_type": spec.get("strategy_type", "pursuit"),
                "prediction_timeout": spec.get("prediction_timeout", 50),
                "observation_window_size": spec.get("observation_window_size", 5),
                "prediction_interval": spec.get("prediction_interval", 1),
            }

            # Add optional parallel-mode parameters
            if env_config:
                width = float(env_config.get("width", 100))
                height = float(env_config.get("height", 100))
                kwargs["grid_size"] = (width, height)
                kwargs["debug_mode"] = env_config.get("debug_mode", False)

            all_agents.append(BlueAgent(**kwargs))
            agent_id_counter += 1

    red_specs = agents_config.get("red_agents", [])
    for spec in red_specs:
        count = spec.get("count", 0)
        for _ in range(count):
            agent_name = f"red_{agent_id_counter}"
            all_agents.append(RedAgent(
                name=agent_name,
                communication_bandwidth=spec.get("communication_bandwidth", 0),
                processing_capability=spec.get("processing_capability", 0),
                detection_radius=spec.get("detection_radius", 15.0),
                strategy_type=spec.get("strategy_type", "center"),
            ))
            agent_id_counter += 1

    logger.debug("Created {} agents from config", len(all_agents))
    return all_agents
