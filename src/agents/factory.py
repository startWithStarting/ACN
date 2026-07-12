"""Agent factory for creating agents from configuration."""

from typing import Optional

from src.agents.action_spaces import (
    ActionSpaceConfig,
    build_movement_action_space,
    resolve_spec_max_speed,
)
from src.agents.red_agent import RedAgent
from src.agents.blue_agent import BlueAgent
from src.utils.logger import get_logger

logger = get_logger("acn.agents.factory")


def create_agents_from_config(
    agents_config: dict,
    env_config: Optional[dict] = None,
    action_space_config: Optional[dict] = None,
) -> list:
    """
    Create all agents from a configuration dict.

    Each agent group's ``max_speed`` is resolved from its spec (default 10.0,
    the historic cap) and written onto every agent in the group, and the
    movement action space is built from ``environment.action_space`` plus that
    resolved cap and passed to the agent constructor.

    Args:
        agents_config: The 'agents' section from the config file containing
                       'blue_agents' and 'red_agents' lists.
        env_config: Optional environment config to pass grid_size and debug_mode
                    to BlueAgent (used in parallel mode). Its ``action_space``
                    block also selects the movement action space.
        action_space_config: Optional raw ``environment.action_space`` block.
                    Overrides ``env_config["action_space"]`` when provided; used
                    by callers that do not forward the full environment config.

    Returns:
        List of agent instances (BlueAgent and RedAgent mixed).
    """
    if env_config is None:
        env_config = {}

    if action_space_config is None:
        action_space_config = env_config.get("action_space")
    space_config = ActionSpaceConfig.from_dict(action_space_config)

    all_agents = []
    agent_id_counter = 0

    blue_specs = agents_config.get("blue_agents", [])
    for spec in blue_specs:
        count = spec.get("count", 0)
        max_speed = resolve_spec_max_speed(spec)
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
                "max_speed": max_speed,
                "action_space": build_movement_action_space(space_config, max_speed),
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
        max_speed = resolve_spec_max_speed(spec)
        for _ in range(count):
            agent_name = f"red_{agent_id_counter}"
            all_agents.append(RedAgent(
                name=agent_name,
                communication_bandwidth=spec.get("communication_bandwidth", 0),
                processing_capability=spec.get("processing_capability", 0),
                detection_radius=spec.get("detection_radius", 15.0),
                strategy_type=spec.get("strategy_type", "center"),
                max_speed=max_speed,
                action_space=build_movement_action_space(space_config, max_speed),
            ))
            agent_id_counter += 1

    logger.debug("Created {} agents from config", len(all_agents))
    return all_agents
