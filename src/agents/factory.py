"""Agent factory for creating agents from configuration."""

import math
from collections.abc import Mapping as MappingABC
from typing import Any, Mapping, Optional

from src.agents.action_spaces import (
    ActionSpaceConfig,
    build_movement_action_space,
    resolve_spec_max_speed,
)
from src.agents.red_agent import RedAgent
from src.agents.blue_agent import BlueAgent
from src.utils.logger import get_logger

logger = get_logger("acn.agents.factory")

#: Environment fallback for the per-agent communication radius, used when
#: neither the agent spec nor the team default configures one (see
#: docs/communication_decision_log.md, "Communication Semantics").
DEFAULT_COMMUNICATION_RADIUS = 15.0


def _validate_radius(value: Any, context: str) -> float:
    """Coerce a configured communication radius to a non-negative finite float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            "{} must be a number, got {} ({!r}).".format(context, type(value).__name__, value)
        )
    radius = float(value)
    if not math.isfinite(radius) or radius < 0.0:
        raise ValueError(
            "{} must be a non-negative finite number, got {}.".format(context, radius)
        )
    return radius


def resolve_spec_communication_radius(
    spec: Optional[Mapping[str, Any]],
    team_default: Optional[Mapping[str, Any]] = None,
    env_config: Optional[Mapping[str, Any]] = None,
) -> float:
    """Resolve one agent group's communication radius from configuration.

    Precedence (docs/communication_decision_log.md, "Communication Semantics"):
    the group spec's ``communication_radius``, else the team default
    (``agents.team_defaults.<team>.communication_radius``), else the environment
    fallback (``environment.communication_radius``, default
    :data:`DEFAULT_COMMUNICATION_RADIUS`). The factory writes the resolved value
    onto every agent in the group as ``communication_radius``, which
    ``src.communication.topology.RadiusTopology`` reads directly.

    Args:
        spec: One entry of ``agents.blue_agents`` / ``agents.red_agents``, or None.
        team_default: The team's ``agents.team_defaults.<team>`` mapping, or None.
        env_config: The environment config providing the ``communication_radius``
            fallback, or None.

    Returns:
        The resolved non-negative finite radius as a float.

    Raises:
        ValueError: If the selected configured value is not a non-negative
            finite number.
    """
    if spec is not None and spec.get("communication_radius") is not None:
        return _validate_radius(spec["communication_radius"], "agent spec communication_radius")
    if team_default is not None and team_default.get("communication_radius") is not None:
        return _validate_radius(
            team_default["communication_radius"], "team default communication_radius"
        )
    if env_config is not None and env_config.get("communication_radius") is not None:
        return _validate_radius(
            env_config["communication_radius"], "environment communication_radius"
        )
    return DEFAULT_COMMUNICATION_RADIUS


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

    Each agent's ``communication_radius`` is likewise resolved here (spec value
    over ``agents.team_defaults.<team>`` over ``environment.communication_radius``,
    default 15.0) and written onto every agent, so topology code never has to
    know about configuration inheritance.

    Args:
        agents_config: The 'agents' section from the config file containing
                       'blue_agents' and 'red_agents' lists, plus the optional
                       'team_defaults' mapping ({'blue': {...}, 'red': {...}}).
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

    team_defaults = agents_config.get("team_defaults") or {}
    if not isinstance(team_defaults, MappingABC):
        raise ValueError(
            "agents.team_defaults must be a mapping of team name ('blue'/'red') to a "
            "defaults mapping, got {} ({!r}).".format(type(team_defaults).__name__, team_defaults)
        )

    all_agents = []
    agent_id_counter = 0

    blue_specs = agents_config.get("blue_agents", [])
    for spec in blue_specs:
        count = spec.get("count", 0)
        max_speed = resolve_spec_max_speed(spec)
        communication_radius = resolve_spec_communication_radius(
            spec, team_defaults.get("blue"), env_config
        )
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

            blue_agent = BlueAgent(**kwargs)
            blue_agent.communication_radius = communication_radius
            all_agents.append(blue_agent)
            agent_id_counter += 1

    red_specs = agents_config.get("red_agents", [])
    for spec in red_specs:
        count = spec.get("count", 0)
        max_speed = resolve_spec_max_speed(spec)
        communication_radius = resolve_spec_communication_radius(
            spec, team_defaults.get("red"), env_config
        )
        for _ in range(count):
            agent_name = f"red_{agent_id_counter}"
            red_agent = RedAgent(
                name=agent_name,
                communication_bandwidth=spec.get("communication_bandwidth", 0),
                processing_capability=spec.get("processing_capability", 0),
                detection_radius=spec.get("detection_radius", 15.0),
                strategy_type=spec.get("strategy_type", "center"),
                max_speed=max_speed,
                action_space=build_movement_action_space(space_config, max_speed),
            )
            red_agent.communication_radius = communication_radius
            all_agents.append(red_agent)
            agent_id_counter += 1

    logger.debug("Created {} agents from config", len(all_agents))
    return all_agents
