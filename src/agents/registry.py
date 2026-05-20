"""Agent and strategy registry for dynamic agent type and strategy selection.

This module provides a decorator-based registration system that allows
adding new agent types and strategies without modifying existing code.

Example:
    Register a new agent type::

        @register_agent("my_agent")
        class MyAgent(BaseAgent):
            ...

    Register a new strategy::

        @register_strategy("my_strategy", side="red")
        def my_strategy(agent, env_state):
            ...

    Create registered objects::

        agent = create_agent("my_agent", name="test", ...)
        strategy_fn = get_strategy("my_strategy", side="red")
"""

from typing import Callable, Dict, Optional, Type, Any

from src.agents.base_agent import BaseAgent
from src.utils.logger import get_logger

logger = get_logger("acn.agents.registry")

# Global registries
_AGENT_REGISTRY: Dict[str, Type[BaseAgent]] = {}
_STRATEGY_REGISTRY: Dict[str, Callable] = {}


def register_agent(agent_type: str) -> Callable[[Type[BaseAgent]], Type[BaseAgent]]:
    """
    Decorator to register an agent class.

    Args:
        agent_type: The string identifier for this agent type.

    Returns:
        A decorator function that registers the class and returns it unchanged.

    Example:
        Register a pursuer agent::

            @register_agent("pursuer")
            class PursuerAgent(BaseAgent):
                ...
    """
    def decorator(cls: Type[BaseAgent]) -> Type[BaseAgent]:
        if agent_type in _AGENT_REGISTRY:
            logger.warning("Overwriting existing agent type '{}' in registry", agent_type)
        _AGENT_REGISTRY[agent_type] = cls
        logger.debug("Registered agent type: {}", agent_type)
        return cls
    return decorator


def register_strategy(name: str, side: str) -> Callable[[Callable], Callable]:
    """
    Decorator to register a strategy function.

    Args:
        name: The string identifier for this strategy.
        side: Which side ("red" or "blue") this strategy applies to.

    Returns:
        A decorator function that registers the function and returns it unchanged.

    Example:
        Register an aggressive red strategy::

            @register_strategy("aggressive", side="red")
            def aggressive_strategy(agent, env_state):
                ...
    """
    if side not in ("red", "blue"):
        raise ValueError(f"Invalid side: {side}. Must be 'red' or 'blue'.")

    key = f"{side}:{name}"

    def decorator(fn: Callable) -> Callable:
        if key in _STRATEGY_REGISTRY:
            logger.warning("Overwriting existing strategy '{}' in registry", key)
        _STRATEGY_REGISTRY[key] = fn
        logger.debug("Registered strategy: {} (side={})", name, side)
        return fn
    return decorator


def create_agent(agent_type: str, **kwargs) -> BaseAgent:
    """
    Create an agent instance by type name.

    Args:
        agent_type: The registered agent type identifier.
        **kwargs: Arguments passed to the agent constructor.

    Returns:
        An instance of the requested agent type.

    Raises:
        ValueError: If the agent_type is not registered.
    """
    if agent_type not in _AGENT_REGISTRY:
        available = list(_AGENT_REGISTRY.keys())
        raise ValueError(
            f"Unknown agent type: '{agent_type}'. Available types: {available}"
        )
    return _AGENT_REGISTRY[agent_type](**kwargs)


def get_strategy(name: str, side: str) -> Callable:
    """
    Get a strategy function by name and side.

    Args:
        name: The strategy name.
        side: Which side ("red" or "blue").

    Returns:
        The strategy function.

    Raises:
        ValueError: If the strategy is not registered.
    """
    key = f"{side}:{name}"
    if key not in _STRATEGY_REGISTRY:
        # Find available strategies for this side
        available = [k.split(":", 1)[1] for k in _STRATEGY_REGISTRY.keys() if k.startswith(f"{side}:")]
        raise ValueError(
            f"Unknown strategy: '{name}' for side '{side}'. Available: {available}"
        )
    return _STRATEGY_REGISTRY[key]


def list_agent_types() -> list[str]:
    """Return list of registered agent type names."""
    return list(_AGENT_REGISTRY.keys())


def list_strategies(side: Optional[str] = None) -> list[str]:
    """
    Return list of registered strategy names.

    Args:
        side: If provided, only return strategies for this side ("red" or "blue").
    """
    if side is None:
        return list(_STRATEGY_REGISTRY.keys())
    prefix = f"{side}:"
    return [k[len(prefix):] for k in _STRATEGY_REGISTRY.keys() if k.startswith(prefix)]


# Auto-register built-in agents and strategies at import time
def _auto_register():
    """Import all agent modules to trigger their decorators."""
    # Import agents to trigger @register_agent decorators
    from src.agents import blue_agent, red_agent  # noqa: F401

    # Import strategies to trigger @register_strategy decorators
    from src.agents.strategies import (  # noqa: F401
        aggressive_red_strategy,
        avoidant_red_strategy,
        flocking_red_strategy,
        red_agent_strategy,
        team_based_red_strategy,
    )
    from src.agents.blue_strategies import (  # noqa: F401
        pursuit_strategy,
        static_strategy,
    )

    logger.debug(
        "Auto-registered agents: {}, strategies: {}",
        list_agent_types(),
        list_strategies()
    )


# Trigger auto-registration when module is imported
_auto_register()
