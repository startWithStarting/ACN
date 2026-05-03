"""Observation builder pattern for ACN environments.

This module provides a builder pattern for constructing agent observations,
replacing the if/elif chains in common_env_logic.py.

Usage:
    # Create builder
    builder = BlueObservationBuilder(env)
    obs = builder.build(agent_obj, step_count)

    # Or use factory
    builder = create_observation_builder(agent_type, env)
    obs = builder.build(agent_obj, step_count)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class ObservationBuilder(ABC):
    """Base class for observation builders."""

    def __init__(self, env):
        """
        Args:
            env: The environment instance (must have grid_width, grid_height,
                 active_red_agents, active_blue_agents attributes).
        """
        self.env = env

    @abstractmethod
    def build(self, agent_obj: Any, step_count: int) -> Dict[str, Any]:
        """Build the observation dictionary for an agent."""
        pass

    def _get_base_observation(self, agent_obj: Any, step_count: int) -> Dict[str, Any]:
        """Common base observation (position, grid_center, timestamp)."""
        agent_pos = np.array([agent_obj.x, agent_obj.y], dtype=np.float32)
        grid_center = np.array(
            [self.env.grid_width / 2, self.env.grid_height / 2], dtype=np.float32
        )
        return {
            "position": agent_pos,
            "grid_center": grid_center,
            "timestamp": step_count,
        }

    def _get_agent_config(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get agent config from environment."""
        if hasattr(self.env, "_get_agent_config"):
            return self.env._get_agent_config(agent_name)
        return None


class BlueObservationBuilder(ObservationBuilder):
    """Observation builder for Blue agents."""

    def build(self, agent_obj: Any, step_count: int) -> Dict[str, Any]:
        """Build observation for a Blue agent."""
        obs = self._get_base_observation(agent_obj, step_count)

        # Add red agents info
        red_agents_info = {}
        for red_agent in getattr(self.env, "active_red_agents", []):
            if (
                hasattr(red_agent, "x")
                and hasattr(red_agent, "y")
                and red_agent.x is not None
                and red_agent.y is not None
            ):
                red_agents_info[red_agent.name] = {"position": (red_agent.x, red_agent.y)}

        obs["red_agents"] = red_agents_info
        return obs


class RedObservationBuilder(ObservationBuilder):
    """Observation builder for Red agents."""

    def build(self, agent_obj: Any, step_count: int) -> Dict[str, Any]:
        """Build observation for a Red agent."""
        obs = self._get_base_observation(agent_obj, step_count)

        # Add blue agents info
        blue_agents_info = {}
        for blue_agent in getattr(self.env, "active_blue_agents", []):
            if (
                hasattr(blue_agent, "x")
                and hasattr(blue_agent, "y")
                and blue_agent.x is not None
                and blue_agent.y is not None
            ):
                blue_agents_info[blue_agent.name] = {"position": (blue_agent.x, blue_agent.y)}
        obs["blue_agents"] = blue_agents_info

        # Add red teammates info
        agent_name = getattr(agent_obj, "name", "")
        red_teammates_info = {}
        for red_agent in getattr(self.env, "active_red_agents", []):
            if (
                red_agent.name != agent_name
                and hasattr(red_agent, "x")
                and hasattr(red_agent, "y")
                and red_agent.x is not None
                and red_agent.y is not None
            ):
                red_teammates_info[red_agent.name] = {"position": (red_agent.x, red_agent.y)}
        obs["red_teammates"] = red_teammates_info

        return obs


class FlockingObservationBuilder(RedObservationBuilder):
    """Observation builder for Red agents using flocking strategy."""

    def build(self, agent_obj: Any, step_count: int) -> Dict[str, Any]:
        """Build observation for a flocking Red agent."""
        obs = super().build(agent_obj, step_count)

        # Add flocking-specific parameters
        if hasattr(agent_obj, "strategy_type") and agent_obj.strategy_type == "flocking":
            agent_config = self._get_agent_config(agent_obj.name)
            if agent_config:
                for param in [
                    "cohesion_weight",
                    "alignment_weight",
                    "separation_weight",
                    "separation_radius",
                    "inertia_weight",
                    "max_speed",
                    "max_force",
                    "wall_avoidance_weight",
                    "wall_detection_radius",
                ]:
                    if param in agent_config:
                        obs[param] = agent_config[param]

            # Add current direction and speed for smoothing
            if hasattr(agent_obj, "direction"):
                obs["current_direction"] = agent_obj.direction
            if hasattr(agent_obj, "speed"):
                obs["current_speed"] = agent_obj.speed

        return obs


def create_observation_builder(agent_type: str, env) -> ObservationBuilder:
    """
    Factory function to create the appropriate observation builder.

    Args:
        agent_type: The agent type ("blue", "red", or "flocking")
        env: The environment instance

    Returns:
        An ObservationBuilder instance
    """
    if agent_type == "blue":
        return BlueObservationBuilder(env)
    elif agent_type == "flocking":
        return FlockingObservationBuilder(env)
    else:
        return RedObservationBuilder(env)
