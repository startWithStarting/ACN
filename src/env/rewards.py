"""Reward function modularity for ACN environments.

This module provides a Protocol-based system for different reward functions,
allowing easy swapping and configuration of reward mechanisms.

Usage:
    # Config-driven selection
    reward_fn = create_reward_function(config.get("reward_function", {}))

    # Use in environment
    reward, details = reward_fn(agent_name, agent_obj, env_state)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, Tuple, Dict, Any, Optional

import numpy as np


class RewardFunction(Protocol):
    """Protocol defining the interface for reward functions."""

    def __call__(self, agent_name: str, agent_obj: Any, env_state: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate reward for an agent.

        Args:
            agent_name: Name of the agent.
            agent_obj: The agent object with position and type info.
            env_state: Dictionary containing environment state (grid dims, other agents, etc.)

        Returns:
            Tuple of (reward_value, details_dict) where details contains debug info.
        """
        ...


@dataclass
class AttractorRewardConfig:
    """Configuration for attractor-style rewards."""
    reward_radius: float = 50.0
    tolerance: float = 1.0
    blue_passive_reward: float = 0.1


class AttractorReward:
    """
    Attractor-ring reward: Red agents get reward when in the attractor zone (ring at radius)
    and not detected by blue agents.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}
        self.reward_radius = config.get("reward_radius", 50.0)
        self.tolerance = config.get("tolerance", 1.0)
        self.blue_passive_reward = config.get("blue_passive_reward", 0.1)

    def __call__(self, agent_name: str, agent_obj: Any, env_state: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Calculate attractor reward."""
        agent_type = getattr(agent_obj.agent_type, "value", None)
        details = {"agent_type": agent_type, "scored": False}

        if agent_type != "red":
            return 0.0, details

        # Get grid center from env_state
        grid_width = env_state.get("grid_width", 100)
        grid_height = env_state.get("grid_height", 100)
        center = (grid_width / 2, grid_height / 2)

        # Calculate distance to center
        distance_to_center = np.sqrt(
            (agent_obj.x - center[0]) ** 2 + (agent_obj.y - center[1]) ** 2
        )
        details["distance_to_center"] = distance_to_center

        # Check if within tolerance of the attractor distance
        if abs(distance_to_center - self.reward_radius) <= self.tolerance:
            # Check if not detected by any blue agent
            detected = env_state.get("red_detected_by", {}).get(agent_name, False)
            if not detected:
                details["scored"] = True
                return 1.0, details

        return 0.0, details


class DistanceReward:
    """
    Distance-based reward: Reward is proportional to distance from center.
    Used for encouraging movement or holding positions.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}
        self.scale = config.get("scale", 0.01)
        self.max_reward = config.get("max_reward", 1.0)

    def __call__(self, agent_name: str, agent_obj: Any, env_state: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Calculate distance-based reward."""
        agent_type = getattr(agent_obj.agent_type, "value", None)
        details = {"agent_type": agent_type}

        # Get grid center
        grid_width = env_state.get("grid_width", 100)
        grid_height = env_state.get("grid_height", 100)
        center = (grid_width / 2, grid_height / 2)

        # Calculate distance
        distance = np.sqrt(
            (agent_obj.x - center[0]) ** 2 + (agent_obj.y - center[1]) ** 2
        )

        # Scale reward
        reward = min(distance * self.scale, self.max_reward)
        details["distance"] = distance
        details["raw_reward"] = reward

        return reward, details


class DetectionReward:
    """
    Detection-based reward: Blue agents get rewards for detecting red agents.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}
        self.detection_reward = config.get("detection_reward", 1.0)
        self.undetected_penalty = config.get("undetected_penalty", 0.0)

    def __call__(self, agent_name: str, agent_obj: Any, env_state: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Calculate detection-based reward."""
        agent_type = getattr(agent_obj.agent_type, "value", None)
        details = {"agent_type": agent_type}

        if agent_type != "blue":
            return 0.0, details

        # Check if this blue agent detected any red
        blue_detected = env_state.get("blue_detected_red", {})
        detected_list = blue_detected.get(agent_name, [])

        if detected_list:
            details["num_detected"] = len(detected_list)
            return self.detection_reward * len(detected_list), details

        return self.undetected_penalty, details


class CompositeReward:
    """Composite reward that combines multiple reward functions."""

    def __init__(self, reward_fns: list[RewardFunction], weights: Optional[list[float]] = None):
        self.reward_fns = reward_fns
        self.weights = weights or [1.0] * len(reward_fns)

    def __call__(self, agent_name: str, agent_obj: Any, env_state: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Calculate composite reward."""
        total = 0.0
        details = {}

        for i, (fn, weight) in enumerate(zip(self.reward_fns, self.weights)):
            reward, fn_details = fn(agent_name, agent_obj, env_state)
            total += weight * reward
            details[f"reward_{i}"] = fn_details

        return total, details


def create_reward_function(config: Optional[Dict[str, Any]] = None) -> RewardFunction:
    """
    Factory function to create a reward function from config.

    Args:
        config: Dict with keys:
            - type: "attractor", "distance", "detection", or "composite"
            - params: Additional parameters for the reward function

    Returns:
        A RewardFunction instance.
    """
    if config is None:
        config = {}

    reward_type = config.get("type", "attractor")
    params = config.get("params", {})

    if reward_type == "attractor":
        return AttractorReward(params)
    elif reward_type == "distance":
        return DistanceReward(params)
    elif reward_type == "detection":
        return DetectionReward(params)
    elif reward_type == "composite":
        sub_configs = params.get("components", [])
        reward_fns = [create_reward_function(sc) for sc in sub_configs]
        weights = params.get("weights", None)
        return CompositeReward(reward_fns, weights)
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")
