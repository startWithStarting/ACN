"""Metric calculators for benchmark results.

This module provides functions for calculating common metrics:
- avg_episode_reward
- detection_rate
- red_score_rate
- communication_utility
- convergence_speed

Usage:
    metrics = calculate_metrics(episodes, ["avg_reward", "detection_rate"])
"""

from typing import Dict, List, Any


def avg_episode_reward(episodes: List[Dict[str, Any]]) -> float:
    """Calculate average reward across episodes."""
    if not episodes:
        return 0.0

    total = sum(ep.get("total_reward", 0) for ep in episodes)
    return total / len(episodes)


def detection_rate(episodes: List[Dict[str, Any]]) -> float:
    """Calculate detection rate (detections / total steps)."""
    if not episodes:
        return 0.0

    total_detections = sum(ep.get("red_detections", 0) for ep in episodes)
    total_steps = sum(ep.get("steps", 0) for ep in episodes)

    if total_steps == 0:
        return 0.0

    return total_detections / total_steps


def red_score_rate(episodes: List[Dict[str, Any]]) -> float:
    """Calculate red agent scoring rate."""
    if not episodes:
        return 0.0

    total_scored = sum(ep.get("red_scored", 0) for ep in episodes)
    return total_scored / len(episodes)


def communication_utility(episodes_with_comms: List[Dict[str, Any]],
                         episodes_without: List[Dict[str, Any]]) -> float:
    """Calculate utility of communication: (with - without) / without."""
    if not episodes_with_comms or not episodes_without:
        return 0.0

    reward_with = avg_episode_reward(episodes_with_comms)
    reward_without = avg_episode_reward(episodes_without)

    if reward_without == 0:
        return 0.0

    return (reward_with - reward_without) / abs(reward_without)


def convergence_speed(episodes: List[Dict[str, Any]], threshold: float = 0.95) -> int:
    """
    Calculate at which episode the metric reaches threshold of its final value.
    Returns -1 if never converges.
    """
    if not episodes:
        return -1

    # Use avg_reward as the convergence metric
    rewards = [ep.get("total_reward", 0) for ep in episodes]

    if not rewards:
        return -1

    final_avg = sum(rewards) / len(rewards)
    target = final_avg * threshold

    for i, reward in enumerate(rewards):
        if reward >= target:
            return i + 1  # 1-indexed episode number

    return -1


def calculate_metrics(episodes: List[Dict[str, Any]], metric_names: List[str]) -> Dict[str, float]:
    """
    Calculate multiple metrics from episode data.

    Args:
        episodes: List of episode result dicts
        metric_names: List of metric names to calculate

    Returns:
        Dict mapping metric names to values
    """
    available_metrics = {
        "avg_reward": lambda e: avg_episode_reward(e),
        "detection_rate": lambda e: detection_rate(e),
        "red_score_rate": lambda e: red_score_rate(e),
        "convergence_step": lambda e: convergence_speed(e),
    }

    results = {}
    for name in metric_names:
        if name in available_metrics:
            results[name] = available_metrics[name](episodes)
        else:
            results[name] = 0.0

    return results
