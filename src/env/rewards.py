"""Reward function modularity for ACN environments.

This module provides a Protocol-based system for different reward functions,
allowing easy swapping and configuration of reward mechanisms.

Usage:
    # Config-driven selection
    reward_fn = create_reward_function(config.get("reward_function", {}))

    # Use in environment
    reward, details = reward_fn(agent_name, agent_obj, env_state)

It also contains the config-gated **benchmark reward mode** (see
``docs/communication_decision_log.md``, sections "Blue Reward Design",
"Dense Reward And Reward/Metric Decoupling", and "Red Reward Design"):
pure functions over a per-step :class:`DetectionState`, wired into
``ACNEnvironmentLogic`` behind ``environment.reward``. The default
(``legacy``) mode reproduces current runtime behavior exactly.
"""

from __future__ import annotations

import math
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from typing import Protocol, Tuple, Dict, Any, FrozenSet, Iterable, Mapping, Optional, Sequence

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


# ---------------------------------------------------------------------------
# Benchmark reward mode (config-gated via ``environment.reward``)
# ---------------------------------------------------------------------------

REWARD_MODE_LEGACY = "legacy"
REWARD_MODE_BENCHMARK = "benchmark"
_VALID_REWARD_MODES = (REWARD_MODE_LEGACY, REWARD_MODE_BENCHMARK)

#: k_r >= 3 detectors means red r is "pinned" (triangulation gate).
PIN_DETECTOR_THRESHOLD = 3
#: streak_r >= 3 consecutive pinned steps means red r is "tracked".
TRACK_STREAK_THRESHOLD = 3

#: Weight keys accepted under ``environment.reward.weights``.
_WEIGHT_KEYS = ("pin", "track", "shape", "score_penalty", "red_score", "red_track", "red_progress")


@dataclass(frozen=True)
class RewardSettings:
    """Parsed and validated ``environment.reward`` configuration.

    Attributes:
        blue_mode: ``"legacy"`` (passive no-score bonus) or ``"benchmark"``
            (pin/track/shape coverage minus scoring penalty).
        red_mode: ``"legacy"`` (sparse undetected ring scoring) or ``"benchmark"``
            (ring score, tracked penalty, potential-based progress shaping).
        pin_weight: Weight on team pin coverage (blue benchmark).
        track_weight: Weight on team track coverage (blue benchmark).
        shape_weight: Weight on the individual undercoverage shaping term
            (blue benchmark).
        score_penalty_weight: Weight on the newly-scoring-red fraction penalty.
            Required when ``blue_mode == "benchmark"``.
        red_score_weight: Weight on undetected on-ring occupancy (red benchmark).
        red_track_weight: Weight on the tracked penalty. Required when
            ``red_mode == "benchmark"``.
        red_progress_weight: Weight on potential-based progress-to-ring shaping.
            Required when ``red_mode == "benchmark"``.
    """

    blue_mode: str = REWARD_MODE_LEGACY
    red_mode: str = REWARD_MODE_LEGACY
    pin_weight: float = 0.5
    track_weight: float = 1.0
    shape_weight: float = 0.1
    score_penalty_weight: Optional[float] = None
    red_score_weight: float = 1.0
    red_track_weight: Optional[float] = None
    red_progress_weight: Optional[float] = None

    @property
    def blue_benchmark(self) -> bool:
        """Whether the blue team uses the benchmark reward."""
        return self.blue_mode == REWARD_MODE_BENCHMARK

    @property
    def red_benchmark(self) -> bool:
        """Whether the red team uses the benchmark reward."""
        return self.red_mode == REWARD_MODE_BENCHMARK

    @property
    def benchmark_enabled(self) -> bool:
        """Whether any team uses the benchmark reward path."""
        return self.blue_benchmark or self.red_benchmark


def parse_reward_settings(raw: Optional[Mapping[str, Any]]) -> RewardSettings:
    """Parse and validate the ``environment.reward`` config block.

    An absent or empty block yields the legacy/legacy default, which reproduces
    current runtime behavior exactly.

    Args:
        raw: The raw ``environment.reward`` mapping from YAML, or ``None``.

    Returns:
        An immutable :class:`RewardSettings`.

    Raises:
        ValueError: On an unknown reward mode, an unknown key, a non-numeric
            weight, or a weight required by an enabled benchmark mode that is
            missing (``score_penalty`` for blue; ``red_track`` and
            ``red_progress`` for red).
    """
    if raw is None:
        raw = {}
    if not isinstance(raw, MappingABC):
        raise ValueError(
            "environment.reward must be a mapping, got {!r}".format(type(raw).__name__)
        )

    unknown = sorted(set(raw) - {"blue", "red", "weights"})
    if unknown:
        raise ValueError(
            "Unknown environment.reward keys {}; expected 'blue', 'red', 'weights'".format(unknown)
        )

    blue_mode = raw.get("blue", REWARD_MODE_LEGACY)
    red_mode = raw.get("red", REWARD_MODE_LEGACY)
    for side, mode in (("blue", blue_mode), ("red", red_mode)):
        if mode not in _VALID_REWARD_MODES:
            raise ValueError(
                "environment.reward.{} must be one of {}, got {!r}".format(
                    side, list(_VALID_REWARD_MODES), mode
                )
            )

    weights = raw.get("weights") or {}
    if not isinstance(weights, MappingABC):
        raise ValueError(
            "environment.reward.weights must be a mapping, got {!r}".format(
                type(weights).__name__
            )
        )
    unknown_weights = sorted(set(weights) - set(_WEIGHT_KEYS))
    if unknown_weights:
        raise ValueError(
            "Unknown environment.reward.weights keys {}; expected keys from {}".format(
                unknown_weights, list(_WEIGHT_KEYS)
            )
        )
    for key, value in weights.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                "environment.reward.weights.{} must be a number, got {!r}".format(key, value)
            )

    if blue_mode == REWARD_MODE_BENCHMARK and "score_penalty" not in weights:
        raise ValueError(
            "environment.reward.weights.score_penalty is required when "
            "environment.reward.blue is 'benchmark'"
        )
    if red_mode == REWARD_MODE_BENCHMARK:
        missing = [key for key in ("red_track", "red_progress") if key not in weights]
        if missing:
            raise ValueError(
                "environment.reward.weights {} are required when "
                "environment.reward.red is 'benchmark'".format(missing)
            )

    def _optional(key: str) -> Optional[float]:
        return float(weights[key]) if key in weights else None

    return RewardSettings(
        blue_mode=blue_mode,
        red_mode=red_mode,
        pin_weight=float(weights.get("pin", 0.5)),
        track_weight=float(weights.get("track", 1.0)),
        shape_weight=float(weights.get("shape", 0.1)),
        score_penalty_weight=_optional("score_penalty"),
        red_score_weight=float(weights.get("red_score", 1.0)),
        red_track_weight=_optional("red_track"),
        red_progress_weight=_optional("red_progress"),
    )


@dataclass(frozen=True)
class DetectionState:
    """Per-step blue-red detection state shared by both teams' benchmark rewards.

    Built once per environment step by :func:`build_detection_state`. All
    mappings are keyed by agent name; ``red_names`` lists ACTIVE reds only, and
    every derived mapping covers exactly those names.

    Attributes:
        blue_names: Active blue agent names, in team order.
        red_names: Active red agent names, in team order.
        detections: Blue name -> frozenset of red names it currently detects.
        k: Red name -> number of blue agents detecting it (``k_r``).
        pinned: Red name -> ``k_r >= 3`` (triangulation gate).
        streaks: Red name -> consecutive pinned steps, including this one.
        tracked: Red name -> ``streak_r >= 3`` (sustained-tracking gate).
    """

    blue_names: Tuple[str, ...]
    red_names: Tuple[str, ...]
    detections: Dict[str, FrozenSet[str]]
    k: Dict[str, int]
    pinned: Dict[str, bool]
    streaks: Dict[str, int]
    tracked: Dict[str, bool]


def build_detection_state(
    active_blue_agents: Sequence[Any],
    active_red_agents: Sequence[Any],
    previous_streaks: Mapping[str, int],
) -> DetectionState:
    """Build the per-step detection matrix and derived pin/streak/track state.

    Pure with respect to its inputs: agents and ``previous_streaks`` are read,
    never mutated. The environment owns streak persistence across steps: it
    passes the previous step's streaks in, stores the returned ``streaks``, and
    clears them on episode reset.

    Detection uses the same test as the legacy detection path
    (``blue.is_within_detection_radius(red position)`` with the same
    missing-attribute guards), so ``k_r == 0`` is exactly "undetected" in the
    legacy scoring sense.

    Args:
        active_blue_agents: Active blue agent objects.
        active_red_agents: Active red agent objects.
        previous_streaks: Red name -> consecutive pinned-step count carried from
            the previous step (empty at episode start).

    Returns:
        A :class:`DetectionState` for this step.
    """
    blue_names = []
    detections: Dict[str, set] = {}
    usable_blues = []
    for blue_agent in active_blue_agents:
        blue_names.append(blue_agent.name)
        detections[blue_agent.name] = set()
        if (
            not getattr(blue_agent, "is_active", False)
            or not hasattr(blue_agent, "is_within_detection_radius")
            or getattr(blue_agent, "x", None) is None
            or getattr(blue_agent, "y", None) is None
        ):
            continue
        usable_blues.append(blue_agent)

    red_names = []
    k: Dict[str, int] = {}
    pinned: Dict[str, bool] = {}
    streaks: Dict[str, int] = {}
    tracked: Dict[str, bool] = {}
    for red_agent in active_red_agents:
        red_name = red_agent.name
        red_names.append(red_name)
        count = 0
        if getattr(red_agent, "x", None) is not None and getattr(red_agent, "y", None) is not None:
            red_pos = (red_agent.x, red_agent.y)
            for blue_agent in usable_blues:
                if blue_agent.is_within_detection_radius(red_pos):
                    detections[blue_agent.name].add(red_name)
                    count += 1
        k[red_name] = count
        is_pinned = count >= PIN_DETECTOR_THRESHOLD
        pinned[red_name] = is_pinned
        streak = previous_streaks.get(red_name, 0) + 1 if is_pinned else 0
        streaks[red_name] = streak
        tracked[red_name] = streak >= TRACK_STREAK_THRESHOLD

    return DetectionState(
        blue_names=tuple(blue_names),
        red_names=tuple(red_names),
        detections={name: frozenset(reds) for name, reds in detections.items()},
        k=k,
        pinned=pinned,
        streaks=streaks,
        tracked=tracked,
    )


def shape_undercoverage_gate(k: int) -> float:
    """Undercoverage factor ``max(0, 4 - (k - 1)) / 2`` from the blue shaping term.

    Yields ``1.0`` at ``k == 3``, ``0.5`` at ``k == 4``, and ``0.0`` at
    ``k >= 5``, so shaping rewards useful participation around three or four
    observers and stops rewarding over-covered reds.

    Args:
        k: Number of blue agents detecting the red (``k_r``).

    Returns:
        The undercoverage factor.
    """
    return max(0.0, 4.0 - (k - 1)) / 2.0


def blue_benchmark_reward(
    blue_name: str,
    state: DetectionState,
    red_score_fraction: float,
    settings: RewardSettings,
) -> float:
    """Benchmark blue reward: team coverage plus shaping minus scoring penalty.

    Implements the decision-log formula exactly::

        blue_reward_i = pin_weight * pin_coverage
                      + track_weight * track_coverage
                      + shape_weight * shape_i
                      - score_penalty_weight * red_score_fraction

    where ``pin_coverage``/``track_coverage`` are means over ACTIVE reds and::

        shape_i = (1 / N_R) * sum over active reds r of
            1[blue_i detects r] * 1[k_r >= 3] * max(0, 4 - (k_r - 1)) / 2

    This REPLACES the legacy passive blue bonus; the two must never both apply.

    Args:
        blue_name: Name of the blue agent receiving the reward.
        state: The shared per-step detection state.
        red_score_fraction: Newly scoring reds this step / initial red count.
        settings: Parsed reward settings (``score_penalty_weight`` must be set
            when blue mode is benchmark; ``None`` is treated as ``0.0``).

    Returns:
        The blue agent's benchmark reward for this step.
    """
    n_red = len(state.red_names)
    if n_red > 0:
        pin_coverage = sum(1.0 for name in state.red_names if state.pinned[name]) / n_red
        track_coverage = sum(1.0 for name in state.red_names if state.tracked[name]) / n_red
        detected = state.detections.get(blue_name, frozenset())
        shape = (
            sum(
                shape_undercoverage_gate(state.k[name])
                for name in state.red_names
                if name in detected and state.pinned[name]
            )
            / n_red
        )
    else:
        pin_coverage = track_coverage = shape = 0.0

    score_penalty_weight = (
        settings.score_penalty_weight if settings.score_penalty_weight is not None else 0.0
    )
    return (
        settings.pin_weight * pin_coverage
        + settings.track_weight * track_coverage
        + settings.shape_weight * shape
        - score_penalty_weight * red_score_fraction
    )


def ring_potential(
    position: Iterable[float],
    center: Iterable[float],
    ring_radius: float,
) -> float:
    """Potential ``phi(s) = -|distance_to_center - ring_radius|``.

    Used for potential-based progress-to-ring shaping in the red benchmark
    reward: per-step contributions ``phi(s') - phi(s)`` telescope over an
    episode segment to ``phi(end) - phi(start)`` and do not move the optimum.

    Args:
        position: Agent ``(x, y)`` position.
        center: Grid center ``(x, y)``.
        ring_radius: Radius of the scoring ring around the center.

    Returns:
        The potential value (always ``<= 0``; ``0`` exactly on the ring).
    """
    x, y = tuple(position)[:2]
    center_x, center_y = tuple(center)[:2]
    distance_to_center = math.hypot(x - center_x, y - center_y)
    return -abs(distance_to_center - ring_radius)


def red_benchmark_reward(
    red_name: str,
    state: DetectionState,
    on_ring: bool,
    potential_delta: float,
    settings: RewardSettings,
) -> float:
    """Benchmark red reward: ring score, tracked penalty, and progress shaping.

    Implements the decision-log formula exactly::

        red_reward_i = red_score_weight * on_ring_undetected_i
                     - red_track_weight * tracked_i
                     + red_progress_weight * (phi(s') - phi(s))

    ``on_ring`` is the geometric ring-band test only (the same test as the
    legacy ``_calculate_reward``); the undetected gate (``k_r == 0``) comes from
    the shared detection state. ``tracked_i`` mirrors blue's ``tracked_r`` with
    a minus sign.

    Args:
        red_name: Name of the red agent receiving the reward.
        state: The shared per-step detection state.
        on_ring: Whether the red is within the ring tolerance band this step.
        potential_delta: ``phi(s') - phi(s)`` from :func:`ring_potential`.
        settings: Parsed reward settings (``red_track_weight`` and
            ``red_progress_weight`` must be set when red mode is benchmark;
            ``None`` is treated as ``0.0``).

    Returns:
        The red agent's benchmark reward for this step.
    """
    on_ring_undetected = 1.0 if (on_ring and state.k.get(red_name, 0) == 0) else 0.0
    tracked = 1.0 if state.tracked.get(red_name, False) else 0.0
    red_track_weight = settings.red_track_weight if settings.red_track_weight is not None else 0.0
    red_progress_weight = (
        settings.red_progress_weight if settings.red_progress_weight is not None else 0.0
    )
    return (
        settings.red_score_weight * on_ring_undetected
        - red_track_weight * tracked
        + red_progress_weight * potential_delta
    )
