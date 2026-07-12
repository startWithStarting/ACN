"""Config-driven movement action-space construction and decoding.

This module owns how a movement action space is built from configuration and,
for the discrete variant, how flat action indices map to movement commands.
Agents receive their action space from :func:`build_movement_action_space` at
construction (the factory passes it); they no longer hardcode spaces.

Two specifications exist (``environment.action_space.type``):

- ``continuous`` (default): the legacy
  ``Dict{direction: Box[-1, 1]^2, speed: Box[0, max_speed]}`` space. With the
  default ``max_speed`` of ``10.0`` this is byte-for-byte the space the agents
  used to hardcode, so existing scenarios are unchanged.
- ``discrete``: a single flat ``Discrete(N)`` with ``N = 1 + H * (S - 1)``.
  ``H`` evenly spaced unit headings (``theta_k = 2*pi*k/H`` starting at 0) are
  combined with ``S`` speed levels evenly spaced from 0 to ``max_speed``.
  Index 0 is the shared ``stay`` action (every speed-0 choice collapses onto
  it); the remaining indices enumerate heading-major (heading, nonzero-speed)
  pairs. Headings are unit-norm, so the speed cap is isotropic, unlike the
  legacy un-normalized ``direction * speed`` parameterization.

``max_speed`` resolves per team from the agent spec
(``agents.<side>[i].max_speed``, default ``10.0`` -- the current cap), and the
factory writes the resolved value onto every agent, mirroring how
``communication_radius`` resolution is specified in
``docs/communication_implementation_plan.md``.

Physics boundary (see docs/communication_decision_log.md, "Action Space"):
in ``velocity`` control mode (the benchmark default) a decoded speed level is
the achieved speed exactly. In ``force`` control mode the decoded levels act as
force magnitudes and the true top speed emerges from the force/drag balance, so
the discrete ``max_speed`` cap does not apply there; a hard cap under force
control would require post-integration magnitude clipping, which is out of
scope for this module.
"""

from __future__ import annotations

import math
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple, Union

import gymnasium.spaces as spaces
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("acn.agents.action_spaces")

__all__ = [
    "ActionSpaceConfigError",
    "VALID_ACTION_SPACE_TYPES",
    "DEFAULT_MAX_SPEED",
    "DEFAULT_HEADINGS",
    "DEFAULT_SPEED_LEVELS",
    "ActionSpaceConfig",
    "DiscreteMovementSpec",
    "build_continuous_movement_space",
    "build_discrete_movement_spec",
    "build_movement_action_space",
    "resolve_spec_max_speed",
]


class ActionSpaceConfigError(ValueError):
    """Raised when an ``environment.action_space`` configuration is invalid."""


#: Accepted values for ``environment.action_space.type``.
VALID_ACTION_SPACE_TYPES: Tuple[str, ...] = ("continuous", "discrete")

#: Historic hardcoded speed cap; the default when an agent spec has no ``max_speed``.
DEFAULT_MAX_SPEED: float = 10.0

#: Default number of evenly spaced unit headings (H).
DEFAULT_HEADINGS: int = 8

#: Default number of speed levels including the zero level (S).
DEFAULT_SPEED_LEVELS: int = 4


def _require_positive_int(value: Any, name: str, minimum: int) -> int:
    """Validate an integer config field, rejecting bools and small values."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ActionSpaceConfigError(
            "action_space.{} must be an integer, got {} ({!r}).".format(
                name, type(value).__name__, value
            )
        )
    if value < minimum:
        raise ActionSpaceConfigError(
            "action_space.{} must be >= {} (got {}).".format(name, minimum, value)
        )
    return value


def _require_positive_number(value: Any, context: str) -> float:
    """Validate a strictly positive, finite numeric value and return it as float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ActionSpaceConfigError(
            "{} must be a number, got {} ({!r}).".format(context, type(value).__name__, value)
        )
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ActionSpaceConfigError(
            "{} must be a positive finite number (got {!r}).".format(context, value)
        )
    return number


@dataclass(frozen=True)
class ActionSpaceConfig:
    """Validated ``environment.action_space`` configuration.

    The default-constructed instance is the canonical legacy configuration:
    continuous ``Dict{direction, speed}`` movement spaces.

    Attributes:
        type: ``"continuous"`` (default) or ``"discrete"``.
        headings: H, the number of evenly spaced unit headings (discrete mode).
        speed_levels: S, the number of speed levels evenly spaced from 0 to
            ``max_speed`` including the zero level (discrete mode).
    """

    type: str = "continuous"
    headings: int = DEFAULT_HEADINGS
    speed_levels: int = DEFAULT_SPEED_LEVELS

    def __post_init__(self) -> None:
        if self.type not in VALID_ACTION_SPACE_TYPES:
            raise ActionSpaceConfigError(
                "action_space.type must be one of {} (got '{}').".format(
                    ", ".join(VALID_ACTION_SPACE_TYPES), self.type
                )
            )
        _require_positive_int(self.headings, "headings", 1)
        _require_positive_int(self.speed_levels, "speed_levels", 2)

    @property
    def is_discrete(self) -> bool:
        """True when the configured movement action space is discrete."""
        return self.type == "discrete"

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "ActionSpaceConfig":
        """Parse the ``environment.action_space`` block of a loaded YAML config.

        Args:
            data: The plain-dict block, or None when absent. An absent block
                yields the legacy continuous configuration.

        Returns:
            A validated, immutable ActionSpaceConfig.

        Raises:
            ActionSpaceConfigError: If the block is malformed (not a mapping,
                unknown type, non-integer or out-of-range headings/speed_levels).
        """
        if data is None:
            return cls()
        if not isinstance(data, MappingABC):
            raise ActionSpaceConfigError(
                "environment.action_space must be a mapping of keys to values, got {} "
                "({!r}). Check the YAML indentation of this block.".format(
                    type(data).__name__, data
                )
            )
        known = ("type", "headings", "speed_levels")
        unknown = sorted(set(data.keys()) - set(known))
        if unknown:
            logger.warning(
                "Ignoring unknown keys in environment.action_space: {}. Known keys: {}",
                ", ".join(unknown),
                ", ".join(known),
            )
        space_type = data.get("type", "continuous")
        if not isinstance(space_type, str):
            raise ActionSpaceConfigError(
                "action_space.type must be a string, got {} ({!r}).".format(
                    type(space_type).__name__, space_type
                )
            )
        return cls(
            type=space_type,
            headings=data.get("headings", DEFAULT_HEADINGS),
            speed_levels=data.get("speed_levels", DEFAULT_SPEED_LEVELS),
        )


@dataclass(frozen=True)
class DiscreteMovementSpec:
    """Flat discrete movement specification and its index <-> command mapping.

    The flat index layout is ``stay`` first, then heading-major moving actions:

    ```text
    index 0                        -> stay (zero velocity)
    index 1 + k*(S-1) + (j-1)      -> heading k in [0, H), speed level j in [1, S)
    ```

    Heading ``k`` decodes to the unit vector ``(cos theta_k, sin theta_k)`` with
    ``theta_k = 2*pi*k/H``; speed level ``j`` decodes to ``j * max_speed/(S-1)``.
    The target velocity is ``speed * unit_direction``. In ``velocity`` physics
    control mode the level is the achieved speed exactly; ``force`` mode is out
    of scope (see module docstring).

    Attributes:
        headings: H, number of evenly spaced unit headings (>= 1).
        speed_levels: S, number of speed levels including zero (>= 2).
        max_speed: Top speed level; positive and finite.
    """

    headings: int
    speed_levels: int
    max_speed: float

    def __post_init__(self) -> None:
        _require_positive_int(self.headings, "headings", 1)
        _require_positive_int(self.speed_levels, "speed_levels", 2)
        object.__setattr__(
            self, "max_speed", _require_positive_number(self.max_speed, "max_speed")
        )

    @property
    def n(self) -> int:
        """Total number of flat actions: ``1 + H * (S - 1)``."""
        return 1 + self.headings * (self.speed_levels - 1)

    @property
    def speed_step(self) -> float:
        """Speed difference between adjacent nonzero levels: ``max_speed / (S - 1)``."""
        return self.max_speed / (self.speed_levels - 1)

    def heading_vector(self, heading_index: int) -> np.ndarray:
        """Return the unit direction for heading ``k``: ``(cos theta_k, sin theta_k)``.

        Args:
            heading_index: Heading index in ``[0, H)``.

        Returns:
            A float64 unit vector of shape (2,), renormalized so its Euclidean
            norm is 1 to within floating-point precision (isotropic speed cap).
        """
        if not 0 <= heading_index < self.headings:
            raise ValueError(
                "heading_index must be in [0, {}) (got {}).".format(
                    self.headings, heading_index
                )
            )
        theta = 2.0 * math.pi * heading_index / self.headings
        vector = np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)
        return vector / np.linalg.norm(vector)

    def decode(self, index: Union[int, np.integer]) -> Tuple[np.ndarray, float]:
        """Decode a flat action index into ``(unit_direction, speed)``.

        Args:
            index: Flat action index in ``[0, N)``. Accepts Python and numpy
                integers; bools are rejected.

        Returns:
            Tuple of a float64 unit direction of shape (2,) (the zero vector
            for the stay action) and the speed as a float. The target velocity
            is ``speed * unit_direction``.

        Raises:
            ValueError: If ``index`` is not an integer or is out of range.
        """
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise ValueError(
                "Discrete movement action must be an integer index, got {} ({!r}).".format(
                    type(index).__name__, index
                )
            )
        idx = int(index)
        if not 0 <= idx < self.n:
            raise ValueError(
                "Discrete movement action {} out of range for Discrete({}) "
                "(headings={}, speed_levels={}).".format(
                    idx, self.n, self.headings, self.speed_levels
                )
            )
        if idx == 0:
            return np.zeros(2, dtype=np.float64), 0.0
        heading_index, level_offset = divmod(idx - 1, self.speed_levels - 1)
        speed = (level_offset + 1) * self.speed_step
        return self.heading_vector(heading_index), speed

    def decode_velocity(self, index: Union[int, np.integer]) -> np.ndarray:
        """Decode a flat action index straight to a target velocity vector.

        Args:
            index: Flat action index in ``[0, N)``.

        Returns:
            Float64 velocity vector of shape (2,): ``speed * unit_direction``.
        """
        unit_direction, speed = self.decode(index)
        return unit_direction * speed

    def encode(self, unit_direction: Any, speed: Any) -> int:
        """Encode ``(unit_direction, speed)`` back into a flat action index.

        Exact inverse of :meth:`decode`: for every index ``i`` in ``[0, N)``,
        ``encode(*decode(i)) == i``. Values must lie on the discrete lattice
        (a decoded heading and speed level); anything else raises.

        Args:
            unit_direction: Array-like of shape (2,); ignored for zero speed.
            speed: Speed value; must match a level ``j * max_speed/(S-1)``.

        Returns:
            The flat action index.

        Raises:
            ValueError: If the speed is not a valid level or the direction does
                not match any heading.
        """
        speed_value = float(speed)
        if not math.isfinite(speed_value) or speed_value < 0.0:
            raise ValueError(
                "Speed must be a finite non-negative number (got {!r}).".format(speed)
            )
        level = int(round(speed_value / self.speed_step))
        if level >= self.speed_levels or not math.isclose(
            level * self.speed_step, speed_value, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise ValueError(
                "Speed {!r} is not one of the {} discrete levels spaced {} apart "
                "(max_speed={}).".format(
                    speed, self.speed_levels, self.speed_step, self.max_speed
                )
            )
        if level == 0:
            return 0
        direction = np.asarray(unit_direction, dtype=np.float64).reshape(-1)
        if direction.shape[0] != 2:
            raise ValueError(
                "unit_direction must have exactly 2 components (got shape {}).".format(
                    direction.shape
                )
            )
        theta = math.atan2(direction[1], direction[0]) % (2.0 * math.pi)
        heading_index = int(round(theta * self.headings / (2.0 * math.pi))) % self.headings
        if not np.allclose(direction, self.heading_vector(heading_index), atol=1e-9):
            raise ValueError(
                "Direction {!r} does not match any of the {} evenly spaced unit "
                "headings.".format(unit_direction, self.headings)
            )
        return 1 + heading_index * (self.speed_levels - 1) + (level - 1)

    def gymnasium_space(self) -> spaces.Discrete:
        """Return the flat gymnasium ``Discrete(N)`` space for this spec."""
        return spaces.Discrete(self.n)


def build_continuous_movement_space(max_speed: float = DEFAULT_MAX_SPEED) -> spaces.Dict:
    """Build the legacy continuous movement space.

    Args:
        max_speed: Upper bound of the speed Box. The default of 10.0 reproduces
            the previously hardcoded agent spaces exactly.

    Returns:
        ``Dict{direction: Box[-1, 1]^2, speed: Box[0, max_speed]}``.
    """
    max_speed = _require_positive_number(max_speed, "max_speed")
    return spaces.Dict(
        {
            "direction": spaces.Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                shape=(2,),
                dtype=np.float32,
            ),
            "speed": spaces.Box(low=0.0, high=max_speed, shape=(1,), dtype=np.float32),
        }
    )


def build_discrete_movement_spec(
    config: ActionSpaceConfig, max_speed: float
) -> DiscreteMovementSpec:
    """Build the discrete movement spec for one agent from config and its cap.

    Args:
        config: Validated action-space configuration (``headings``/``speed_levels``).
        max_speed: The agent's resolved per-team speed cap.

    Returns:
        A DiscreteMovementSpec combining the shared layout with the agent's cap.
    """
    return DiscreteMovementSpec(
        headings=config.headings,
        speed_levels=config.speed_levels,
        max_speed=max_speed,
    )


def build_movement_action_space(
    config: Optional[ActionSpaceConfig], max_speed: float = DEFAULT_MAX_SPEED
) -> spaces.Space:
    """Build the gymnasium movement action space for one agent.

    Args:
        config: Validated action-space configuration; None means the default
            (legacy continuous) configuration.
        max_speed: The agent's resolved per-team speed cap.

    Returns:
        The legacy ``Dict`` space in continuous mode, or ``Discrete(N)`` with
        ``N = 1 + headings * (speed_levels - 1)`` in discrete mode.
    """
    if config is None:
        config = ActionSpaceConfig()
    if config.is_discrete:
        return build_discrete_movement_spec(config, max_speed).gymnasium_space()
    return build_continuous_movement_space(max_speed)


def resolve_spec_max_speed(spec: Optional[Mapping[str, Any]]) -> float:
    """Resolve an agent group's speed cap from its config spec.

    Resolution order: the group spec's ``max_speed`` value, else
    :data:`DEFAULT_MAX_SPEED` (10.0, the historic hardcoded cap). The factory
    writes the resolved value onto every agent in the group.

    Args:
        spec: One entry of ``agents.blue_agents`` / ``agents.red_agents``, or None.

    Returns:
        The resolved positive, finite cap as a float.

    Raises:
        ActionSpaceConfigError: If the configured value is not a positive
            finite number.
    """
    if spec is None:
        return DEFAULT_MAX_SPEED
    value = spec.get("max_speed", DEFAULT_MAX_SPEED)
    return _require_positive_number(value, "agent spec max_speed")
