"""Message sources: outbox generation from local observations (Phase 1).

A message source implements the decision-log rule ``u_i(t) = g_comm(o_i(t))``:
outgoing source messages are derived from the agent's current local
observation only — never from privileged simulator state, other agents'
observations, or the movement policy.

The first source is the engineered bearing report of
``docs/communication_decision_log.md`` ("Sensor Model") and
``docs/communication_implementation_plan.md`` ("Engineered Payload"): one
anonymous contact report per locally visible opponent with payload
``[observer_x, observer_y, direction_x, direction_y]`` where the direction is
a global-frame unit bearing ``(cos(theta), sin(theta))`` from the observer to
the detection. The payload contains no opponent identity, range, velocity, or
confidence. Frame metadata carries the originating teammate id and the
observation step; the ground-truth opponent id is attached only as privileged
(never policy-visible) frame metadata for evaluation.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

import torch

from src.communication.transport import Frame
from src.utils.logger import get_logger

logger = get_logger("acn.communication")

__all__ = ["MessageSource", "EngineeredBearingSource", "create_message_source"]

#: Observation keys holding visible-opponent mappings, tried in order.
_OPPONENT_KEYS: Tuple[str, ...] = ("red_agents", "blue_agents")


@runtime_checkable
class MessageSource(Protocol):
    """Creates per-agent outboxes from local observations only."""

    def build_outboxes(
        self,
        local_observations: Mapping[str, Mapping[str, object]],
        step: int,
        context: Optional[Mapping[str, object]] = None,
    ) -> Dict[str, List[Frame]]:
        """Derive each agent's outgoing frames from its local observation.

        Args:
            local_observations: Policy-visible local observation per agent id.
            step: Current movement-step index.
            context: Optional public runtime context.

        Returns:
            A mapping from agent id to its (possibly empty) list of frames.
        """
        ...


def _position_xy(value: object, context: str) -> Tuple[float, float]:
    """Coerce a position value (sequence or array of length 2) to floats."""
    try:
        x, y = value[0], value[1]  # type: ignore[index]
        return float(x), float(y)
    except (TypeError, IndexError, KeyError, ValueError):
        raise ValueError(
            "{} must be an (x, y) pair, got {!r}.".format(context, value)
        ) from None


class EngineeredBearingSource:
    """Deterministic anonymous bearing-report source.

    For every locally visible opponent in an agent's observation, emits one
    :class:`~src.communication.transport.Frame` whose payload is
    ``[observer_x, observer_y, direction_x, direction_y]`` (global frame,
    unit bearing). Reports are anonymous: multiple detections produce a
    variable-size collection of frames without padding to a global opponent
    count, and nothing in the payload identifies, ranges, or tracks the
    detected opponent.

    Observations follow the environment's dict format: observer position
    under ``"position"`` and visible opponents as a mapping under
    ``"red_agents"`` (blue observers) or ``"blue_agents"`` (red observers),
    each entry carrying a ``"position"`` pair. Teammate entries such as
    ``"red_teammates"`` are never read. An explicit ``opponent_key`` can
    override the automatic key detection.
    """

    #: Payload layout of one contact report.
    PAYLOAD_DIMENSION = 4

    def __init__(self, opponent_key: Optional[str] = None) -> None:
        """Create the source.

        Args:
            opponent_key: Optional explicit observation key holding the
                visible-opponent mapping. ``None`` auto-detects
                ``"red_agents"`` then ``"blue_agents"``.
        """
        self._opponent_key = opponent_key

    def _visible_opponents(
        self, observation: Mapping[str, object], agent_id: str
    ) -> Mapping[str, Mapping[str, object]]:
        """Return the visible-opponent mapping of one observation (may be empty)."""
        keys: Sequence[str]
        keys = (self._opponent_key,) if self._opponent_key is not None else _OPPONENT_KEYS
        for key in keys:
            value = observation.get(key)
            if value is not None:
                if not isinstance(value, Mapping):
                    raise ValueError(
                        "Observation['{}'] of agent '{}' must be a mapping of visible "
                        "opponents, got {}.".format(key, agent_id, type(value).__name__)
                    )
                return value
        return {}

    def build_frames(
        self, agent_id: str, observation: Mapping[str, object], step: int
    ) -> List[Frame]:
        """Build the contact-report frames of one agent from its observation.

        Args:
            agent_id: The observing agent's id (becomes the frame origin).
            observation: The agent's local observation dict.
            step: Current movement-step index (becomes the observation step).

        Returns:
            One frame per visible opponent, ordered by opponent id for
            determinism. Zero-distance detections carry the degenerate
            direction ``(0.0, 0.0)``.
        """
        opponents = self._visible_opponents(observation, agent_id)
        if not opponents:
            return []
        if "position" not in observation:
            raise ValueError(
                "Observation of agent '{}' has no 'position'; the bearing report "
                "requires the observer's own global position.".format(agent_id)
            )
        observer_x, observer_y = _position_xy(
            observation["position"], "Observation['position'] of agent '{}'".format(agent_id)
        )

        frames: List[Frame] = []
        for opponent_id in sorted(opponents):
            info = opponents[opponent_id]
            if not isinstance(info, Mapping) or "position" not in info:
                raise ValueError(
                    "Visible-opponent entry '{}' of agent '{}' must be a mapping with a "
                    "'position' pair, got {!r}.".format(opponent_id, agent_id, info)
                )
            target_x, target_y = _position_xy(
                info["position"],
                "Opponent '{}' position in observation of '{}'".format(opponent_id, agent_id),
            )
            delta_x = target_x - observer_x
            delta_y = target_y - observer_y
            norm = math.hypot(delta_x, delta_y)
            if norm > 0.0:
                direction_x, direction_y = delta_x / norm, delta_y / norm
            else:
                direction_x, direction_y = 0.0, 0.0
            frames.append(
                Frame(
                    payload=torch.tensor(
                        [observer_x, observer_y, direction_x, direction_y],
                        dtype=torch.float32,
                    ),
                    origin=agent_id,
                    created_step=step,
                    created_round=0,
                    observation_step=step,
                    # Privileged ground-truth identity for evaluation only;
                    # never enters the payload or policy-visible metadata.
                    privileged={"opponent_id": opponent_id},
                )
            )
        return frames

    def build_outboxes(
        self,
        local_observations: Mapping[str, Mapping[str, object]],
        step: int,
        context: Optional[Mapping[str, object]] = None,
    ) -> Dict[str, List[Frame]]:
        """Derive every agent's outbox from its local observation.

        Args:
            local_observations: Policy-visible local observation per agent id.
            step: Current movement-step index.
            context: Ignored (accepted for source-signature uniformity).

        Returns:
            A mapping from agent id (sorted for determinism) to its frames;
            agents with no visible opponents map to an empty list.
        """
        outboxes: Dict[str, List[Frame]] = {}
        for agent_id in sorted(local_observations):
            outboxes[agent_id] = self.build_frames(
                agent_id, local_observations[agent_id], step
            )
        total = sum(len(frames) for frames in outboxes.values())
        logger.debug(
            "EngineeredBearingSource built {} contact report(s) across {} agent(s) at step {}",
            total,
            len(outboxes),
            step,
        )
        return outboxes


def create_message_source(payload_config: Optional[object] = None) -> MessageSource:
    """Build the message source selected by a communication payload config.

    Maps the ``communication.payload`` section onto the source that derives
    per-agent outboxes from local observations (``u_i(t) = g_comm(o_i(t))``).
    ``engineered_vector`` — the Phase 1 baseline — yields the deterministic
    :class:`EngineeredBearingSource`.

    Args:
        payload_config: The ``communication.payload`` section — a typed
            :class:`~src.communication.config.PayloadConfig`, a mapping with
            the documented field names, or ``None`` for the defaults
            (``engineered_vector``).

    Returns:
        The :class:`MessageSource` for the configured payload type.

    Raises:
        NotImplementedError: For payload types defined by the implementation
            plan but without a source yet (``learned``).
        ValueError: For unknown payload types.
    """
    if payload_config is None:
        payload_type: object = "engineered_vector"
    elif isinstance(payload_config, Mapping):
        payload_type = payload_config.get("type", "engineered_vector")
    else:
        payload_type = getattr(payload_config, "type", "engineered_vector")

    if payload_type == "engineered_vector":
        return EngineeredBearingSource()
    if payload_type == "learned":
        raise NotImplementedError(
            "Communication payload type 'learned' has no message source yet; learned "
            "payloads are delivered in Phase 4/5 of "
            "docs/communication_implementation_plan.md."
        )
    raise ValueError(
        "Unknown communication payload type {!r}. Known types: engineered_vector, "
        "learned.".format(payload_type)
    )
