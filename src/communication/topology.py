"""Radius-based communication topology (Phase 1).

Implements the ``RadiusTopology`` builder described in
``docs/communication_implementation_plan.md`` ("Topology Layer"): directed
same-team edges exist when the sender-receiver distance satisfies the
configured radius-ownership rule. The builder is deliberately ignorant of
configuration inheritance: communication-radius resolution
(agent > team default > environment fallback) happens *outside* this module,
either by writing the resolved value onto each agent as
``communication_radius`` or by passing an explicit per-agent radius mapping.

Determinism contract: node index order equals the order of the ``agents``
sequence passed to :meth:`RadiusTopology.build` (callers must supply a
deterministic order), and edges are emitted sorted by
``(source index, receiver index)``.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from src.communication.config import VALID_RADIUS_RULES
from src.communication.types import CommunicationGraph
from src.utils.logger import get_logger

logger = get_logger("acn.communication")

__all__ = ["RadiusTopology"]


def _resolve_team_label(agent: object) -> str:
    """Return the team label of an agent object.

    Resolution order: an explicit ``team`` attribute, then
    ``agent_type.value`` (the :class:`~src.agents.base_agent.AgentType`
    enum), then a plain-string ``agent_type``.

    Args:
        agent: Agent object.

    Returns:
        The team label as a string.

    Raises:
        ValueError: If no team information can be resolved.
    """
    team = getattr(agent, "team", None)
    if team is not None:
        return str(team)
    agent_type = getattr(agent, "agent_type", None)
    if agent_type is not None:
        value = getattr(agent_type, "value", agent_type)
        if isinstance(value, str) and value:
            return value
    raise ValueError(
        "Cannot resolve the team of agent {!r}: expected a 'team' attribute or an "
        "'agent_type' attribute (enum with .value or string).".format(
            getattr(agent, "name", agent)
        )
    )


class RadiusTopology:
    """Same-team radius graph builder with configurable radius-ownership rule.

    A directed edge ``source -> receiver`` exists when both agents are active,
    belong to the same team, and the Euclidean distance ``d`` between them
    satisfies the configured rule:

    - ``"sender"`` (default): ``d <= radius(source)`` — the transmission range
      of the sender controls the edge.
    - ``"receiver"``: ``d <= radius(receiver)``.
    - ``"mutual"``: both ranges must cover the link, i.e.
      ``d <= radius(source) and d <= radius(receiver)``.
    - ``"minimum"``: ``d <= min(radius(source), radius(receiver))``.

    ``"mutual"`` and ``"minimum"`` are currently equivalent for symmetric
    Euclidean distances; they are kept as distinct named rules because later
    extensions (directional antennas, attenuation) can make them diverge.

    Self-edges are disabled by default and, when enabled, have distance 0.

    Radius resolution happens outside this class: pass a complete per-agent
    ``radius_by_agent`` mapping, or leave it ``None`` to read each agent's
    already-resolved ``communication_radius`` attribute.
    """

    def __init__(
        self,
        radius_by_agent: Optional[Mapping[str, float]] = None,
        radius_rule: str = "sender",
        include_self_edges: bool = False,
    ) -> None:
        """Create a radius topology builder.

        Args:
            radius_by_agent: Optional explicit per-agent communication radius,
                keyed by agent name. When provided it takes precedence over
                (and fully replaces) the ``communication_radius`` attribute.
            radius_rule: One of ``mutual``, ``sender``, ``receiver``,
                ``minimum``. Defaults to ``sender``.
            include_self_edges: Whether to emit ``i -> i`` edges. Defaults to
                ``False`` (transport default per the implementation plan).

        Raises:
            ValueError: If ``radius_rule`` is not a known rule.
        """
        if radius_rule not in VALID_RADIUS_RULES:
            raise ValueError(
                "radius_rule must be one of {} (got {!r}).".format(
                    ", ".join(VALID_RADIUS_RULES), radius_rule
                )
            )
        self._radius_by_agent = dict(radius_by_agent) if radius_by_agent is not None else None
        self._radius_rule = radius_rule
        self._include_self_edges = include_self_edges

    @property
    def radius_rule(self) -> str:
        """The configured radius-ownership rule."""
        return self._radius_rule

    @property
    def include_self_edges(self) -> bool:
        """Whether self-edges are included."""
        return self._include_self_edges

    def _resolve_radius(self, name: str, agent: object) -> float:
        """Return the resolved communication radius for one agent.

        Args:
            name: The agent's name.
            agent: The agent object.

        Returns:
            The non-negative communication radius.

        Raises:
            ValueError: If no radius can be resolved or it is negative.
        """
        if self._radius_by_agent is not None:
            if name not in self._radius_by_agent:
                raise ValueError(
                    "No communication radius for agent '{}' in the provided radius mapping. "
                    "Radius resolution (agent > team default > environment fallback) must "
                    "cover every active agent.".format(name)
                )
            radius = self._radius_by_agent[name]
        else:
            radius = getattr(agent, "communication_radius", None)
            if radius is None:
                raise ValueError(
                    "Agent '{}' has no resolved 'communication_radius' attribute and no "
                    "radius mapping was provided. Resolve the radius (agent > team default "
                    "> environment fallback) before building the topology.".format(name)
                )
        radius = float(radius)
        if radius < 0.0 or math.isnan(radius):
            raise ValueError(
                "Communication radius of agent '{}' must be a non-negative number, "
                "got {}.".format(name, radius)
            )
        return radius

    def _edge_allowed(self, distance: float, source_radius: float, receiver_radius: float) -> bool:
        """Return whether a directed edge is valid under the configured rule."""
        if self._radius_rule == "sender":
            return distance <= source_radius
        if self._radius_rule == "receiver":
            return distance <= receiver_radius
        if self._radius_rule == "mutual":
            return distance <= source_radius and distance <= receiver_radius
        # "minimum" (validated in __init__, so no other value can reach here).
        return distance <= min(source_radius, receiver_radius)

    def build(self, agents: Sequence[object], step: int) -> CommunicationGraph:
        """Build the frozen same-team radius graph for one movement step.

        Inactive agents (``is_active`` is falsy) are excluded from the node
        set entirely. Node index order equals the order of ``agents`` after
        that filtering; edges are sorted by ``(source index, receiver index)``.

        Args:
            agents: Agent objects with ``name``, ``x``, ``y``, team
                information (``team`` or ``agent_type``), and optionally
                ``is_active`` (missing means active).
            step: Current movement-step index, recorded on the snapshot.

        Returns:
            A :class:`CommunicationGraph` snapshot. Empty and disconnected
            graphs are valid results.

        Raises:
            ValueError: On duplicate agent names, missing positions, missing
                team labels, or unresolvable radii.
        """
        active = [agent for agent in agents if getattr(agent, "is_active", True)]

        names: List[str] = []
        positions: List[Tuple[float, float]] = []
        team_labels: List[str] = []
        radii: List[float] = []
        seen: Dict[str, int] = {}
        for agent in active:
            name = getattr(agent, "name", None)
            if not isinstance(name, str) or not name:
                raise ValueError(
                    "Every agent must expose a non-empty string 'name', got {!r}.".format(name)
                )
            if name in seen:
                raise ValueError("Duplicate agent name '{}' in topology input.".format(name))
            seen[name] = len(names)
            x = getattr(agent, "x", None)
            y = getattr(agent, "y", None)
            if x is None or y is None:
                raise ValueError(
                    "Active agent '{}' has no position (x={!r}, y={!r}); an agent without "
                    "a position cannot be placed on the communication graph.".format(name, x, y)
                )
            names.append(name)
            positions.append((float(x), float(y)))
            team_labels.append(_resolve_team_label(agent))
            radii.append(self._resolve_radius(name, agent))

        # Deterministic team indexing: sorted unique labels -> 0..T-1
        # ("blue" -> 0, "red" -> 1 for the standard two-team setup).
        label_to_index = {label: i for i, label in enumerate(sorted(set(team_labels)))}
        team_index = [label_to_index[label] for label in team_labels]

        sources: List[int] = []
        receivers: List[int] = []
        distances: List[float] = []
        num_agents = len(names)
        for i in range(num_agents):
            for j in range(num_agents):
                if i == j and not self._include_self_edges:
                    continue
                if team_index[i] != team_index[j]:
                    continue
                distance = math.dist(positions[i], positions[j]) if i != j else 0.0
                if self._edge_allowed(distance, radii[i], radii[j]):
                    sources.append(i)
                    receivers.append(j)
                    distances.append(distance)

        if sources:
            edge_index = torch.tensor([sources, receivers], dtype=torch.long)
            edge_distance = torch.tensor(distances, dtype=torch.float32)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_distance = torch.zeros((0,), dtype=torch.float32)

        logger.debug(
            "RadiusTopology built graph at step {}: {} agent(s), {} edge(s), rule={}",
            step,
            num_agents,
            len(sources),
            self._radius_rule,
        )
        return CommunicationGraph(
            agent_ids=tuple(names),
            edge_index=edge_index,
            edge_distance=edge_distance,
            edge_features=None,
            team_index=torch.tensor(team_index, dtype=torch.long),
            step=step,
        )
