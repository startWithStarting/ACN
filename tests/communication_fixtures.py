"""Deterministic ``CommunicationGraph`` fixtures for communication tests (Phase 0).

These plain functions build small, fully hardcoded graph snapshots against the
``CommunicationGraph`` contract in ``src.communication.types``. They exist so
topology, transport, relay, and registry tests can share one set of known
graphs with exactly documented edges.

Conventions shared by every fixture:

- Positions are hardcoded module constants (``*_POSITIONS``) together with the
  communication radius (``*_RADIUS``) that justifies the edge set. Fixtures do
  not run a topology builder; the edge lists are written out explicitly so the
  fixtures stay valid oracles for future ``RadiusTopology`` tests.
- ``agent_ids`` are listed in lexicographic order, so agent index order equals
  agent id order.
- Edge ordering is deterministic: edges are sorted by
  ``(source index, receiver index)``, which for these fixtures is identical to
  lexicographic ``(source id, receiver id)`` order.
- All edges are directed and same-team only; self-edges are never included
  (self-edges are disabled by default for transport).
- ``edge_distance`` holds the exact Euclidean distance between the hardcoded
  positions of each edge's endpoints.
- ``edge_features`` is ``None`` and ``step`` defaults to ``0``.
- Team indices use ``BLUE_TEAM = 0`` and ``RED_TEAM = 1``.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch

from src.communication.types import CommunicationGraph

BLUE_TEAM = 0
RED_TEAM = 1

# --- chain_graph -------------------------------------------------------------
# A - B - C on one line, 1.0 apart. Radius 1.5 covers adjacent pairs only:
# |AB| = |BC| = 1.0 <= 1.5 < |AC| = 2.0, so there is no A<->C edge.
CHAIN_POSITIONS: Dict[str, Tuple[float, float]] = {
    "A": (0.0, 0.0),
    "B": (1.0, 0.0),
    "C": (2.0, 0.0),
}
CHAIN_RADIUS: float = 1.5

# --- disconnected_graph ------------------------------------------------------
# Two same-team components far apart: {A, B} and {C, D}. Radius 1.5 covers the
# intra-component pairs (distance 1.0) but no cross-component pair
# (minimum cross-component distance |BC| = 9.0).
DISCONNECTED_POSITIONS: Dict[str, Tuple[float, float]] = {
    "A": (0.0, 0.0),
    "B": (1.0, 0.0),
    "C": (10.0, 0.0),
    "D": (11.0, 0.0),
}
DISCONNECTED_RADIUS: float = 1.5
DISCONNECTED_COMPONENTS: Tuple[Tuple[str, ...], ...] = (("A", "B"), ("C", "D"))

# --- empty_graph -------------------------------------------------------------
# Three agents pairwise farther apart (10.0) than the radius, so E = 0.
EMPTY_POSITIONS: Dict[str, Tuple[float, float]] = {
    "A": (0.0, 0.0),
    "B": (10.0, 0.0),
    "C": (20.0, 0.0),
}
EMPTY_RADIUS: float = 1.5

# --- two_team_graph ----------------------------------------------------------
# Two blue and two red agents spatially interleaved. Every cross-team nearest
# pair is INSIDE the radius (e.g. |blue_0 red_0| = sqrt(0.5) ~ 0.707 <= 1.5),
# yet no cross-team edge exists: edges are same-team only by construction.
TWO_TEAM_POSITIONS: Dict[str, Tuple[float, float]] = {
    "blue_0": (0.0, 0.0),
    "blue_1": (1.0, 0.0),
    "red_0": (0.5, 0.5),
    "red_1": (1.5, 0.5),
}
TWO_TEAM_RADIUS: float = 1.5
TWO_TEAM_TEAMS: Dict[str, int] = {
    "blue_0": BLUE_TEAM,
    "blue_1": BLUE_TEAM,
    "red_0": RED_TEAM,
    "red_1": RED_TEAM,
}


def _distance(positions: Dict[str, Tuple[float, float]], source: str, receiver: str) -> float:
    """Return the Euclidean distance between two hardcoded agent positions."""
    sx, sy = positions[source]
    rx, ry = positions[receiver]
    return float(((sx - rx) ** 2 + (sy - ry) ** 2) ** 0.5)


def _build_graph(
    agent_ids: Tuple[str, ...],
    team_index: Sequence[int],
    edges: Sequence[Tuple[str, str]],
    positions: Dict[str, Tuple[float, float]],
    step: int,
) -> CommunicationGraph:
    """Assemble a ``CommunicationGraph`` from explicit id-level edges.

    Args:
        agent_ids: Agent ids in lexicographic order; index order follows it.
        team_index: Team label per agent, aligned with ``agent_ids``.
        edges: Directed ``(source id, receiver id)`` pairs.
        positions: Hardcoded positions used to derive exact edge distances.
        step: Movement step the snapshot belongs to.

    Returns:
        A frozen graph whose edges are sorted by (source index, receiver
        index) and whose empty case uses valid zero-length tensors.
    """
    index_of = {agent_id: i for i, agent_id in enumerate(agent_ids)}
    indexed = sorted((index_of[src], index_of[rcv]) for src, rcv in edges)
    if indexed:
        edge_index = torch.tensor(
            [[src for src, _ in indexed], [rcv for _, rcv in indexed]],
            dtype=torch.long,
        )
        edge_distance = torch.tensor(
            [_distance(positions, agent_ids[src], agent_ids[rcv]) for src, rcv in indexed],
            dtype=torch.float32,
        )
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_distance = torch.zeros((0,), dtype=torch.float32)
    return CommunicationGraph(
        agent_ids=agent_ids,
        edge_index=edge_index,
        edge_distance=edge_distance,
        edge_features=None,
        team_index=torch.tensor(list(team_index), dtype=torch.long),
        step=step,
    )


def chain_graph(step: int = 0) -> CommunicationGraph:
    """Build the A-B-C chain graph on one blue team.

    Edges, in deterministic (source, receiver) sorted order:
    ``A->B, B->A, B->C, C->B``. The radius covers adjacent pairs only, so
    there is no ``A->C`` or ``C->A`` edge; reaching C from A requires two
    hops (two communication rounds).

    Args:
        step: Movement step recorded on the snapshot.

    Returns:
        Chain graph with 3 agents and 4 directed edges.
    """
    return _build_graph(
        agent_ids=("A", "B", "C"),
        team_index=(BLUE_TEAM, BLUE_TEAM, BLUE_TEAM),
        edges=(("A", "B"), ("B", "A"), ("B", "C"), ("C", "B")),
        positions=CHAIN_POSITIONS,
        step=step,
    )


def disconnected_graph(step: int = 0) -> CommunicationGraph:
    """Build a one-team graph with two disconnected components {A, B} and {C, D}.

    Edges, in deterministic (source, receiver) sorted order:
    ``A->B, B->A, C->D, D->C``. No edge crosses the component boundary, so no
    round count can move information between components.

    Args:
        step: Movement step recorded on the snapshot.

    Returns:
        Disconnected graph with 4 agents and 4 directed edges.
    """
    return _build_graph(
        agent_ids=("A", "B", "C", "D"),
        team_index=(BLUE_TEAM, BLUE_TEAM, BLUE_TEAM, BLUE_TEAM),
        edges=(("A", "B"), ("B", "A"), ("C", "D"), ("D", "C")),
        positions=DISCONNECTED_POSITIONS,
        step=step,
    )


def empty_graph(step: int = 0) -> CommunicationGraph:
    """Build a one-team graph with agents but zero edges.

    All pairwise distances exceed the radius, so ``edge_index`` is a valid
    ``[2, 0]`` long tensor and ``edge_distance`` a valid ``[0]`` tensor.

    Args:
        step: Movement step recorded on the snapshot.

    Returns:
        Edgeless graph with 3 agents.
    """
    return _build_graph(
        agent_ids=("A", "B", "C"),
        team_index=(BLUE_TEAM, BLUE_TEAM, BLUE_TEAM),
        edges=(),
        positions=EMPTY_POSITIONS,
        step=step,
    )


def two_team_graph(step: int = 0) -> CommunicationGraph:
    """Build a graph with blue and red teams and same-team edges only.

    Edges, in deterministic (source, receiver) sorted order:
    ``blue_0->blue_1, blue_1->blue_0, red_0->red_1, red_1->red_0``. The teams
    are spatially interleaved so cross-team pairs sit inside the radius, yet
    no cross-team edge exists: edge validity requires same team membership,
    not just proximity.

    Args:
        step: Movement step recorded on the snapshot.

    Returns:
        Two-team graph with 4 agents and 4 directed edges.
    """
    agent_ids = ("blue_0", "blue_1", "red_0", "red_1")
    return _build_graph(
        agent_ids=agent_ids,
        team_index=tuple(TWO_TEAM_TEAMS[agent_id] for agent_id in agent_ids),
        edges=(
            ("blue_0", "blue_1"),
            ("blue_1", "blue_0"),
            ("red_0", "red_1"),
            ("red_1", "red_0"),
        ),
        positions=TWO_TEAM_POSITIONS,
        step=step,
    )
