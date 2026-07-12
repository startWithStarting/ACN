"""PettingZoo dict observations <-> TensorDict adapters for the trainable team.

The PettingZoo dict/list observation contract stays the simulator's source of
truth; TorchRL consumes TensorDict batches. These adapters convert between the
two without changing the environment API (``docs/communication_decision_log.md``,
"Training Algorithm Boundary": "An adapter converts ACN observations and
transitions into the TensorDict representation ... without changing the
simulator's dict/list observation contract").

Privileged-state boundary: :func:`build_privileged_state` reads simulator
state (ALL agents' positions/velocities plus team ids) directly from the live
agent objects. Its output is CRITIC-ONLY: the runner stores it under the
clearly named ``privileged_state`` TensorDict key, only a ``critic: global``
value network ever consumes it, and actors never see the key.

Differentiable-communication inputs: for a differentiable scheme (e.g.
``multihop_gnn``) the per-agent policy input splits into (a) the local
features exactly as today and (b) the team's raw node features plus the
frozen per-step :class:`~src.communication.types.CommunicationGraph`
edge_index, attached under the :data:`COMMUNICATION_INPUT_KEYS`.
:func:`build_team_communication_graph` builds that graph with the compiled
plan's OWN topology (the same ``RadiusTopology``/round definitions the env
would use, reading the resolved ``communication_radius`` off the agents), so
policy-side communication and env semantics can never drift apart. The node
features are the policy-visible local features themselves — no privileged
field ever enters the graph path.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import torch
from tensordict import TensorDict

from src.communication.types import CommunicationGraph
from src.training.marl.encoders import PolicyEncoder
from src.utils.logger import get_logger

logger = get_logger("acn.training.adapters")

__all__ = [
    "PRIVILEGED_STATE_KEY",
    "PRIVILEGED_FEATURES_PER_AGENT",
    "COMMUNICATION_INPUT_KEYS",
    "encode_team_observations",
    "build_privileged_state",
    "privileged_state_dim",
    "actions_to_env",
    "max_team_edges",
    "build_team_communication_graph",
    "attach_communication_inputs",
]

#: TensorDict key under which the critic-only privileged state travels.
PRIVILEGED_STATE_KEY: str = "privileged_state"

#: Per-agent privileged features: [x_norm, y_norm, vx_norm, vy_norm, team_id].
PRIVILEGED_FEATURES_PER_AGENT: int = 5

#: TensorDict keys carrying the differentiable-communication inputs:
#: ``comm_nodes`` ``[N, N, F]`` (each agent row carries the whole team's raw
#: node features for its step), ``comm_edges`` ``[N, 2, E_max]`` (the frozen
#: edge_index, zero-padded), ``comm_num_edges`` ``[N]`` (valid edge count),
#: and ``comm_node_index`` ``[N]`` (which node each row is).
COMMUNICATION_INPUT_KEYS: Tuple[str, ...] = (
    "comm_nodes",
    "comm_edges",
    "comm_num_edges",
    "comm_node_index",
)


def encode_team_observations(
    observations: Mapping[str, Mapping[str, Any]],
    team_names: Sequence[str],
    encoder: PolicyEncoder,
) -> TensorDict:
    """Encode the trainable team's observations into one TensorDict batch.

    Args:
        observations: Full per-agent observation mapping from the environment.
        team_names: The trainable team's agent names, in fixed team order (the
            batch row order).
        encoder: The run's policy encoder.

    Returns:
        A :class:`TensorDict` with batch size ``[N]`` and keys ``features``
        ``[N, F]``, ``base`` ``[N, 4]``, ``contacts`` ``[N, K, 4]``,
        ``contacts_mask`` ``[N, K]``, and ``comm`` ``[N, D]``.

    Raises:
        KeyError: If a team agent is missing from ``observations`` (the
            parallel environment returns observations for every live agent, so
            a missing name indicates a bug or a finished episode).
    """
    rows = []
    for name in team_names:
        if name not in observations:
            raise KeyError(
                "No observation for trainable agent {!r}; available: {}. The runner "
                "must reset the environment before encoding a finished episode.".format(
                    name, sorted(observations.keys())
                )
            )
        rows.append(encoder.encode(observations[name]))
    stacked: Dict[str, torch.Tensor] = {}
    for key in ("features", "base", "contacts", "contacts_mask", "comm"):
        stacked[key] = torch.stack([row[key] for row in rows], dim=0)
    return TensorDict(stacked, batch_size=[len(rows)])


def privileged_state_dim(num_agents: int) -> int:
    """Return the flat privileged-state width for ``num_agents`` simulator agents."""
    return PRIVILEGED_FEATURES_PER_AGENT * num_agents


def build_privileged_state(
    agent_objects: Mapping[str, Any],
    ordered_names: Sequence[str],
    grid_width: float,
    grid_height: float,
) -> torch.Tensor:
    """Build the critic-only global state from live simulator objects.

    Per agent (in ``ordered_names`` order, normally ``env.possible_agents`` so
    the layout is fixed for the run): normalized position ``(x/W, y/H)``,
    velocity ``direction * speed`` normalized by the agent's ``max_speed``,
    and the team id (blue = 0.0, red = 1.0). Inactive agents keep their last
    simulator state, which is exactly what a centralized critic may see.

    Args:
        agent_objects: Agent name -> live agent object.
        ordered_names: Fixed agent order defining the layout.
        grid_width: Grid width for position normalization.
        grid_height: Grid height for position normalization.

    Returns:
        Float32 tensor of shape ``[5 * len(ordered_names)]``.
    """
    values = torch.zeros(
        (len(ordered_names), PRIVILEGED_FEATURES_PER_AGENT), dtype=torch.float32
    )
    for index, name in enumerate(ordered_names):
        agent_obj = agent_objects.get(name)
        if agent_obj is None:
            continue
        x = getattr(agent_obj, "x", None)
        y = getattr(agent_obj, "y", None)
        if x is not None and y is not None:
            values[index, 0] = float(x) / float(grid_width)
            values[index, 1] = float(y) / float(grid_height)
        direction = getattr(agent_obj, "direction", None)
        speed = float(getattr(agent_obj, "speed", 0.0) or 0.0)
        max_speed = float(getattr(agent_obj, "max_speed", 10.0) or 10.0)
        if direction is not None and max_speed > 0.0:
            try:
                dx = float(direction[0])
                dy = float(direction[1])
            except (TypeError, IndexError, ValueError):
                dx = dy = 0.0
            values[index, 2] = dx * speed / max_speed
            values[index, 3] = dy * speed / max_speed
        team = getattr(getattr(agent_obj, "agent_type", None), "value", None)
        values[index, 4] = 1.0 if team == "red" else 0.0
    return values.reshape(-1)


def max_team_edges(num_agents: int, include_self_edges: bool = False) -> int:
    """Return the maximum possible same-team edge count (the padding width).

    Args:
        num_agents: Team size ``N``.
        include_self_edges: Whether the topology emits ``i -> i`` edges.

    Returns:
        ``N * (N - 1)`` directed edges, plus ``N`` when self-edges are on.
        Fixed for a run, so per-step edge tensors stack rectangularly.
    """
    return num_agents * (num_agents - 1) + (num_agents if include_self_edges else 0)


def build_team_communication_graph(
    agent_objects: Mapping[str, Any],
    team_names: Sequence[str],
    topology: Any,
    step: int,
) -> CommunicationGraph:
    """Build the trainable team's frozen communication graph for one step.

    Uses the compiled communication plan's own topology builder — the exact
    same plan/topology/round definitions the environment would use — over the
    live team agent objects (which carry their resolved
    ``communication_radius``).

    Args:
        agent_objects: The environment's agent name -> live object mapping.
        team_names: The trainable team's agent names, in fixed team order.
        topology: The compiled plan's topology builder (``plan.topology``).
        step: Current movement-step index.

    Returns:
        The frozen :class:`CommunicationGraph`; its node order equals
        ``team_names``.

    Raises:
        ValueError: If the built graph's node set/order does not match the
            team order (e.g. an inactive team agent was dropped) — the
            rollout layout requires a stable ``N``-node graph.
    """
    graph = topology.build([agent_objects[name] for name in team_names], step)
    if tuple(graph.agent_ids) != tuple(team_names):
        raise ValueError(
            "Team communication graph nodes {} do not match the team order {}; the "
            "trainer requires every trainable-team agent to stay on the graph for the "
            "whole rollout.".format(graph.agent_ids, tuple(team_names))
        )
    return graph


def attach_communication_inputs(
    policy_inputs: TensorDict,
    graph: CommunicationGraph,
    max_edges: int,
) -> TensorDict:
    """Attach the differentiable-communication inputs to one step's batch.

    Stores RAW graph-building results — the team's node features (the local
    ``features`` already in the batch) and the frozen edge_index — never a
    computed or detached communication embedding: PPO updates must be able to
    reconstruct the full communication forward pass with current parameters
    (``docs/communication_implementation_plan.md``, "Differentiable Shared
    Communication").

    Args:
        policy_inputs: The step's encoded team batch (batch size ``[N]``,
            from :func:`encode_team_observations`); modified in place.
        graph: The step's frozen team communication graph (node order must
            equal the batch row order).
        max_edges: Fixed padding width from :func:`max_team_edges`.

    Returns:
        The same TensorDict with the :data:`COMMUNICATION_INPUT_KEYS` added.

    Raises:
        ValueError: If the graph has more edges than ``max_edges`` or its
            node count differs from the batch size.
    """
    features = policy_inputs["features"]
    num_agents = features.size(0)
    if graph.num_agents != num_agents:
        raise ValueError(
            "Communication graph has {} node(s) for a {}-agent policy batch.".format(
                graph.num_agents, num_agents
            )
        )
    edge_index = graph.edge_index.to(torch.long)
    num_edges = edge_index.size(1)
    if num_edges > max_edges:
        raise ValueError(
            "Communication graph has {} edge(s), exceeding the padding width {}.".format(
                num_edges, max_edges
            )
        )
    padded_edges = torch.zeros((2, max_edges), dtype=torch.long)
    if num_edges > 0:
        padded_edges[:, :num_edges] = edge_index
    feature_dim = features.size(1)
    policy_inputs.set(
        "comm_nodes",
        features.unsqueeze(0).expand(num_agents, num_agents, feature_dim).clone(),
    )
    policy_inputs.set(
        "comm_edges", padded_edges.unsqueeze(0).expand(num_agents, 2, max_edges).clone()
    )
    policy_inputs.set(
        "comm_num_edges", torch.full((num_agents,), num_edges, dtype=torch.long)
    )
    policy_inputs.set("comm_node_index", torch.arange(num_agents, dtype=torch.long))
    return policy_inputs


def actions_to_env(actions: torch.Tensor, team_names: Sequence[str]) -> Dict[str, int]:
    """Convert an ``[N]`` action tensor back to the env's per-agent dict form.

    Args:
        actions: Long tensor of flat discrete movement indices, aligned with
            ``team_names``.
        team_names: The team's agent names, in the same order as ``actions``.

    Returns:
        Agent name -> plain Python int action, ready for ``env.step``.
    """
    if actions.numel() != len(team_names):
        raise ValueError(
            "Action tensor has {} entries for {} team agents.".format(
                actions.numel(), len(team_names)
            )
        )
    return {name: int(actions[index].item()) for index, name in enumerate(team_names)}
