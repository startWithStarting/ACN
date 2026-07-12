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
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import torch
from tensordict import TensorDict

from src.training.marl.encoders import PolicyEncoder
from src.utils.logger import get_logger

logger = get_logger("acn.training.adapters")

__all__ = [
    "PRIVILEGED_STATE_KEY",
    "PRIVILEGED_FEATURES_PER_AGENT",
    "encode_team_observations",
    "build_privileged_state",
    "privileged_state_dim",
    "actions_to_env",
]

#: TensorDict key under which the critic-only privileged state travels.
PRIVILEGED_STATE_KEY: str = "privileged_state"

#: Per-agent privileged features: [x_norm, y_norm, vx_norm, vy_norm, team_id].
PRIVILEGED_FEATURES_PER_AGENT: int = 5


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
