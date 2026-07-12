"""On-policy rollout storage for the team PPO trainer.

Stores complete :class:`~src.training.marl.contract.TeamTransition` objects
keyed by collection order and converts them to the ``[N, T]`` TensorDict
layout TorchRL's GAE and ClipPPOLoss consume (``N`` team agents, ``T``
collected steps, time on the last batch dimension).

Stored per step: encoded policy inputs (``features``, ``contacts``,
``contacts_mask``), actions, log-probs, rollout-time values, rewards, dones,
terminated flags, the critic-only privileged state, and the NEXT-step
snapshots of policy inputs and privileged state — enough to recompute
advantages from scratch with the current critic.

Engineered communication schemes (``one_hop_direct``, ``one_hop_mean``,
``multihop_relay``) need NO communication-graph storage here: their
communication runs inside the environment step and no gradient flows through
env-side deliveries, so the encoded communication features in
``features``/``comm`` are plain inputs. The later learned ``multihop_gnn``
scheme runs its message passing inside the actor's forward pass and will add
rollout fields (graph or graph-building inputs, communication state, message
log-probs) so updates can recompute the communication forward pass.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
from tensordict import TensorDict

from src.training.marl.adapters import PRIVILEGED_STATE_KEY
from src.training.marl.contract import TeamTransition
from src.utils.logger import get_logger

logger = get_logger("acn.training.buffer")

__all__ = ["RolloutBuffer"]

#: Extras keys every stored transition must carry (PPO contract).
_REQUIRED_EXTRAS = ("sample_log_prob", "state_value")


class RolloutBuffer:
    """Ordered on-policy storage of one rollout's team transitions.

    Args:
        team_names: The trainable team's agent names (fixes ``N`` and the
            batch row order).
    """

    def __init__(self, team_names: Sequence[str]) -> None:
        self._team_names = list(team_names)
        self._transitions: List[TeamTransition] = []

    def __len__(self) -> int:
        """Number of stored steps ``T``."""
        return len(self._transitions)

    @property
    def team_names(self) -> List[str]:
        """The team agent names defining the batch row order."""
        return list(self._team_names)

    def add(self, transition: TeamTransition) -> None:
        """Append one complete transition.

        Args:
            transition: The step's transition; its tensors must be aligned
                with ``team_names`` and its extras must contain
                ``sample_log_prob`` and ``state_value``.

        Raises:
            ValueError: If the transition's batch size or extras are wrong.
        """
        expected = len(self._team_names)
        if transition.actions.numel() != expected:
            raise ValueError(
                "Transition actions have {} entries for a {}-agent team.".format(
                    transition.actions.numel(), expected
                )
            )
        missing = [key for key in _REQUIRED_EXTRAS if key not in transition.extras]
        if missing:
            raise ValueError(
                "Transition extras are missing {} (required by the PPO rollout "
                "contract).".format(", ".join(missing))
            )
        self._transitions.append(transition)

    def clear(self) -> None:
        """Drop all stored transitions (after an update consumed them)."""
        self._transitions = []

    def as_tensordict(self) -> TensorDict:
        """Assemble the rollout as an ``[N, T]`` TensorDict.

        Returns:
            TensorDict with batch size ``[N, T]`` and keys:

            - ``features`` ``[N, T, F]``, ``contacts_mask`` ``[N, T, K]``;
            - ``action`` ``[N, T]``, ``sample_log_prob`` ``[N, T]``,
              ``rollout_state_value`` ``[N, T]`` (diagnostic; advantages are
              recomputed with the current critic);
            - ``privileged_state`` ``[N, T, P]`` (critic-only, broadcast per
              agent so a global critic can be evaluated batch-wise);
            - ``next``: ``features``, ``privileged_state``, ``reward``
              ``[N, T, 1]``, ``done`` ``[N, T, 1]``, ``terminated``
              ``[N, T, 1]``.

        Raises:
            ValueError: If the buffer is empty.
        """
        if not self._transitions:
            raise ValueError("RolloutBuffer is empty; collect transitions before update().")
        num_agents = len(self._team_names)

        def stack_policy(key: str, next_inputs: bool) -> torch.Tensor:
            source = (
                [t.next_policy_inputs[key] for t in self._transitions]
                if next_inputs
                else [t.policy_inputs[key] for t in self._transitions]
            )
            return torch.stack(source, dim=0).transpose(0, 1).contiguous()

        privileged = torch.stack(
            [t.privileged_state for t in self._transitions], dim=0
        )  # [T, P]
        next_privileged = torch.stack(
            [t.next_privileged_state for t in self._transitions], dim=0
        )
        num_steps = privileged.size(0)

        def per_agent(values: List[torch.Tensor]) -> torch.Tensor:
            return torch.stack(values, dim=0).transpose(0, 1).contiguous()  # [N, T]

        rollout = TensorDict(
            {
                "features": stack_policy("features", next_inputs=False),
                "contacts_mask": stack_policy("contacts_mask", next_inputs=False),
                "action": per_agent([t.actions for t in self._transitions]),
                "sample_log_prob": per_agent(
                    [t.extras["sample_log_prob"] for t in self._transitions]
                ),
                "rollout_state_value": per_agent(
                    [t.extras["state_value"] for t in self._transitions]
                ),
                PRIVILEGED_STATE_KEY: privileged.unsqueeze(0)
                .expand(num_agents, num_steps, privileged.size(1))
                .contiguous(),
                "next": TensorDict(
                    {
                        "features": stack_policy("features", next_inputs=True),
                        PRIVILEGED_STATE_KEY: next_privileged.unsqueeze(0)
                        .expand(num_agents, num_steps, next_privileged.size(1))
                        .contiguous(),
                        "reward": per_agent(
                            [t.rewards.to(torch.float32) for t in self._transitions]
                        ).unsqueeze(-1),
                        "done": per_agent([t.dones for t in self._transitions])
                        .to(torch.bool)
                        .unsqueeze(-1),
                        "terminated": per_agent(
                            [t.terminateds for t in self._transitions]
                        )
                        .to(torch.bool)
                        .unsqueeze(-1),
                    },
                    batch_size=[num_agents, num_steps],
                ),
            },
            batch_size=[num_agents, num_steps],
        )
        return rollout
