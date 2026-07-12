"""Common online-method interface and per-step transition contract.

Implements the training algorithm boundary from
``docs/communication_decision_log.md`` ("Training Algorithm Boundary"):
environment interaction and algorithm-specific learning are separated through
one transition contract executed per environment step:

1. Snapshot policy inputs and any training-only privileged state at time
   ``t``.
2. Ask the selected method for environment actions.
3. Apply ``env.step(actions)`` once.
4. Snapshot next observations, privileged state, rewards, and termination
   flags at ``t + 1``.
5. Build ONE complete :class:`TeamTransition` and hand it to the method.

The :class:`OnlineMethod` interface provides episode reset, action selection,
transition ingestion, update readiness, optimization, and checkpointing. It
does NOT standardize network architecture, storage type, loss function, or
update schedule: PPO appends transitions to an on-policy rollout buffer,
value-based methods would insert them into replay storage, and non-gradient
methods (scripted strategies, the VAR-pursuit blue) implement the same
interface with no-op updates (:class:`ScriptedTeamMethod`).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
from tensordict import TensorDict

from src.utils.logger import get_logger

logger = get_logger("acn.training.contract")

__all__ = [
    "ActionSelection",
    "TeamTransition",
    "OnlineMethod",
    "ScriptedTeamMethod",
]


@dataclass
class ActionSelection:
    """Result of one method's action selection at decision epoch ``t``.

    Attributes:
        env_actions: Agent name -> action in the environment's declared action
            space (an integer index for the discrete movement space, or the
            legacy ``{'direction', 'speed'}`` dict from scripted controllers).
        actions: Optional ``[N]`` action tensor aligned with the method's team
            order, for gradient methods that store tensor rollouts. ``None``
            for scripted methods.
        extras: Method-specific per-step tensors to carry into the transition
            (e.g. ``sample_log_prob`` and ``state_value`` for PPO). Empty for
            scripted methods.
    """

    env_actions: Dict[str, Any]
    actions: Optional[torch.Tensor] = None
    extras: Dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class TeamTransition:
    """One complete per-step transition for the trainable team.

    Built by the runner exactly once per environment step, after the single
    ``env.step(actions)`` call, from the pre-step and post-step snapshots.

    Attributes:
        step: Zero-based collection step index within the current rollout.
        policy_inputs: Encoded policy-visible inputs at ``t`` for the team, a
            :class:`~tensordict.TensorDict` with batch size ``[N]`` (keys
            ``features``, ``contacts``, ``contacts_mask``, ``comm``).
        privileged_state: Training-only privileged simulator state at ``t``
            (flat tensor). CRITIC-ONLY: never fed to actors.
        actions: ``[N]`` long tensor of the team's environment actions.
        extras: Per-step method tensors captured at selection time (PPO stores
            ``sample_log_prob`` ``[N]`` and ``state_value`` ``[N]``).
        rewards: ``[N]`` float tensor of per-agent rewards at ``t + 1``.
        dones: ``[N]`` bool tensor; True when the agent's episode ended at
            ``t + 1`` (terminated or truncated).
        terminateds: ``[N]`` bool tensor; True only for genuine termination
            (bootstrapping is cut). Truncation keeps this False so the value
            of the final observation still bootstraps.
        next_policy_inputs: Encoded policy inputs at ``t + 1`` (the terminal
            observation when the episode ended).
        next_privileged_state: Privileged state at ``t + 1``.
    """

    step: int
    policy_inputs: TensorDict
    privileged_state: torch.Tensor
    actions: torch.Tensor
    extras: Dict[str, torch.Tensor]
    rewards: torch.Tensor
    dones: torch.Tensor
    terminateds: torch.Tensor
    next_policy_inputs: TensorDict
    next_privileged_state: torch.Tensor


class OnlineMethod(abc.ABC):
    """Common interface every online decision/learning method implements.

    Gradient methods (PPO) and non-gradient methods (scripted strategies,
    VAR baselines) present the same surface to the runner; non-gradient
    methods simply never become ready to update.
    """

    @abc.abstractmethod
    def reset_episode(self) -> None:
        """Notify the method that a new episode begins (clear episode state)."""

    @abc.abstractmethod
    def select_actions(
        self,
        decision_inputs: Any,
        privileged_state: Optional[torch.Tensor] = None,
    ) -> ActionSelection:
        """Select environment actions for the method's team at epoch ``t``.

        Args:
            decision_inputs: Method-defined decision inputs. Gradient methods
                receive the encoded policy-input :class:`TensorDict` built by
                the adapters; scripted methods receive the raw per-agent
                observation mapping (their controllers consume the native
                dict observation, exactly as in ``main_parallel.py``).
            privileged_state: Training-only privileged state at ``t``. Only a
                global critic may consume it; actors never do.

        Returns:
            An :class:`ActionSelection` with per-agent environment actions.
        """

    @abc.abstractmethod
    def ingest_transition(self, transition: TeamTransition) -> None:
        """Ingest one complete per-step transition (see the module docstring)."""

    @abc.abstractmethod
    def ready_to_update(self) -> bool:
        """Return whether the method has enough data to run an update."""

    @abc.abstractmethod
    def update(self) -> Dict[str, float]:
        """Run one optimization update and return scalar metrics."""

    @abc.abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """Return the method's complete checkpoint state."""

    @abc.abstractmethod
    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore the method from :meth:`state_dict` output."""


class ScriptedTeamMethod(OnlineMethod):
    """Non-gradient method wrapping the scripted per-agent controllers.

    Presents the same interface as a learned method while delegating action
    selection to each agent object's ``choose_action(observation)``, exactly
    like the scripted loop in ``main_parallel.py``. The VAR-pursuit blue works
    unchanged through this wrapper (including its privileged observation mode,
    which travels inside the raw observation dict, never through
    ``privileged_state``). All learning hooks are no-ops.
    """

    def __init__(self, agent_objects: Mapping[str, Any], team_names: Sequence[str]) -> None:
        """Create the wrapper.

        Args:
            agent_objects: Agent name -> live agent object (the environment's
                ``agent_objects`` mapping).
            team_names: Names of this method's team, in team order.
        """
        self._agent_objects = agent_objects
        self._team_names = list(team_names)

    def reset_episode(self) -> None:
        """No-op: scripted controllers own their internal state."""

    def select_actions(
        self,
        decision_inputs: Any,
        privileged_state: Optional[torch.Tensor] = None,
    ) -> ActionSelection:
        """Delegate to each scripted controller's ``choose_action``.

        Args:
            decision_inputs: Raw per-agent observation mapping (agent name ->
                native dict observation).
            privileged_state: Ignored; scripted controllers never consume the
                critic-only privileged tensor.

        Returns:
            An :class:`ActionSelection` whose ``env_actions`` carry the
            controllers' native action dicts.
        """
        env_actions: Dict[str, Any] = {}
        for name in self._team_names:
            observation = decision_inputs.get(name)
            if observation is None:
                continue
            agent_obj = self._agent_objects.get(name)
            if agent_obj is None:
                continue
            env_actions[name] = agent_obj.choose_action(observation)
        return ActionSelection(env_actions=env_actions)

    def ingest_transition(self, transition: TeamTransition) -> None:
        """No-op: scripted methods store nothing."""

    def ready_to_update(self) -> bool:
        """Scripted methods never request an update."""
        return False

    def update(self) -> Dict[str, float]:
        """No-op update returning no metrics."""
        return {}

    def state_dict(self) -> Dict[str, Any]:
        """Scripted methods have no checkpointable state."""
        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """No-op restore."""
