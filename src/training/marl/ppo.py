"""One PPO implementation for the trainable team (TorchRL backend).

Supports all four combinations of the two independent configuration axes from
``docs/communication_decision_log.md`` ("Training Algorithm Boundary"):

- ``actor: shared`` — ONE categorical actor network and optimizer for the
  whole team (parameter sharing; shared weights never share observations,
  message memory, or state — each agent's row is its own policy-visible
  encoding);
- ``actor: separate`` — per-agent actor copies, each with its own optimizer;
- ``critic: local`` — the critic consumes the actor's own policy-visible
  encoding (IPPO-style). Local critics follow the actor assignment: one
  critic when the actor is shared, per-agent critics when separate;
- ``critic: global`` — ONE central critic consumes the training-only
  privileged simulator state (MAPPO/CTDE-style). The privileged tensor lives
  under the dedicated ``privileged_state`` key and never reaches an actor.

TorchRL supplies :class:`~torchrl.objectives.ClipPPOLoss` and GAE over
TensorDict rollouts; ACN keeps the transition contract, actor assignment,
critic scope, and the update schedule.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn
from torch.distributions import Categorical
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from src.training.marl.adapters import PRIVILEGED_STATE_KEY, actions_to_env
from src.training.marl.buffer import RolloutBuffer
from src.training.marl.config import TrainingSettings
from src.training.marl.contract import ActionSelection, OnlineMethod, TeamTransition
from src.utils.logger import get_logger

logger = get_logger("acn.training.ppo")

__all__ = ["TeamPPO"]

_SHARED_KEY = "__shared__"
_LOSS_KEYS = ("loss_objective", "loss_critic", "loss_entropy")


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    """Build the small tanh MLP used for actors and critics."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
        nn.Linear(hidden, out_dim),
    )


class _Learner:
    """One (actor, critic, loss, GAE, optimizer) bundle plus its team rows."""

    def __init__(
        self,
        name: str,
        rows: List[int],
        actor: ProbabilisticActor,
        critic: ValueOperator,
        settings: TrainingSettings,
        optimizer_params: Sequence[torch.nn.Parameter],
    ) -> None:
        self.name = name
        self.rows = rows
        self.actor = actor
        self.critic = critic
        self.loss = ClipPPOLoss(
            actor_network=actor,
            critic_network=critic,
            clip_epsilon=settings.clip_epsilon,
            entropy_bonus=settings.entropy_coef > 0.0,
            entropy_coeff=settings.entropy_coef,
            critic_coeff=settings.value_coef,
            normalize_advantage=False,
        )
        self.loss.set_keys(sample_log_prob="sample_log_prob")
        self.gae = GAE(
            gamma=settings.gamma,
            lmbda=settings.gae_lambda,
            value_network=critic,
            average_gae=False,
        )
        self.optimizer = torch.optim.Adam(list(optimizer_params), lr=settings.lr)


class TeamPPO(OnlineMethod):
    """Team PPO over the environment's flat discrete movement space.

    Args:
        settings: The validated ``training:`` block.
        team_names: The trainable team's agent names, in fixed team order.
        feature_dim: Width of the flat policy-input feature vector.
        privileged_dim: Width of the flat privileged-state vector.
        num_actions: Size of the flat ``Discrete(N)`` movement space.
    """

    def __init__(
        self,
        settings: TrainingSettings,
        team_names: Sequence[str],
        feature_dim: int,
        privileged_dim: int,
        num_actions: int,
    ) -> None:
        if not team_names:
            raise ValueError("TeamPPO needs at least one trainable agent.")
        self._settings = settings
        self._team_names = list(team_names)
        self._feature_dim = int(feature_dim)
        self._privileged_dim = int(privileged_dim)
        self._num_actions = int(num_actions)
        self._device = settings.resolved_device()
        self._buffer = RolloutBuffer(self._team_names)
        self._updates_done = 0
        # Dedicated generator for minibatch permutations so update order is
        # part of the checkpointable state.
        self._perm_generator = torch.Generator(device="cpu")
        self._perm_generator.manual_seed(settings.seed)

        self._actor_nets: Dict[str, nn.Module] = {}
        self._critic_nets: Dict[str, nn.Module] = {}
        self._actors: Dict[str, ProbabilisticActor] = {}
        self._critics: Dict[str, ValueOperator] = {}
        self._learners: List[_Learner] = []
        self._build_networks()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_actor(self) -> Tuple[ProbabilisticActor, nn.Module]:
        """Build one categorical actor over the flat discrete movement space."""
        net = _mlp(self._feature_dim, self._settings.hidden_size, self._num_actions).to(
            self._device
        )
        module = TensorDictModule(net, in_keys=["features"], out_keys=["logits"])
        actor = ProbabilisticActor(
            module,
            in_keys=["logits"],
            out_keys=["action"],
            distribution_class=Categorical,
            return_log_prob=True,
            log_prob_key="sample_log_prob",
        )
        return actor, net

    def _build_critic(self, scope: str) -> Tuple[ValueOperator, nn.Module]:
        """Build one critic for the given scope (``local`` or ``global``)."""
        in_dim = self._privileged_dim if scope == "global" else self._feature_dim
        in_key = PRIVILEGED_STATE_KEY if scope == "global" else "features"
        net = _mlp(in_dim, self._settings.hidden_size, 1).to(self._device)
        return ValueOperator(net, in_keys=[in_key]), net

    def _build_networks(self) -> None:
        settings = self._settings
        shared_actor = settings.actor == "shared"
        global_critic = settings.critic == "global"

        central_critic = None
        if global_critic:
            central_critic, central_net = self._build_critic("global")
            self._critics[_SHARED_KEY] = central_critic
            self._critic_nets[_SHARED_KEY] = central_net

        if shared_actor:
            actor, actor_net = self._build_actor()
            self._actors[_SHARED_KEY] = actor
            self._actor_nets[_SHARED_KEY] = actor_net
            if global_critic:
                critic, critic_net = central_critic, self._critic_nets[_SHARED_KEY]
            else:
                critic, critic_net = self._build_critic("local")
                self._critics[_SHARED_KEY] = critic
                self._critic_nets[_SHARED_KEY] = critic_net
            params = list(actor_net.parameters()) + list(critic_net.parameters())
            self._learners.append(
                _Learner(
                    name="team",
                    rows=list(range(len(self._team_names))),
                    actor=actor,
                    critic=critic,
                    settings=settings,
                    optimizer_params=params,
                )
            )
        else:
            for row, agent_name in enumerate(self._team_names):
                actor, actor_net = self._build_actor()
                self._actors[agent_name] = actor
                self._actor_nets[agent_name] = actor_net
                params = list(actor_net.parameters())
                if global_critic:
                    critic = central_critic
                    if row == 0:
                        # The central critic's parameters join exactly one
                        # optimizer; every learner's loss still backpropagates
                        # into the same shared parameter objects.
                        params = params + list(self._critic_nets[_SHARED_KEY].parameters())
                else:
                    critic, critic_net = self._build_critic("local")
                    self._critics[agent_name] = critic
                    self._critic_nets[agent_name] = critic_net
                    params = params + list(critic_net.parameters())
                self._learners.append(
                    _Learner(
                        name=agent_name,
                        rows=[row],
                        actor=actor,
                        critic=critic,
                        settings=settings,
                        optimizer_params=params,
                    )
                )
        logger.debug(
            "TeamPPO built: actor={} critic={} learners={} feature_dim={} "
            "privileged_dim={} actions={}",
            settings.actor,
            settings.critic,
            len(self._learners),
            self._feature_dim,
            self._privileged_dim,
            self._num_actions,
        )

    # ------------------------------------------------------------------
    # OnlineMethod interface
    # ------------------------------------------------------------------

    def reset_episode(self) -> None:
        """No per-episode state: PPO rollouts may span episode boundaries."""

    def select_actions(
        self,
        decision_inputs: TensorDict,
        privileged_state: Optional[torch.Tensor] = None,
    ) -> ActionSelection:
        """Sample actions, log-probs, and values for the team at epoch ``t``.

        Args:
            decision_inputs: Encoded policy inputs, batch ``[N]`` (keys from
                :func:`~src.training.marl.adapters.encode_team_observations`).
            privileged_state: Flat privileged tensor at ``t``; required when
                the critic scope is global.

        Returns:
            An :class:`ActionSelection` whose extras carry ``sample_log_prob``
            and ``state_value`` (both ``[N]``).
        """
        num_agents = len(self._team_names)
        td = decision_inputs.to(self._device)
        actions = torch.zeros(num_agents, dtype=torch.long)
        log_probs = torch.zeros(num_agents, dtype=torch.float32)
        values = torch.zeros(num_agents, dtype=torch.float32)
        with torch.no_grad():
            for learner in self._learners:
                rows = torch.as_tensor(learner.rows, dtype=torch.long)
                sub = td[rows]
                out = learner.actor(sub)
                actions[rows] = out["action"].to(torch.long).cpu()
                log_probs[rows] = out["sample_log_prob"].to(torch.float32).cpu()
            values = self._state_values(td, privileged_state)
        return ActionSelection(
            env_actions=actions_to_env(actions, self._team_names),
            actions=actions,
            extras={"sample_log_prob": log_probs, "state_value": values},
        )

    def _state_values(
        self, td: TensorDict, privileged_state: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute per-agent values with the configured critic scope."""
        num_agents = len(self._team_names)
        values = torch.zeros(num_agents, dtype=torch.float32)
        if self._settings.critic == "global":
            if privileged_state is None:
                raise ValueError(
                    "critic: global requires the privileged_state tensor at every "
                    "decision epoch."
                )
            state_td = TensorDict(
                {PRIVILEGED_STATE_KEY: privileged_state.unsqueeze(0).to(self._device)},
                batch_size=[1],
            )
            central = self._critics[_SHARED_KEY](state_td)["state_value"]
            values[:] = float(central.reshape(-1)[0].item())
            return values
        for learner in self._learners:
            rows = torch.as_tensor(learner.rows, dtype=torch.long)
            sub = learner.critic(td[rows])
            values[rows] = sub["state_value"].reshape(-1).to(torch.float32).cpu()
        return values

    def ingest_transition(self, transition: TeamTransition) -> None:
        """Append the step's transition to the on-policy rollout buffer."""
        self._buffer.add(transition)

    def ready_to_update(self) -> bool:
        """True once the rollout buffer holds ``rollout_length`` steps."""
        return len(self._buffer) >= self._settings.rollout_length

    def update(self) -> Dict[str, float]:
        """Run one PPO update over the collected rollout.

        Advantages/value targets are recomputed per learner with the current
        critic (GAE over the ``[n, T]`` sub-rollout, time on the last batch
        dim), then ``epochs x minibatches`` clipped-PPO steps run with a
        synchronized minibatch schedule: minibatch ``k`` of every learner is
        combined into one backward/step so shared parameters (the central
        critic under ``actor: separate``) receive exactly ``epochs *
        minibatches`` optimization steps like every other parameter.

        Returns:
            Scalar metrics: mean rollout reward, per-loss means, entropy, and
            the update counter.
        """
        settings = self._settings
        rollout = self._buffer.as_tensordict().to(self._device)
        num_steps = rollout.batch_size[1]

        flats: List[TensorDict] = []
        for learner in self._learners:
            rows = torch.as_tensor(learner.rows, dtype=torch.long)
            sub = rollout[rows]
            with torch.no_grad():
                learner.gae(sub)
            flats.append(sub.reshape(-1))

        sums = {key: 0.0 for key in _LOSS_KEYS}
        entropy_sum = 0.0
        num_minibatch_steps = 0
        for _epoch in range(settings.epochs):
            perms = [
                torch.randperm(flat.batch_size[0], generator=self._perm_generator)
                for flat in flats
            ]
            for k in range(settings.minibatches):
                for learner in self._learners:
                    learner.optimizer.zero_grad(set_to_none=True)
                total = None
                entropy_val = 0.0
                for flat, perm, learner in zip(flats, perms, self._learners):
                    index = perm.chunk(settings.minibatches)[k]
                    if index.numel() == 0:
                        continue
                    loss_td = learner.loss(flat[index])
                    part = sum(
                        loss_td[key] for key in _LOSS_KEYS if key in loss_td.keys()
                    )
                    total = part if total is None else total + part
                    for key in _LOSS_KEYS:
                        if key in loss_td.keys():
                            sums[key] += float(loss_td[key].item())
                    if "entropy" in loss_td.keys():
                        entropy_val += float(loss_td["entropy"].mean().item())
                if total is None:
                    continue
                total.backward()
                for learner in self._learners:
                    torch.nn.utils.clip_grad_norm_(
                        [p for group in learner.optimizer.param_groups for p in group["params"]],
                        settings.max_grad_norm,
                    )
                for learner in self._learners:
                    learner.optimizer.step()
                entropy_sum += entropy_val
                num_minibatch_steps += 1

        mean_reward = float(rollout[("next", "reward")].mean().item())
        self._buffer.clear()
        self._updates_done += 1
        denominator = max(num_minibatch_steps, 1)
        metrics = {
            "update": float(self._updates_done),
            "rollout_steps": float(num_steps),
            "mean_reward": mean_reward,
            "loss_objective": sums["loss_objective"] / denominator,
            "loss_critic": sums["loss_critic"] / denominator,
            "loss_entropy": sums["loss_entropy"] / denominator,
            "entropy": entropy_sum / denominator,
        }
        logger.debug(
            "PPO update {}: mean_reward={:.4f} loss_objective={:.4f} loss_critic={:.4f}",
            self._updates_done,
            metrics["mean_reward"],
            metrics["loss_objective"],
            metrics["loss_critic"],
        )
        return metrics

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    @property
    def updates_done(self) -> int:
        """Number of completed optimization updates."""
        return self._updates_done

    def state_dict(self) -> Dict[str, Any]:
        """Return actors, critics, optimizers, permutation RNG, and counters."""
        return {
            "actor_nets": {key: net.state_dict() for key, net in self._actor_nets.items()},
            "critic_nets": {key: net.state_dict() for key, net in self._critic_nets.items()},
            "optimizers": {
                learner.name: learner.optimizer.state_dict() for learner in self._learners
            },
            "perm_generator": self._perm_generator.get_state(),
            "updates_done": self._updates_done,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore from :meth:`state_dict` output (exact continuation)."""
        for key, net_state in state["actor_nets"].items():
            self._actor_nets[key].load_state_dict(net_state)
        for key, net_state in state["critic_nets"].items():
            self._critic_nets[key].load_state_dict(net_state)
        for learner in self._learners:
            learner.optimizer.load_state_dict(state["optimizers"][learner.name])
        self._perm_generator.set_state(state["perm_generator"])
        self._updates_done = int(state["updates_done"])
