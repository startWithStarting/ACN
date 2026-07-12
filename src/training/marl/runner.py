"""Team-selection training runner for the MARL (TorchRL) backend.

Binds the configured algorithm to exactly ONE trainable team and runs the
opposing team through its scripted ``choose_action`` exactly like the
scripted loop in ``main_parallel.py`` (the VAR-pursuit blue keeps working via
the existing privileged observation mode). The runner owns the per-step
transition contract (snapshot at ``t`` -> actions -> one ``env.step`` ->
snapshot at ``t + 1`` -> one complete transition), the collection/update
loop, deterministic seeding, CSV metric logging, and checkpoint/resume.

Determinism and resume model: every rollout starts from a FRESH environment
(new agent objects, new physics state) reset with a seed derived from the
master seed and a persistent reset counter; mid-rollout episode ends rebuild
the environment the same way. Scripted controllers therefore never leak state
across episodes, and a checkpoint (networks, optimizers, RNG states, counters,
config snapshot) reproduces the remaining run exactly — no live environment
state needs to be serialized.

Fully headless: rendering and GIF capture are forcibly disabled for training
runs regardless of the scenario config. The entrypoint is cloud-runnable by
construction (``docs/communication_decision_log.md``, "Remote Training
Infrastructure"): config-driven, seeded, checkpointed, artifacts under the
normal results directory.
"""

from __future__ import annotations

import csv
import os
import random
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch

from src.agents.action_spaces import ActionSpaceConfig
from src.agents.factory import create_agents_from_config
from src.communication.config import parse_communication_config
from src.communication.registry import create_communication_plan
from src.env.parallel_env import ParallelGameEnv
from src.env.rewards import parse_reward_settings
from src.env.sensors import ObservationConfig
from src.training.marl.adapters import (
    attach_communication_inputs,
    build_privileged_state,
    build_team_communication_graph,
    encode_team_observations,
    max_team_edges,
    privileged_state_dim,
)
from src.training.marl.config import (
    TrainingConfigError,
    TrainingSettings,
    parse_training_config,
)
from src.training.marl.contract import ScriptedTeamMethod, TeamTransition
from src.training.marl.encoders import PolicyEncoder
from src.training.marl.ppo import GnnCommSettings, TeamPPO
from src.utils.config_loader import load_config
from src.utils.experiment import build_file_run_id, setup_experiment_results_dir
from src.utils.logger import get_logger

logger = get_logger("acn.training.runner")

__all__ = ["MarlTrainingRunner", "run_marl_training"]

_METRIC_FIELDS = (
    "update",
    "rollout_steps",
    "mean_reward",
    "loss_objective",
    "loss_critic",
    "loss_entropy",
    "entropy",
)


def _derive_reset_seed(master_seed: int, reset_counter: int) -> int:
    """Derive a per-reset environment seed from the master seed and counter."""
    return (master_seed * 100003 + reset_counter) % (2**31 - 1)


class MarlTrainingRunner:
    """Config-driven single-team PPO training over ``ParallelGameEnv``.

    Args:
        config: The full loaded scenario config (the ``training:`` block must
            exist with ``backend: marl``).
        config_path: Path to the YAML file (copied into the results dir and
            recorded in checkpoints).
        results_dir: Optional explicit results directory; created through the
            standard experiment layout when omitted.
        resume_path: Optional checkpoint file to resume from (exact
            continuation).
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        config_path: Optional[str] = None,
        results_dir: Optional[str] = None,
        resume_path: Optional[str] = None,
    ) -> None:
        self.config = dict(config)
        self.config_path = config_path
        self.settings: TrainingSettings = parse_training_config(self.config.get("training"))
        self.env_config = dict(self.config.get("environment", {}))
        self.agents_config = self.config.get("agents", {})
        self._validate_environment_for_training()

        # Headless enforcement: training never renders or captures GIFs.
        if self.env_config.get("render_mode") or self.env_config.get("save_episode_gifs", True):
            logger.info("Training runs headless: disabling render_mode and episode GIFs.")
        self.env_config["render_mode"] = None
        self.env_config["save_episode_gifs"] = False

        if results_dir is None:
            experiment_name = self.config.get("experiment_name", "marl_training")
            base_results_dir = self.config.get("results_base_dir", "results")
            run_id = build_file_run_id(experiment_name, mode_suffix="_train")
            results_dir = setup_experiment_results_dir(
                base_results_dir,
                experiment_name,
                config_path,
                mode_suffix="_train",
                run_id=run_id,
            )
        self.results_dir = results_dir
        self.checkpoint_dir = os.path.join(results_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.metrics_path = os.path.join(results_dir, "training_metrics.csv")

        self._seed_everything(self.settings.seed)
        self._reset_counter = 0
        self._start_update = 0
        self._env: Optional[ParallelGameEnv] = None

        (
            self.team_names,
            self.opponent_names,
            self.all_names,
        ) = self._resolve_team_names()
        # Differentiable communication (multihop_gnn): compile the SAME plan
        # the environment compiles (and skips executing) so the trainer runs
        # the communication forward pass with identical topology/round
        # definitions inside the actor.
        self.communication_config = parse_communication_config(
            self.env_config.get("communication")
        )
        self._communication_plan = (
            create_communication_plan(self.communication_config)
            if self.communication_config.is_active
            else None
        )
        self._differentiable_comm = bool(
            self._communication_plan is not None and self._communication_plan.differentiable
        )
        self._max_team_edges = max_team_edges(
            len(self.team_names),
            include_self_edges=self.communication_config.topology.include_self_edges,
        )
        comm_settings = (
            GnnCommSettings.from_plan(self._communication_plan)
            if self._differentiable_comm
            else None
        )
        self.encoder = self._build_encoder()
        num_actions = self._num_actions()
        self.method = TeamPPO(
            settings=self.settings,
            team_names=self.team_names,
            feature_dim=self.encoder.feature_dim,
            privileged_dim=privileged_state_dim(len(self.all_names)),
            num_actions=num_actions,
            comm=comm_settings,
        )
        if resume_path is not None:
            self._load_checkpoint(resume_path)

        logger.info(
            "MARL trainer ready: team={} ({} agents) vs scripted {} ({} agents), "
            "actor={} critic={} actions={} features={} device={}",
            self.settings.trainable_team,
            len(self.team_names),
            "red" if self.settings.trainable_team == "blue" else "blue",
            len(self.opponent_names),
            self.settings.actor,
            self.settings.critic,
            num_actions,
            self.encoder.feature_dim,
            str(self.settings.resolved_device()),
        )

    # ------------------------------------------------------------------
    # Validation / construction helpers
    # ------------------------------------------------------------------

    def _validate_environment_for_training(self) -> None:
        """Reject environment configs the v1 trainer cannot learn from."""
        action_config = ActionSpaceConfig.from_dict(self.env_config.get("action_space"))
        if not action_config.is_discrete:
            raise TrainingConfigError(
                "The MARL trainer needs the discrete movement action space: set "
                "environment.action_space.type: 'discrete' (the benchmark decision; "
                "scripted opponents keep emitting dict actions unchanged)."
            )
        observation = ObservationConfig.from_dict(self.env_config.get("observation"))
        if self.settings.trainable_team == "blue":
            if not observation.bearing_only:
                raise TrainingConfigError(
                    "trainable_team: blue requires the bearing-only blue sensor: set "
                    "environment.observation.blue_sensor: 'bearing_only'. The v1 policy "
                    "encoder consumes contact_reports, not the legacy red_agents dict."
                )
            if observation.scripted_blue_privileged:
                raise TrainingConfigError(
                    "trainable_team: blue requires "
                    "environment.observation.scripted_blue_privileged: false. The "
                    "privileged mode exists for SCRIPTED blues only; when blue is the "
                    "trainable team no privileged key may exist in blue observations."
                )
        reward = parse_reward_settings(self.env_config.get("reward"))
        if self.settings.trainable_team == "blue" and not reward.blue_benchmark:
            logger.warning(
                "trainable_team is blue but environment.reward.blue is 'legacy' (the "
                "passive no-score bonus); the benchmark blue reward is the intended "
                "learning signal."
            )
        if self.settings.trainable_team == "red" and not reward.red_benchmark:
            logger.warning(
                "trainable_team is red but environment.reward.red is 'legacy' (sparse "
                "undetected ring scoring); the benchmark red reward adds the dense "
                "shaping a learner needs."
            )

    def _seed_everything(self, seed: int) -> None:
        """Seed python, numpy, and torch RNGs from the master seed."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _resolve_team_names(self) -> Tuple[List[str], List[str], List[str]]:
        """Build one throwaway agent set to fix the run's agent name order."""
        agents = create_agents_from_config(self.agents_config, dict(self.env_config))
        if not agents:
            raise TrainingConfigError(
                "No agents defined in the config's agents: section; the trainer needs "
                "both a trainable team and a scripted opponent team."
            )
        all_names = [agent.name for agent in agents]
        prefix = self.settings.trainable_team
        team = [a.name for a in agents if getattr(a.agent_type, "value", None) == prefix]
        opponents = [name for name in all_names if name not in team]
        if not team:
            raise TrainingConfigError(
                "training.trainable_team is {!r} but the config defines no {} agents. "
                "Add them under agents.{}_agents.".format(prefix, prefix, prefix)
            )
        if not opponents:
            raise TrainingConfigError(
                "The config defines no scripted opponent agents for trainable team "
                "{!r}; the baseline runner requires a scripted opposing team.".format(prefix)
            )
        return team, opponents, all_names

    def _build_encoder(self) -> PolicyEncoder:
        """Build the run's policy encoder from environment + training config.

        For a differentiable scheme the observations carry NO communication
        view (the env skips execution), so the encoder's comm feature width
        is 0: the actor computes communication itself from the raw graph
        inputs instead of consuming an env-delivered encoding.
        """
        communication = self.communication_config
        comm_dim = 0 if self._differentiable_comm else communication.payload.dimension
        return PolicyEncoder(
            grid_width=float(self.env_config.get("width", 100)),
            grid_height=float(self.env_config.get("height", 80)),
            contact_slots=self.settings.encoder.contact_slots,
            comm_dim=comm_dim,
            comm_scheme=communication.scheme,
        )

    def _num_actions(self) -> int:
        """Size of the flat discrete movement space (uniform across the team)."""
        action_config = ActionSpaceConfig.from_dict(self.env_config.get("action_space"))
        return 1 + action_config.headings * (action_config.speed_levels - 1)

    # ------------------------------------------------------------------
    # Environment lifecycle
    # ------------------------------------------------------------------

    def _start_episode(self) -> Dict[str, Any]:
        """Build a fresh environment and reset it with the next derived seed.

        Rebuilding agents+env every episode keeps scripted controllers free of
        cross-episode state and makes each episode a pure function of (config,
        derived seed, current policy), which is what checkpoint/resume
        exactness relies on.

        Returns:
            The reset observations mapping.
        """
        if self._env is not None:
            self._env.close()
        agents = create_agents_from_config(self.agents_config, dict(self.env_config))
        env = ParallelGameEnv(agents=agents, **self.env_config)
        env.env_config["agents"] = self.agents_config
        seed = _derive_reset_seed(self.settings.seed, self._reset_counter)
        self._reset_counter += 1
        observations, _infos = env.reset(seed=seed)
        self._env = env
        self.method.reset_episode()
        self._scripted = ScriptedTeamMethod(env.agent_objects, self.opponent_names)
        return observations

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, float]:
        """Run the configured number of updates; returns the last metrics."""
        settings = self.settings
        grid_w = self.encoder.grid_width
        grid_h = self.encoder.grid_height
        last_metrics: Dict[str, float] = {}
        write_header = not os.path.exists(self.metrics_path)
        with open(self.metrics_path, "a", newline="") as metrics_file:
            writer = csv.DictWriter(metrics_file, fieldnames=_METRIC_FIELDS)
            if write_header:
                writer.writeheader()
            for update_index in range(self._start_update, settings.updates):
                observations = self._start_episode()
                for step in range(settings.rollout_length):
                    if not self._env.agents:
                        observations = self._start_episode()
                    env = self._env
                    # --- snapshot t ---
                    policy_inputs = encode_team_observations(
                        observations, self.team_names, self.encoder
                    )
                    if self._differentiable_comm:
                        # Raw graph inputs for the in-actor communication
                        # stage: the frozen team graph at the pre-move
                        # positions (the plan's own topology) plus the raw
                        # node features. Stored in the rollout so updates
                        # RECOMPUTE the communication forward pass.
                        graph = build_team_communication_graph(
                            env.agent_objects,
                            self.team_names,
                            self._communication_plan.topology,
                            env.steps,
                        )
                        attach_communication_inputs(
                            policy_inputs, graph, self._max_team_edges
                        )
                    privileged = build_privileged_state(
                        env.agent_objects, self.all_names, grid_w, grid_h
                    )
                    # --- action selection ---
                    selection = self.method.select_actions(policy_inputs, privileged)
                    scripted = self._scripted.select_actions(
                        {name: observations.get(name) for name in self.opponent_names}
                    )
                    actions = dict(selection.env_actions)
                    actions.update(scripted.env_actions)
                    # --- one env.step ---
                    next_observations, rewards, terminations, truncations, _infos = env.step(
                        actions
                    )
                    # --- snapshot t + 1 ---
                    next_policy_inputs = encode_team_observations(
                        next_observations, self.team_names, self.encoder
                    )
                    next_privileged = build_privileged_state(
                        env.agent_objects, self.all_names, grid_w, grid_h
                    )
                    dones = torch.tensor(
                        [
                            bool(terminations.get(name, False) or truncations.get(name, False))
                            for name in self.team_names
                        ],
                        dtype=torch.bool,
                    )
                    terminateds = torch.tensor(
                        [bool(terminations.get(name, False)) for name in self.team_names],
                        dtype=torch.bool,
                    )
                    reward_tensor = torch.tensor(
                        [float(rewards.get(name, 0.0)) for name in self.team_names],
                        dtype=torch.float32,
                    )
                    transition = TeamTransition(
                        step=step,
                        policy_inputs=policy_inputs,
                        privileged_state=privileged,
                        actions=selection.actions,
                        extras=selection.extras,
                        rewards=reward_tensor,
                        dones=dones,
                        terminateds=terminateds,
                        next_policy_inputs=next_policy_inputs,
                        next_privileged_state=next_privileged,
                    )
                    self.method.ingest_transition(transition)
                    observations = next_observations
                if not self.method.ready_to_update():
                    raise RuntimeError(
                        "Rollout collected {} steps but the method is not ready to "
                        "update; this indicates a contract bug.".format(
                            settings.rollout_length
                        )
                    )
                metrics = self.method.update()
                last_metrics = metrics
                writer.writerow({key: metrics.get(key, 0.0) for key in _METRIC_FIELDS})
                metrics_file.flush()
                logger.info(
                    "update {}/{}: mean_reward={:.4f} loss_objective={:.4f} "
                    "loss_critic={:.4f} entropy={:.4f}",
                    update_index + 1,
                    settings.updates,
                    metrics["mean_reward"],
                    metrics["loss_objective"],
                    metrics["loss_critic"],
                    metrics["entropy"],
                )
                if (update_index + 1) % settings.checkpoint_every == 0:
                    self._save_checkpoint(update_index + 1)
        self._save_checkpoint(settings.updates, final=True)
        if self._env is not None:
            self._env.close()
            self._env = None
        logger.info("Training finished; artifacts in {}", self.results_dir)
        return last_metrics

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _checkpoint_payload(self, updates_completed: int) -> Dict[str, Any]:
        """Assemble the complete checkpoint dictionary."""
        return {
            "method_state": self.method.state_dict(),
            "updates_completed": updates_completed,
            "reset_counter": self._reset_counter,
            "config_snapshot": self.config,
            "config_path": self.config_path,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
            },
            "team_names": self.team_names,
            "settings": {
                "trainable_team": self.settings.trainable_team,
                "actor": self.settings.actor,
                "critic": self.settings.critic,
                "feature_dim": self.encoder.feature_dim,
                "comm_scheme": self.communication_config.scheme,
            },
        }

    def _save_checkpoint(self, updates_completed: int, final: bool = False) -> str:
        """Write a checkpoint file and refresh ``checkpoint_latest.pt``."""
        name = "checkpoint_final.pt" if final else "checkpoint_{:05d}.pt".format(
            updates_completed
        )
        path = os.path.join(self.checkpoint_dir, name)
        payload = self._checkpoint_payload(updates_completed)
        torch.save(payload, path)
        torch.save(payload, os.path.join(self.checkpoint_dir, "checkpoint_latest.pt"))
        logger.debug("Saved checkpoint after {} update(s): {}", updates_completed, path)
        return path

    def _load_checkpoint(self, resume_path: str) -> None:
        """Restore method state, counters, and RNG streams for exact resume."""
        if not os.path.exists(resume_path):
            raise TrainingConfigError(
                "--resume points to {!r}, which does not exist.".format(resume_path)
            )
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        saved = payload.get("settings", {})
        mismatches = []
        for key, current in (
            ("trainable_team", self.settings.trainable_team),
            ("actor", self.settings.actor),
            ("critic", self.settings.critic),
            ("feature_dim", self.encoder.feature_dim),
            ("comm_scheme", self.communication_config.scheme),
        ):
            if saved.get(key) != current:
                mismatches.append(
                    "{}: checkpoint={!r} config={!r}".format(key, saved.get(key), current)
                )
        if mismatches:
            raise TrainingConfigError(
                "Checkpoint {!r} is incompatible with the current config ({}). Resume "
                "with the config the checkpoint was trained with.".format(
                    resume_path, "; ".join(mismatches)
                )
            )
        self.method.load_state_dict(payload["method_state"])
        self._reset_counter = int(payload["reset_counter"])
        self._start_update = int(payload["updates_completed"])
        rng = payload["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        logger.info(
            "Resumed from {} after {} update(s); continuing to {}.",
            resume_path,
            self._start_update,
            self.settings.updates,
        )


def run_marl_training(
    config_path: str,
    resume: Optional[str] = None,
    results_dir: Optional[str] = None,
) -> Dict[str, float]:
    """Entry point used by ``run.py --mode train`` for the MARL backend.

    Args:
        config_path: Path to the scenario YAML (must contain a ``training:``
            block with ``backend: marl``).
        resume: Optional checkpoint path for exact continuation.
        results_dir: Optional explicit results directory (tests use this).

    Returns:
        The final update's metrics.
    """
    config = load_config(config_path)
    if not config:
        raise TrainingConfigError(
            "Failed to load configuration from {!r}.".format(config_path)
        )
    runner = MarlTrainingRunner(
        config=config,
        config_path=config_path,
        results_dir=results_dir,
        resume_path=resume,
    )
    return runner.run()
