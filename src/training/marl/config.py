"""Validated schema for the ``training:`` config block (MARL backend).

The block binds one training algorithm to exactly ONE trainable team per run
(``docs/communication_decision_log.md``, "First MARL Task"): learned blue
against scripted red, or learned red against scripted blue. Marking both (or
neither) team as trainable is rejected with an actionable error — simultaneous
or alternating self-play is a later training mode, never an implicit
consequence of this schema.

Reference block (all keys shown; defaults in parentheses)::

    training:
      backend: "marl"              # required to route into this trainer
      trainable_team: "blue"       # required: "blue" | "red" (exactly one)
      actor: "shared"              # "shared" | "separate"        (shared)
      critic: "global"             # "global" | "local"           (global)
      rollout_length: 128          # env steps collected per update
      updates: 100                 # number of PPO updates to run
      epochs: 4                    # optimization epochs per update
      minibatches: 4               # minibatches per epoch
      lr: 3.0e-4                   # Adam learning rate
      gamma: 0.99                  # discount
      gae_lambda: 0.95             # GAE lambda
      clip_epsilon: 0.2            # PPO clip range
      entropy_coef: 0.01           # entropy bonus coefficient
      value_coef: 0.5              # critic loss coefficient
      max_grad_norm: 0.5           # global gradient-norm clip
      seed: 0                      # master seed (python/numpy/torch)
      device: "cpu"                # torch device string
      checkpoint_every: 10         # save a checkpoint every N updates
      hidden_size: 64              # actor/critic MLP hidden width
      encoder:
        contact_slots: 8           # K contact slots in the policy encoder

``hidden_size`` is a small extension beyond the decision-log field list so
tests can use tiny networks; everything else matches the documented schema.

The default actor/critic combination (shared actor + global privileged
critic) is the decision log's first trainable benchmark: MAPPO with parameter
sharing. ``critic: local`` gives the shared-actor IPPO control; ``actor:
separate`` keeps per-agent actor copies supported.
"""

from __future__ import annotations

import math
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

import torch

from src.utils.logger import get_logger

logger = get_logger("acn.training.config")

__all__ = [
    "TrainingConfigError",
    "VALID_TRAINABLE_TEAMS",
    "VALID_ACTOR_MODES",
    "VALID_CRITIC_SCOPES",
    "MARL_BACKEND",
    "EncoderSettings",
    "TrainingSettings",
    "parse_training_config",
]


class TrainingConfigError(ValueError):
    """Raised when the ``training:`` configuration block is invalid."""


#: Teams a run may train. Exactly one must be selected (XOR).
VALID_TRAINABLE_TEAMS: Tuple[str, ...] = ("blue", "red")

#: Actor parameter-assignment modes (decision log, "Training Algorithm Boundary").
VALID_ACTOR_MODES: Tuple[str, ...] = ("shared", "separate")

#: Critic information scopes: local = IPPO-style, global = MAPPO/CTDE-style.
VALID_CRITIC_SCOPES: Tuple[str, ...] = ("local", "global")

#: The backend value that routes ``run.py --mode train`` into this trainer.
MARL_BACKEND: str = "marl"

_KNOWN_KEYS: Tuple[str, ...] = (
    "backend",
    "trainable_team",
    "actor",
    "critic",
    "rollout_length",
    "updates",
    "epochs",
    "minibatches",
    "lr",
    "gamma",
    "gae_lambda",
    "clip_epsilon",
    "entropy_coef",
    "value_coef",
    "max_grad_norm",
    "seed",
    "device",
    "checkpoint_every",
    "hidden_size",
    "encoder",
)

_KNOWN_ENCODER_KEYS: Tuple[str, ...] = ("contact_slots",)


def _require_int(value: Any, name: str, minimum: int) -> int:
    """Validate an integer field, rejecting bools and values below ``minimum``."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TrainingConfigError(
            "training.{} must be an integer, got {} ({!r}).".format(
                name, type(value).__name__, value
            )
        )
    if value < minimum:
        raise TrainingConfigError(
            "training.{} must be >= {} (got {}).".format(name, minimum, value)
        )
    return value


def _require_number(
    value: Any,
    name: str,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    exclusive_minimum: bool = False,
) -> float:
    """Validate a finite numeric field within an optional closed/open range."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingConfigError(
            "training.{} must be a number, got {} ({!r}).".format(
                name, type(value).__name__, value
            )
        )
    number = float(value)
    if not math.isfinite(number):
        raise TrainingConfigError("training.{} must be finite (got {!r}).".format(name, value))
    if minimum is not None:
        if exclusive_minimum and number <= minimum:
            raise TrainingConfigError(
                "training.{} must be > {} (got {}).".format(name, minimum, number)
            )
        if not exclusive_minimum and number < minimum:
            raise TrainingConfigError(
                "training.{} must be >= {} (got {}).".format(name, minimum, number)
            )
    if maximum is not None and number > maximum:
        raise TrainingConfigError(
            "training.{} must be <= {} (got {}).".format(name, maximum, number)
        )
    return number


@dataclass(frozen=True)
class EncoderSettings:
    """Validated ``training.encoder`` sub-block.

    Attributes:
        contact_slots: K, the number of padded contact-report slots the policy
            encoder exposes (reports beyond K are dropped deterministically in
            report order).
    """

    contact_slots: int = 8

    def __post_init__(self) -> None:
        _require_int(self.contact_slots, "encoder.contact_slots", 1)


@dataclass(frozen=True)
class TrainingSettings:
    """Validated, immutable ``training:`` block for the MARL backend.

    See the module docstring for the full reference block. ``trainable_team``
    is required and must name exactly one team; everything else has benchmark
    defaults.
    """

    trainable_team: str
    backend: str = MARL_BACKEND
    actor: str = "shared"
    critic: str = "global"
    rollout_length: int = 128
    updates: int = 100
    epochs: int = 4
    minibatches: int = 4
    lr: float = 3.0e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    seed: int = 0
    device: str = "cpu"
    checkpoint_every: int = 10
    hidden_size: int = 64
    encoder: EncoderSettings = field(default_factory=EncoderSettings)

    def __post_init__(self) -> None:
        if self.backend != MARL_BACKEND:
            raise TrainingConfigError(
                "training.backend must be {!r} for the MARL trainer, got {!r}. "
                "Remove the block or set backend: 'marl'.".format(MARL_BACKEND, self.backend)
            )
        _validate_trainable_team(self.trainable_team)
        if self.actor not in VALID_ACTOR_MODES:
            raise TrainingConfigError(
                "training.actor must be one of {} (got {!r}).".format(
                    ", ".join(VALID_ACTOR_MODES), self.actor
                )
            )
        if self.critic not in VALID_CRITIC_SCOPES:
            raise TrainingConfigError(
                "training.critic must be one of {} (got {!r}).".format(
                    ", ".join(VALID_CRITIC_SCOPES), self.critic
                )
            )
        _require_int(self.rollout_length, "rollout_length", 1)
        _require_int(self.updates, "updates", 1)
        _require_int(self.epochs, "epochs", 1)
        _require_int(self.minibatches, "minibatches", 1)
        if self.minibatches > self.rollout_length:
            raise TrainingConfigError(
                "training.minibatches ({}) cannot exceed training.rollout_length ({}): "
                "every minibatch needs at least one sample.".format(
                    self.minibatches, self.rollout_length
                )
            )
        _require_number(self.lr, "lr", minimum=0.0, exclusive_minimum=True)
        _require_number(self.gamma, "gamma", minimum=0.0, maximum=1.0, exclusive_minimum=True)
        _require_number(
            self.gae_lambda, "gae_lambda", minimum=0.0, maximum=1.0, exclusive_minimum=False
        )
        _require_number(self.clip_epsilon, "clip_epsilon", minimum=0.0, exclusive_minimum=True)
        _require_number(self.entropy_coef, "entropy_coef", minimum=0.0)
        _require_number(self.value_coef, "value_coef", minimum=0.0)
        _require_number(self.max_grad_norm, "max_grad_norm", minimum=0.0, exclusive_minimum=True)
        _require_int(self.seed, "seed", 0)
        _require_int(self.checkpoint_every, "checkpoint_every", 1)
        _require_int(self.hidden_size, "hidden_size", 1)
        if not isinstance(self.device, str):
            raise TrainingConfigError(
                "training.device must be a torch device string (e.g. 'cpu', 'cuda'), "
                "got {} ({!r}).".format(type(self.device).__name__, self.device)
            )
        try:
            torch.device(self.device)
        except (RuntimeError, ValueError) as exc:
            raise TrainingConfigError(
                "training.device {!r} is not a valid torch device: {}".format(self.device, exc)
            )
        if not isinstance(self.encoder, EncoderSettings):
            raise TrainingConfigError(
                "training.encoder must parse to EncoderSettings, got {!r}.".format(self.encoder)
            )


def _validate_trainable_team(value: Any) -> None:
    """Enforce the exclusive team choice with actionable errors.

    The valid values are exactly ``"blue"`` and ``"red"``. Anything that reads
    as "both", "neither", or a collection is rejected explicitly so a
    misconfigured self-play attempt fails loudly.
    """
    if value is None:
        raise TrainingConfigError(
            "training.trainable_team is required: set it to 'blue' (learned blue vs "
            "scripted red) or 'red' (learned red vs scripted blue). Exactly one team "
            "trains per run."
        )
    if isinstance(value, (list, tuple, set)):
        raise TrainingConfigError(
            "training.trainable_team must be a single team, got {!r}. Training both "
            "teams in one run (self-play) is not supported by the baseline runner; "
            "pick 'blue' or 'red'.".format(value)
        )
    if not isinstance(value, str):
        raise TrainingConfigError(
            "training.trainable_team must be the string 'blue' or 'red', got {} "
            "({!r}).".format(type(value).__name__, value)
        )
    lowered = value.lower()
    if lowered in ("both", "all", "blue+red", "red+blue"):
        raise TrainingConfigError(
            "training.trainable_team={!r} requests both teams, but exactly one team "
            "trains per run (decision log, 'First MARL Task'). Self-play is a later "
            "training mode; pick 'blue' or 'red'.".format(value)
        )
    if lowered in ("none", "neither", ""):
        raise TrainingConfigError(
            "training.trainable_team={!r} selects no team; the trainer needs exactly "
            "one of 'blue' or 'red'. To run without training use --mode parallel "
            "instead.".format(value)
        )
    if value not in VALID_TRAINABLE_TEAMS:
        raise TrainingConfigError(
            "training.trainable_team must be one of {} (got {!r}).".format(
                ", ".join(VALID_TRAINABLE_TEAMS), value
            )
        )


def parse_training_config(data: Optional[Mapping[str, Any]]) -> TrainingSettings:
    """Parse and validate the ``training:`` block of a loaded YAML config.

    Args:
        data: The plain-dict ``training`` block. ``None`` (absent) is rejected:
            callers must only invoke the MARL trainer when the block exists
            with ``backend: marl``.

    Returns:
        A validated, immutable :class:`TrainingSettings`.

    Raises:
        TrainingConfigError: If the block is missing, is not a mapping, has
            unknown keys, violates the exclusive trainable-team choice, or any
            field fails validation.
    """
    if data is None:
        raise TrainingConfigError(
            "The config has no training: block. Add one with backend: 'marl' and "
            "trainable_team: 'blue' or 'red' to use the MARL trainer."
        )
    if not isinstance(data, MappingABC):
        raise TrainingConfigError(
            "training must be a mapping of keys to values, got {} ({!r}). Check the "
            "YAML indentation of this block.".format(type(data).__name__, data)
        )
    unknown = sorted(set(data.keys()) - set(_KNOWN_KEYS))
    if unknown:
        raise TrainingConfigError(
            "Unknown training keys: {}. Known keys: {}.".format(
                ", ".join(unknown), ", ".join(_KNOWN_KEYS)
            )
        )

    encoder_raw = data.get("encoder")
    if encoder_raw is None:
        encoder = EncoderSettings()
    else:
        if not isinstance(encoder_raw, MappingABC):
            raise TrainingConfigError(
                "training.encoder must be a mapping, got {} ({!r}).".format(
                    type(encoder_raw).__name__, encoder_raw
                )
            )
        unknown_encoder = sorted(set(encoder_raw.keys()) - set(_KNOWN_ENCODER_KEYS))
        if unknown_encoder:
            raise TrainingConfigError(
                "Unknown training.encoder keys: {}. Known keys: {}.".format(
                    ", ".join(unknown_encoder), ", ".join(_KNOWN_ENCODER_KEYS)
                )
            )
        encoder = EncoderSettings(
            contact_slots=encoder_raw.get("contact_slots", 8),
        )

    # Validate the exclusive team choice before dataclass construction so a
    # missing key raises the actionable error, not a TypeError.
    _validate_trainable_team(data.get("trainable_team"))

    kwargs = {key: data[key] for key in _KNOWN_KEYS if key in data and key != "encoder"}
    settings = TrainingSettings(encoder=encoder, **kwargs)
    logger.debug(
        "Parsed training config: team={} actor={} critic={} rollout={} updates={}",
        settings.trainable_team,
        settings.actor,
        settings.critic,
        settings.rollout_length,
        settings.updates,
    )
    return settings
