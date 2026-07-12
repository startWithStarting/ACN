"""ACN MARL trainer package (TorchRL-backed, config-driven, opt-in).

This package implements the first trainable benchmark from
``docs/communication_decision_log.md`` ("Training Algorithm Boundary",
"First MARL Task"): single-team PPO (IPPO/MAPPO) for a configured trainable
team against a scripted opponent team, driven entirely by the ``training:``
config block and routed through ``run.py --mode train`` when
``training.backend`` is ``"marl"``.

Public API:

- :func:`~src.training.marl.config.parse_training_config` /
  :class:`~src.training.marl.config.TrainingSettings`: the validated
  ``training:`` block schema.
- :class:`~src.training.marl.contract.OnlineMethod` /
  :class:`~src.training.marl.contract.TeamTransition`: the common online
  method interface and per-step transition contract.
- :class:`~src.training.marl.encoders.PolicyEncoder`: policy-visible
  observation -> fixed-size tensors + masks.
- :class:`~src.training.marl.ppo.TeamPPO`: the one PPO implementation
  (shared/separate actor x local/global critic).
- :class:`~src.training.marl.runner.MarlTrainingRunner` /
  :func:`~src.training.marl.runner.run_marl_training`: the team-selection
  training runner and entry point.
"""

from src.training.marl.config import TrainingConfigError, TrainingSettings, parse_training_config
from src.training.marl.contract import ActionSelection, OnlineMethod, ScriptedTeamMethod, TeamTransition
from src.training.marl.encoders import PolicyEncoder
from src.training.marl.ppo import TeamPPO
from src.training.marl.runner import MarlTrainingRunner, run_marl_training

__all__ = [
    "ActionSelection",
    "MarlTrainingRunner",
    "OnlineMethod",
    "PolicyEncoder",
    "ScriptedTeamMethod",
    "TeamPPO",
    "TeamTransition",
    "TrainingConfigError",
    "TrainingSettings",
    "parse_training_config",
    "run_marl_training",
]
