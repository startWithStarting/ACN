"""Learning-sanity tests: the trainer measurably improves on micro scenarios.

Both tests are fully seeded, bounded in steps, and tuned to complete in
seconds with tiny networks.

- **Red**: learned red vs one distant static scripted blue, with the
  potential-based progress-to-ring shaping dominant (the decision log's
  recommended dense signal). A random policy's mean progress is negative
  (a random step from any point increases ``|distance - ring|`` on average),
  so the mean red return over a rollout strictly improves as the policy
  learns to move onto the ring. Asserted directly on the training curve.

- **Blue**: learned blue vs one stationary-at-center scripted red with a
  generous detection radius. The pin gate (``k >= 3`` detectors) is a team
  AND over the blues' positions, so the per-update training curve is
  dominated by random spawn variance and is NOT reliably monotone over a
  handful of updates; a first-vs-last curve assertion proved flaky. Per the
  documented fallback this is therefore a strict-improvement-over-baseline
  test: the trained policy and the run's own INITIAL policy are evaluated on
  the same paired evaluation seeds, and the trained policy must achieve a
  strictly higher mean pin coverage. Paired seeds remove the spawn variance
  that made the curve assertion flaky.
"""

import copy
import csv
import os
import tempfile
import unittest

import torch

from src.agents.factory import create_agents_from_config
from src.env.parallel_env import ParallelGameEnv
from src.env.rewards import build_detection_state
from src.training.marl.adapters import (
    build_privileged_state,
    encode_team_observations,
    privileged_state_dim,
)
from src.training.marl.contract import ScriptedTeamMethod
from src.training.marl.ppo import TeamPPO
from src.training.marl.runner import MarlTrainingRunner


def read_rewards(results_dir):
    with open(os.path.join(results_dir, "training_metrics.csv")) as handle:
        return [float(row["mean_reward"]) for row in csv.DictReader(handle)]


def evaluate_pin_coverage(method, runner, eval_seeds):
    """Mean per-step pin coverage of ``method`` over the given episode seeds."""
    total, steps = 0.0, 0
    env_config = dict(runner.env_config)
    grid_w = runner.encoder.grid_width
    grid_h = runner.encoder.grid_height
    for eval_seed in eval_seeds:
        agents = create_agents_from_config(runner.agents_config, dict(env_config))
        env = ParallelGameEnv(agents=agents, **env_config)
        env.env_config["agents"] = runner.agents_config
        observations, _ = env.reset(seed=eval_seed)
        scripted = ScriptedTeamMethod(env.agent_objects, runner.opponent_names)
        while env.agents:
            inputs = encode_team_observations(observations, runner.team_names, runner.encoder)
            privileged = build_privileged_state(
                env.agent_objects, runner.all_names, grid_w, grid_h
            )
            selection = method.select_actions(inputs, privileged)
            actions = dict(selection.env_actions)
            actions.update(
                scripted.select_actions(
                    {name: observations.get(name) for name in runner.opponent_names}
                ).env_actions
            )
            observations, _, _, _, _ = env.step(actions)
            state = build_detection_state(env.active_blue_agents, env.active_red_agents, {})
            if state.red_names:
                total += sum(
                    1.0 for name in state.red_names if state.pinned[name]
                ) / len(state.red_names)
            steps += 1
        env.close()
    return total / steps


class TestRedLearningImproves(unittest.TestCase):
    """Learned red return strictly improves against a static scripted blue."""

    def test_mean_red_return_improves_over_updates(self):
        config = {
            "experiment_name": "overfit_red",
            "agents": {
                "blue_agents": [
                    {
                        "count": 1,
                        "communication_bandwidth": 0,
                        "processing_capability": 0,
                        "detection_radius": 5.0,
                        "strategy_type": "static",
                        "max_speed": 1.0,
                    }
                ],
                "red_agents": [
                    {
                        "count": 1,
                        "communication_bandwidth": 0,
                        "processing_capability": 0,
                        "detection_radius": 10.0,
                        "strategy_type": "avoidant",
                        "max_speed": 10.0,
                    }
                ],
            },
            "environment": {
                # 120x120 keeps the legacy scoring ring (radius 50 around the
                # center) fully inside the arena.
                "width": 120,
                "height": 120,
                "max_cycles": 32,
                "save_episode_gifs": False,
                "render_mode": None,
                "action_space": {"type": "discrete", "headings": 8, "speed_levels": 2},
                "observation": {
                    "blue_sensor": "bearing_only",
                    "scripted_blue_privileged": True,
                },
                "reward": {
                    "blue": "legacy",
                    "red": "benchmark",
                    # Progress-to-ring shaping dominant, per the test design.
                    "weights": {"red_score": 1.0, "red_track": 0.0, "red_progress": 1.0},
                },
                "physics": {"enabled": False},
            },
            "training": {
                "backend": "marl",
                "trainable_team": "red",
                "actor": "shared",
                "critic": "global",
                "rollout_length": 64,
                "updates": 6,
                "epochs": 4,
                "minibatches": 4,
                "lr": 3.0e-3,
                "gamma": 0.95,
                "gae_lambda": 0.95,
                "entropy_coef": 0.001,
                "seed": 3,
                "checkpoint_every": 100,
                "hidden_size": 32,
                "encoder": {"contact_slots": 4},
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            MarlTrainingRunner(config, results_dir=tmp).run()
            rewards = read_rewards(tmp)
        self.assertEqual(len(rewards), 6)
        early = sum(rewards[:2]) / 2.0
        late = sum(rewards[-2:]) / 2.0
        self.assertGreater(
            late,
            early + 0.2,
            "mean red return did not improve: curve={}".format(
                [round(value, 3) for value in rewards]
            ),
        )


class TestBlueLearningImproves(unittest.TestCase):
    """Trained blue pin coverage strictly beats its own initial policy."""

    def test_pin_coverage_beats_initial_policy_on_paired_seeds(self):
        config = {
            "experiment_name": "overfit_blue",
            "agents": {
                "blue_agents": [
                    {
                        "count": 4,
                        "communication_bandwidth": 0,
                        "processing_capability": 0,
                        # Generous detection radius on a 60x60 grid.
                        "detection_radius": 35.0,
                        "strategy_type": "pursuit",
                        "max_speed": 10.0,
                    }
                ],
                "red_agents": [
                    {
                        "count": 1,
                        "communication_bandwidth": 0,
                        "processing_capability": 0,
                        "detection_radius": 5.0,
                        # Moves to the grid center and stays: effectively
                        # stationary after a few steps.
                        "strategy_type": "center",
                        "max_speed": 6.0,
                    }
                ],
            },
            "environment": {
                "width": 60,
                "height": 60,
                "max_cycles": 16,
                "save_episode_gifs": False,
                "render_mode": None,
                "action_space": {"type": "discrete", "headings": 8, "speed_levels": 2},
                "observation": {
                    "blue_sensor": "bearing_only",
                    "scripted_blue_privileged": False,
                },
                "reward": {
                    "blue": "benchmark",
                    "red": "legacy",
                    # Shape term boosted for individual credit assignment on
                    # the micro task; pin coverage is what the test measures.
                    "weights": {"pin": 0.5, "track": 0.0, "shape": 1.0, "score_penalty": 0.0},
                },
                "physics": {"enabled": False},
            },
            "training": {
                "backend": "marl",
                "trainable_team": "blue",
                "actor": "shared",
                "critic": "global",
                "rollout_length": 128,
                "updates": 12,
                "epochs": 4,
                "minibatches": 4,
                "lr": 3.0e-3,
                "gamma": 0.95,
                "gae_lambda": 0.95,
                "entropy_coef": 0.001,
                "seed": 11,
                "checkpoint_every": 100,
                "hidden_size": 32,
                "encoder": {"contact_slots": 4},
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            runner = MarlTrainingRunner(config, results_dir=tmp)
            initial_state = copy.deepcopy(runner.method.state_dict())
            runner.run()

            baseline = TeamPPO(
                runner.settings,
                team_names=runner.team_names,
                feature_dim=runner.encoder.feature_dim,
                privileged_dim=privileged_state_dim(len(runner.all_names)),
                num_actions=runner._num_actions(),
            )
            baseline.load_state_dict(initial_state)

            eval_seeds = list(range(5000, 5020))
            torch.manual_seed(99)
            trained = evaluate_pin_coverage(runner.method, runner, eval_seeds)
            torch.manual_seed(99)
            initial = evaluate_pin_coverage(baseline, runner, eval_seeds)
        self.assertGreater(
            trained,
            initial + 0.05,
            "trained pin coverage {:.3f} did not beat the initial policy's "
            "{:.3f}".format(trained, initial),
        )


if __name__ == "__main__":
    unittest.main()
