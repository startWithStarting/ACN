"""Runner tests: environment-prerequisite validation and checkpoint/resume.

The exact-continuation test relies on the runner's determinism model: every
episode starts from a fresh environment reset with a seed derived from the
master seed and a checkpointed reset counter, and all RNG streams (python,
numpy, torch, and the PPO minibatch permutation generator) are saved in the
checkpoint, so resuming after update ``k`` reproduces updates ``k+1..N``
bit-for-bit.
"""

import csv
import os
import tempfile
import unittest
from pathlib import Path

import yaml

from src.training.marl.config import TrainingConfigError
from src.training.marl.runner import MarlTrainingRunner

REPO_ROOT = Path(__file__).resolve().parents[1]


def micro_config(**training_overrides):
    """A tiny red-trainable scenario used across the runner tests."""
    training = {
        "backend": "marl",
        "trainable_team": "red",
        "actor": "shared",
        "critic": "global",
        "rollout_length": 8,
        "updates": 4,
        "epochs": 2,
        "minibatches": 2,
        "lr": 1.0e-3,
        "seed": 5,
        "checkpoint_every": 2,
        "hidden_size": 16,
        "encoder": {"contact_slots": 2},
    }
    training.update(training_overrides)
    return {
        "experiment_name": "marl_runner_test",
        "agents": {
            "blue_agents": [
                {
                    "count": 1,
                    "communication_bandwidth": 0,
                    "processing_capability": 0,
                    "detection_radius": 10.0,
                    "strategy_type": "static",
                    "max_speed": 5.0,
                }
            ],
            "red_agents": [
                {
                    "count": 2,
                    "communication_bandwidth": 0,
                    "processing_capability": 0,
                    "detection_radius": 10.0,
                    "strategy_type": "avoidant",
                    "max_speed": 10.0,
                }
            ],
        },
        "environment": {
            "width": 60,
            "height": 60,
            "max_cycles": 8,
            "save_episode_gifs": False,
            "render_mode": None,
            "action_space": {"type": "discrete", "headings": 4, "speed_levels": 2},
            "observation": {"blue_sensor": "bearing_only", "scripted_blue_privileged": True},
            "reward": {
                "blue": "legacy",
                "red": "benchmark",
                "weights": {"red_score": 1.0, "red_track": 0.1, "red_progress": 0.5},
            },
            "physics": {"enabled": False},
        },
        "training": training,
    }


def read_metrics(results_dir):
    with open(os.path.join(results_dir, "training_metrics.csv")) as handle:
        return list(csv.DictReader(handle))


class TestEnvironmentValidation(unittest.TestCase):
    def test_continuous_action_space_is_rejected(self):
        config = micro_config()
        del config["environment"]["action_space"]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(TrainingConfigError, "discrete"):
                MarlTrainingRunner(config, results_dir=tmp)

    def test_trainable_blue_requires_bearing_only(self):
        config = micro_config(trainable_team="blue")
        config["environment"]["observation"] = {"blue_sensor": "legacy"}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(TrainingConfigError, "bearing_only"):
                MarlTrainingRunner(config, results_dir=tmp)

    def test_trainable_blue_rejects_privileged_observations(self):
        config = micro_config(trainable_team="blue")
        config["environment"]["observation"] = {
            "blue_sensor": "bearing_only",
            "scripted_blue_privileged": True,
        }
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(TrainingConfigError, "scripted_blue_privileged"):
                MarlTrainingRunner(config, results_dir=tmp)

    def test_missing_trainable_agents_is_actionable(self):
        config = micro_config()
        config["agents"]["red_agents"] = []
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(TrainingConfigError, "no red agents"):
                MarlTrainingRunner(config, results_dir=tmp)

    def test_missing_opponent_agents_is_actionable(self):
        config = micro_config()
        config["agents"]["blue_agents"] = []
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(TrainingConfigError, "scripted opponent"):
                MarlTrainingRunner(config, results_dir=tmp)

    def test_missing_resume_checkpoint_is_actionable(self):
        config = micro_config()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(TrainingConfigError, "does not exist"):
                MarlTrainingRunner(config, results_dir=tmp, resume_path="/nope.pt")


class TestCheckpointResume(unittest.TestCase):
    def test_resume_reproduces_the_loss_trajectory_exactly(self):
        config = micro_config(updates=4, checkpoint_every=2)
        with tempfile.TemporaryDirectory() as tmp:
            full_dir = os.path.join(tmp, "full")
            resumed_dir = os.path.join(tmp, "resumed")

            MarlTrainingRunner(config, results_dir=full_dir).run()
            full_rows = read_metrics(full_dir)
            self.assertEqual(len(full_rows), 4)

            checkpoint = os.path.join(full_dir, "checkpoints", "checkpoint_00002.pt")
            self.assertTrue(os.path.exists(checkpoint))
            MarlTrainingRunner(
                config, results_dir=resumed_dir, resume_path=checkpoint
            ).run()
            resumed_rows = read_metrics(resumed_dir)
            self.assertEqual(len(resumed_rows), 2)

            for full_row, resumed_row in zip(full_rows[2:], resumed_rows):
                for key in (
                    "update",
                    "mean_reward",
                    "loss_objective",
                    "loss_critic",
                    "loss_entropy",
                ):
                    self.assertEqual(full_row[key], resumed_row[key], key)

    def test_incompatible_checkpoint_is_rejected(self):
        config = micro_config(updates=2, checkpoint_every=1)
        with tempfile.TemporaryDirectory() as tmp:
            train_dir = os.path.join(tmp, "train")
            MarlTrainingRunner(config, results_dir=train_dir).run()
            checkpoint = os.path.join(train_dir, "checkpoints", "checkpoint_latest.pt")
            other = micro_config(updates=2, actor="separate")
            with self.assertRaisesRegex(TrainingConfigError, "incompatible"):
                MarlTrainingRunner(
                    other, results_dir=os.path.join(tmp, "other"), resume_path=checkpoint
                )

    def test_checkpoints_and_metrics_are_written(self):
        config = micro_config(updates=2, checkpoint_every=1)
        with tempfile.TemporaryDirectory() as tmp:
            runner = MarlTrainingRunner(config, results_dir=tmp)
            runner.run()
            names = sorted(os.listdir(os.path.join(tmp, "checkpoints")))
            self.assertIn("checkpoint_00001.pt", names)
            self.assertIn("checkpoint_00002.pt", names)
            self.assertIn("checkpoint_final.pt", names)
            self.assertIn("checkpoint_latest.pt", names)
            self.assertEqual(len(read_metrics(tmp)), 2)


class TestBenchmarkConfigsAreRunnable(unittest.TestCase):
    """The shipped benchmark YAMLs construct and run one tiny update each."""

    def run_one_update(self, config_name):
        with open(REPO_ROOT / "config" / config_name) as handle:
            config = yaml.safe_load(handle)
        config["training"].update(
            {"updates": 1, "rollout_length": 8, "epochs": 1, "minibatches": 2,
             "hidden_size": 16, "checkpoint_every": 10}
        )
        config["environment"]["max_cycles"] = 8
        with tempfile.TemporaryDirectory() as tmp:
            metrics = MarlTrainingRunner(config, results_dir=tmp).run()
        self.assertEqual(metrics["update"], 1.0)

    def test_benchmark_blue_mappo(self):
        self.run_one_update("benchmark_blue_mappo.yaml")

    def test_benchmark_red_mappo(self):
        self.run_one_update("benchmark_red_mappo.yaml")


if __name__ == "__main__":
    unittest.main()
