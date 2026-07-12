"""Validation tests for the ``training:`` config schema (MARL backend)."""

import unittest

from src.training.marl.config import (
    EncoderSettings,
    TrainingConfigError,
    TrainingSettings,
    parse_training_config,
)


def valid_block(**overrides):
    """Return a minimal valid training block; overrides are merged on top."""
    block = {"backend": "marl", "trainable_team": "red"}
    block.update(overrides)
    return block


class TestTrainableTeamXor(unittest.TestCase):
    """Exactly one team trains per run; both/neither are rejected loudly."""

    def test_blue_and_red_are_the_only_valid_teams(self):
        for team in ("blue", "red"):
            settings = parse_training_config(valid_block(trainable_team=team))
            self.assertEqual(settings.trainable_team, team)

    def test_missing_team_is_actionable(self):
        with self.assertRaisesRegex(TrainingConfigError, "trainable_team is required"):
            parse_training_config({"backend": "marl"})

    def test_both_teams_rejected(self):
        with self.assertRaisesRegex(TrainingConfigError, "exactly one team"):
            parse_training_config(valid_block(trainable_team="both"))

    def test_team_list_rejected(self):
        with self.assertRaisesRegex(TrainingConfigError, "single team"):
            parse_training_config(valid_block(trainable_team=["blue", "red"]))

    def test_neither_team_rejected(self):
        with self.assertRaisesRegex(TrainingConfigError, "selects no team"):
            parse_training_config(valid_block(trainable_team="none"))

    def test_unknown_team_rejected(self):
        with self.assertRaisesRegex(TrainingConfigError, "must be one of"):
            parse_training_config(valid_block(trainable_team="green"))

    def test_non_string_team_rejected(self):
        with self.assertRaisesRegex(TrainingConfigError, "must be the string"):
            parse_training_config(valid_block(trainable_team=1))


class TestSchemaValidation(unittest.TestCase):
    def test_absent_block_is_rejected(self):
        with self.assertRaisesRegex(TrainingConfigError, "no training: block"):
            parse_training_config(None)

    def test_non_mapping_block_is_rejected(self):
        with self.assertRaisesRegex(TrainingConfigError, "must be a mapping"):
            parse_training_config(["marl"])

    def test_unknown_keys_are_rejected(self):
        with self.assertRaisesRegex(TrainingConfigError, "Unknown training keys: banana"):
            parse_training_config(valid_block(banana=1))

    def test_wrong_backend_is_rejected(self):
        with self.assertRaisesRegex(TrainingConfigError, "backend"):
            parse_training_config(valid_block(backend="sb3"))

    def test_invalid_actor_and_critic(self):
        with self.assertRaisesRegex(TrainingConfigError, "training.actor"):
            parse_training_config(valid_block(actor="grouped"))
        with self.assertRaisesRegex(TrainingConfigError, "training.critic"):
            parse_training_config(valid_block(critic="central"))

    def test_numeric_field_validation(self):
        with self.assertRaisesRegex(TrainingConfigError, "rollout_length"):
            parse_training_config(valid_block(rollout_length=0))
        with self.assertRaisesRegex(TrainingConfigError, "lr"):
            parse_training_config(valid_block(lr=0.0))
        with self.assertRaisesRegex(TrainingConfigError, "gamma"):
            parse_training_config(valid_block(gamma=1.5))
        with self.assertRaisesRegex(TrainingConfigError, "seed"):
            parse_training_config(valid_block(seed=-1))
        with self.assertRaisesRegex(TrainingConfigError, "must be an integer"):
            parse_training_config(valid_block(epochs=2.5))

    def test_minibatches_cannot_exceed_rollout(self):
        with self.assertRaisesRegex(TrainingConfigError, "cannot exceed"):
            parse_training_config(valid_block(rollout_length=4, minibatches=8))

    def test_invalid_device_is_rejected(self):
        with self.assertRaisesRegex(TrainingConfigError, "torch device"):
            parse_training_config(valid_block(device="warp-drive"))

    def test_unknown_encoder_key_is_rejected(self):
        with self.assertRaisesRegex(TrainingConfigError, "training.encoder keys"):
            parse_training_config(valid_block(encoder={"slots": 4}))

    def test_encoder_contact_slots_validated(self):
        with self.assertRaisesRegex(TrainingConfigError, "contact_slots"):
            parse_training_config(valid_block(encoder={"contact_slots": 0}))

    def test_defaults_are_the_benchmark_mappo(self):
        settings = parse_training_config(valid_block())
        self.assertEqual(settings.actor, "shared")
        self.assertEqual(settings.critic, "global")
        self.assertEqual(settings.encoder, EncoderSettings(contact_slots=8))
        self.assertEqual(settings.device, "cpu")

    def test_settings_is_immutable(self):
        settings = parse_training_config(valid_block())
        with self.assertRaises(AttributeError):
            settings.trainable_team = "blue"

    def test_direct_construction_validates_too(self):
        with self.assertRaises(TrainingConfigError):
            TrainingSettings(trainable_team="blue", clip_epsilon=0.0)


if __name__ == "__main__":
    unittest.main()
