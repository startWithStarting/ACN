"""Shape and contract tests for the on-policy rollout buffer."""

import unittest

import torch
from tensordict import TensorDict

from src.training.marl.buffer import RolloutBuffer
from src.training.marl.contract import TeamTransition

NUM_AGENTS = 2
FEATURES = 6
SLOTS = 3
PRIV = 10


def policy_inputs(fill=0.0):
    return TensorDict(
        {
            "features": torch.full((NUM_AGENTS, FEATURES), fill),
            "contacts_mask": torch.zeros(NUM_AGENTS, SLOTS, dtype=torch.bool),
        },
        batch_size=[NUM_AGENTS],
    )


def transition(step, reward=1.0, done=False, extras=None):
    if extras is None:
        extras = {
            "sample_log_prob": torch.zeros(NUM_AGENTS),
            "state_value": torch.zeros(NUM_AGENTS),
        }
    return TeamTransition(
        step=step,
        policy_inputs=policy_inputs(float(step)),
        privileged_state=torch.full((PRIV,), float(step)),
        actions=torch.zeros(NUM_AGENTS, dtype=torch.long),
        extras=extras,
        rewards=torch.full((NUM_AGENTS,), reward),
        dones=torch.tensor([done] * NUM_AGENTS),
        terminateds=torch.tensor([False] * NUM_AGENTS),
        next_policy_inputs=policy_inputs(float(step) + 0.5),
        next_privileged_state=torch.full((PRIV,), float(step) + 0.5),
    )


class TestRolloutBuffer(unittest.TestCase):
    def test_tensordict_shapes(self):
        buffer = RolloutBuffer(["a_0", "a_1"])
        for step in range(4):
            buffer.add(transition(step))
        self.assertEqual(len(buffer), 4)
        td = buffer.as_tensordict()
        self.assertEqual(list(td.batch_size), [NUM_AGENTS, 4])
        self.assertEqual(tuple(td["features"].shape), (NUM_AGENTS, 4, FEATURES))
        self.assertEqual(tuple(td["contacts_mask"].shape), (NUM_AGENTS, 4, SLOTS))
        self.assertEqual(tuple(td["action"].shape), (NUM_AGENTS, 4))
        self.assertEqual(tuple(td["sample_log_prob"].shape), (NUM_AGENTS, 4))
        self.assertEqual(tuple(td["privileged_state"].shape), (NUM_AGENTS, 4, PRIV))
        self.assertEqual(tuple(td[("next", "reward")].shape), (NUM_AGENTS, 4, 1))
        self.assertEqual(tuple(td[("next", "done")].shape), (NUM_AGENTS, 4, 1))
        self.assertEqual(tuple(td[("next", "terminated")].shape), (NUM_AGENTS, 4, 1))
        self.assertEqual(tuple(td[("next", "features")].shape), (NUM_AGENTS, 4, FEATURES))
        self.assertEqual(tuple(td[("next", "privileged_state")].shape), (NUM_AGENTS, 4, PRIV))

    def test_time_axis_ordering_and_next_alignment(self):
        buffer = RolloutBuffer(["a_0", "a_1"])
        for step in range(3):
            buffer.add(transition(step))
        td = buffer.as_tensordict()
        self.assertEqual(float(td["features"][0, 2, 0]), 2.0)
        self.assertEqual(float(td[("next", "features")][0, 2, 0]), 2.5)
        self.assertEqual(float(td["privileged_state"][1, 1, 0]), 1.0)

    def test_clear_empties_the_buffer(self):
        buffer = RolloutBuffer(["a_0", "a_1"])
        buffer.add(transition(0))
        buffer.clear()
        self.assertEqual(len(buffer), 0)
        with self.assertRaisesRegex(ValueError, "empty"):
            buffer.as_tensordict()

    def test_wrong_team_size_rejected(self):
        buffer = RolloutBuffer(["a_0", "a_1", "a_2"])
        with self.assertRaisesRegex(ValueError, "3-agent team"):
            buffer.add(transition(0))

    def test_missing_extras_rejected(self):
        buffer = RolloutBuffer(["a_0", "a_1"])
        with self.assertRaisesRegex(ValueError, "state_value"):
            buffer.add(transition(0, extras={"sample_log_prob": torch.zeros(NUM_AGENTS)}))


if __name__ == "__main__":
    unittest.main()
