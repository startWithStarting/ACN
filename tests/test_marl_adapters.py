"""Tests for the PettingZoo dict <-> TensorDict adapters and privileged state."""

import unittest

import torch

from src.agents.base_agent import AgentType
from src.training.marl.adapters import (
    PRIVILEGED_FEATURES_PER_AGENT,
    actions_to_env,
    build_privileged_state,
    encode_team_observations,
    privileged_state_dim,
)
from src.training.marl.encoders import PolicyEncoder


class FakeAgent:
    """Minimal stand-in for a live agent object."""

    def __init__(self, name, team, x, y, direction=(0.0, 0.0), speed=0.0, max_speed=10.0):
        self.name = name
        self.agent_type = AgentType.BLUE if team == "blue" else AgentType.RED
        self.x = x
        self.y = y
        self.direction = direction
        self.speed = speed
        self.max_speed = max_speed


def observation(x, y, reports=()):
    return {
        "position": (x, y),
        "grid_center": (50.0, 50.0),
        "timestamp": 0,
        "contact_reports": list(reports),
    }


class TestEncodeTeamObservations(unittest.TestCase):
    def setUp(self):
        self.encoder = PolicyEncoder(grid_width=100.0, grid_height=100.0, contact_slots=2)

    def test_round_trip_shapes_and_order(self):
        observations = {
            "blue_0": observation(10.0, 0.0),
            "blue_1": observation(20.0, 0.0),
            "red_2": {"position": (0.0, 0.0), "grid_center": (50.0, 50.0)},
        }
        td = encode_team_observations(observations, ["blue_0", "blue_1"], self.encoder)
        self.assertEqual(list(td.batch_size), [2])
        self.assertEqual(tuple(td["features"].shape), (2, self.encoder.feature_dim))
        self.assertEqual(tuple(td["contacts"].shape), (2, 2, 4))
        self.assertEqual(tuple(td["contacts_mask"].shape), (2, 2))
        # Row order follows team_names, not dict order.
        self.assertAlmostEqual(float(td["base"][0][0]), 0.1)
        self.assertAlmostEqual(float(td["base"][1][0]), 0.2)

    def test_missing_agent_is_actionable(self):
        with self.assertRaisesRegex(KeyError, "blue_1"):
            encode_team_observations(
                {"blue_0": observation(1.0, 1.0)}, ["blue_0", "blue_1"], self.encoder
            )


class TestPrivilegedState(unittest.TestCase):
    def test_layout_positions_velocities_teams(self):
        agents = {
            "blue_0": FakeAgent("blue_0", "blue", 25.0, 50.0, direction=(1.0, 0.0), speed=5.0),
            "red_1": FakeAgent("red_1", "red", 100.0, 0.0),
        }
        state = build_privileged_state(agents, ["blue_0", "red_1"], 100.0, 100.0)
        self.assertEqual(state.numel(), privileged_state_dim(2))
        self.assertEqual(privileged_state_dim(2), 2 * PRIVILEGED_FEATURES_PER_AGENT)
        torch.testing.assert_close(
            state,
            torch.tensor([0.25, 0.5, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
        )

    def test_missing_agent_row_stays_zero(self):
        state = build_privileged_state({}, ["ghost_0"], 100.0, 100.0)
        self.assertTrue(torch.all(state == 0))


class TestActionsToEnv(unittest.TestCase):
    def test_round_trip(self):
        actions = torch.tensor([3, 0, 7], dtype=torch.long)
        env_actions = actions_to_env(actions, ["red_0", "red_1", "red_2"])
        self.assertEqual(env_actions, {"red_0": 3, "red_1": 0, "red_2": 7})
        self.assertTrue(all(isinstance(v, int) for v in env_actions.values()))

    def test_size_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "2 entries"):
            actions_to_env(torch.tensor([1, 2]), ["red_0"])


if __name__ == "__main__":
    unittest.main()
