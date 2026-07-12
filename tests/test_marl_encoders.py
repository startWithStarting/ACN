"""Tests for the v1 policy-input encoders: padding, masks, comm views, leakage."""

import math
import unittest

import torch

from src.communication.processors.pyg import AggregatedVectorView
from src.communication.types import EdgeMessageBatch
from src.training.marl.encoders import PolicyEncoder


def blue_observation(reports, communication=None):
    """Build a minimal bearing-only blue observation."""
    observation = {
        "position": (10.0, 20.0),
        "grid_center": (50.0, 40.0),
        "timestamp": 3,
        "contact_reports": reports,
    }
    if communication is not None:
        observation["communication"] = communication
    return observation


def report(observer_x, observer_y, angle):
    """Build one bearing-only contact report at the given angle."""
    return {
        "payload": [observer_x, observer_y, math.cos(angle), math.sin(angle)],
        "metadata": {"observer": "blue_0", "step": 3},
    }


def inbox_batch(payloads):
    """Build a delivered EdgeMessageBatch with the given payload rows."""
    count = len(payloads)
    return EdgeMessageBatch(
        payload=torch.tensor(payloads, dtype=torch.float32),
        sender_index=torch.zeros(count, dtype=torch.long),
        receiver_index=torch.zeros(count, dtype=torch.long),
        origin_index=torch.zeros(count, dtype=torch.long),
        message_id=torch.arange(count, dtype=torch.long),
        round_index=torch.zeros(count, dtype=torch.long),
    )


class TestContactPadding(unittest.TestCase):
    def setUp(self):
        self.encoder = PolicyEncoder(
            grid_width=100.0, grid_height=80.0, contact_slots=3, comm_dim=4
        )

    def test_padding_and_mask(self):
        encoded = self.encoder.encode(blue_observation([report(10.0, 20.0, 0.0)]))
        self.assertEqual(tuple(encoded["contacts"].shape), (3, 4))
        self.assertEqual(encoded["contacts_mask"].tolist(), [True, False, False])
        torch.testing.assert_close(
            encoded["contacts"][0],
            torch.tensor([10.0 / 100.0, 20.0 / 80.0, 1.0, 0.0]),
        )
        # Padded slots stay zero.
        self.assertTrue(torch.all(encoded["contacts"][1:] == 0))

    def test_deterministic_order_is_report_order(self):
        reports = [report(10.0, 20.0, -1.0), report(10.0, 20.0, 0.5)]
        encoded = self.encoder.encode(blue_observation(reports))
        self.assertAlmostEqual(float(encoded["contacts"][0][2]), math.cos(-1.0), places=6)
        self.assertAlmostEqual(float(encoded["contacts"][1][2]), math.cos(0.5), places=6)

    def test_overflow_truncates_in_order(self):
        reports = [report(10.0, 20.0, 0.1 * i) for i in range(5)]
        encoded = self.encoder.encode(blue_observation(reports))
        self.assertEqual(encoded["contacts_mask"].tolist(), [True, True, True])
        self.assertAlmostEqual(float(encoded["contacts"][2][2]), math.cos(0.2), places=6)

    def test_no_contacts_yields_empty_mask(self):
        encoded = self.encoder.encode(blue_observation([]))
        self.assertEqual(encoded["contacts_mask"].tolist(), [False, False, False])

    def test_feature_layout_dim(self):
        encoded = self.encoder.encode(blue_observation([]))
        self.assertEqual(
            encoded["features"].numel(),
            self.encoder.feature_dim,
        )
        # base 4 + contacts 3*4 + mask 3 + comm 4
        self.assertEqual(self.encoder.feature_dim, 4 + 12 + 3 + 4)


class TestRedDerivedContacts(unittest.TestCase):
    def test_opponent_positions_become_anonymous_bearing_slots(self):
        encoder = PolicyEncoder(grid_width=100.0, grid_height=100.0, contact_slots=2, comm_dim=4)
        observation = {
            "position": (50.0, 50.0),
            "grid_center": (50.0, 50.0),
            "blue_agents": {
                "blue_9": {"position": (60.0, 50.0), "distance": 10.0},  # bearing 0
                "blue_1": {"position": (50.0, 40.0), "distance": 10.0},  # bearing -pi/2
            },
            "red_teammates": {},
        }
        encoded = encoder.encode(observation)
        self.assertEqual(encoded["contacts_mask"].tolist(), [True, True])
        # Sorted by bearing angle: -pi/2 before 0, regardless of dict key order.
        torch.testing.assert_close(
            encoded["contacts"][0], torch.tensor([0.5, 0.5, 0.0, -1.0]), atol=1e-6, rtol=0
        )
        torch.testing.assert_close(
            encoded["contacts"][1], torch.tensor([0.5, 0.5, 1.0, 0.0]), atol=1e-6, rtol=0
        )


class TestCommunicationEncoding(unittest.TestCase):
    def setUp(self):
        self.encoder = PolicyEncoder(
            grid_width=100.0, grid_height=80.0, contact_slots=2, comm_dim=4
        )

    def view(self, inbox, scheme="one_hop_direct"):
        return {"scheme": scheme, "inbox": inbox, "agent_ids": (), "cache": None,
                "cache_window": 0}

    def test_absent_view_yields_zeros(self):
        encoded = self.encoder.encode(blue_observation([]))
        self.assertTrue(torch.all(encoded["comm"] == 0))
        self.assertEqual(encoded["comm"].numel(), 4)

    def test_empty_tuple_inbox_yields_zeros(self):
        encoded = self.encoder.encode(blue_observation([], communication=self.view(())))
        self.assertTrue(torch.all(encoded["comm"] == 0))

    def test_aggregated_vector_view_passes_through(self):
        vector = torch.tensor([1.0, 2.0, 3.0, 4.0])
        view = self.view(AggregatedVectorView(vector=vector, count=2), scheme="one_hop_mean")
        encoded = self.encoder.encode(blue_observation([], communication=view))
        torch.testing.assert_close(encoded["comm"], vector)

    def test_preserved_inbox_is_mean_pooled_at_policy_boundary(self):
        batch = inbox_batch([[1.0, 0.0, 0.0, 0.0], [3.0, 2.0, 0.0, 0.0]])
        encoded = self.encoder.encode(blue_observation([], communication=self.view(batch)))
        torch.testing.assert_close(encoded["comm"], torch.tensor([2.0, 1.0, 0.0, 0.0]))

    def test_empty_inbox_batch_yields_zeros(self):
        batch = EdgeMessageBatch.empty(payload_dim=4)
        encoded = self.encoder.encode(blue_observation([], communication=self.view(batch)))
        self.assertTrue(torch.all(encoded["comm"] == 0))

    def test_dimension_mismatch_is_actionable(self):
        view = self.view(AggregatedVectorView(vector=torch.zeros(7), count=1), "one_hop_mean")
        with self.assertRaisesRegex(ValueError, "comm_dim"):
            self.encoder.encode(blue_observation([], communication=view))


class TestPrivilegedLeakageGuard(unittest.TestCase):
    def test_privileged_red_agents_is_refused(self):
        encoder = PolicyEncoder(grid_width=100.0, grid_height=80.0)
        observation = blue_observation([])
        observation["privileged_red_agents"] = {"red_1": {"position": (1.0, 2.0)}}
        with self.assertRaisesRegex(ValueError, "privileged"):
            encoder.encode(observation)

    def test_ground_truth_contacts_is_refused(self):
        encoder = PolicyEncoder(grid_width=100.0, grid_height=80.0)
        observation = blue_observation([])
        observation["ground_truth_contacts"] = {0: "red_1"}
        with self.assertRaisesRegex(ValueError, "privileged"):
            encoder.encode(observation)


if __name__ == "__main__":
    unittest.main()
