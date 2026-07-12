"""Tests for the Phase 1 engineered bearing-report message source."""

import math
import unittest

import numpy as np
import torch

from src.communication.sources import EngineeredBearingSource, MessageSource
from src.communication.transport import Frame


def observation(position, opponents, key="red_agents"):
    """Build an env-format observation dict with visible opponents."""
    return {
        "position": np.array(position, dtype=np.float32),
        key: {
            name: {"position": pos, "distance": math.dist(position, pos)}
            for name, pos in opponents.items()
        },
    }


class TestBearingPayload(unittest.TestCase):
    """Payload layout and anonymity of one contact report."""

    def setUp(self):
        self.source = EngineeredBearingSource()

    def test_payload_is_observer_position_plus_unit_bearing(self):
        obs = observation((1.0, 2.0), {"red_0": (4.0, 6.0)})
        (report,) = self.source.build_frames("blue_0", obs, step=3)
        self.assertTrue(
            torch.allclose(report.payload, torch.tensor([1.0, 2.0, 0.6, 0.8]))
        )

    def test_bearing_is_unit_length(self):
        obs = observation((0.0, 0.0), {"red_0": (-7.0, 3.0)})
        (report,) = self.source.build_frames("blue_0", obs, step=0)
        direction = report.payload[2:]
        self.assertAlmostEqual(float(direction.norm()), 1.0, places=6)

    def test_payload_contains_no_opponent_identity_range_or_velocity(self):
        """Exactly 4 numbers: observer x/y and the bearing — nothing else."""
        opponent_position = (30.0, 40.0)  # distance 50: must NOT appear anywhere
        obs = observation((0.0, 0.0), {"red_9": opponent_position})
        (report,) = self.source.build_frames("blue_0", obs, step=0)
        self.assertEqual(report.payload.shape, (4,))
        self.assertEqual(report.payload.shape[0], EngineeredBearingSource.PAYLOAD_DIMENSION)
        self.assertNotIn(50.0, report.payload.tolist())
        self.assertNotIn(30.0, report.payload.tolist())
        self.assertNotIn(40.0, report.payload.tolist())

    def test_metadata_carries_origin_and_observation_step(self):
        obs = observation((0.0, 0.0), {"red_0": (1.0, 0.0)})
        (report,) = self.source.build_frames("blue_3", obs, step=11)
        self.assertEqual(report.origin, "blue_3")
        self.assertEqual(report.observation_step, 11)
        self.assertEqual(report.created_step, 11)
        self.assertIsInstance(report, Frame)

    def test_ground_truth_identity_is_privileged_only(self):
        obs = observation((0.0, 0.0), {"red_5": (1.0, 0.0)})
        (report,) = self.source.build_frames("blue_0", obs, step=0)
        self.assertEqual(report.privileged["opponent_id"], "red_5")

    def test_zero_distance_detection_uses_degenerate_direction(self):
        obs = observation((2.0, 2.0), {"red_0": (2.0, 2.0)})
        (report,) = self.source.build_frames("blue_0", obs, step=0)
        self.assertTrue(torch.equal(report.payload, torch.tensor([2.0, 2.0, 0.0, 0.0])))


class TestReportCollections(unittest.TestCase):
    """Variable-size report collections and outbox construction."""

    def setUp(self):
        self.source = EngineeredBearingSource()

    def test_one_report_per_visible_opponent(self):
        obs = observation((0.0, 0.0), {"red_0": (1.0, 0.0), "red_1": (0.0, 1.0), "red_2": (2.0, 2.0)})
        reports = self.source.build_frames("blue_0", obs, step=0)
        self.assertEqual(len(reports), 3)

    def test_reports_ordered_by_opponent_id(self):
        obs = observation((0.0, 0.0), {"red_1": (0.0, 1.0), "red_0": (1.0, 0.0)})
        reports = self.source.build_frames("blue_0", obs, step=0)
        self.assertEqual(
            [r.privileged["opponent_id"] for r in reports], ["red_0", "red_1"]
        )

    def test_no_visible_opponents_produces_empty_outbox(self):
        obs = observation((0.0, 0.0), {})
        self.assertEqual(self.source.build_frames("blue_0", obs, step=0), [])

    def test_outboxes_cover_every_observer(self):
        observations = {
            "blue_1": observation((1.0, 0.0), {"red_0": (2.0, 0.0)}),
            "blue_0": observation((0.0, 0.0), {}),
        }
        outboxes = self.source.build_outboxes(observations, step=2)
        self.assertEqual(sorted(outboxes.keys()), ["blue_0", "blue_1"])
        self.assertEqual(outboxes["blue_0"], [])
        self.assertEqual(len(outboxes["blue_1"]), 1)

    def test_red_observer_blue_agents_key_auto_detected(self):
        obs = observation((0.0, 0.0), {"blue_0": (0.0, 3.0)}, key="blue_agents")
        obs["red_teammates"] = {"red_1": {"position": (1.0, 0.0), "distance": 1.0}}
        reports = self.source.build_frames("red_0", obs, step=0)
        self.assertEqual(len(reports), 1, "teammate entries must never produce reports")
        self.assertEqual(reports[0].privileged["opponent_id"], "blue_0")

    def test_explicit_opponent_key_overrides_detection(self):
        obs = observation((0.0, 0.0), {"x_0": (1.0, 0.0)}, key="custom_contacts")
        source = EngineeredBearingSource(opponent_key="custom_contacts")
        self.assertEqual(len(source.build_frames("blue_0", obs, step=0)), 1)

    def test_deterministic_across_two_calls(self):
        obs = {"blue_0": observation((0.5, -0.5), {"red_0": (3.0, 1.0), "red_1": (-1.0, 2.0)})}
        first = self.source.build_outboxes(obs, step=1)
        second = self.source.build_outboxes(obs, step=1)
        for a, b in zip(first["blue_0"], second["blue_0"]):
            self.assertTrue(torch.equal(a.payload, b.payload))
            self.assertEqual(a.origin, b.origin)

    def test_source_only_reads_local_observation(self):
        """The source satisfies the MessageSource protocol: u_i(t) = g_comm(o_i(t))."""
        self.assertIsInstance(self.source, MessageSource)

    def test_missing_observer_position_rejected_when_opponents_visible(self):
        obs = {"red_agents": {"red_0": {"position": (1.0, 0.0), "distance": 1.0}}}
        with self.assertRaises(ValueError):
            self.source.build_frames("blue_0", obs, step=0)

    def test_malformed_opponent_entry_rejected(self):
        obs = {"position": (0.0, 0.0), "red_agents": {"red_0": {"distance": 1.0}}}
        with self.assertRaises(ValueError):
            self.source.build_frames("blue_0", obs, step=0)


if __name__ == "__main__":
    unittest.main()
