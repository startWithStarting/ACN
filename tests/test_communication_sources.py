"""Tests for the Phase 1 engineered bearing-report message source."""

import math
import unittest
from collections.abc import Mapping

import numpy as np
import torch

from src.agents.factory import create_agents_from_config
from src.communication.sources import EngineeredBearingSource, MessageSource
from src.communication.transport import Frame
from src.env.parallel_env import ParallelGameEnv
from src.env.sensors import build_contact_reports


def observation(position, opponents, key="red_agents"):
    """Build an env-format observation dict with visible opponents."""
    return {
        "position": np.array(position, dtype=np.float32),
        key: {
            name: {"position": pos, "distance": math.dist(position, pos)}
            for name, pos in opponents.items()
        },
    }


def bearing_observation(position, opponents, observer="blue_0", step=0):
    """Build a bearing-only observation with the sensor's real report layout."""
    reports, _ = build_contact_reports(
        observer, position, sorted(opponents.items()), step
    )
    return {
        "position": np.array(position, dtype=np.float32),
        "contact_reports": reports,
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


class TestContactReportFrames(unittest.TestCase):
    """Frames built from bearing-only contact_reports observations."""

    #: Non-degenerate geometry: id order (red_0, red_1) equals bearing order.
    POSITION = (1.0, 2.0)
    OPPONENTS = {"red_0": (4.0, 6.0), "red_1": (-2.0, 6.0)}

    def setUp(self):
        self.source = EngineeredBearingSource()

    def test_same_geometry_both_ways_yields_identical_payloads(self):
        legacy = observation(self.POSITION, self.OPPONENTS)
        bearing = bearing_observation(self.POSITION, self.OPPONENTS, step=3)
        legacy_frames = self.source.build_frames("blue_0", legacy, step=3)
        report_frames = self.source.build_frames("blue_0", bearing, step=3)
        self.assertEqual(len(legacy_frames), 2)
        self.assertEqual(len(report_frames), 2)
        for legacy_frame, report_frame in zip(legacy_frames, report_frames):
            self.assertTrue(
                torch.equal(legacy_frame.payload, report_frame.payload),
                "{} != {}".format(legacy_frame.payload, report_frame.payload),
            )
            self.assertEqual(legacy_frame.origin, report_frame.origin)
            self.assertEqual(
                legacy_frame.observation_step, report_frame.observation_step
            )

    def test_axis_aligned_geometry_matches_to_float32_precision(self):
        """cos(atan2) vs delta/norm differ by ~1e-17 on axis-aligned bearings."""
        opponents = {"red_0": (1.0, 7.0)}
        legacy = observation(self.POSITION, opponents)
        bearing = bearing_observation(self.POSITION, opponents)
        (legacy_frame,) = self.source.build_frames("blue_0", legacy, step=0)
        (report_frame,) = self.source.build_frames("blue_0", bearing, step=0)
        self.assertTrue(
            torch.allclose(legacy_frame.payload, report_frame.payload, atol=1e-7)
        )

    def test_report_frames_are_anonymous_with_empty_privileged(self):
        bearing = bearing_observation(self.POSITION, self.OPPONENTS)
        for frame in self.source.build_frames("blue_0", bearing, step=0):
            self.assertEqual(dict(frame.privileged), {})

    def test_report_metadata_sets_origin_and_observation_step(self):
        bearing = bearing_observation(
            self.POSITION, self.OPPONENTS, observer="blue_3", step=11
        )
        frames = self.source.build_frames("blue_3", bearing, step=11)
        for frame in frames:
            self.assertEqual(frame.origin, "blue_3")
            self.assertEqual(frame.observation_step, 11)
            self.assertEqual(frame.created_step, 11)

    def test_frame_order_equals_report_order(self):
        """Frames follow the (bearing-angle sorted) report order verbatim."""
        # red_9 is at a smaller bearing angle than red_0, so the sensor emits
        # red_9's report first even though red_0 sorts first by id.
        opponents = {"red_0": (-2.0, 6.0), "red_9": (4.0, 6.0)}
        bearing = bearing_observation(self.POSITION, opponents)
        reports = bearing["contact_reports"]
        frames = self.source.build_frames("blue_0", bearing, step=0)
        self.assertEqual(len(frames), 2)
        for report, frame in zip(reports, frames):
            self.assertTrue(
                torch.equal(
                    frame.payload,
                    torch.tensor(report["payload"], dtype=torch.float32),
                )
            )

    def test_empty_contact_reports_produce_empty_outbox(self):
        bearing = bearing_observation(self.POSITION, {})
        self.assertEqual(self.source.build_frames("blue_0", bearing, step=0), [])

    def test_position_mapping_takes_precedence_over_reports(self):
        """The legacy path is byte-identical: a mapping wins over reports."""
        obs = observation(self.POSITION, self.OPPONENTS)
        obs["contact_reports"] = []
        frames = self.source.build_frames("blue_0", obs, step=0)
        self.assertEqual(len(frames), 2)
        self.assertEqual(
            [f.privileged["opponent_id"] for f in frames], ["red_0", "red_1"]
        )

    def test_privileged_red_agents_never_consulted(self):
        """Privileged observation fields must not become message payloads."""
        bearing = bearing_observation(self.POSITION, {})
        bearing["privileged_red_agents"] = {
            "red_0": {"position": (4.0, 6.0), "distance": 5.0}
        }
        self.assertEqual(self.source.build_frames("blue_0", bearing, step=0), [])

    def test_report_without_payload_rejected(self):
        obs = {"position": (0.0, 0.0), "contact_reports": [{"metadata": {}}]}
        with self.assertRaises(ValueError):
            self.source.build_frames("blue_0", obs, step=0)

    def test_report_with_wrong_payload_dimension_rejected(self):
        obs = {"position": (0.0, 0.0), "contact_reports": [{"payload": [1.0, 2.0]}]}
        with self.assertRaises(ValueError):
            self.source.build_frames("blue_0", obs, step=0)

    def test_report_from_other_observer_rejected(self):
        bearing = bearing_observation(self.POSITION, self.OPPONENTS, observer="blue_9")
        with self.assertRaises(ValueError):
            self.source.build_frames("blue_0", bearing, step=0)

    def test_non_sequence_reports_rejected(self):
        obs = {"position": (0.0, 0.0), "contact_reports": {"payload": [0.0] * 4}}
        with self.assertRaises(ValueError):
            self.source.build_frames("blue_0", obs, step=0)


class TestBearingOnlyCommunicationComposition(unittest.TestCase):
    """Combined mode: bearing_only blue sensor AND one_hop_direct comms."""

    def _make_env(self):
        env_config = {
            "width": 100,
            "height": 80,
            "max_cycles": 10,
            "save_episode_gifs": False,
            "communication_radius": 1000.0,
            "observation": {
                "blue_sensor": "bearing_only",
                "scripted_blue_privileged": False,  # strict: no privileged fields
            },
            "communication": {
                "enabled": True,
                "scheme": "one_hop_direct",
                "rounds_per_step": 1,
                "cache_window": 0,
            },
        }
        agents = create_agents_from_config(
            {
                "blue_agents": [{"count": 2, "detection_radius": 1000.0}],
                "red_agents": [{"count": 1, "detection_radius": 1000.0}],
            },
            env_config=env_config,
        )
        return ParallelGameEnv(agents, **env_config)

    def _assert_no_red_strings(self, value, context):
        if isinstance(value, str):
            self.assertFalse(
                value.startswith("red"),
                "red identity {!r} leaked into policy-visible {}".format(value, context),
            )
        elif isinstance(value, Mapping):
            for key, item in value.items():
                self._assert_no_red_strings(key, context)
                self._assert_no_red_strings(item, context)
        elif isinstance(value, (list, tuple)):
            for item in value:
                self._assert_no_red_strings(item, context)

    def test_blues_deliver_bearing_report_frames(self):
        env = self._make_env()
        env.reset(seed=7)
        actions = {
            name: {"direction": (0.0, 0.0), "speed": 0.0} for name in env.agents
        }
        observations, _, _, _, infos = env.step(actions)
        for name, teammate in (("blue_0", "blue_1"), ("blue_1", "blue_0")):
            # Nonzero deliveries: the teammate saw the red and shared a report.
            self.assertEqual(infos[name]["communication"]["messages_delivered"], 1)
            view = observations[name]["communication"]
            inbox = view["inbox"]
            self.assertEqual(inbox.num_messages, 1)
            self.assertEqual(tuple(inbox.payload.shape), (1, 4))
            # The direction part of the delivered payload is a unit bearing.
            self.assertAlmostEqual(float(inbox.payload[0, 2:].norm()), 1.0, places=6)
            # Senders and origins decode to blue teammates only.
            for index in inbox.sender_index.tolist() + inbox.origin_index.tolist():
                self.assertEqual(view["agent_ids"][index], teammate)
            # No red identity anywhere policy-visible in the blue observation
            # (the communication view is checked via its decoded senders above:
            # its agent_ids decode table is runtime metadata, not features).
            self.assertNotIn("privileged_red_agents", observations[name])
            policy_visible = {
                key: value
                for key, value in observations[name].items()
                if key != "communication"
            }
            self._assert_no_red_strings(policy_visible, "observation of " + name)
        # Ground-truth evaluation joins through infos, never the observation.
        for name in ("blue_0", "blue_1"):
            self.assertIn("ground_truth_contacts", infos[name])
        env.close()


if __name__ == "__main__":
    unittest.main()
