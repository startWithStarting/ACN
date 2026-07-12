"""Tests for the config-gated bearing-only blue sensor.

Covers the Track C observation contract: anonymous bearing-sorted contact
reports, privileged ground truth via infos, scripted-blue privileged access,
strict-mode identity hiding, and byte-identical legacy default behavior.
"""

import json
import math
import unittest

import numpy as np

from src.agents.blue_agent import BlueAgent
from src.agents.red_agent import RedAgent
from src.env.aec_env import AECGameEnv
from src.env.parallel_env import ParallelGameEnv
from src.env.sensors import ObservationConfig, build_contact_reports

GRID_WIDTH = 100
GRID_HEIGHT = 80
DETECTION_RADIUS = 20.0

# Fixed layout: blue_0 at the grid attractor-free spot (50, 40), four reds on
# the axes at distance 10 (all visible), one far red (not visible).
BLUE_POSITION = (50.0, 40.0)
RED_POSITIONS = {
    "red_0": (60.0, 40.0),  # bearing 0
    "red_1": (50.0, 50.0),  # bearing +pi/2
    "red_2": (40.0, 40.0),  # bearing pi
    "red_3": (50.0, 30.0),  # bearing -pi/2
    "red_4": (95.0, 75.0),  # out of detection range
}
VISIBLE_REDS = ("red_0", "red_1", "red_2", "red_3")
# Reports sorted by bearing angle in (-pi, pi].
EXPECTED_ORDER = ("red_3", "red_0", "red_1", "red_2")


def _make_agents():
    """Build one pursuit blue and five center reds with fixed names."""
    blue = BlueAgent(
        name="blue_0",
        communication_bandwidth=10,
        processing_capability=3,
        detection_radius=DETECTION_RADIUS,
        strategy_type="pursuit",
        grid_size=(GRID_WIDTH, GRID_HEIGHT),
    )
    reds = [
        RedAgent(
            name=name,
            communication_bandwidth=10,
            processing_capability=1,
            strategy_type="center",
        )
        for name in sorted(RED_POSITIONS)
    ]
    return [blue] + reds


def _make_env(observation=None, env_class=ParallelGameEnv):
    """Build a small headless kinematic env, optionally with an observation block."""
    env_config = {
        "width": GRID_WIDTH,
        "height": GRID_HEIGHT,
        "max_cycles": 10,
        "save_episode_gifs": False,
        "physics": {"enabled": False},
    }
    if observation is not None:
        env_config["observation"] = observation
    return env_class(agents=_make_agents(), render_mode=None, **env_config)


def _place_fixed_layout(env):
    """Pin every agent to the fixed test layout."""
    blue = env.agent_objects["blue_0"]
    blue.x, blue.y = BLUE_POSITION
    for name, (x, y) in RED_POSITIONS.items():
        env.agent_objects[name].x = x
        env.agent_objects[name].y = y


def _stationary_actions(env):
    """Zero-movement actions for every live agent."""
    return {
        name: {
            "direction": np.array([0.0, 0.0], dtype=np.float32),
            "speed": np.array([0.0], dtype=np.float32),
        }
        for name in env.agents
    }


def _canonical(value):
    """Normalize an observation tree into JSON-serializable plain Python."""
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _serialize(value):
    """Canonical byte-comparable serialization of an observation tree."""
    return json.dumps(_canonical(value), sort_keys=True)


def _collect_strings(value, found):
    """Recursively collect every string in keys and values of a tree."""
    if isinstance(value, str):
        found.append(value)
    elif isinstance(value, dict):
        for key, item in value.items():
            _collect_strings(key, found)
            _collect_strings(item, found)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_strings(item, found)


class TestObservationConfig(unittest.TestCase):
    """Parsing and validation of the environment.observation block."""

    def test_absent_block_is_legacy_default(self):
        config = ObservationConfig.from_dict(None)
        self.assertEqual(config.blue_sensor, "legacy")
        self.assertTrue(config.scripted_blue_privileged)
        self.assertFalse(config.bearing_only)

    def test_bearing_only_parses(self):
        config = ObservationConfig.from_dict({"blue_sensor": "bearing_only"})
        self.assertTrue(config.bearing_only)
        self.assertTrue(config.scripted_blue_privileged)

    def test_strict_mode_parses(self):
        config = ObservationConfig.from_dict(
            {"blue_sensor": "bearing_only", "scripted_blue_privileged": False}
        )
        self.assertTrue(config.bearing_only)
        self.assertFalse(config.scripted_blue_privileged)

    def test_invalid_sensor_rejected(self):
        with self.assertRaises(ValueError):
            ObservationConfig.from_dict({"blue_sensor": "xray"})

    def test_non_mapping_rejected(self):
        with self.assertRaises(ValueError):
            ObservationConfig.from_dict("bearing_only")

    def test_non_bool_privileged_rejected(self):
        with self.assertRaises(ValueError):
            ObservationConfig.from_dict(
                {"blue_sensor": "bearing_only", "scripted_blue_privileged": "yes"}
            )


class TestContactReports(unittest.TestCase):
    """Bearing-only observations in the parallel environment."""

    def setUp(self):
        self.env = _make_env(observation={"blue_sensor": "bearing_only"})
        self.env.reset(seed=7)
        _place_fixed_layout(self.env)
        self.observation = self.env._get_observation("blue_0", self.env.steps)

    def test_report_count_matches_visible_reds(self):
        reports = self.observation["contact_reports"]
        self.assertEqual(len(reports), len(VISIBLE_REDS))

    def test_red_agents_dict_replaced(self):
        self.assertNotIn("red_agents", self.observation)
        self.assertIn("contact_reports", self.observation)

    def test_base_fields_unchanged(self):
        np.testing.assert_allclose(self.observation["position"], BLUE_POSITION)
        np.testing.assert_allclose(
            self.observation["grid_center"], (GRID_WIDTH / 2, GRID_HEIGHT / 2)
        )
        self.assertEqual(self.observation["timestamp"], self.env.steps)

    def test_payload_is_four_floats_with_unit_direction(self):
        for report in self.observation["contact_reports"]:
            payload = report["payload"]
            self.assertIsInstance(payload, list)
            self.assertEqual(len(payload), 4)
            for value in payload:
                self.assertIsInstance(value, float)
            observer_x, observer_y, direction_x, direction_y = payload
            self.assertEqual((observer_x, observer_y), BLUE_POSITION)
            self.assertAlmostEqual(
                direction_x**2 + direction_y**2, 1.0, delta=1e-6
            )

    def test_metadata_has_observer_and_step_only(self):
        for report in self.observation["contact_reports"]:
            self.assertEqual(
                report["metadata"], {"observer": "blue_0", "step": self.env.steps}
            )
            self.assertEqual(set(report.keys()), {"payload", "metadata"})

    def test_reports_sorted_by_bearing_angle(self):
        angles = [
            math.atan2(report["payload"][3], report["payload"][2])
            for report in self.observation["contact_reports"]
        ]
        self.assertEqual(angles, sorted(angles))

    def test_bearings_match_hand_computed_atan2(self):
        reports = self.observation["contact_reports"]
        for index, red_name in enumerate(EXPECTED_ORDER):
            red_x, red_y = RED_POSITIONS[red_name]
            expected_angle = math.atan2(red_y - BLUE_POSITION[1], red_x - BLUE_POSITION[0])
            self.assertAlmostEqual(
                reports[index]["payload"][2], math.cos(expected_angle), places=12
            )
            self.assertAlmostEqual(
                reports[index]["payload"][3], math.sin(expected_angle), places=12
            )

    def test_privileged_dict_present_by_default_and_matches_legacy(self):
        privileged = self.observation["privileged_red_agents"]
        self.assertEqual(set(privileged.keys()), set(VISIBLE_REDS))
        for red_name in VISIBLE_REDS:
            self.assertEqual(privileged[red_name]["position"], RED_POSITIONS[red_name])
            self.assertAlmostEqual(privileged[red_name]["distance"], 10.0, places=12)

    def test_red_observations_unchanged(self):
        red_observation = self.env._get_observation("red_0", self.env.steps)
        self.assertIn("blue_agents", red_observation)
        self.assertIn("red_teammates", red_observation)
        self.assertNotIn("contact_reports", red_observation)
        self.assertNotIn("privileged_red_agents", red_observation)
        self.assertEqual(
            red_observation["blue_agents"]["blue_0"]["position"], BLUE_POSITION
        )


class TestGroundTruthContacts(unittest.TestCase):
    """Privileged report-index -> red-name mapping through infos."""

    def test_step_infos_align_with_report_order(self):
        env = _make_env(observation={"blue_sensor": "bearing_only"})
        env.reset(seed=7)
        _place_fixed_layout(env)
        observations, _, _, _, infos = env.step(_stationary_actions(env))

        ground_truth = infos["blue_0"]["ground_truth_contacts"]
        self.assertEqual(ground_truth, dict(enumerate(EXPECTED_ORDER)))

        reports = observations["blue_0"]["contact_reports"]
        self.assertEqual(len(reports), len(ground_truth))
        for index, red_name in ground_truth.items():
            red_x, red_y = RED_POSITIONS[red_name]
            expected_angle = math.atan2(red_y - BLUE_POSITION[1], red_x - BLUE_POSITION[0])
            self.assertAlmostEqual(
                reports[index]["payload"][2], math.cos(expected_angle), places=12
            )
            self.assertAlmostEqual(
                reports[index]["payload"][3], math.sin(expected_angle), places=12
            )

    def test_reset_infos_carry_ground_truth(self):
        env = _make_env(observation={"blue_sensor": "bearing_only"})
        observations, infos = env.reset(seed=7)
        self.assertIn("ground_truth_contacts", infos["blue_0"])
        self.assertEqual(
            len(infos["blue_0"]["ground_truth_contacts"]),
            len(observations["blue_0"]["contact_reports"]),
        )
        # Red agents get no ground-truth mapping.
        self.assertNotIn("ground_truth_contacts", infos["red_0"])

    def test_legacy_mode_adds_nothing_to_infos(self):
        env = _make_env()
        _, infos = env.reset(seed=7)
        for agent_infos in infos.values():
            self.assertEqual(agent_infos, {})

    def test_aec_observe_populates_infos(self):
        env = _make_env(observation={"blue_sensor": "bearing_only"}, env_class=AECGameEnv)
        env.reset(seed=7)
        _place_fixed_layout(env)
        observation = env.observe("blue_0")
        ground_truth = env.infos["blue_0"]["ground_truth_contacts"]
        self.assertEqual(ground_truth, dict(enumerate(EXPECTED_ORDER)))
        self.assertEqual(
            len(observation["contact_reports"]), len(ground_truth)
        )


class TestStrictModeIdentityBoundary(unittest.TestCase):
    """No red identity in any policy-visible field when privileged access is off."""

    def setUp(self):
        self.env = _make_env(
            observation={"blue_sensor": "bearing_only", "scripted_blue_privileged": False}
        )
        self.env.reset(seed=7)
        _place_fixed_layout(self.env)

    def test_no_privileged_dict_and_no_red_names(self):
        observation = self.env._get_observation("blue_0", self.env.steps)
        self.assertNotIn("red_agents", observation)
        self.assertNotIn("privileged_red_agents", observation)

        strings = []
        _collect_strings(_canonical(observation), strings)
        for red_name in RED_POSITIONS:
            self.assertNotIn(red_name, strings)
        # Positive control: teammate identity (the observer) is policy-visible.
        self.assertIn("blue_0", strings)

    def test_scripted_blue_does_not_crash_without_privileged_access(self):
        observation = self.env._get_observation("blue_0", self.env.steps)
        blue = self.env.agent_objects["blue_0"]
        action = blue.choose_action(observation)
        self.assertIn("direction", action)
        self.assertIn("speed", action)
        # Nothing was recorded: no red positions were policy-visible.
        self.assertEqual(dict(blue.observed_red_agents), {})


class TestScriptedBluePrivilegedAccess(unittest.TestCase):
    """Scripted VAR blues consume the privileged channel when granted."""

    def test_choose_action_records_privileged_positions(self):
        env = _make_env(observation={"blue_sensor": "bearing_only"})
        env.reset(seed=7)
        _place_fixed_layout(env)
        observation = env._get_observation("blue_0", env.steps)
        blue = env.agent_objects["blue_0"]
        action = blue.choose_action(observation)
        self.assertIn("direction", action)
        self.assertEqual(set(blue.observed_red_agents.keys()), set(VISIBLE_REDS))

    def test_legacy_observation_path_unaffected(self):
        env = _make_env()
        env.reset(seed=7)
        _place_fixed_layout(env)
        observation = env._get_observation("blue_0", env.steps)
        blue = env.agent_objects["blue_0"]
        blue.choose_action(observation)
        self.assertEqual(set(blue.observed_red_agents.keys()), set(VISIBLE_REDS))


class TestLegacyModeUnchanged(unittest.TestCase):
    """Explicit legacy mode is byte-identical to an absent observation block."""

    def _run_and_serialize(self, observation):
        env = _make_env(observation=observation)
        observations, infos = env.reset(seed=123)
        trace = [_serialize(observations), _serialize(infos)]
        for _ in range(3):
            actions = {
                name: {
                    "direction": np.array([0.6, -0.8], dtype=np.float32),
                    "speed": np.array([1.5], dtype=np.float32),
                }
                for name in env.agents
            }
            observations, rewards, terminations, truncations, infos = env.step(actions)
            trace.append(_serialize(observations))
            trace.append(_serialize(rewards))
            trace.append(_serialize(infos))
        env.close()
        return trace

    def test_byte_identical_observations(self):
        baseline = self._run_and_serialize(observation=None)
        explicit_legacy = self._run_and_serialize(observation={"blue_sensor": "legacy"})
        self.assertEqual(baseline, explicit_legacy)

    def test_legacy_observation_has_red_agents_dict(self):
        env = _make_env(observation={"blue_sensor": "legacy"})
        env.reset(seed=7)
        _place_fixed_layout(env)
        observation = env._get_observation("blue_0", env.steps)
        self.assertIn("red_agents", observation)
        self.assertNotIn("contact_reports", observation)
        self.assertNotIn("privileged_red_agents", observation)


class TestBuildContactReportsFunction(unittest.TestCase):
    """Direct unit tests of the report builder."""

    def test_empty_visible_set(self):
        reports, ground_truth = build_contact_reports("blue_0", (1.0, 2.0), [], 5)
        self.assertEqual(reports, [])
        self.assertEqual(ground_truth, {})

    def test_hand_computed_directions_and_order(self):
        observer = (10.0, 10.0)
        visible = [
            ("red_a", (13.0, 14.0)),  # atan2(4, 3) ~ 0.9273
            ("red_b", (7.0, 6.0)),    # atan2(-4, -3) ~ -2.2143
            ("red_c", (10.0, 25.0)),  # atan2(15, 0) = pi/2
        ]
        reports, ground_truth = build_contact_reports("blue_0", observer, visible, 3)
        self.assertEqual(ground_truth, {0: "red_b", 1: "red_a", 2: "red_c"})
        self.assertAlmostEqual(reports[1]["payload"][2], 0.6, places=12)
        self.assertAlmostEqual(reports[1]["payload"][3], 0.8, places=12)
        self.assertAlmostEqual(reports[2]["payload"][2], 0.0, places=12)
        self.assertAlmostEqual(reports[2]["payload"][3], 1.0, places=12)
        expected_b = math.atan2(-4.0, -3.0)
        self.assertAlmostEqual(reports[0]["payload"][2], math.cos(expected_b), places=12)
        self.assertAlmostEqual(reports[0]["payload"][3], math.sin(expected_b), places=12)
        for report in reports:
            self.assertEqual(report["payload"][0], observer[0])
            self.assertEqual(report["payload"][1], observer[1])
            self.assertEqual(report["metadata"], {"observer": "blue_0", "step": 3})

    def test_coincident_target_uses_atan2_zero_convention(self):
        reports, _ = build_contact_reports("blue_0", (5.0, 5.0), [("red_0", (5.0, 5.0))], 0)
        self.assertEqual(reports[0]["payload"][2:], [1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
