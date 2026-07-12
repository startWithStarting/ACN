"""Tests for the config-driven movement action-space builder and env decoding."""

import os
import sys
import unittest

import numpy as np
from gymnasium.spaces import Box, Dict as DictSpace, Discrete

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.action_spaces import (
    DEFAULT_MAX_SPEED,
    ActionSpaceConfig,
    ActionSpaceConfigError,
    DiscreteMovementSpec,
    build_continuous_movement_space,
    build_discrete_movement_spec,
    build_movement_action_space,
    resolve_spec_max_speed,
)
from src.agents.blue_agent import BlueAgent
from src.agents.factory import create_agents_from_config
from src.agents.red_agent import RedAgent
from src.env.parallel_env import ParallelGameEnv


def legacy_dict_space():
    """The Dict space RedAgent/BlueAgent hardcoded before the builder existed."""
    return DictSpace({
        'direction': Box(low=np.array([-1.0, -1.0], dtype=np.float32),
                         high=np.array([1.0, 1.0], dtype=np.float32),
                         shape=(2,), dtype=np.float32),
        'speed': Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32),
    })


class TestDiscreteMovementSpec(unittest.TestCase):
    def setUp(self):
        self.spec = DiscreteMovementSpec(headings=8, speed_levels=4, max_speed=10.0)

    def test_n_is_25_for_default_shape(self):
        """N = 1 + H*(S-1) = 25 for H=8, S=4."""
        self.assertEqual(self.spec.n, 25)
        self.assertEqual(self.spec.gymnasium_space(), Discrete(25))

    def test_index_zero_is_stay(self):
        direction, speed = self.spec.decode(0)
        self.assertEqual(speed, 0.0)
        np.testing.assert_array_equal(direction, np.zeros(2))
        np.testing.assert_array_equal(self.spec.decode_velocity(0), np.zeros(2))

    def test_headings_are_unit_norm(self):
        for k in range(self.spec.headings):
            norm = float(np.linalg.norm(self.spec.heading_vector(k)))
            self.assertAlmostEqual(norm, 1.0, places=12,
                                   msg="heading {} is not unit norm".format(k))

    def test_first_heading_is_positive_x(self):
        """theta_k = 2*pi*k/H starts at 0, i.e. heading 0 is (1, 0)."""
        np.testing.assert_allclose(self.spec.heading_vector(0), [1.0, 0.0], atol=1e-12)

    def test_isotropy_max_velocity_magnitude_identical_across_headings(self):
        top_level_offset = self.spec.speed_levels - 2  # offset of the max level
        magnitudes = []
        for k in range(self.spec.headings):
            index = 1 + k * (self.spec.speed_levels - 1) + top_level_offset
            magnitudes.append(float(np.linalg.norm(self.spec.decode_velocity(index))))
        self.assertAlmostEqual(max(magnitudes), min(magnitudes), places=9)
        self.assertAlmostEqual(magnitudes[0], self.spec.max_speed, places=9)

    def test_encode_decode_bijection_over_all_indices(self):
        for index in range(self.spec.n):
            direction, speed = self.spec.decode(index)
            self.assertEqual(self.spec.encode(direction, speed), index)

    def test_decode_speed_levels_evenly_spaced(self):
        # Heading 0 indices are 1, 2, 3 -> speeds {1/3, 2/3, 1} * max_speed.
        speeds = [self.spec.decode(i)[1] for i in (1, 2, 3)]
        expected = [10.0 / 3.0, 20.0 / 3.0, 10.0]
        for got, want in zip(speeds, expected):
            self.assertAlmostEqual(got, want, places=12)

    def test_decode_accepts_numpy_integers(self):
        direction, speed = self.spec.decode(np.int64(3))
        self.assertAlmostEqual(speed, 10.0, places=12)
        np.testing.assert_allclose(direction, [1.0, 0.0], atol=1e-12)

    def test_decode_rejects_invalid_indices(self):
        for bad in (-1, 25, 100):
            with self.assertRaises(ValueError):
                self.spec.decode(bad)
        with self.assertRaises(ValueError):
            self.spec.decode(True)
        with self.assertRaises(ValueError):
            self.spec.decode(1.5)

    def test_encode_rejects_off_lattice_values(self):
        with self.assertRaises(ValueError):
            self.spec.encode(np.array([1.0, 0.0]), 4.0)  # not a speed level
        off_heading = np.array([np.cos(np.pi / 8), np.sin(np.pi / 8)])
        with self.assertRaises(ValueError):
            self.spec.encode(off_heading, 10.0)  # between two headings

    def test_invalid_construction_rejected(self):
        with self.assertRaises(ActionSpaceConfigError):
            DiscreteMovementSpec(headings=0, speed_levels=4, max_speed=10.0)
        with self.assertRaises(ActionSpaceConfigError):
            DiscreteMovementSpec(headings=8, speed_levels=1, max_speed=10.0)
        with self.assertRaises(ActionSpaceConfigError):
            DiscreteMovementSpec(headings=8, speed_levels=4, max_speed=0.0)


class TestActionSpaceConfig(unittest.TestCase):
    def test_default_is_continuous(self):
        config = ActionSpaceConfig()
        self.assertEqual(config.type, "continuous")
        self.assertFalse(config.is_discrete)
        self.assertEqual(config.headings, 8)
        self.assertEqual(config.speed_levels, 4)

    def test_from_dict_none_is_continuous(self):
        self.assertEqual(ActionSpaceConfig.from_dict(None), ActionSpaceConfig())

    def test_from_dict_discrete(self):
        config = ActionSpaceConfig.from_dict(
            {"type": "discrete", "headings": 12, "speed_levels": 3}
        )
        self.assertTrue(config.is_discrete)
        self.assertEqual(config.headings, 12)
        self.assertEqual(config.speed_levels, 3)

    def test_invalid_values_rejected(self):
        with self.assertRaises(ActionSpaceConfigError):
            ActionSpaceConfig.from_dict({"type": "hexagonal"})
        with self.assertRaises(ActionSpaceConfigError):
            ActionSpaceConfig.from_dict({"type": "discrete", "headings": 0})
        with self.assertRaises(ActionSpaceConfigError):
            ActionSpaceConfig.from_dict({"type": "discrete", "speed_levels": 1})
        with self.assertRaises(ActionSpaceConfigError):
            ActionSpaceConfig.from_dict("discrete")


class TestBuilders(unittest.TestCase):
    def test_default_continuous_space_matches_legacy(self):
        self.assertEqual(build_movement_action_space(None), legacy_dict_space())
        self.assertEqual(build_continuous_movement_space(), legacy_dict_space())

    def test_discrete_space_size(self):
        config = ActionSpaceConfig.from_dict({"type": "discrete"})
        self.assertEqual(build_movement_action_space(config, 10.0), Discrete(25))

    def test_resolve_spec_max_speed(self):
        self.assertEqual(resolve_spec_max_speed(None), DEFAULT_MAX_SPEED)
        self.assertEqual(resolve_spec_max_speed({}), DEFAULT_MAX_SPEED)
        self.assertEqual(resolve_spec_max_speed({"max_speed": 6.0}), 6.0)
        with self.assertRaises(ActionSpaceConfigError):
            resolve_spec_max_speed({"max_speed": 0.0})
        with self.assertRaises(ActionSpaceConfigError):
            resolve_spec_max_speed({"max_speed": "fast"})


class TestAgentDefaultsUnchanged(unittest.TestCase):
    """Directly constructed agents keep the exact spaces they used to hardcode."""

    def test_red_agent_default_space(self):
        agent = RedAgent("red_0", 10, 10)
        self.assertEqual(agent.action_space, legacy_dict_space())
        self.assertEqual(agent.max_speed, 10.0)

    def test_blue_agent_default_space(self):
        agent = BlueAgent("blue_0", 10, 10)
        self.assertEqual(agent.action_space, legacy_dict_space())
        self.assertEqual(agent.max_speed, 10.0)


def _agents_config(blue_max_speed=None, red_max_speed=None):
    blue = {
        "count": 1,
        "communication_bandwidth": 1,
        "processing_capability": 1,
        "strategy_type": "static",
    }
    red = {
        "count": 1,
        "communication_bandwidth": 1,
        "processing_capability": 1,
        "strategy_type": "center",
    }
    if blue_max_speed is not None:
        blue["max_speed"] = blue_max_speed
    if red_max_speed is not None:
        red["max_speed"] = red_max_speed
    return {"blue_agents": [blue], "red_agents": [red]}


def _env_config(action_space=None):
    config = {
        "width": 200,
        "height": 200,
        "max_cycles": 10,
        "save_episode_gifs": False,
        "physics": {"enabled": False},
    }
    if action_space is not None:
        config["action_space"] = action_space
    return config


class TestFactoryPerTeamMaxSpeed(unittest.TestCase):
    def test_per_team_caps_resolved_and_written_onto_agents(self):
        env_config = _env_config({"type": "discrete"})
        agents = create_agents_from_config(
            _agents_config(blue_max_speed=12.0, red_max_speed=6.0), env_config
        )
        by_team = {getattr(a.agent_type, "value", None): a for a in agents}
        self.assertEqual(by_team["blue"].max_speed, 12.0)
        self.assertEqual(by_team["red"].max_speed, 6.0)
        # Same flat layout for both teams...
        self.assertEqual(by_team["blue"].action_space, Discrete(25))
        self.assertEqual(by_team["red"].action_space, Discrete(25))
        # ...but the same index decodes to team-specific speeds.
        config = ActionSpaceConfig.from_dict({"type": "discrete"})
        blue_spec = build_discrete_movement_spec(config, by_team["blue"].max_speed)
        red_spec = build_discrete_movement_spec(config, by_team["red"].max_speed)
        self.assertAlmostEqual(blue_spec.decode(3)[1], 12.0, places=12)
        self.assertAlmostEqual(red_spec.decode(3)[1], 6.0, places=12)

    def test_default_config_yields_legacy_dict_spaces(self):
        agents = create_agents_from_config(_agents_config(), _env_config())
        for agent in agents:
            self.assertEqual(agent.action_space, legacy_dict_space())
            self.assertEqual(agent.max_speed, DEFAULT_MAX_SPEED)


class TestEnvActionDecoding(unittest.TestCase):
    def _make_env(self, action_space=None, blue_max_speed=None, red_max_speed=None):
        env_config = _env_config(action_space)
        agents = create_agents_from_config(
            _agents_config(blue_max_speed, red_max_speed), env_config
        )
        env = ParallelGameEnv(agents=agents, **env_config)
        env.reset(seed=7)
        # Deterministic starting positions away from the boundary.
        for agent_obj in env.agent_objects.values():
            agent_obj.x = 100.0
            agent_obj.y = 100.0
        return env

    def test_integer_actions_rejected_in_continuous_mode(self):
        env = self._make_env()  # no action_space block -> continuous
        blue_obj = env.agent_objects["blue_0"]
        with self.assertRaises(ValueError):
            env._parse_movement_action(3, blue_obj)
        with self.assertRaises(ValueError):
            env.step({
                "blue_0": 3,
                "red_1": {'direction': (1.0, 0.0), 'speed': 1.0},
            })
        env.close()

    def test_dict_actions_still_parse_in_discrete_mode(self):
        env = self._make_env({"type": "discrete"})
        env.step({
            "blue_0": {'direction': (0.0, 1.0), 'speed': 2.0},
            "red_1": 0,
        })
        self.assertAlmostEqual(env.agent_objects["blue_0"].y, 102.0, places=5)
        self.assertAlmostEqual(env.agent_objects["blue_0"].x, 100.0, places=5)
        env.close()

    def test_env_declares_discrete_spaces_in_discrete_mode(self):
        env = self._make_env({"type": "discrete"})
        self.assertEqual(env.action_space("blue_0"), Discrete(25))
        self.assertEqual(env.action_space("red_1"), Discrete(25))
        env.close()

    def test_step_applies_decoded_velocity_per_team(self):
        env = self._make_env(
            {"type": "discrete"}, blue_max_speed=12.0, red_max_speed=6.0
        )
        # Index 3 = heading 0 (+x) at the top speed level for both teams.
        env.step({"blue_0": 3, "red_1": 3})
        self.assertAlmostEqual(env.agent_objects["blue_0"].x, 112.0, places=5)
        self.assertAlmostEqual(env.agent_objects["red_1"].x, 106.0, places=5)
        self.assertAlmostEqual(env.agent_objects["blue_0"].y, 100.0, places=5)
        self.assertAlmostEqual(env.agent_objects["red_1"].y, 100.0, places=5)
        # Agent kinematic state reflects the decoded command.
        self.assertAlmostEqual(float(env.agent_objects["red_1"].speed), 6.0, places=5)
        np.testing.assert_allclose(
            np.asarray(env.agent_objects["red_1"].direction, dtype=np.float64),
            [1.0, 0.0],
            atol=1e-9,
        )
        env.close()

    def test_stay_action_keeps_position(self):
        env = self._make_env({"type": "discrete"})
        env.step({"blue_0": 0, "red_1": 0})
        self.assertAlmostEqual(env.agent_objects["blue_0"].x, 100.0, places=6)
        self.assertAlmostEqual(env.agent_objects["blue_0"].y, 100.0, places=6)
        env.close()

    def test_heading_two_moves_positive_y(self):
        env = self._make_env({"type": "discrete"}, red_max_speed=6.0)
        # Index 9 = heading 2 (theta = pi/2, +y) at the top speed level.
        env.step({"blue_0": 0, "red_1": 9})
        self.assertAlmostEqual(env.agent_objects["red_1"].x, 100.0, places=5)
        self.assertAlmostEqual(env.agent_objects["red_1"].y, 106.0, places=5)
        env.close()


if __name__ == "__main__":
    unittest.main()
