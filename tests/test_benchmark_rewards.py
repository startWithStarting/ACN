"""Tests for the config-gated benchmark reward modes (``environment.reward``).

Hand-computed fixture geometry (100x100 grid, center (50, 50), scoring ring of
radius 50 with a +/-1 tolerance band, blue detection radius 10):

    b0 (45, 50), b1 (55, 50), b2 (50, 45), b3 (50, 80)
    red_0 (50, 50): detected by b0, b1, b2 -> k = 3 (pinned)
    red_1 (20, 20): detected by nobody    -> k = 0
    red_1 moved to (50, 0): on the ring (distance 50) and undetected -> scores

Covers: k_r counts and pinned, streak accumulation with tracked flipping at
streak 3, the shape undercoverage gate (1.0 @ k=3, 0.5 @ k=4, 0 @ k>=5),
coverage normalization by active reds, the score penalty firing only on scoring
steps, telescoping of the red progress term, the tracked penalty mirroring
blue's tracked_r, and legacy mode remaining unchanged.
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import numpy as np

from src.agents.blue_agent import BlueAgent
from src.agents.red_agent import RedAgent
from src.env.aec_env import AECGameEnv
from src.env.common_env_logic import BLUE_AGENT_PASSIVE_REWARD
from src.env.parallel_env import ParallelGameEnv
from src.env.rewards import (
    RewardSettings,
    blue_benchmark_reward,
    build_detection_state,
    parse_reward_settings,
    red_benchmark_reward,
    ring_potential,
    shape_undercoverage_gate,
)


class _StubAgent:
    """Minimal agent exposing the attributes the detection state builder reads."""

    def __init__(self, name, team, x, y, detection_radius=10.0):
        self.name = name
        self.agent_type = SimpleNamespace(value=team)
        self.x = x
        self.y = y
        self.is_active = True
        self.detection_radius = detection_radius

    def is_within_detection_radius(self, pos):
        return math.hypot(self.x - pos[0], self.y - pos[1]) <= self.detection_radius


def _fixture_blues():
    return [
        _StubAgent("blue_0", "blue", 45.0, 50.0),
        _StubAgent("blue_1", "blue", 55.0, 50.0),
        _StubAgent("blue_2", "blue", 50.0, 45.0),
        _StubAgent("blue_3", "blue", 50.0, 80.0),
    ]


def _fixture_reds():
    return [
        _StubAgent("red_0", "red", 50.0, 50.0),
        _StubAgent("red_1", "red", 20.0, 20.0),
    ]


class TestParseRewardSettings(unittest.TestCase):
    def test_absent_block_defaults_to_legacy(self):
        for raw in (None, {}):
            settings = parse_reward_settings(raw)
            self.assertEqual(settings.blue_mode, "legacy")
            self.assertEqual(settings.red_mode, "legacy")
            self.assertFalse(settings.benchmark_enabled)

    def test_default_weights(self):
        settings = parse_reward_settings(
            {"blue": "benchmark", "weights": {"score_penalty": 2.0}}
        )
        self.assertEqual(settings.pin_weight, 0.5)
        self.assertEqual(settings.track_weight, 1.0)
        self.assertEqual(settings.shape_weight, 0.1)
        self.assertEqual(settings.red_score_weight, 1.0)
        self.assertEqual(settings.score_penalty_weight, 2.0)

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            parse_reward_settings({"blue": "dense"})
        with self.assertRaises(ValueError):
            parse_reward_settings({"red": True})

    def test_unknown_keys_rejected(self):
        with self.assertRaises(ValueError):
            parse_reward_settings({"green": "legacy"})
        with self.assertRaises(ValueError):
            parse_reward_settings({"weights": {"pinn": 0.5}})

    def test_non_numeric_weight_rejected(self):
        with self.assertRaises(ValueError):
            parse_reward_settings({"weights": {"pin": "high"}})
        with self.assertRaises(ValueError):
            parse_reward_settings({"weights": {"pin": True}})

    def test_blue_benchmark_requires_score_penalty(self):
        with self.assertRaises(ValueError):
            parse_reward_settings({"blue": "benchmark"})
        parse_reward_settings({"blue": "benchmark", "weights": {"score_penalty": 1.0}})

    def test_red_benchmark_requires_track_and_progress(self):
        with self.assertRaises(ValueError):
            parse_reward_settings({"red": "benchmark", "weights": {"red_track": 0.5}})
        with self.assertRaises(ValueError):
            parse_reward_settings({"red": "benchmark", "weights": {"red_progress": 0.1}})
        parse_reward_settings(
            {"red": "benchmark", "weights": {"red_track": 0.5, "red_progress": 0.1}}
        )


class TestDetectionState(unittest.TestCase):
    def test_detection_matrix_k_and_pinned(self):
        state = build_detection_state(_fixture_blues(), _fixture_reds(), {})
        self.assertEqual(state.blue_names, ("blue_0", "blue_1", "blue_2", "blue_3"))
        self.assertEqual(state.red_names, ("red_0", "red_1"))
        self.assertEqual(state.detections["blue_0"], frozenset({"red_0"}))
        self.assertEqual(state.detections["blue_1"], frozenset({"red_0"}))
        self.assertEqual(state.detections["blue_2"], frozenset({"red_0"}))
        self.assertEqual(state.detections["blue_3"], frozenset())
        self.assertEqual(state.k, {"red_0": 3, "red_1": 0})
        self.assertEqual(state.pinned, {"red_0": True, "red_1": False})

    def test_streak_accumulation_and_tracked_flip_at_three(self):
        blues, reds = _fixture_blues(), _fixture_reds()
        streaks = {}
        expected = [(1, False), (2, False), (3, True), (4, True)]
        for step, (want_streak, want_tracked) in enumerate(expected):
            state = build_detection_state(blues, reds, streaks)
            self.assertEqual(state.streaks["red_0"], want_streak, "step {}".format(step))
            self.assertEqual(state.tracked["red_0"], want_tracked, "step {}".format(step))
            self.assertEqual(state.streaks["red_1"], 0)
            self.assertFalse(state.tracked["red_1"])
            streaks = state.streaks

        # Losing the pin resets the streak (and tracked) immediately.
        state = build_detection_state(blues[:2], reds, streaks)  # only 2 detectors -> k=2
        self.assertEqual(state.k["red_0"], 2)
        self.assertEqual(state.streaks["red_0"], 0)
        self.assertFalse(state.tracked["red_0"])

    def test_previous_streaks_not_mutated(self):
        previous = {"red_0": 2, "red_1": 5}
        snapshot = dict(previous)
        build_detection_state(_fixture_blues(), _fixture_reds(), previous)
        self.assertEqual(previous, snapshot)

    def test_inactive_blue_does_not_detect(self):
        blues = _fixture_blues()
        blues[0].is_active = False
        state = build_detection_state(blues, _fixture_reds(), {})
        self.assertEqual(state.k["red_0"], 2)
        self.assertEqual(state.detections["blue_0"], frozenset())


class TestBlueBenchmarkReward(unittest.TestCase):
    SETTINGS = RewardSettings(
        blue_mode="benchmark",
        pin_weight=0.5,
        track_weight=1.0,
        shape_weight=0.1,
        score_penalty_weight=2.0,
    )

    def test_hand_computed_fixture_first_step(self):
        state = build_detection_state(_fixture_blues(), _fixture_reds(), {})
        # pin_coverage = 1/2, track_coverage = 0, shape_b0 = (1/2)*1.0
        for name in ("blue_0", "blue_1", "blue_2"):
            self.assertAlmostEqual(
                blue_benchmark_reward(name, state, 0.0, self.SETTINGS), 0.30, places=9
            )
        # blue_3 detects nothing: team components only.
        self.assertAlmostEqual(
            blue_benchmark_reward("blue_3", state, 0.0, self.SETTINGS), 0.25, places=9
        )

    def test_hand_computed_fixture_tracked_step(self):
        # Streak 2 carried in -> streak 3 this step -> tracked contributes 1.0 * 1/2.
        state = build_detection_state(_fixture_blues(), _fixture_reds(), {"red_0": 2})
        self.assertAlmostEqual(
            blue_benchmark_reward("blue_0", state, 0.0, self.SETTINGS), 0.80, places=9
        )
        self.assertAlmostEqual(
            blue_benchmark_reward("blue_3", state, 0.0, self.SETTINGS), 0.75, places=9
        )

    def test_shape_gate_values(self):
        self.assertEqual(shape_undercoverage_gate(3), 1.0)
        self.assertEqual(shape_undercoverage_gate(4), 0.5)
        self.assertEqual(shape_undercoverage_gate(5), 0.0)
        self.assertEqual(shape_undercoverage_gate(7), 0.0)

    def test_shape_term_at_k4_and_k5(self):
        settings = RewardSettings(
            blue_mode="benchmark",
            pin_weight=0.0,
            track_weight=0.0,
            shape_weight=1.0,
            score_penalty_weight=0.0,
        )
        blues = _fixture_blues()
        reds = _fixture_reds()
        # Fourth detector at (50, 55): k(red_0) = 4 -> gate 0.5, shape = 0.5/2.
        blues.append(_StubAgent("blue_4", "blue", 50.0, 55.0))
        state = build_detection_state(blues, reds, {})
        self.assertEqual(state.k["red_0"], 4)
        self.assertAlmostEqual(
            blue_benchmark_reward("blue_0", state, 0.0, settings), 0.25, places=9
        )
        # Fifth detector at (47, 47): k(red_0) = 5 -> gate 0, shape = 0.
        blues.append(_StubAgent("blue_5", "blue", 47.0, 47.0))
        state = build_detection_state(blues, reds, {})
        self.assertEqual(state.k["red_0"], 5)
        self.assertAlmostEqual(
            blue_benchmark_reward("blue_0", state, 0.0, settings), 0.0, places=9
        )
        # A blue that does not detect the red earns no shaping either way.
        self.assertAlmostEqual(
            blue_benchmark_reward("blue_3", state, 0.0, settings), 0.0, places=9
        )

    def test_coverage_normalized_by_active_reds(self):
        blues = _fixture_blues()
        # Both reds active: pinned mean = 1/2.
        state_two = build_detection_state(blues, _fixture_reds(), {})
        # Only the pinned red active: pinned mean = 1.
        state_one = build_detection_state(blues, _fixture_reds()[:1], {})
        settings = RewardSettings(
            blue_mode="benchmark",
            pin_weight=1.0,
            track_weight=0.0,
            shape_weight=0.0,
            score_penalty_weight=0.0,
        )
        self.assertAlmostEqual(
            blue_benchmark_reward("blue_3", state_two, 0.0, settings), 0.5, places=9
        )
        self.assertAlmostEqual(
            blue_benchmark_reward("blue_3", state_one, 0.0, settings), 1.0, places=9
        )

    def test_score_penalty_scales_with_fraction(self):
        state = build_detection_state(_fixture_blues(), _fixture_reds(), {})
        base = blue_benchmark_reward("blue_3", state, 0.0, self.SETTINGS)
        penalized = blue_benchmark_reward("blue_3", state, 0.5, self.SETTINGS)
        self.assertAlmostEqual(base - penalized, 2.0 * 0.5, places=9)

    def test_no_active_reds_gives_zero_coverage(self):
        state = build_detection_state(_fixture_blues(), [], {})
        self.assertAlmostEqual(
            blue_benchmark_reward("blue_0", state, 0.0, self.SETTINGS), 0.0, places=9
        )


class TestRedBenchmarkReward(unittest.TestCase):
    SETTINGS = RewardSettings(
        red_mode="benchmark",
        red_score_weight=1.0,
        red_track_weight=0.5,
        red_progress_weight=0.25,
    )

    def test_on_ring_undetected_scores(self):
        state = build_detection_state(_fixture_blues(), _fixture_reds(), {})
        # red_1 is undetected: on-ring pays the score weight.
        self.assertAlmostEqual(
            red_benchmark_reward("red_1", state, True, 0.0, self.SETTINGS), 1.0, places=9
        )
        # red_0 is detected (k=3): on-ring pays nothing.
        self.assertAlmostEqual(
            red_benchmark_reward("red_0", state, True, 0.0, self.SETTINGS), 0.0, places=9
        )

    def test_tracked_penalty_mirrors_blue_tracked(self):
        # Same streak history blue uses: tracked flips at streak 3.
        state = build_detection_state(_fixture_blues(), _fixture_reds(), {"red_0": 2})
        self.assertTrue(state.tracked["red_0"])
        self.assertAlmostEqual(
            red_benchmark_reward("red_0", state, False, 0.0, self.SETTINGS), -0.5, places=9
        )
        untracked = build_detection_state(_fixture_blues(), _fixture_reds(), {})
        self.assertFalse(untracked.tracked["red_0"])
        self.assertAlmostEqual(
            red_benchmark_reward("red_0", untracked, False, 0.0, self.SETTINGS), 0.0, places=9
        )

    def test_progress_term(self):
        state = build_detection_state(_fixture_blues(), _fixture_reds(), {})
        self.assertAlmostEqual(
            red_benchmark_reward("red_1", state, False, 4.0, self.SETTINGS), 1.0, places=9
        )

    def test_ring_potential_values(self):
        center = (50.0, 50.0)
        self.assertAlmostEqual(ring_potential((50.0, 0.0), center, 50.0), 0.0, places=9)
        self.assertAlmostEqual(ring_potential((10.0, 50.0), center, 50.0), -10.0, places=9)
        self.assertAlmostEqual(ring_potential((50.0, 50.0), center, 50.0), -50.0, places=9)

    def test_potential_deltas_telescope(self):
        center = (50.0, 50.0)
        path = [(10.0, 50.0), (15.0, 50.0), (20.0, 50.0), (25.0, 50.0), (30.0, 50.0)]
        potentials = [ring_potential(p, center, 50.0) for p in path]
        deltas = [b - a for a, b in zip(potentials, potentials[1:])]
        self.assertAlmostEqual(sum(deltas), potentials[-1] - potentials[0], places=9)


def _make_blue(name):
    return BlueAgent(
        name=name,
        communication_bandwidth=1,
        processing_capability=3,
        detection_radius=10.0,
        strategy_type="static",
    )


def _make_red(name):
    return RedAgent(
        name=name,
        communication_bandwidth=1,
        processing_capability=1,
        detection_radius=15.0,
        strategy_type="center",
    )


def _make_parallel_env(positions, reward=None):
    """Build a physics-off parallel env and pin agents to the given positions.

    Args:
        positions: Mapping of agent name -> (x, y). Names starting with "blue"
            become BlueAgents, the rest RedAgents.
        reward: Optional ``environment.reward`` block.

    Returns:
        A reset ParallelGameEnv with agents at the requested positions and the
        benchmark per-episode state re-anchored to those positions.
    """
    agents = [
        _make_blue(name) if name.startswith("blue") else _make_red(name)
        for name in positions
    ]
    env = ParallelGameEnv(
        agents=agents,
        render_mode=None,
        width=100,
        height=100,
        max_cycles=500,
        save_episode_gifs=False,
        physics={"enabled": False},
        reward=reward,
    )
    env.reset(seed=7)
    _place(env, positions)
    return env


def _place(env, positions):
    """Teleport agents and re-anchor streaks/potentials to the new positions."""
    for name, (x, y) in positions.items():
        agent_obj = env.agent_objects[name]
        agent_obj.x = float(x)
        agent_obj.y = float(y)
        agent_obj.speed = 0.0
        agent_obj.direction = (0.0, 0.0)
    env._reset_benchmark_reward_state()


FIXTURE_POSITIONS = {
    "blue_0": (45.0, 50.0),
    "blue_1": (55.0, 50.0),
    "blue_2": (50.0, 45.0),
    "blue_3": (50.0, 80.0),
    "red_0": (50.0, 50.0),
    "red_1": (20.0, 20.0),
}


class TestParallelEnvLegacyUnchanged(unittest.TestCase):
    def test_default_config_keeps_legacy_rewards(self):
        env = _make_parallel_env(FIXTURE_POSITIONS)
        self.assertFalse(env.reward_settings.benchmark_enabled)
        _obs, rewards, _term, _trunc, _infos = env.step({})
        for name in ("red_0", "red_1"):
            self.assertEqual(rewards[name], 0)
        for name in ("blue_0", "blue_1", "blue_2", "blue_3"):
            self.assertAlmostEqual(rewards[name], BLUE_AGENT_PASSIVE_REWARD, places=9)

    def test_explicit_legacy_matches_default(self):
        legacy = {"blue": "legacy", "red": "legacy"}
        env_default = _make_parallel_env(FIXTURE_POSITIONS)
        env_legacy = _make_parallel_env(FIXTURE_POSITIONS, reward=legacy)
        _, rewards_default, _, _, _ = env_default.step({})
        _, rewards_legacy, _, _, _ = env_legacy.step({})
        self.assertEqual(rewards_default, rewards_legacy)

    def test_legacy_scoring_red_suppresses_passive_bonus(self):
        positions = dict(FIXTURE_POSITIONS, red_1=(50.0, 0.0))  # on ring, undetected
        env = _make_parallel_env(positions)
        _obs, rewards, _term, _trunc, _infos = env.step({})
        self.assertEqual(rewards["red_1"], 1)
        for name in ("blue_0", "blue_1", "blue_2", "blue_3"):
            self.assertEqual(rewards[name], 0)


class TestParallelEnvBlueBenchmark(unittest.TestCase):
    REWARD = {"blue": "benchmark", "weights": {"score_penalty": 2.0}}

    def test_hand_computed_rewards_and_tracked_flip(self):
        env = _make_parallel_env(FIXTURE_POSITIONS, reward=self.REWARD)
        # Steps 1 and 2: pinned but not yet tracked.
        for _ in range(2):
            _obs, rewards, _term, _trunc, _infos = env.step({})
            for name in ("blue_0", "blue_1", "blue_2"):
                self.assertAlmostEqual(rewards[name], 0.30, places=9)
            self.assertAlmostEqual(rewards["blue_3"], 0.25, places=9)
            for name in ("red_0", "red_1"):
                self.assertEqual(rewards[name], 0)  # red stays legacy
        # Step 3: streak reaches 3 -> tracked coverage kicks in.
        _obs, rewards, _term, _trunc, _infos = env.step({})
        for name in ("blue_0", "blue_1", "blue_2"):
            self.assertAlmostEqual(rewards[name], 0.80, places=9)
        self.assertAlmostEqual(rewards["blue_3"], 0.75, places=9)

    def test_benchmark_replaces_passive_bonus(self):
        # No blue detects anything and no red scores: benchmark blues earn
        # exactly 0.0, not the legacy passive 0.1.
        positions = {
            "blue_0": (90.0, 10.0),
            "blue_1": (90.0, 20.0),
            "blue_2": (90.0, 30.0),
            "blue_3": (90.0, 40.0),
            "red_0": (20.0, 20.0),
            "red_1": (25.0, 25.0),
        }
        env = _make_parallel_env(positions, reward=self.REWARD)
        _obs, rewards, _term, _trunc, _infos = env.step({})
        for name in ("blue_0", "blue_1", "blue_2", "blue_3"):
            self.assertEqual(rewards[name], 0.0)

    def test_score_penalty_only_on_scoring_steps(self):
        positions = dict(FIXTURE_POSITIONS, red_1=(50.0, 0.0))  # scoring step
        env = _make_parallel_env(positions, reward=self.REWARD)
        _obs, rewards, _term, _trunc, _infos = env.step({})
        # red_score_fraction = 1/2, penalty = 2.0 * 0.5 = 1.0 for every blue.
        for name in ("blue_0", "blue_1", "blue_2"):
            self.assertAlmostEqual(rewards[name], 0.30 - 1.0, places=9)
        self.assertAlmostEqual(rewards["blue_3"], 0.25 - 1.0, places=9)
        self.assertEqual(rewards["red_1"], 1)  # legacy red still scores
        # Move the red off the ring: the penalty disappears immediately.
        env.agent_objects["red_1"].x, env.agent_objects["red_1"].y = 20.0, 20.0
        _obs, rewards, _term, _trunc, _infos = env.step({})
        for name in ("blue_0", "blue_1", "blue_2"):
            self.assertAlmostEqual(rewards[name], 0.30, places=9)
        self.assertAlmostEqual(rewards["blue_3"], 0.25, places=9)
        self.assertEqual(rewards["red_1"], 0)


class TestParallelEnvRedBenchmark(unittest.TestCase):
    REWARD = {"red": "benchmark", "weights": {"red_track": 0.5, "red_progress": 0.25}}

    def test_tracked_penalty_flips_at_streak_three(self):
        env = _make_parallel_env(FIXTURE_POSITIONS, reward=self.REWARD)
        expected_red_0 = [0.0, 0.0, -0.5, -0.5]
        for step, expected in enumerate(expected_red_0):
            _obs, rewards, _term, _trunc, _infos = env.step({})
            self.assertAlmostEqual(rewards["red_0"], expected, places=9, msg="step {}".format(step))
            self.assertAlmostEqual(rewards["red_1"], 0.0, places=9)
            # Blue stays legacy: passive bonus still applies (no red scored).
            for name in ("blue_0", "blue_1", "blue_2", "blue_3"):
                self.assertAlmostEqual(rewards[name], BLUE_AGENT_PASSIVE_REWARD, places=9)

    def test_on_ring_undetected_scores_and_suppresses_passive_bonus(self):
        positions = dict(FIXTURE_POSITIONS, red_1=(50.0, 0.0))
        env = _make_parallel_env(positions, reward=self.REWARD)
        _obs, rewards, _term, _trunc, _infos = env.step({})
        self.assertAlmostEqual(rewards["red_1"], 1.0, places=9)
        for name in ("blue_0", "blue_1", "blue_2", "blue_3"):
            self.assertEqual(rewards[name], 0)

    def test_progress_term_telescopes_over_episode(self):
        positions = {
            "blue_0": (90.0, 10.0),
            "blue_1": (90.0, 20.0),
            "blue_2": (90.0, 30.0),
            "blue_3": (90.0, 90.0),
            "red_0": (10.0, 50.0),
            "red_1": (90.0, 60.0),
        }
        env = _make_parallel_env(positions, reward=self.REWARD)
        move_right = {
            "direction": np.array([1.0, 0.0], dtype=np.float32),
            "speed": np.array([5.0], dtype=np.float32),
        }
        center = (50.0, 50.0)
        phi_start = ring_potential(positions["red_0"], center, 50.0)
        total = 0.0
        for step in range(4):
            _obs, rewards, _term, _trunc, _infos = env.step({"red_0": move_right})
            expected_x = 10.0 + 5.0 * (step + 1)
            self.assertAlmostEqual(env.agent_objects["red_0"].x, expected_x, places=6)
            total += rewards["red_0"]
        phi_end = ring_potential(
            (env.agent_objects["red_0"].x, env.agent_objects["red_0"].y), center, 50.0
        )
        # Never on the ring, never tracked: reward is purely the progress term,
        # and the per-step potential differences telescope.
        self.assertAlmostEqual(total, 0.25 * (phi_end - phi_start), places=6)
        self.assertAlmostEqual(total, 0.25 * (-30.0 - (-10.0)), places=6)


class TestAECBenchmarkUnsupported(unittest.TestCase):
    def _agents(self):
        return [_make_blue("blue_0"), _make_red("red_0")]

    def test_benchmark_mode_raises(self):
        with self.assertRaises(NotImplementedError):
            AECGameEnv(
                agents=self._agents(),
                max_cycles=5,
                save_episode_gifs=False,
                reward={"blue": "benchmark", "weights": {"score_penalty": 1.0}},
            )
        with self.assertRaises(NotImplementedError):
            AECGameEnv(
                agents=self._agents(),
                max_cycles=5,
                save_episode_gifs=False,
                reward={"red": "benchmark", "weights": {"red_track": 1.0, "red_progress": 1.0}},
            )

    def test_legacy_mode_constructs(self):
        env = AECGameEnv(
            agents=self._agents(),
            max_cycles=5,
            save_episode_gifs=False,
            reward={"blue": "legacy", "red": "legacy"},
        )
        self.assertFalse(env.reward_settings.benchmark_enabled)


if __name__ == "__main__":
    unittest.main()
