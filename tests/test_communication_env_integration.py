"""Environment-integration tests for communication Phase 1.

Covers the parallel environment's config-gated communication phase (the R
synchronous rounds run before movement over the frozen pre-move graph, with
deliveries attached to the NEXT decision's observations), per-agent radius
resolution in the agent factory, cache clearing on episode reset, trace
determinism, the untouched disabled path, and the AEC environment reporting
enabled communication as unsupported.
"""

import json
import unittest

from src.agents.factory import (
    DEFAULT_COMMUNICATION_RADIUS,
    create_agents_from_config,
    resolve_spec_communication_radius,
)
from src.communication.tracing import DELIVERY_RECORD_TYPE, GRAPH_RECORD_TYPE
from src.env.aec_env import AECGameEnv
from src.env.parallel_env import ParallelGameEnv
from src.utils.history import to_jsonable

BLUE_NAMES = ("blue_0", "blue_1", "blue_2")
RED_NAMES = ("red_3", "red_4")


def base_env_config(**overrides):
    """Build a small env config; keyword overrides are merged on top."""
    config = {
        "width": 100,
        "height": 80,
        "max_cycles": 20,
        "save_episode_gifs": False,
    }
    config.update(overrides)
    return config


def communication_block(**overrides):
    """Build an enabled one_hop_direct communication block."""
    block = {
        "enabled": True,
        "scheme": "one_hop_direct",
        "rounds_per_step": 1,
        "cache_window": 0,
    }
    block.update(overrides)
    return block


def make_agents(blue_specs, red_specs, env_config, team_defaults=None):
    """Create agents through the factory (the radius-resolution path)."""
    agents_config = {"blue_agents": blue_specs, "red_agents": red_specs}
    if team_defaults is not None:
        agents_config["team_defaults"] = team_defaults
    return create_agents_from_config(agents_config, env_config=env_config)


def stay_actions(env):
    """Deterministic zero-movement actions for every current agent."""
    return {name: {"direction": (0.0, 0.0), "speed": 0.0} for name in env.agents}


def make_comm_env(cache_window=0, communication_radius=1000.0, **env_overrides):
    """Build a 3-blue/2-red parallel env with generous radii and communication on."""
    env_config = base_env_config(
        communication=communication_block(cache_window=cache_window),
        communication_radius=communication_radius,
        **env_overrides,
    )
    agents = make_agents(
        [{"count": 3, "detection_radius": 1000.0}],
        [{"count": 2, "detection_radius": 1000.0}],
        env_config,
    )
    return ParallelGameEnv(agents, **env_config)


class TestOneHopDirectDelivery(unittest.TestCase):
    """Delivered bearing reports reach the next decision, same-team only."""

    def test_reset_carries_explicit_empty_view(self):
        env = make_comm_env()
        observations, infos = env.reset(seed=11)
        for name in BLUE_NAMES + RED_NAMES:
            view = observations[name]["communication"]
            self.assertEqual(view["scheme"], "one_hop_direct")
            self.assertEqual(len(view["inbox"]), 0)
            self.assertEqual(len(view["cache"]), 0)
            self.assertEqual(view["cache_window"], 0)
            self.assertEqual(set(view["agent_ids"]), set(BLUE_NAMES + RED_NAMES))
        env.close()

    def test_blue_views_carry_teammate_bearing_reports(self):
        env = make_comm_env()
        env.reset(seed=11)
        observations, _, _, _, infos = env.step(stay_actions(env))
        for name in BLUE_NAMES:
            view = observations[name]["communication"]
            inbox = view["inbox"]
            # Two teammates each see both reds: 2 x 2 distinct reports arrive.
            self.assertEqual(inbox.num_messages, 4)
            # The engineered bearing payload is exactly 4-dimensional.
            self.assertEqual(tuple(inbox.payload.shape), (4, 4))
            senders = {view["agent_ids"][i] for i in inbox.sender_index.tolist()}
            self.assertEqual(senders, set(BLUE_NAMES) - {name})
            self.assertEqual(infos[name]["communication"]["messages_delivered"], 4)
        env.close()

    def test_red_views_carry_red_team_traffic_only(self):
        env = make_comm_env()
        env.reset(seed=11)
        observations, _, _, _, infos = env.step(stay_actions(env))
        for name in RED_NAMES:
            view = observations[name]["communication"]
            inbox = view["inbox"]
            # The other red sees all three blues: 3 distinct reports arrive.
            self.assertEqual(inbox.num_messages, 3)
            senders = {view["agent_ids"][i] for i in inbox.sender_index.tolist()}
            origins = {view["agent_ids"][i] for i in inbox.origin_index.tolist()}
            self.assertEqual(senders, set(RED_NAMES) - {name})
            self.assertEqual(origins, set(RED_NAMES) - {name})
            self.assertEqual(infos[name]["communication"]["messages_delivered"], 3)
        env.close()

    def test_no_cross_team_delivery(self):
        env = make_comm_env()
        env.reset(seed=11)
        observations, _, _, _, _ = env.step(stay_actions(env))
        for name in BLUE_NAMES + RED_NAMES:
            view = observations[name]["communication"]
            inbox = view["inbox"]
            team = name.split("_")[0]
            for index in inbox.sender_index.tolist() + inbox.origin_index.tolist():
                sender = view["agent_ids"][index]
                self.assertTrue(
                    sender.startswith(team),
                    "{} received cross-team traffic from {}".format(name, sender),
                )
        env.close()


class TestRadiusResolutionPrecedence(unittest.TestCase):
    """Spec value beats team default beats environment fallback."""

    def test_precedence_chain(self):
        env_config = {"width": 100, "height": 80, "communication_radius": 7.5}
        agents = create_agents_from_config(
            {
                "blue_agents": [
                    {"count": 1, "communication_radius": 20.0},  # spec value
                    {"count": 1},  # falls to the blue team default
                ],
                "red_agents": [{"count": 1}],  # falls to the env fallback
                "team_defaults": {"blue": {"communication_radius": 12.0}},
            },
            env_config=env_config,
        )
        by_name = {agent.name: agent for agent in agents}
        self.assertEqual(by_name["blue_0"].communication_radius, 20.0)
        self.assertEqual(by_name["blue_1"].communication_radius, 12.0)
        self.assertEqual(by_name["red_2"].communication_radius, 7.5)

    def test_default_environment_fallback(self):
        agents = create_agents_from_config(
            {"blue_agents": [{"count": 1}], "red_agents": [{"count": 1}]},
            env_config={},
        )
        for agent in agents:
            self.assertEqual(agent.communication_radius, DEFAULT_COMMUNICATION_RADIUS)

    def test_invalid_radius_rejected(self):
        with self.assertRaises(ValueError):
            resolve_spec_communication_radius({"communication_radius": -1.0})
        with self.assertRaises(ValueError):
            resolve_spec_communication_radius({"communication_radius": "wide"})


class TestOutOfRangeDelivery(unittest.TestCase):
    """A teammate outside every sender's radius receives nothing."""

    def test_out_of_range_teammate_receives_nothing(self):
        env_config = base_env_config(
            communication=communication_block(),
            communication_radius=10.0,
            physics={"enabled": False},
        )
        agents = make_agents(
            [{"count": 3, "detection_radius": 1000.0}],
            [{"count": 1, "detection_radius": 1000.0}],
            env_config,
        )
        env = ParallelGameEnv(agents, **env_config)
        env.reset(seed=3)
        positions = {
            "blue_0": (0.0, 0.0),
            "blue_1": (5.0, 0.0),
            "blue_2": (90.0, 70.0),  # far outside every teammate's radius
            "red_3": (2.0, 2.0),
        }
        for name, (x, y) in positions.items():
            env.agent_objects[name].x = x
            env.agent_objects[name].y = y

        observations, _, _, _, infos = env.step(stay_actions(env))

        # blue_2 is isolated: nothing arrives despite its teammates emitting.
        view = observations["blue_2"]["communication"]
        self.assertEqual(view["inbox"].num_messages, 0)
        self.assertEqual(infos["blue_2"]["communication"]["messages_delivered"], 0)

        # blue_0 and blue_1 exchange their single red report; blue_2's report
        # reaches nobody (no edges), so each in-range blue receives exactly one.
        for name, sender in (("blue_0", "blue_1"), ("blue_1", "blue_0")):
            view = observations[name]["communication"]
            self.assertEqual(view["inbox"].num_messages, 1)
            sender_ids = {view["agent_ids"][i] for i in view["inbox"].sender_index.tolist()}
            self.assertEqual(sender_ids, {sender})

        # The single red has no teammate at all.
        self.assertEqual(observations["red_3"]["communication"]["inbox"].num_messages, 0)
        env.close()


class TestDisabledCommunicationPath(unittest.TestCase):
    """Disabled/absent communication leaves observations structurally legacy."""

    LEGACY_BLUE_KEYS = {"position", "grid_center", "timestamp", "red_agents"}
    LEGACY_RED_KEYS = {"position", "grid_center", "timestamp", "blue_agents", "red_teammates"}

    def _assert_legacy_shapes(self, env):
        self.assertIsNone(env.communication_runtime)
        observations, infos = env.reset(seed=5)
        self._assert_no_communication(observations, infos)
        observations, _, _, _, infos = env.step(stay_actions(env))
        self._assert_no_communication(observations, infos)
        env.close()

    def _assert_no_communication(self, observations, infos):
        for name, observation in observations.items():
            expected = (
                self.LEGACY_BLUE_KEYS if name.startswith("blue") else self.LEGACY_RED_KEYS
            )
            self.assertEqual(set(observation.keys()), expected)
        for agent_infos in infos.values():
            self.assertNotIn("communication", agent_infos)

    def test_absent_communication_block(self):
        env_config = base_env_config()
        agents = make_agents([{"count": 2}], [{"count": 1}], env_config)
        self._assert_legacy_shapes(ParallelGameEnv(agents, **env_config))

    def test_disabled_communication_block(self):
        env_config = base_env_config(
            communication={"enabled": False, "scheme": "one_hop_direct"}
        )
        agents = make_agents([{"count": 2}], [{"count": 1}], env_config)
        self._assert_legacy_shapes(ParallelGameEnv(agents, **env_config))

    def test_scheme_none_block(self):
        env_config = base_env_config(communication={"enabled": True, "scheme": "none"})
        agents = make_agents([{"count": 2}], [{"count": 1}], env_config)
        self._assert_legacy_shapes(ParallelGameEnv(agents, **env_config))


class TestEpisodeResetClearsCaches(unittest.TestCase):
    """A message delivered in episode 1 is not visible after reset."""

    def test_reset_clears_caches_and_views(self):
        env = make_comm_env(cache_window=4)
        env.reset(seed=11)
        observations, _, _, _, _ = env.step(stay_actions(env))
        view = observations["blue_0"]["communication"]
        self.assertEqual(view["inbox"].num_messages, 4)
        self.assertEqual(len(view["cache"]), 4)
        self.assertEqual(view["cache_window"], 4)

        observations, _ = env.reset(seed=12)
        view = observations["blue_0"]["communication"]
        self.assertEqual(len(view["inbox"]), 0)
        self.assertEqual(len(view["cache"]), 0)
        for cache in env.communication_runtime.caches.values():
            self.assertEqual(len(cache), 0)
        self.assertEqual(env.last_communication_trace_records, [])

        # A fresh episode delivers again with no leaked state (cache labels
        # restart at step 0 without conflicting with episode 1's rounds).
        observations, _, _, _, _ = env.step(stay_actions(env))
        view = observations["blue_0"]["communication"]
        self.assertEqual(view["inbox"].num_messages, 4)
        self.assertEqual(len(view["cache"]), 4)
        env.close()


class TestTraceRecordingOfCommunicationViews(unittest.TestCase):
    """Observations carrying live communication views stay trace-recordable.

    The recorder embeds full observation dicts in transition JSONL rows;
    the live view objects (EdgeMessageBatch inbox, MessageCache handle)
    must serialize to compact summaries, never crash json.dumps.
    """

    def test_step_observation_with_view_is_json_serializable(self):
        env = make_comm_env(cache_window=2)
        env.reset(seed=11)
        observations, _, _, _, infos = env.step(stay_actions(env))
        row = json.dumps(to_jsonable(observations["blue_0"]))
        summary = json.loads(row)["communication"]
        self.assertEqual(summary["inbox"]["type"], "EdgeMessageBatch")
        self.assertEqual(summary["inbox"]["num_messages"], 4)
        self.assertEqual(summary["inbox"]["payload_dim"], 4)
        self.assertEqual(summary["cache"]["type"], "MessageCache")
        self.assertEqual(summary["cache"]["capacity_rounds"], 2)
        json.dumps(to_jsonable(infos["blue_0"]))  # infos must serialize too
        env.close()


class TestAECCommunicationUnsupported(unittest.TestCase):
    """AEC must not emulate communication rounds via agent iteration order."""

    def test_enabled_communication_raises_at_construction(self):
        env_config = base_env_config(communication=communication_block())
        agents = make_agents([{"count": 1}], [{"count": 1}], env_config)
        with self.assertRaises(NotImplementedError):
            AECGameEnv(agents, **env_config)

    def test_disabled_communication_still_constructs(self):
        env_config = base_env_config(communication={"enabled": False})
        agents = make_agents([{"count": 1}], [{"count": 1}], env_config)
        env = AECGameEnv(agents, **env_config)
        self.assertIsNone(env.communication_runtime)
        env.close()


class TestTraceRecords(unittest.TestCase):
    """Trace records are present and deterministic under a fixed seed."""

    def _run_and_collect(self, seed, steps=3):
        env = make_comm_env()
        env.reset(seed=seed)
        per_step_records = []
        for _ in range(steps):
            env.step(stay_actions(env))
            per_step_records.append(env.last_communication_trace_records)
        env.close()
        return per_step_records

    def test_records_present_with_expected_types(self):
        records = self._run_and_collect(seed=21)[0]
        self.assertTrue(records)
        record_types = {record["record_type"] for record in records}
        self.assertIn(GRAPH_RECORD_TYPE, record_types)
        self.assertIn(DELIVERY_RECORD_TYPE, record_types)
        graph_records = [r for r in records if r["record_type"] == GRAPH_RECORD_TYPE]
        self.assertEqual(len(graph_records), 1)
        self.assertEqual(graph_records[0]["scheme"], "one_hop_direct")

    def test_records_deterministic_across_identically_seeded_runs(self):
        first = self._run_and_collect(seed=21)
        second = self._run_and_collect(seed=21)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
