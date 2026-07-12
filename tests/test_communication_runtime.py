"""Integration tests for the fixed-round communication runtime and one_hop_direct."""

import unittest

import torch

from src.agents.base_agent import AgentType, BaseAgent
from src.communication.plans import CommunicationPlan
from src.communication.processors.identity import DirectProcessor
from src.communication.registry import create_communication_plan, list_communication_schemes
from src.communication.runtime import CachedDelivery, CommunicationRuntime
from src.communication.topology import RadiusTopology
from src.communication.transport import Frame, SlottedRadiusTransport
from tests.communication_fixtures import (
    CHAIN_POSITIONS,
    CHAIN_RADIUS,
    DISCONNECTED_POSITIONS,
    DISCONNECTED_RADIUS,
    EMPTY_POSITIONS,
    EMPTY_RADIUS,
    TWO_TEAM_POSITIONS,
    TWO_TEAM_RADIUS,
    TWO_TEAM_TEAMS,
    chain_graph,
)


def make_agent(name, x, y, team="blue", radius=1.5):
    """Build a real BaseAgent with a resolved communication radius."""
    agent = BaseAgent(
        name=name,
        agent_type=AgentType.BLUE if team == "blue" else AgentType.RED,
        communication_bandwidth=1,
        processing_capability=1,
        x=x,
        y=y,
    )
    agent.communication_radius = radius
    return agent


def agents_from_positions(positions, radius, teams=None):
    """Build agents (in lexicographic id order) from a fixture position table."""
    teams = teams or {}
    return [
        make_agent(name, x, y, team=teams.get(name, "blue"), radius=radius)
        for name, (x, y) in sorted(positions.items())
    ]


def one_hop_direct_plan(cache_window=0, **overrides):
    """Compile a one_hop_direct plan through the public registry."""
    config = {
        "enabled": True,
        "scheme": "one_hop_direct",
        "rounds_per_step": 1,
        "cache_window": cache_window,
    }
    config.update(overrides)
    return create_communication_plan(config)


def source_frame(payload, origin, step=0):
    """Build a round-0 source frame."""
    return Frame(payload=payload, origin=origin, created_step=step)


class EchoProcessor(DirectProcessor):
    """Test double: a naive relay that re-emits everything it received.

    It inherits the inbox-preserving bookkeeping of DirectProcessor and
    overrides only ``emit_round`` to re-emit every frame delivered in the
    just-finished round. Used to verify double buffering: whatever it emits
    in round r must only be transported in round r + 1.
    """

    def __init__(self):
        self._pending = {}

    def process_round(self, state, inbox, graph, round_index, context):
        state = super().process_round(state, inbox, graph, round_index, context)
        self._pending = {}
        if inbox.num_messages > 0:
            receivers = inbox.receiver_index.tolist()
            origins = inbox.origin_index.tolist()
            message_ids = inbox.message_id.tolist()
            created_steps = inbox.metadata["created_step"].tolist()
            created_rounds = inbox.metadata["created_round"].tolist()
            hops = inbox.metadata["hop_count"].tolist()
            for row in range(inbox.num_messages):
                sender_id = graph.agent_ids[receivers[row]]
                self._pending.setdefault(sender_id, []).append(
                    Frame(
                        payload=inbox.payload[row],
                        origin=graph.agent_ids[origins[row]],
                        created_step=int(created_steps[row]),
                        created_round=int(created_rounds[row]),
                        message_id=int(message_ids[row]),
                        hop_count=int(hops[row]),
                    )
                )
        return state

    def emit_round(self, state, graph, round_index, context):
        return self._pending


def echo_plan(rounds, cache_window=0):
    """Build a plan around the EchoProcessor with the real topology/transport."""
    return CommunicationPlan(
        topology=RadiusTopology(),
        transport=SlottedRadiusTransport(),
        processor=EchoProcessor(),
        rounds=rounds,
        cache_window=cache_window,
        graph_update="frozen",
        differentiable=False,
        output_contract="preserved-inbox",
    )


class TestOneHopDirectOnChain(unittest.TestCase):
    """A-B-C chain: one_hop_direct reaches B after one round and never C."""

    def setUp(self):
        self.agents = agents_from_positions(CHAIN_POSITIONS, CHAIN_RADIUS)
        self.plan = one_hop_direct_plan(cache_window=2)
        self.runtime = CommunicationRuntime(self.plan, scheme_name="one_hop_direct")
        self.payload = [0.0, 0.0, 1.0, 0.0]
        self.result = self.runtime.run_step(
            local_observations={},
            outboxes={"A": [source_frame(self.payload, "A")]},
            step=0,
            agents=self.agents,
        )

    def test_direct_neighbour_receives_the_distinct_payload(self):
        inbox = self.result.agent_output["B"]
        self.assertEqual(inbox.num_messages, 1)
        self.assertTrue(torch.equal(inbox.payload[0], torch.tensor(self.payload)))

    def test_non_neighbour_receives_nothing(self):
        self.assertEqual(self.result.agent_output["C"].num_messages, 0)

    def test_sender_receives_nothing_back(self):
        self.assertEqual(self.result.agent_output["A"].num_messages, 0)

    def test_no_second_hop_delivery_occurs(self):
        """Every delivered row is a one-hop copy of A's frame to B."""
        delivered = self.result.delivered_messages
        self.assertEqual(delivered.num_messages, 1)
        self.assertEqual(delivered.metadata["hop_count"].tolist(), [1])

    def test_graph_matches_the_fixture_oracle(self):
        graph = self.result.diagnostics["graph"]
        self.assertTrue(torch.equal(graph.edge_index, chain_graph().edge_index))

    def test_sender_identity_is_available_to_the_receiver(self):
        inbox = self.result.agent_output["B"]
        graph = self.result.diagnostics["graph"]
        self.assertEqual(graph.agent_ids[int(inbox.sender_index[0])], "A")
        self.assertEqual(graph.agent_ids[int(inbox.origin_index[0])], "A")

    def test_extra_rounds_still_never_reach_c_without_relay(self):
        """Even with R=2, the direct processor never re-emits."""
        plan = CommunicationPlan(
            topology=RadiusTopology(),
            transport=SlottedRadiusTransport(),
            processor=DirectProcessor(),
            rounds=2,
            cache_window=0,
            graph_update="frozen",
            differentiable=False,
            output_contract="preserved-inbox",
        )
        runtime = CommunicationRuntime(plan)
        result = runtime.run_step(
            local_observations={},
            outboxes={"A": [source_frame(self.payload, "A")]},
            step=0,
            agents=agents_from_positions(CHAIN_POSITIONS, CHAIN_RADIUS),
        )
        self.assertEqual(result.agent_output["C"].num_messages, 0)
        self.assertEqual(result.delivered_messages.num_messages, 1)


class TestTeamBoundary(unittest.TestCase):
    """Opponents never receive teammate messages."""

    def test_no_opponent_delivery(self):
        agents = agents_from_positions(
            TWO_TEAM_POSITIONS, TWO_TEAM_RADIUS,
            teams={name: ("red" if team == 1 else "blue") for name, team in TWO_TEAM_TEAMS.items()},
        )
        runtime = CommunicationRuntime(one_hop_direct_plan(), scheme_name="one_hop_direct")
        result = runtime.run_step(
            local_observations={},
            outboxes={"blue_0": [source_frame([1.0, 2.0, 3.0, 4.0], "blue_0")]},
            step=0,
            agents=agents,
        )
        self.assertEqual(result.agent_output["blue_1"].num_messages, 1)
        self.assertEqual(result.agent_output["red_0"].num_messages, 0)
        self.assertEqual(result.agent_output["red_1"].num_messages, 0)
        graph = result.diagnostics["graph"]
        receiver_teams = {
            int(graph.team_index[r])
            for r in result.delivered_messages.receiver_index.tolist()
        }
        self.assertEqual(receiver_teams, {0})


class TestDistinctMessagesPreserved(unittest.TestCase):
    """No aggregation: distinct payloads stay distinct rows."""

    def test_two_senders_two_distinct_rows(self):
        agents = agents_from_positions(CHAIN_POSITIONS, CHAIN_RADIUS)
        runtime = CommunicationRuntime(one_hop_direct_plan())
        result = runtime.run_step(
            local_observations={},
            outboxes={
                "A": [source_frame([1.0, 0.0, 0.0, 0.0], "A")],
                "C": [source_frame([0.0, 0.0, 0.0, 2.0], "C")],
            },
            step=0,
            agents=agents,
        )
        inbox = result.agent_output["B"]
        self.assertEqual(inbox.num_messages, 2)
        self.assertTrue(torch.equal(inbox.payload[0], torch.tensor([1.0, 0.0, 0.0, 0.0])))
        self.assertTrue(torch.equal(inbox.payload[1], torch.tensor([0.0, 0.0, 0.0, 2.0])))
        self.assertEqual(len(set(inbox.message_id.tolist())), 2)


class TestAsymmetricRadiusRules(unittest.TestCase):
    """Radius rules flow from config through the compiled plan to delivery."""

    def _run(self, radius_rule):
        agents = [
            make_agent("A", 0.0, 0.0, radius=2.0),
            make_agent("B", 1.0, 0.0, radius=0.5),
        ]
        plan = one_hop_direct_plan(topology={"radius_rule": radius_rule})
        runtime = CommunicationRuntime(plan)
        return runtime.run_step(
            local_observations={},
            outboxes={
                "A": [source_frame([1.0, 0.0, 0.0, 0.0], "A")],
                "B": [source_frame([0.0, 1.0, 0.0, 0.0], "B")],
            },
            step=0,
            agents=agents,
        )

    def test_sender_rule_delivers_only_a_to_b(self):
        result = self._run("sender")
        self.assertEqual(result.agent_output["B"].num_messages, 1)
        self.assertEqual(result.agent_output["A"].num_messages, 0)

    def test_receiver_rule_delivers_only_b_to_a(self):
        result = self._run("receiver")
        self.assertEqual(result.agent_output["A"].num_messages, 1)
        self.assertEqual(result.agent_output["B"].num_messages, 0)

    def test_mutual_rule_delivers_nothing(self):
        result = self._run("mutual")
        self.assertEqual(result.delivered_messages.num_messages, 0)


class TestEmptyAndDisconnectedGraphs(unittest.TestCase):
    """Degenerate topologies are safe, with explicit empty outputs."""

    def test_edgeless_graph_is_safe(self):
        agents = agents_from_positions(EMPTY_POSITIONS, EMPTY_RADIUS)
        runtime = CommunicationRuntime(one_hop_direct_plan(cache_window=1))
        result = runtime.run_step(
            local_observations={},
            outboxes={"A": [source_frame([1.0, 0.0, 0.0, 0.0], "A")]},
            step=0,
            agents=agents,
        )
        self.assertEqual(result.delivered_messages.num_messages, 0)
        for agent_id in ("A", "B", "C"):
            self.assertEqual(result.agent_output[agent_id].num_messages, 0)
            # Empty rounds still advance every agent's cache window.
            self.assertEqual(runtime.cache_for(agent_id).last_completed_round, (0, 0))

    def test_disconnected_components_do_not_leak(self):
        agents = agents_from_positions(DISCONNECTED_POSITIONS, DISCONNECTED_RADIUS)
        runtime = CommunicationRuntime(one_hop_direct_plan())
        result = runtime.run_step(
            local_observations={},
            outboxes={"A": [source_frame([9.0, 0.0, 0.0, 0.0], "A")]},
            step=0,
            agents=agents,
        )
        self.assertEqual(result.agent_output["B"].num_messages, 1)
        self.assertEqual(result.agent_output["C"].num_messages, 0)
        self.assertEqual(result.agent_output["D"].num_messages, 0)

    def test_no_agents_at_all_is_safe(self):
        runtime = CommunicationRuntime(one_hop_direct_plan())
        result = runtime.run_step(
            local_observations={}, outboxes={}, step=0, agents=[]
        )
        self.assertEqual(result.agent_output, {})
        self.assertEqual(result.delivered_messages.num_messages, 0)


class TestDoubleBuffering(unittest.TestCase):
    """A message emitted in round r is not readable (or forwardable) until r+1."""

    def _run(self, rounds):
        runtime = CommunicationRuntime(echo_plan(rounds))
        result = runtime.run_step(
            local_observations={},
            outboxes={"A": [source_frame([5.0], "A")]},
            step=0,
            agents=agents_from_positions(CHAIN_POSITIONS, CHAIN_RADIUS),
        )
        return result

    def test_one_round_cannot_cross_two_hops_even_with_a_relay(self):
        """With R=1 the echo emission for round 1 is never transported."""
        result = self._run(rounds=1)
        self.assertEqual(result.agent_output["C"].num_messages, 0)
        self.assertEqual(result.agent_output["B"].num_messages, 1)
        self.assertEqual(result.agent_output["B"].round_index.tolist(), [0])

    def test_second_round_delivers_the_relayed_copy(self):
        """With R=2 the copy reaches C, and only in round 1."""
        result = self._run(rounds=2)
        inbox_c = result.agent_output["C"]
        self.assertEqual(inbox_c.num_messages, 1)
        self.assertEqual(inbox_c.round_index.tolist(), [1])
        self.assertEqual(inbox_c.metadata["hop_count"].tolist(), [2])
        graph = result.diagnostics["graph"]
        self.assertEqual(graph.agent_ids[int(inbox_c.origin_index[0])], "A")
        self.assertEqual(graph.agent_ids[int(inbox_c.sender_index[0])], "B")

    def test_round_zero_deliveries_are_never_transported_in_round_zero(self):
        """No row delivered in round 0 may have crossed more than one hop."""
        result = self._run(rounds=2)
        delivered = result.delivered_messages
        for round_value, hops in zip(
            delivered.round_index.tolist(), delivered.metadata["hop_count"].tolist()
        ):
            self.assertLessEqual(hops, round_value + 1)


class TestMessageCacheIntegration(unittest.TestCase):
    """Per-agent caches: C-round window, one advance per round, cross-step aging."""

    def test_delivery_lands_in_the_receiver_cache(self):
        runtime = CommunicationRuntime(one_hop_direct_plan(cache_window=2))
        runtime.run_step(
            local_observations={},
            outboxes={"A": [source_frame([1.0, 0.0, 0.0, 0.0], "A", step=0)]},
            step=0,
            agents=agents_from_positions(CHAIN_POSITIONS, CHAIN_RADIUS),
        )
        entries = runtime.cache_for("B").entries()
        self.assertEqual(len(entries), 1)
        cached = entries[0].message
        self.assertIsInstance(cached, CachedDelivery)
        self.assertEqual(cached.origin, "A")
        self.assertEqual(cached.sender, "A")
        self.assertEqual(cached.receiver, "B")
        self.assertEqual(cached.observation_step, 0)
        self.assertEqual(entries[0].age(current_step=3), 3)

    def test_window_spans_steps_and_evicts_oldest_round_first(self):
        runtime = CommunicationRuntime(one_hop_direct_plan(cache_window=1))
        agents = agents_from_positions(CHAIN_POSITIONS, CHAIN_RADIUS)
        runtime.run_step(
            local_observations={},
            outboxes={"A": [source_frame([1.0, 0.0, 0.0, 0.0], "A", step=0)]},
            step=0,
            agents=agents,
        )
        self.assertEqual(len(runtime.cache_for("B")), 1)
        # Step 1 delivers nothing; the 1-round window evicts step 0's entry.
        runtime.run_step(local_observations={}, outboxes={}, step=1, agents=agents)
        self.assertEqual(len(runtime.cache_for("B")), 0)
        self.assertEqual(runtime.cache_for("B").last_completed_round, (1, 0))

    def test_zero_cache_window_retains_nothing(self):
        runtime = CommunicationRuntime(one_hop_direct_plan(cache_window=0))
        runtime.run_step(
            local_observations={},
            outboxes={"A": [source_frame([1.0, 0.0, 0.0, 0.0], "A")]},
            step=0,
            agents=agents_from_positions(CHAIN_POSITIONS, CHAIN_RADIUS),
        )
        self.assertEqual(len(runtime.cache_for("B")), 0)

    def test_reset_clears_caches(self):
        runtime = CommunicationRuntime(one_hop_direct_plan(cache_window=3))
        runtime.run_step(
            local_observations={},
            outboxes={"A": [source_frame([1.0, 0.0, 0.0, 0.0], "A")]},
            step=0,
            agents=agents_from_positions(CHAIN_POSITIONS, CHAIN_RADIUS),
        )
        runtime.reset()
        self.assertEqual(len(runtime.cache_for("B")), 0)
        self.assertIsNone(runtime.cache_for("B").last_completed_round)

    def test_cache_handles_are_exposed_on_the_result(self):
        runtime = CommunicationRuntime(one_hop_direct_plan(cache_window=2))
        result = runtime.run_step(
            local_observations={},
            outboxes={},
            step=0,
            agents=agents_from_positions(CHAIN_POSITIONS, CHAIN_RADIUS),
        )
        caches = result.diagnostics["caches"]
        self.assertEqual(sorted(caches.keys()), ["A", "B", "C"])
        self.assertIs(caches["B"], runtime.cache_for("B"))


class TestDeterministicTraces(unittest.TestCase):
    """Two identical runs produce identical communication traces."""

    def _run_episode(self):
        runtime = CommunicationRuntime(
            one_hop_direct_plan(cache_window=2), scheme_name="one_hop_direct"
        )
        agents = agents_from_positions(CHAIN_POSITIONS, CHAIN_RADIUS)
        records = []
        for step in range(3):
            result = runtime.run_step(
                local_observations={},
                outboxes={
                    "A": [source_frame([1.0, 0.0, 0.5, float(step)], "A", step=step)],
                    "C": [source_frame([2.0, 0.0, 0.5, float(step)], "C", step=step)],
                },
                step=step,
                agents=agents,
            )
            records.extend(result.diagnostics["trace_records"])
        return records

    def test_traces_identical_across_runs(self):
        first, second = self._run_episode(), self._run_episode()
        self.assertEqual(first, second)

    def test_trace_records_carry_the_required_fields(self):
        records = self._run_episode()
        graph_records = [r for r in records if r["record_type"] == "communication_graph"]
        delivery_records = [r for r in records if r["record_type"] == "communication_delivery"]
        self.assertEqual(len(graph_records), 3)
        self.assertGreater(len(delivery_records), 0)
        for record in graph_records:
            self.assertEqual(record["scheme"], "one_hop_direct")
            for edge in record["edges"]:
                self.assertIn("source", edge)
                self.assertIn("receiver", edge)
                self.assertIn("distance", edge)
        for record in delivery_records:
            for key in ("step", "round", "message_id", "origin", "sender", "receiver", "scheme"):
                self.assertIn(key, record)


class TestRegistryIntegration(unittest.TestCase):
    """one_hop_direct compiles through the registry with validation."""

    def test_scheme_is_registered(self):
        self.assertIn("one_hop_direct", list_communication_schemes())

    def test_compiles_with_defaults(self):
        plan = one_hop_direct_plan()
        self.assertEqual(plan.rounds, 1)
        self.assertEqual(plan.output_contract, "preserved-inbox")
        self.assertEqual(plan.graph_update, "frozen")
        self.assertFalse(plan.differentiable)

    def test_none_scheme_still_compiles(self):
        plan = create_communication_plan({"enabled": False})
        self.assertEqual(plan.rounds, 0)
        self.assertEqual(plan.output_contract, "empty")

    def test_multiple_rounds_rejected(self):
        with self.assertRaises(ValueError):
            one_hop_direct_plan(rounds_per_step=2)

    def test_aggregation_rejected(self):
        with self.assertRaises(ValueError):
            one_hop_direct_plan(processor={"aggregation": "mean"})

    def test_relay_settings_rejected(self):
        with self.assertRaises(ValueError):
            one_hop_direct_plan(cache_window=3, processor={"ttl": 2})

    def test_dynamic_graph_rejected(self):
        with self.assertRaises(ValueError):
            one_hop_direct_plan(topology={"freeze_within_step": False})

    def test_wrong_engineered_payload_dimension_rejected(self):
        with self.assertRaises(ValueError):
            one_hop_direct_plan(payload={"type": "engineered_vector", "dimension": 6})


if __name__ == "__main__":
    unittest.main()
