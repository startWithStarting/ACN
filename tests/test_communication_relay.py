"""Tests for the Phase 3 multi-hop unchanged relay (``multihop_relay``).

Covers the acceptance criteria of docs/communication_implementation_plan.md
("Phase 3: Multi-Hop Unchanged Relay") against the live runtime: one hop per
round, duplicate suppression under cycles, TTL exhaustion, previous-hop
exclusion, origin vs immediate sender, byte-identical payloads end to end,
cross-step continuation when ``TTL > R``, agents leaving the graph while a
carryover is pending, the ``C == TTL`` window boundary, input-order
determinism, scheme compilation, and env integration.
"""

import json
import unittest

import torch

from src.agents.base_agent import AgentType, BaseAgent
from src.agents.factory import create_agents_from_config
from src.communication.registry import create_communication_plan, list_communication_schemes
from src.communication.runtime import CommunicationRuntime
from src.communication.tracing import RELAY_RECORD_TYPE, relay_decision_record
from src.communication.transport import Frame
from src.env.parallel_env import ParallelGameEnv
from src.utils.history import to_jsonable
from tests.communication_fixtures import CHAIN_POSITIONS, CHAIN_RADIUS

# A - B - C - D on one line, 1.0 apart; radius 1.5 covers adjacent pairs only.
LINE4_POSITIONS = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (2.0, 0.0), "D": (3.0, 0.0)}
LINE4_RADIUS = 1.5

# Equilateral-ish triangle, all pairs 1.0 apart: a fully connected cycle.
TRIANGLE_POSITIONS = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (0.5, 0.866)}
TRIANGLE_RADIUS = 1.5

# Isolated bidirectional pair.
PAIR_POSITIONS = {"A": (0.0, 0.0), "B": (1.0, 0.0)}
PAIR_RADIUS = 1.5

# 4-cycle A - P - Q - X - A (unit square): diagonals (1.414) exceed the
# radius, so each agent is adjacent to exactly its two ring neighbours.
SQUARE_POSITIONS = {"A": (0.0, 0.0), "P": (0.0, 1.0), "Q": (1.0, 1.0), "X": (1.0, 0.0)}
SQUARE_RADIUS = 1.2

PAYLOAD = [1.0, 2.0, 3.0, 4.0]


def make_agent(name, x, y, radius):
    """Build a real BaseAgent with a resolved communication radius."""
    agent = BaseAgent(
        name=name,
        agent_type=AgentType.BLUE,
        communication_bandwidth=1,
        processing_capability=1,
        x=x,
        y=y,
    )
    agent.communication_radius = radius
    return agent


def agents_from_positions(positions, radius, order=None):
    """Build agents from a position table, in ``order`` (default lexicographic)."""
    names = order if order is not None else sorted(positions)
    return [make_agent(name, positions[name][0], positions[name][1], radius) for name in names]


def relay_plan(rounds, ttl, cache_window, **overrides):
    """Compile a multihop_relay plan through the public registry."""
    config = {
        "enabled": True,
        "scheme": "multihop_relay",
        "rounds_per_step": rounds,
        "cache_window": cache_window,
        "processor": {"ttl": ttl},
    }
    config.update(overrides)
    return create_communication_plan(config)


def relay_runtime(rounds, ttl, cache_window):
    """Build a runtime around a freshly compiled relay plan."""
    plan = relay_plan(rounds, ttl, cache_window)
    return CommunicationRuntime(plan, scheme_name="multihop_relay")


def source_frame(payload, origin, step=0):
    """Build a round-0 source frame."""
    return Frame(payload=payload, origin=origin, created_step=step)


def relay_records(result, decision=None):
    """Extract the relay decision records of one step, optionally by decision."""
    records = [
        record
        for record in result.diagnostics["trace_records"]
        if record["record_type"] == RELAY_RECORD_TYPE
    ]
    if decision is not None:
        records = [record for record in records if record["decision"] == decision]
    return records


def decode_inbox(result, agent_id):
    """Return one agent's app inbox as (origin, sender, round, hop, payload) tuples."""
    graph = result.diagnostics["graph"]
    inbox = result.agent_output[agent_id]
    rows = []
    for row in range(inbox.num_messages):
        rows.append(
            (
                graph.agent_ids[int(inbox.origin_index[row])],
                graph.agent_ids[int(inbox.sender_index[row])],
                int(inbox.round_index[row]),
                int(inbox.metadata["hop_count"][row]),
                tuple(inbox.payload[row].tolist()),
            )
        )
    return rows


class TestChainRelay(unittest.TestCase):
    """A-B-C chain with R=2, TTL=2: two hops in two rounds, never fewer."""

    def setUp(self):
        self.runtime = relay_runtime(rounds=2, ttl=2, cache_window=2)
        self.agents = agents_from_positions(CHAIN_POSITIONS, CHAIN_RADIUS)
        self.result = self.runtime.run_step(
            local_observations={},
            outboxes={"A": [source_frame(PAYLOAD, "A")]},
            step=0,
            agents=self.agents,
        )

    def test_c_receives_in_the_second_round_and_never_the_first(self):
        inbox_c = self.result.agent_output["C"]
        self.assertEqual(inbox_c.num_messages, 1)
        self.assertEqual(inbox_c.round_index.tolist(), [1])

    def test_b_receives_in_the_first_round(self):
        inbox_b = self.result.agent_output["B"]
        self.assertEqual(inbox_b.num_messages, 1)
        self.assertEqual(inbox_b.round_index.tolist(), [0])

    def test_packets_travel_one_graph_edge_per_round(self):
        """Every transported copy has crossed exactly round + 1 hops."""
        delivered = self.result.delivered_messages
        self.assertGreater(delivered.num_messages, 0)
        for round_value, hops in zip(
            delivered.round_index.tolist(), delivered.metadata["hop_count"].tolist()
        ):
            self.assertEqual(hops, round_value + 1)

    def test_origin_and_immediate_sender_differ_on_the_second_hop(self):
        self.assertEqual(decode_inbox(self.result, "C"), [("A", "B", 1, 2, tuple(PAYLOAD))])

    def test_payload_is_byte_identical_end_to_end(self):
        expected = torch.tensor(PAYLOAD)
        self.assertTrue(torch.equal(self.result.agent_output["B"].payload[0], expected))
        self.assertTrue(torch.equal(self.result.agent_output["C"].payload[0], expected))

    def test_message_identity_survives_the_hop(self):
        id_at_b = int(self.result.agent_output["B"].message_id[0])
        id_at_c = int(self.result.agent_output["C"].message_id[0])
        self.assertEqual(id_at_b, id_at_c)

    def test_origin_never_app_receives_its_own_message(self):
        self.assertEqual(self.result.agent_output["A"].num_messages, 0)
        echoes = [
            record
            for record in relay_records(self.result, "dropped_duplicate")
            if record["receiver"] == "A"
        ]
        self.assertEqual(len(echoes), 1, "B's broadcast echo to A must be a traced duplicate")

    def test_forwarded_record_carries_previous_hop(self):
        forwarded = relay_records(self.result, "forwarded")
        self.assertEqual(len(forwarded), 1)
        record = forwarded[0]
        self.assertEqual(record["sender"], "B")
        self.assertEqual(record["receiver"], "C")
        self.assertEqual(record["previous_hop"], "A")
        self.assertEqual(record["round"], 1)
        self.assertEqual(record["scheme"], "multihop_relay")

    def test_exhausted_copy_is_delivered_but_never_re_emitted(self):
        dropped = relay_records(self.result, "dropped_ttl")
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0]["receiver"], "C")
        self.assertEqual(dropped[0]["ttl"], 0)
        # C's copy was still app-delivered (TTL governs re-emission only).
        self.assertEqual(self.result.agent_output["C"].num_messages, 1)


class TestTriangleCycle(unittest.TestCase):
    """Fully connected A-B-C cycle: no unbounded duplicates under generous TTL."""

    def setUp(self):
        self.runtime = relay_runtime(rounds=3, ttl=4, cache_window=4)
        self.agents = agents_from_positions(TRIANGLE_POSITIONS, TRIANGLE_RADIUS)
        self.payloads = {
            "A": [1.0, 0.0, 0.0, 0.0],
            "B": [0.0, 2.0, 0.0, 0.0],
            "C": [0.0, 0.0, 3.0, 0.0],
        }
        self.result = self.runtime.run_step(
            local_observations={},
            outboxes={
                name: [source_frame(payload, name)] for name, payload in self.payloads.items()
            },
            step=0,
            agents=self.agents,
        )

    def test_every_agent_receives_each_other_origin_exactly_once(self):
        for receiver in ("A", "B", "C"):
            origins = [row[0] for row in decode_inbox(self.result, receiver)]
            self.assertEqual(
                sorted(origins),
                sorted(set(self.payloads) - {receiver}),
                "receiver {} must get each other origin exactly once".format(receiver),
            )

    def test_no_agent_receives_its_own_message_back(self):
        for receiver in ("A", "B", "C"):
            for origin, _, _, _, _ in decode_inbox(self.result, receiver):
                self.assertNotEqual(origin, receiver)

    def test_cycle_duplicates_are_dropped_with_a_traced_reason(self):
        self.assertGreater(len(relay_records(self.result, "dropped_duplicate")), 0)

    def test_payloads_stay_distinct_and_identical_to_the_source(self):
        for receiver in ("A", "B", "C"):
            for origin, _, _, _, payload in decode_inbox(self.result, receiver):
                self.assertEqual(payload, tuple(self.payloads[origin]))


class TestTTLExhaustion(unittest.TestCase):
    """TTL=1 delivers one hop and never forwards."""

    def test_second_hop_never_happens_and_the_drop_is_traced(self):
        runtime = relay_runtime(rounds=2, ttl=1, cache_window=1)
        result = runtime.run_step(
            local_observations={},
            outboxes={"A": [source_frame(PAYLOAD, "A")]},
            step=0,
            agents=agents_from_positions(CHAIN_POSITIONS, CHAIN_RADIUS),
        )
        self.assertEqual(result.agent_output["B"].num_messages, 1)
        self.assertEqual(result.agent_output["C"].num_messages, 0)
        self.assertEqual(result.delivered_messages.num_messages, 1)
        self.assertEqual(relay_records(result, "forwarded"), [])
        dropped = relay_records(result, "dropped_ttl")
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0]["receiver"], "B")
        self.assertEqual(dropped[0]["ttl"], 0)


class TestPreviousHopExclusion(unittest.TestCase):
    """A<->B bidirectional pair: the previous hop is never a forward target."""

    def test_no_ping_pong_transmission_ever_occurs(self):
        runtime = relay_runtime(rounds=3, ttl=3, cache_window=3)
        agents = agents_from_positions(PAIR_POSITIONS, PAIR_RADIUS)
        result = runtime.run_step(
            local_observations={},
            outboxes={"A": [source_frame(PAYLOAD, "A")]},
            step=0,
            agents=agents,
        )
        # One physical transmission in total: A -> B in round 0. B's only
        # out-neighbour is its previous hop, so B never emits at all.
        self.assertEqual(result.delivered_messages.num_messages, 1)
        self.assertEqual(result.agent_output["B"].num_messages, 1)
        self.assertEqual(result.agent_output["A"].num_messages, 0)
        self.assertEqual(relay_records(result, "forwarded"), [])
        # And nothing lingers into the next step.
        follow_up = runtime.run_step(
            local_observations={}, outboxes={}, step=1, agents=agents
        )
        self.assertEqual(follow_up.delivered_messages.num_messages, 0)

    def test_the_no_target_drop_is_traced(self):
        runtime = relay_runtime(rounds=2, ttl=2, cache_window=2)
        result = runtime.run_step(
            local_observations={},
            outboxes={"A": [source_frame(PAYLOAD, "A")]},
            step=0,
            agents=agents_from_positions(PAIR_POSITIONS, PAIR_RADIUS),
        )
        dropped = relay_records(result, "dropped_no_target")
        self.assertEqual(len(dropped), 1)
        record = dropped[0]
        self.assertEqual(record["sender"], "B")
        self.assertIsNone(record["receiver"])
        self.assertEqual(record["previous_hop"], "A")
        self.assertEqual(record["origin"], "A")
        self.assertEqual(record["round"], 1)
        self.assertEqual(record["ttl"], 1)  # The budget the frame still held.


class TestCrossStepContinuation(unittest.TestCase):
    """TTL > R: the relay continues over the movement-step boundary."""

    def setUp(self):
        self.runtime = relay_runtime(rounds=2, ttl=3, cache_window=3)
        self.agents = agents_from_positions(LINE4_POSITIONS, LINE4_RADIUS)
        self.step0 = self.runtime.run_step(
            local_observations={},
            outboxes={"A": [source_frame(PAYLOAD, "A")]},
            step=0,
            agents=self.agents,
        )
        self.step1 = self.runtime.run_step(
            local_observations={}, outboxes={}, step=1, agents=self.agents
        )

    def test_d_is_out_of_reach_within_the_first_step(self):
        self.assertEqual(self.step0.agent_output["D"].num_messages, 0)

    def test_d_receives_at_the_next_step_round_zero(self):
        self.assertEqual(
            decode_inbox(self.step1, "D"), [("A", "C", 0, 3, tuple(PAYLOAD))]
        )

    def test_cross_step_forward_is_traced_at_the_new_step(self):
        forwarded = relay_records(self.step1, "forwarded")
        self.assertEqual(len(forwarded), 1)
        record = forwarded[0]
        self.assertEqual(record["step"], 1)
        self.assertEqual(record["round"], 0)
        self.assertEqual(record["sender"], "C")
        self.assertEqual(record["receiver"], "D")
        self.assertEqual(record["previous_hop"], "B")

    def test_exhausted_relay_stops_after_the_third_hop(self):
        step2 = self.runtime.run_step(
            local_observations={}, outboxes={}, step=2, agents=self.agents
        )
        self.assertEqual(step2.delivered_messages.num_messages, 0)

    def test_episode_reset_clears_the_carryover_queue(self):
        runtime = relay_runtime(rounds=2, ttl=3, cache_window=3)
        runtime.run_step(
            local_observations={},
            outboxes={"A": [source_frame(PAYLOAD, "A")]},
            step=0,
            agents=self.agents,
        )
        runtime.reset()
        fresh = runtime.run_step(
            local_observations={}, outboxes={}, step=0, agents=self.agents
        )
        self.assertEqual(fresh.delivered_messages.num_messages, 0)


class TestAgentLeavesGraphMidRelay(unittest.TestCase):
    """A pending cross-step carryover survives agents leaving the graph.

    4-line A-B-C-D with R=2, TTL=3, C=3: step 0 leaves C holding a queued
    forward of A's message for step 1. If the frame's origin (A) or its
    forwarder (C) is not on step 1's graph, the forward must be consumed
    with a traced drop — not crash the transport (which can encode neither
    an off-graph origin nor an off-graph sender).
    """

    def _step0(self, runtime, agents):
        return runtime.run_step(
            local_observations={},
            outboxes={"A": [source_frame(PAYLOAD, "A")]},
            step=0,
            agents=agents,
        )

    def test_origin_departure_drops_the_carried_forward_without_crashing(self):
        runtime = relay_runtime(rounds=2, ttl=3, cache_window=3)
        agents = agents_from_positions(LINE4_POSITIONS, LINE4_RADIUS)
        self._step0(runtime, agents)
        without_a = [agent for agent in agents if agent.name != "A"]
        step1 = runtime.run_step(
            local_observations={}, outboxes={}, step=1, agents=without_a
        )
        self.assertEqual(step1.delivered_messages.num_messages, 0)
        self.assertEqual(step1.agent_output["D"].num_messages, 0)
        dropped = relay_records(step1, "dropped_off_graph")
        self.assertEqual(len(dropped), 1)
        record = dropped[0]
        self.assertEqual(record["origin"], "A")
        self.assertEqual(record["sender"], "C")
        self.assertIsNone(record["receiver"])
        self.assertEqual(record["previous_hop"], "B")
        self.assertEqual(record["step"], 1)
        self.assertEqual(record["round"], 0)
        # The frame was consumed for good: nothing re-emerges when A returns.
        step2 = runtime.run_step(
            local_observations={}, outboxes={}, step=2, agents=agents
        )
        self.assertEqual(step2.delivered_messages.num_messages, 0)

    def test_forwarder_departure_drops_the_carried_forward_with_a_trace(self):
        runtime = relay_runtime(rounds=2, ttl=3, cache_window=3)
        agents = agents_from_positions(LINE4_POSITIONS, LINE4_RADIUS)
        self._step0(runtime, agents)
        without_c = [agent for agent in agents if agent.name != "C"]
        step1 = runtime.run_step(
            local_observations={}, outboxes={}, step=1, agents=without_c
        )
        self.assertEqual(step1.delivered_messages.num_messages, 0)
        dropped = relay_records(step1, "dropped_off_graph")
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0]["sender"], "C")
        self.assertEqual(dropped[0]["origin"], "A")


class TestCacheWindowBoundary(unittest.TestCase):
    """C == TTL: a message still live at the window edge is not re-accepted.

    4-cycle A-P-Q-X with R=2, TTL=3, C=3. X first records the message when
    round 0 completes; the cycle's counter-rotating copy (A->P->Q->X) reaches
    X at absolute round 2 = TTL - 1 rounds later — the latest arrival any
    live copy can achieve — crossing a movement-step boundary on the way.
    An off-by-one window would re-accept it as new.
    """

    def setUp(self):
        self.runtime = relay_runtime(rounds=2, ttl=3, cache_window=3)
        self.agents = agents_from_positions(SQUARE_POSITIONS, SQUARE_RADIUS)
        self.step0 = self.runtime.run_step(
            local_observations={},
            outboxes={"A": [source_frame(PAYLOAD, "A")]},
            step=0,
            agents=self.agents,
        )
        self.step1 = self.runtime.run_step(
            local_observations={}, outboxes={}, step=1, agents=self.agents
        )

    def test_first_step_deliveries_are_exactly_once(self):
        self.assertEqual([row[2] for row in decode_inbox(self.step0, "P")], [0])
        self.assertEqual([row[2] for row in decode_inbox(self.step0, "X")], [0])
        self.assertEqual([row[2] for row in decode_inbox(self.step0, "Q")], [1])

    def test_window_edge_copy_is_dropped_as_duplicate_not_re_accepted(self):
        for agent_id in ("A", "P", "Q", "X"):
            self.assertEqual(
                self.step1.agent_output[agent_id].num_messages,
                0,
                "agent {} re-accepted a still-live message as new".format(agent_id),
            )
        edge_drops = [
            record
            for record in relay_records(self.step1, "dropped_duplicate")
            if record["receiver"] == "X"
        ]
        self.assertEqual(len(edge_drops), 1)
        self.assertEqual(edge_drops[0]["step"], 1)
        self.assertEqual(edge_drops[0]["round"], 0)

    def test_the_window_edge_copy_is_never_forwarded_again(self):
        self.assertEqual(
            [record for record in relay_records(self.step1, "forwarded") if record["sender"] == "X"],
            [],
        )


class TestInputOrderDeterminism(unittest.TestCase):
    """Results are identical under permuted agent input order."""

    def _run(self, outbox_names, agent_order=None):
        runtime = relay_runtime(rounds=3, ttl=4, cache_window=4)
        agents = agents_from_positions(TRIANGLE_POSITIONS, TRIANGLE_RADIUS, order=agent_order)
        payloads = {
            "A": [1.0, 0.0, 0.0, 0.0],
            "B": [0.0, 2.0, 0.0, 0.0],
            "C": [0.0, 0.0, 3.0, 0.0],
        }
        outboxes = {name: [source_frame(payloads[name], name)] for name in outbox_names}
        observations = {name: object() for name in outbox_names}
        return runtime.run_step(
            local_observations=observations, outboxes=outboxes, step=0, agents=agents
        )

    def test_outbox_mapping_order_does_not_change_anything(self):
        first = self._run(("A", "B", "C"))
        second = self._run(("C", "A", "B"))
        self.assertEqual(first.diagnostics["trace_records"], second.diagnostics["trace_records"])
        for agent_id in ("A", "B", "C"):
            self.assertEqual(decode_inbox(first, agent_id), decode_inbox(second, agent_id))

    def test_agent_list_order_does_not_change_who_gets_what(self):
        first = self._run(("A", "B", "C"))
        second = self._run(("A", "B", "C"), agent_order=["C", "A", "B"])
        for agent_id in ("A", "B", "C"):
            first_view = sorted((row[0], row[4]) for row in decode_inbox(first, agent_id))
            second_view = sorted((row[0], row[4]) for row in decode_inbox(second, agent_id))
            self.assertEqual(first_view, second_view)


class TestSchemeCompilation(unittest.TestCase):
    """multihop_relay compiles through the registry with validation."""

    def test_scheme_is_registered(self):
        self.assertIn("multihop_relay", list_communication_schemes())

    def test_compiles_with_relay_defaults(self):
        plan = relay_plan(rounds=2, ttl=2, cache_window=3)
        self.assertEqual(plan.rounds, 2)
        self.assertEqual(plan.cache_window, 3)
        self.assertEqual(plan.output_contract, "preserved-inbox")
        self.assertEqual(plan.graph_update, "frozen")
        self.assertFalse(plan.differentiable)

    def test_missing_ttl_rejected_with_actionable_error(self):
        with self.assertRaises(ValueError) as ctx:
            relay_plan(rounds=2, ttl=2, cache_window=2, processor={})
        self.assertIn("ttl", str(ctx.exception))

    def test_cache_window_below_ttl_rejected_with_actionable_error(self):
        with self.assertRaises(ValueError) as ctx:
            relay_plan(rounds=2, ttl=3, cache_window=2)
        message = str(ctx.exception)
        self.assertIn("C >= TTL", message)
        self.assertIn("cache_window", message)

    def test_ttl_may_exceed_rounds(self):
        plan = relay_plan(rounds=1, ttl=4, cache_window=4)
        self.assertEqual(plan.rounds, 1)

    def test_aggregation_rejected(self):
        with self.assertRaises(ValueError):
            relay_plan(rounds=2, ttl=2, cache_window=2, processor={"ttl": 2, "aggregation": "mean"})

    def test_disabled_duplicate_suppression_rejected(self):
        with self.assertRaises(ValueError):
            relay_plan(
                rounds=2,
                ttl=2,
                cache_window=2,
                processor={"ttl": 2, "duplicate_suppression": False},
            )

    def test_unknown_forwarding_rule_rejected(self):
        with self.assertRaises(ValueError):
            relay_plan(
                rounds=2, ttl=2, cache_window=2, processor={"ttl": 2, "forwarding": "flooding"}
            )

    def test_zero_rounds_rejected(self):
        with self.assertRaises(ValueError):
            relay_plan(rounds=0, ttl=1, cache_window=1)

    def test_wrong_engineered_payload_dimension_rejected(self):
        with self.assertRaises(ValueError):
            relay_plan(
                rounds=2,
                ttl=2,
                cache_window=2,
                payload={"type": "engineered_vector", "dimension": 6},
            )


class TestRelayTraceRecords(unittest.TestCase):
    """Relay decision records are well-formed and JSON-serializable."""

    def test_unknown_decision_rejected(self):
        with self.assertRaises(ValueError):
            relay_decision_record(
                decision="teleported",
                step=0,
                round_index=0,
                message_id=0,
                origin="A",
                sender="A",
                receiver="B",
                previous_hop="A",
                ttl=1,
            )

    def test_records_carry_all_required_fields_and_serialize(self):
        runtime = relay_runtime(rounds=2, ttl=2, cache_window=2)
        result = runtime.run_step(
            local_observations={},
            outboxes={"A": [source_frame(PAYLOAD, "A")]},
            step=0,
            agents=agents_from_positions(CHAIN_POSITIONS, CHAIN_RADIUS),
        )
        records = relay_records(result)
        self.assertGreater(len(records), 0)
        required = (
            "decision",
            "step",
            "round",
            "message_id",
            "origin",
            "sender",
            "receiver",
            "previous_hop",
            "ttl",
            "scheme",
        )
        for record in records:
            for key in required:
                self.assertIn(key, record)
        json.dumps(records)  # Must not raise.

    def test_traces_identical_across_identical_runs(self):
        def run_episode():
            runtime = relay_runtime(rounds=2, ttl=3, cache_window=3)
            agents = agents_from_positions(LINE4_POSITIONS, LINE4_RADIUS)
            records = []
            for step in range(3):
                result = runtime.run_step(
                    local_observations={},
                    outboxes={"A": [source_frame([1.0, 0.0, 0.5, float(step)], "A", step=step)]},
                    step=step,
                    agents=agents,
                )
                records.extend(result.diagnostics["trace_records"])
            return records

        self.assertEqual(run_episode(), run_episode())


class TestEnvIntegration(unittest.TestCase):
    """multihop_relay runs inside the parallel environment and serializes."""

    def _make_env(self):
        env_config = {
            "width": 100,
            "height": 80,
            "max_cycles": 20,
            "save_episode_gifs": False,
            "communication_radius": 1000.0,
            "communication": {
                "enabled": True,
                "scheme": "multihop_relay",
                "rounds_per_step": 2,
                "cache_window": 2,
                "processor": {"ttl": 2},
            },
        }
        agents = create_agents_from_config(
            {
                "blue_agents": [{"count": 3, "detection_radius": 1000.0}],
                "red_agents": [{"count": 2, "detection_radius": 1000.0}],
            },
            env_config=env_config,
        )
        return ParallelGameEnv(agents, **env_config)

    def test_env_steps_deliver_relay_views_and_serialize_to_json(self):
        env = self._make_env()
        observations, _ = env.reset(seed=7)
        json.dumps(to_jsonable(observations))  # Reset views must serialize.
        for step in range(3):
            actions = {
                name: {"direction": (0.0, 0.0), "speed": 0.0} for name in env.agents
            }
            observations, _, _, _, infos = env.step(actions)
            for name in env.agents:
                view = observations[name]["communication"]
                self.assertEqual(view["scheme"], "multihop_relay")
            # The full observation dict (live inbox/cache handles included)
            # must serialize through the recorder path without raising.
            json.dumps(to_jsonable(observations))
            json.dumps(to_jsonable(infos))
        record_types = {
            record["record_type"] for record in env.last_communication_trace_records
        }
        self.assertIn(RELAY_RECORD_TYPE, record_types)
        json.dumps(env.last_communication_trace_records)


if __name__ == "__main__":
    unittest.main()
