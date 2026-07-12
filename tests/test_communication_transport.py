"""Tests for the Phase 1 synchronous slotted radius transport."""

import unittest

import torch

from src.communication.transport import (
    Frame,
    SlottedRadiusTransport,
    concat_edge_message_batches,
)
from src.communication.types import EdgeMessageBatch
from tests.communication_fixtures import chain_graph, empty_graph, two_team_graph

CONTEXT = {}


def frame(payload, origin, step=0, **kwargs):
    """Build a frame with a list payload."""
    return Frame(payload=payload, origin=origin, created_step=step, **kwargs)


def rows(batch, graph):
    """Return delivered rows as (sender, receiver, origin, message_id) id tuples."""
    ids = graph.agent_ids
    return [
        (ids[s], ids[r], ids[o], m)
        for s, r, o, m in zip(
            batch.sender_index.tolist(),
            batch.receiver_index.tolist(),
            batch.origin_index.tolist(),
            batch.message_id.tolist(),
        )
    ]


class TestOneHopDelivery(unittest.TestCase):
    """Frames move exactly one hop, to graph neighbours only."""

    def setUp(self):
        self.graph = chain_graph()
        self.transport = SlottedRadiusTransport()
        self.state = self.transport.reset(self.graph, CONTEXT)

    def test_neighbour_receives_and_non_neighbour_does_not(self):
        """A's frame reaches B (one hop) and never C (two hops away)."""
        delivered = self.transport.transmit_round(
            {"A": [frame([1.0, 2.0], "A")]}, self.graph, self.state, 0, CONTEXT
        )
        self.assertEqual(rows(delivered, self.graph), [("A", "B", "A", 0)])

    def test_broadcast_copies_to_every_outgoing_edge(self):
        """B has edges to A and C: one frame, two deliveries, one message id."""
        delivered = self.transport.transmit_round(
            {"B": [frame([3.0], "B")]}, self.graph, self.state, 0, CONTEXT
        )
        self.assertEqual(
            rows(delivered, self.graph), [("B", "A", "B", 0), ("B", "C", "B", 0)]
        )

    def test_distinct_payloads_are_preserved_not_aggregated(self):
        """Two senders to one receiver: B gets two distinct rows, unchanged."""
        delivered = self.transport.transmit_round(
            {"A": [frame([1.0, 0.0], "A")], "C": [frame([0.0, 1.0], "C")]},
            self.graph,
            self.state,
            0,
            CONTEXT,
        )
        to_b = [
            i for i, r in enumerate(delivered.receiver_index.tolist())
            if self.graph.agent_ids[r] == "B"
        ]
        self.assertEqual(len(to_b), 2)
        payloads = delivered.payload[to_b]
        self.assertTrue(torch.equal(payloads[0], torch.tensor([1.0, 0.0])))
        self.assertTrue(torch.equal(payloads[1], torch.tensor([0.0, 1.0])))
        message_ids = delivered.message_id[to_b].tolist()
        self.assertEqual(len(set(message_ids)), 2, "distinct messages keep distinct ids")

    def test_payload_transported_unchanged(self):
        payload = [0.25, -1.5, 3.0, 0.0]
        delivered = self.transport.transmit_round(
            {"A": [frame(payload, "A")]}, self.graph, self.state, 0, CONTEXT
        )
        self.assertTrue(torch.equal(delivered.payload[0], torch.tensor(payload)))

    def test_metadata_kept_separate_from_payload(self):
        """Ids, origin, hops, and creation labels live outside the payload."""
        delivered = self.transport.transmit_round(
            {"A": [frame([9.0], "A", step=4, created_round=0, hop_count=0)]},
            self.graph,
            self.state,
            2,
            CONTEXT,
        )
        self.assertEqual(delivered.payload.shape, (1, 1))
        self.assertEqual(delivered.round_index.tolist(), [2])
        self.assertEqual(delivered.metadata["created_step"].tolist(), [4])
        self.assertEqual(delivered.metadata["created_round"].tolist(), [0])
        self.assertEqual(delivered.metadata["observation_step"].tolist(), [4])
        self.assertEqual(delivered.metadata["hop_count"].tolist(), [1])

    def test_relayed_frame_keeps_origin_and_message_id(self):
        """A re-emitted frame preserves origin and id; sender differs."""
        delivered = self.transport.transmit_round(
            {"B": [frame([1.0], "A", message_id=41, hop_count=1)]},
            self.graph,
            self.state,
            1,
            CONTEXT,
        )
        self.assertEqual(
            rows(delivered, self.graph), [("B", "A", "A", 41), ("B", "C", "A", 41)]
        )
        self.assertEqual(delivered.metadata["hop_count"].tolist(), [2, 2])


class TestTeamAndGraphBoundaries(unittest.TestCase):
    """Deliveries obey the graph: no cross-team rows, no edgeless deliveries."""

    def test_opponents_never_receive_teammate_messages(self):
        graph = two_team_graph()
        transport = SlottedRadiusTransport()
        delivered = transport.transmit_round(
            {"blue_0": [frame([1.0], "blue_0")]}, graph, transport.reset(graph, CONTEXT),
            0, CONTEXT,
        )
        self.assertEqual(rows(delivered, graph), [("blue_0", "blue_1", "blue_0", 0)])
        blue_teams = {int(graph.team_index[r]) for r in delivered.receiver_index.tolist()}
        self.assertEqual(blue_teams, {0}, "no red agent may appear as receiver")

    def test_edgeless_graph_delivers_nothing(self):
        graph = empty_graph()
        transport = SlottedRadiusTransport()
        delivered = transport.transmit_round(
            {"A": [frame([1.0], "A")]}, graph, transport.reset(graph, CONTEXT), 0, CONTEXT
        )
        self.assertEqual(delivered.num_messages, 0)

    def test_empty_outboxes_deliver_nothing(self):
        graph = chain_graph()
        transport = SlottedRadiusTransport()
        delivered = transport.transmit_round(
            {}, graph, transport.reset(graph, CONTEXT), 0, CONTEXT
        )
        self.assertEqual(delivered.num_messages, 0)

    def test_unknown_sender_rejected(self):
        graph = chain_graph()
        transport = SlottedRadiusTransport()
        with self.assertRaises(ValueError):
            transport.transmit_round(
                {"Z": [frame([1.0], "Z")]}, graph, None, 0, CONTEXT
            )

    def test_unknown_origin_rejected(self):
        graph = chain_graph()
        transport = SlottedRadiusTransport()
        with self.assertRaises(ValueError):
            transport.transmit_round(
                {"A": [frame([1.0], "Z")]}, graph, None, 0, CONTEXT
            )

    def test_inconsistent_payload_dimensions_rejected(self):
        graph = chain_graph()
        transport = SlottedRadiusTransport()
        with self.assertRaises(ValueError):
            transport.transmit_round(
                {"A": [frame([1.0], "A")], "B": [frame([1.0, 2.0], "B")]},
                graph, None, 0, CONTEXT,
            )


class TestDeterminismAndIdentity(unittest.TestCase):
    """Message-id allocation and row ordering are deterministic."""

    def test_two_identical_runs_produce_identical_batches(self):
        graph = chain_graph()

        def run():
            transport = SlottedRadiusTransport()
            return transport.transmit_round(
                {"A": [frame([1.0], "A")], "B": [frame([2.0], "B")], "C": [frame([3.0], "C")]},
                graph, transport.reset(graph, CONTEXT), 0, CONTEXT,
            )

        first, second = run(), run()
        self.assertTrue(torch.equal(first.payload, second.payload))
        self.assertTrue(torch.equal(first.message_id, second.message_id))
        self.assertTrue(torch.equal(first.sender_index, second.sender_index))
        self.assertTrue(torch.equal(first.receiver_index, second.receiver_index))

    def test_id_allocation_independent_of_outbox_mapping_order(self):
        graph = chain_graph()
        forward = {"A": [frame([1.0], "A")], "C": [frame([2.0], "C")]}
        backward = {"C": [frame([2.0], "C")], "A": [frame([1.0], "A")]}
        t1, t2 = SlottedRadiusTransport(), SlottedRadiusTransport()
        first = t1.transmit_round(forward, graph, None, 0, CONTEXT)
        second = t2.transmit_round(backward, graph, None, 0, CONTEXT)
        self.assertEqual(rows(first, graph), rows(second, graph))

    def test_ids_stay_unique_across_rounds_and_steps(self):
        graph = chain_graph()
        transport = SlottedRadiusTransport()
        first = transport.transmit_round({"A": [frame([1.0], "A")]}, graph, None, 0, CONTEXT)
        transport.reset(graph, CONTEXT)  # per-step reset must NOT reset ids
        second = transport.transmit_round({"A": [frame([1.0], "A")]}, graph, None, 0, CONTEXT)
        self.assertNotEqual(first.message_id.tolist(), second.message_id.tolist())
        transport.reset_message_ids()  # explicit episode reset does
        third = transport.transmit_round({"A": [frame([1.0], "A")]}, graph, None, 0, CONTEXT)
        self.assertEqual(third.message_id.tolist(), first.message_id.tolist())


class TestFrameValidation(unittest.TestCase):
    """Frame construction contract."""

    def test_payload_coerced_to_1d_float_tensor(self):
        built = frame([1, 2, 3], "A")
        self.assertEqual(built.payload.dtype, torch.float32)
        self.assertEqual(built.payload.shape, (3,))

    def test_observation_step_defaults_to_created_step(self):
        self.assertEqual(frame([1.0], "A", step=7).observation_step, 7)

    def test_non_1d_payload_rejected(self):
        with self.assertRaises(ValueError):
            Frame(payload=torch.zeros((2, 2)), origin="A", created_step=0)

    def test_empty_origin_rejected(self):
        with self.assertRaises(ValueError):
            Frame(payload=torch.zeros(1), origin="", created_step=0)

    def test_negative_counters_rejected(self):
        with self.assertRaises(ValueError):
            Frame(payload=torch.zeros(1), origin="A", created_step=-1)


class TestConcatBatches(unittest.TestCase):
    """concat_edge_message_batches preserves order and handles empties."""

    def _batch(self, payloads, receiver):
        count = len(payloads)
        return EdgeMessageBatch(
            payload=torch.tensor(payloads),
            sender_index=torch.zeros(count, dtype=torch.long),
            receiver_index=torch.full((count,), receiver, dtype=torch.long),
            origin_index=torch.zeros(count, dtype=torch.long),
            message_id=torch.arange(count, dtype=torch.long),
            round_index=torch.zeros(count, dtype=torch.long),
            metadata={},
        )

    def test_concat_preserves_row_order(self):
        merged = concat_edge_message_batches(
            [self._batch([[1.0]], 1), self._batch([[2.0], [3.0]], 2)]
        )
        self.assertTrue(torch.equal(merged.payload, torch.tensor([[1.0], [2.0], [3.0]])))
        self.assertEqual(merged.receiver_index.tolist(), [1, 2, 2])

    def test_all_empty_batches_concat_to_empty(self):
        merged = concat_edge_message_batches([EdgeMessageBatch.empty(), EdgeMessageBatch.empty()])
        self.assertEqual(merged.num_messages, 0)

    def test_no_batches_concat_to_empty(self):
        self.assertEqual(concat_edge_message_batches([]).num_messages, 0)

    def test_empty_batches_are_skipped_regardless_of_dimension(self):
        merged = concat_edge_message_batches(
            [EdgeMessageBatch.empty(payload_dim=0), self._batch([[5.0]], 0)]
        )
        self.assertEqual(merged.num_messages, 1)


if __name__ == "__main__":
    unittest.main()
