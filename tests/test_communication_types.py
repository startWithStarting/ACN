"""Phase 0 acceptance tests for the communication core data types.

These tests pin the documented contract of ``src.communication.types`` from
``docs/communication_implementation_plan.md`` ("Core Data Types", "Runtime
Model") and ``docs/communication_decision_log.md`` ("Round And Cache Model",
"Communication Semantics"):

- All batch types construct with empty (0-length) tensors and support
  ``.to(device)`` without hidden NumPy conversion.
- ``InboxBatch`` supports dense payload-plus-mask conversion in which
  zero-message receivers are masked out.
- ``MessageCache`` implements the C-round sliding window: a round's messages
  are delivered before eviction runs, eviction removes the oldest round first
  with equal-age ties broken by delivery sequence, ``C = 0`` stores nothing,
  the window spans movement-step boundaries, and message age is derived at
  read time as ``current_step - observation_step`` without mutating stored
  payloads.
- ``PacketBatch`` keeps protocol metadata separate from the payload tensor.

Field names for ``CommunicationGraph``, ``EdgeMessageBatch``, ``InboxBatch``,
``PacketBatch``, and ``CommunicationResult`` follow the plan document
verbatim. Method names the plan leaves open (dense conversion, cache calls)
follow the implemented Phase 0 module; the semantics asserted here are the
doc-pinned ones, so a behavioral deviation fails these tests even where a
method name matches.
"""

from __future__ import annotations

import unittest
from typing import List, Optional, Sequence, Tuple

import torch

from src.communication.types import (
    BROADCAST_DESTINATION,
    CacheEntry,
    CommunicationGraph,
    CommunicationResult,
    EdgeMessageBatch,
    InboxBatch,
    MessageCache,
    PacketBatch,
)
from tests.communication_fixtures import chain_graph

CPU = torch.device("cpu")

_EDGE_INDEX_FIELDS = (
    "sender_index",
    "receiver_index",
    "origin_index",
    "message_id",
    "round_index",
)

_PACKET_PROTOCOL_FIELDS = (
    "packet_id",
    "message_id",
    "origin_index",
    "previous_hop_index",
    "destination_index",
    "sequence_number",
    "ttl",
    "fragment_index",
    "fragment_count",
    "created_step",
    "created_round",
)


# ---------------------------------------------------------------------------
# Construction helpers (documented plan field names, verbatim)
# ---------------------------------------------------------------------------


def _empty_communication_graph(step: int = 0) -> CommunicationGraph:
    """Build the canonical zero-agent, zero-edge communication graph."""
    return CommunicationGraph(
        agent_ids=(),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_distance=torch.zeros(0),
        edge_features=None,
        team_index=torch.zeros(0, dtype=torch.long),
        step=step,
    )


def _empty_edge_message_batch(dimension: int = 4) -> EdgeMessageBatch:
    """Build an EdgeMessageBatch with zero messages and the given payload width."""
    return EdgeMessageBatch(
        payload=torch.zeros((0, dimension)),
        sender_index=torch.zeros(0, dtype=torch.long),
        receiver_index=torch.zeros(0, dtype=torch.long),
        origin_index=torch.zeros(0, dtype=torch.long),
        message_id=torch.zeros(0, dtype=torch.long),
        round_index=torch.zeros(0, dtype=torch.long),
        metadata={},
    )


def _edge_message_batch(
    payload: torch.Tensor,
    receivers: Sequence[int],
    metadata: Optional[dict] = None,
) -> EdgeMessageBatch:
    """Build a small populated EdgeMessageBatch.

    Args:
        payload: Message payload tensor of shape [M, D].
        receivers: Receiver agent index per message, length M.
        metadata: Optional extra tensor metadata (leading dimension M).

    Returns:
        An EdgeMessageBatch with sender/origin 0, sequential message ids, and
        round index 0 for every message.
    """
    count = payload.shape[0]
    return EdgeMessageBatch(
        payload=payload,
        sender_index=torch.zeros(count, dtype=torch.long),
        receiver_index=torch.tensor(list(receivers), dtype=torch.long),
        origin_index=torch.zeros(count, dtype=torch.long),
        message_id=torch.arange(count, dtype=torch.long),
        round_index=torch.zeros(count, dtype=torch.long),
        metadata=metadata if metadata is not None else {},
    )


def _empty_packet_batch(dimension: int = 4) -> PacketBatch:
    """Build a PacketBatch with zero packets and the given payload width."""
    return PacketBatch(
        payload=torch.zeros((0, dimension)),
        packet_id=torch.zeros(0, dtype=torch.long),
        message_id=torch.zeros(0, dtype=torch.long),
        origin_index=torch.zeros(0, dtype=torch.long),
        previous_hop_index=torch.zeros(0, dtype=torch.long),
        destination_index=torch.zeros(0, dtype=torch.long),
        sequence_number=torch.zeros(0, dtype=torch.long),
        ttl=torch.zeros(0, dtype=torch.long),
        fragment_index=torch.zeros(0, dtype=torch.long),
        fragment_count=torch.zeros(0, dtype=torch.long),
        created_step=torch.zeros(0, dtype=torch.long),
        created_round=torch.zeros(0, dtype=torch.long),
    )


def _packet_batch(count: int = 2, dimension: int = 4) -> PacketBatch:
    """Build a small populated broadcast PacketBatch with distinguishable payloads.

    Args:
        count: Number of packets P.
        dimension: Payload width D.

    Returns:
        A PacketBatch whose payload is float [P, D] and whose protocol fields
        are long [P] tensors.
    """
    return PacketBatch(
        payload=torch.arange(count * dimension, dtype=torch.float32).reshape(count, dimension),
        packet_id=torch.arange(count, dtype=torch.long),
        message_id=torch.arange(count, dtype=torch.long),
        origin_index=torch.zeros(count, dtype=torch.long),
        previous_hop_index=torch.zeros(count, dtype=torch.long),
        destination_index=torch.full((count,), BROADCAST_DESTINATION, dtype=torch.long),
        sequence_number=torch.arange(count, dtype=torch.long),
        ttl=torch.full((count,), 3, dtype=torch.long),
        fragment_index=torch.zeros(count, dtype=torch.long),
        fragment_count=torch.ones(count, dtype=torch.long),
        created_step=torch.zeros(count, dtype=torch.long),
        created_round=torch.zeros(count, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# MessageCache helpers
# ---------------------------------------------------------------------------


def _push_round(
    cache: MessageCache,
    values: Sequence[float],
    step: int,
    round_index: int,
    observation_step: Optional[int] = None,
) -> List[CacheEntry]:
    """Insert one round's scalar payloads in delivery order, then complete the round.

    Args:
        cache: Cache under test.
        values: Scalar payload values; each becomes a one-element tensor message.
        step: Movement step of the round.
        round_index: Round index within the step.
        observation_step: Observation step recorded on each entry; defaults to
            the emission step (a source message).

    Returns:
        Entries evicted when the round completed.
    """
    for value in values:
        cache.insert(
            torch.tensor([value]),
            emitted_step=step,
            emitted_round=round_index,
            observation_step=observation_step,
        )
    return cache.advance_round(step, round_index)


def _entry_values(entries: Sequence[CacheEntry]) -> List[float]:
    """Return each entry's scalar payload value, preserving entry order."""
    return [float(entry.message[0]) for entry in entries]


def _row_multiset(rows: torch.Tensor) -> List[Tuple[float, ...]]:
    """Return a [K, D] tensor's rows as a sorted list of tuples (order-independent)."""
    return sorted(tuple(float(value) for value in row) for row in rows.tolist())


# ---------------------------------------------------------------------------
# (1) Empty construction
# ---------------------------------------------------------------------------


class TestEmptyConstruction(unittest.TestCase):
    """Every core type constructs with valid zero-length tensors (plan: Core Data Types)."""

    def test_empty_communication_graph(self):
        """A zero-agent, zero-edge graph is a valid snapshot."""
        graph = _empty_communication_graph()
        self.assertEqual(graph.agent_ids, ())
        self.assertEqual(graph.edge_index.shape, (2, 0))
        self.assertEqual(graph.edge_index.dtype, torch.long)
        self.assertEqual(graph.edge_distance.shape, (0,))
        self.assertEqual(graph.team_index.shape, (0,))
        self.assertEqual(graph.team_index.dtype, torch.long)
        self.assertIsNone(graph.edge_features)
        self.assertEqual(graph.step, 0)

    def test_empty_edge_message_batch(self):
        """A zero-message batch keeps well-formed [0, D] payload and [0] index tensors."""
        batch = _empty_edge_message_batch(dimension=4)
        self.assertEqual(batch.payload.shape, (0, 4))
        for name in _EDGE_INDEX_FIELDS:
            field_tensor = getattr(batch, name)
            self.assertEqual(field_tensor.shape, (0,), msg=name)
            self.assertEqual(field_tensor.dtype, torch.long, msg=name)
        self.assertEqual(batch.metadata, {})

    def test_empty_inbox_batch(self):
        """An inbox over zero messages constructs, with receiver_ptr allowed to be None."""
        inbox = InboxBatch(messages=_empty_edge_message_batch(), receiver_ptr=None)
        self.assertEqual(inbox.messages.payload.shape[0], 0)
        self.assertIsNone(inbox.receiver_ptr)

    def test_empty_packet_batch(self):
        """A zero-packet batch keeps well-formed [0, D] payload and [0] protocol tensors."""
        batch = _empty_packet_batch(dimension=4)
        self.assertEqual(batch.payload.shape, (0, 4))
        for name in _PACKET_PROTOCOL_FIELDS:
            field_tensor = getattr(batch, name)
            self.assertEqual(field_tensor.shape, (0,), msg=name)
            self.assertEqual(field_tensor.dtype, torch.long, msg=name)

    def test_empty_communication_result(self):
        """The explicit empty result constructs around an empty delivered batch."""
        result = CommunicationResult(
            agent_output={},
            final_state=None,
            delivered_messages=_empty_edge_message_batch(),
            protocol_state=None,
            diagnostics={},
        )
        self.assertEqual(result.agent_output, {})
        self.assertIsNone(result.final_state)
        self.assertIsNone(result.protocol_state)
        self.assertEqual(result.diagnostics, {})
        self.assertEqual(result.delivered_messages.payload.shape[0], 0)


# ---------------------------------------------------------------------------
# (2) Device movement
# ---------------------------------------------------------------------------


class TestDeviceMovement(unittest.TestCase):
    """Tensor-bearing types round-trip through ``.to(device)`` on CPU.

    The plan pins device support ("Tensor fields must support CPU and
    accelerator devices without hidden NumPy conversion"). Only the CPU round
    trip is exercised here; accelerator devices are not available in CI.
    """

    def test_communication_graph_to_cpu_round_trip(self):
        """Moving a populated graph to CPU preserves type, tensors, and metadata."""
        graph = chain_graph()
        moved = graph.to(CPU)
        self.assertIsInstance(moved, CommunicationGraph)
        self.assertEqual(moved.agent_ids, graph.agent_ids)
        self.assertEqual(moved.step, graph.step)
        self.assertIsNone(moved.edge_features)
        self.assertTrue(torch.equal(moved.edge_index, graph.edge_index))
        self.assertTrue(torch.equal(moved.edge_distance, graph.edge_distance))
        self.assertTrue(torch.equal(moved.team_index, graph.team_index))
        self.assertEqual(moved.edge_index.device.type, "cpu")
        again = moved.to(CPU)
        self.assertTrue(torch.equal(again.edge_index, graph.edge_index))

    def test_edge_message_batch_to_cpu_round_trip(self):
        """Moving a message batch preserves payload, index fields, and tensor metadata."""
        batch = _edge_message_batch(
            payload=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            receivers=[1, 0],
            metadata={"observation_step": torch.tensor([0, 1], dtype=torch.long)},
        )
        moved = batch.to(CPU)
        self.assertIsInstance(moved, EdgeMessageBatch)
        self.assertTrue(torch.equal(moved.payload, batch.payload))
        for name in _EDGE_INDEX_FIELDS:
            self.assertTrue(
                torch.equal(getattr(moved, name), getattr(batch, name)), msg=name
            )
        self.assertIn("observation_step", moved.metadata)
        self.assertTrue(
            torch.equal(moved.metadata["observation_step"], batch.metadata["observation_step"])
        )
        self.assertEqual(moved.payload.device.type, "cpu")

    def test_inbox_batch_to_cpu_round_trip(self):
        """Moving an inbox moves both the nested message batch and the receiver pointer."""
        messages = _edge_message_batch(
            payload=torch.tensor([[1.0], [2.0], [3.0]]),
            receivers=[0, 0, 1],
        )
        inbox = InboxBatch(messages=messages, receiver_ptr=torch.tensor([0, 2, 3]))
        moved = inbox.to(CPU)
        self.assertIsInstance(moved, InboxBatch)
        self.assertIsInstance(moved.messages, EdgeMessageBatch)
        self.assertTrue(torch.equal(moved.messages.payload, messages.payload))
        self.assertTrue(torch.equal(moved.messages.receiver_index, messages.receiver_index))
        self.assertIsNotNone(moved.receiver_ptr)
        self.assertTrue(torch.equal(moved.receiver_ptr, inbox.receiver_ptr))

    def test_packet_batch_to_cpu_round_trip(self):
        """Moving a packet batch preserves the payload and every protocol field."""
        batch = _packet_batch(count=2, dimension=4)
        moved = batch.to(CPU)
        self.assertIsInstance(moved, PacketBatch)
        self.assertTrue(torch.equal(moved.payload, batch.payload))
        for name in _PACKET_PROTOCOL_FIELDS:
            self.assertTrue(
                torch.equal(getattr(moved, name), getattr(batch, name)), msg=name
            )
        self.assertEqual(moved.payload.device.type, "cpu")

    def test_empty_types_move_to_cpu(self):
        """Empty tensors remain valid under device movement (Phase 0 acceptance)."""
        movables = (
            _empty_communication_graph(),
            _empty_edge_message_batch(),
            InboxBatch(messages=_empty_edge_message_batch(), receiver_ptr=None),
            _empty_packet_batch(),
        )
        for obj in movables:
            with self.subTest(type=type(obj).__name__):
                moved = obj.to(CPU)
                self.assertIsInstance(moved, type(obj))

    def test_communication_result_to_moves_batch_agent_outputs(self):
        """A preserved inbox in agent_output moves with the result, opaque values pass through."""
        messages = _edge_message_batch(
            payload=torch.tensor([[1.0], [2.0]]),
            receivers=[0, 1],
        )
        result = CommunicationResult(
            agent_output={
                "blue_0": InboxBatch(messages=messages, receiver_ptr=None),
                "blue_1": torch.tensor([3.0]),
                "blue_2": "opaque",
            },
            final_state=None,
            delivered_messages=messages,
        )
        moved = result.to(CPU)
        moved_inbox = moved.agent_output["blue_0"]
        self.assertIsInstance(moved_inbox, InboxBatch)
        self.assertTrue(torch.equal(moved_inbox.messages.payload, messages.payload))
        self.assertEqual(moved_inbox.messages.payload.device.type, "cpu")
        moved_tensor = moved.agent_output["blue_1"]
        self.assertIsInstance(moved_tensor, torch.Tensor)
        self.assertEqual(moved_tensor.device.type, "cpu")
        self.assertEqual(moved.agent_output["blue_2"], "opaque")


# ---------------------------------------------------------------------------
# (2b) Identity equality and index validation
# ---------------------------------------------------------------------------


class TestIdentityEqualitySemantics(unittest.TestCase):
    """Tensor-bearing types compare by identity (eq=False); ``==`` never raises."""

    def _cache_entry(self) -> CacheEntry:
        """Build a CacheEntry whose message payload is a tensor."""
        return CacheEntry(
            message=torch.tensor([1.0]),
            emitted_step=0,
            emitted_round=0,
            delivery_sequence=0,
            observation_step=0,
        )

    def test_equality_is_identity_and_never_raises(self):
        """Comparing content-identical instances returns a bool instead of raising."""
        pairs = (
            (chain_graph(), chain_graph()),
            (EdgeMessageBatch.empty(4), EdgeMessageBatch.empty(4)),
            (
                InboxBatch(messages=_empty_edge_message_batch(), receiver_ptr=None),
                InboxBatch(messages=_empty_edge_message_batch(), receiver_ptr=None),
            ),
            (_packet_batch(), _packet_batch()),
            (CommunicationResult.empty(("A",)), CommunicationResult.empty(("A",))),
            (self._cache_entry(), self._cache_entry()),
        )
        for left, right in pairs:
            with self.subTest(type=type(left).__name__):
                self.assertTrue(left == left)
                self.assertFalse(left == right)
                self.assertTrue(left != right)

    def test_frozen_types_hash_by_identity(self):
        """Frozen dataclasses stay hashable without hashing their tensor fields."""
        graph = chain_graph()
        entry = self._cache_entry()
        self.assertEqual(hash(graph), hash(graph))
        self.assertIn(graph, {graph})
        self.assertEqual(len({graph, chain_graph()}), 2)
        self.assertIn(entry, {entry})
        self.assertEqual(len({entry, self._cache_entry()}), 2)


class TestSelectIndexValidation(unittest.TestCase):
    """select() rejects boolean masks and float indices instead of mis-selecting rows."""

    def test_boolean_mask_select_is_rejected(self):
        """A bool mask raises TypeError naming the dtype; nonzero() indices still work."""
        batch = _edge_message_batch(
            payload=torch.tensor([[1.0], [2.0], [3.0]]),
            receivers=[0, 1, 2],
        )
        mask = torch.tensor([False, True, True])
        with self.assertRaises(TypeError) as ctx:
            batch.select(mask)
        self.assertIn("bool", str(ctx.exception))
        picked = batch.select(mask.nonzero(as_tuple=False).flatten())
        self.assertTrue(torch.equal(picked.payload, torch.tensor([[2.0], [3.0]])))

    def test_float_index_select_is_rejected(self):
        """A floating-point index tensor raises TypeError instead of truncating."""
        batch = _edge_message_batch(payload=torch.tensor([[1.0], [2.0]]), receivers=[0, 1])
        with self.assertRaises(TypeError):
            batch.select(torch.tensor([0.5, 1.0]))

    def test_packet_batch_boolean_mask_select_is_rejected(self):
        """PacketBatch.select shares the same index validation."""
        with self.assertRaises(TypeError):
            _packet_batch(count=2).select(torch.tensor([True, False]))


# ---------------------------------------------------------------------------
# (3) InboxBatch dense conversion
# ---------------------------------------------------------------------------


class TestInboxDenseConversion(unittest.TestCase):
    """Dense payload-plus-mask views mask out zero-message receivers."""

    def _three_message_inbox(self) -> InboxBatch:
        """Inbox where receiver 0 gets two messages, receiver 1 none, receiver 2 one."""
        messages = _edge_message_batch(
            payload=torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            receivers=[0, 0, 2],
        )
        return InboxBatch(messages=messages, receiver_ptr=None)

    def test_dense_conversion_masks_zero_message_receivers(self):
        """The mask is True exactly at delivered slots; empty inboxes are all-False rows."""
        dense, mask = self._three_message_inbox().to_dense(num_receivers=3)
        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(dense.shape[0], 3)
        self.assertEqual(dense.shape[2], 2)
        self.assertEqual(mask.shape, dense.shape[:2])
        self.assertEqual(int(mask[0].sum()), 2)
        self.assertEqual(int(mask[1].sum()), 0)
        self.assertEqual(int(mask[2].sum()), 1)

    def test_dense_payload_rows_match_delivered_messages(self):
        """Masked-in dense rows are exactly the delivered payloads, kept distinct."""
        inbox = self._three_message_inbox()
        dense, mask = inbox.to_dense(num_receivers=3)
        self.assertEqual(_row_multiset(dense[0][mask[0]]), [(1.0, 2.0), (3.0, 4.0)])
        self.assertEqual(_row_multiset(dense[2][mask[2]]), [(5.0, 6.0)])
        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.assertTrue(torch.equal(inbox.messages.payload, expected))

    def test_dense_conversion_of_empty_inbox(self):
        """A fully empty inbox converts to an all-masked-out dense view."""
        inbox = InboxBatch(messages=_empty_edge_message_batch(dimension=2), receiver_ptr=None)
        dense, mask = inbox.to_dense(num_receivers=2)
        self.assertEqual(dense.shape[0], 2)
        self.assertEqual(mask.shape[0], 2)
        self.assertEqual(int(mask.sum()), 0)


# ---------------------------------------------------------------------------
# (4) MessageCache round window
# ---------------------------------------------------------------------------


class TestMessageCache(unittest.TestCase):
    """The C-round sliding cache window (decision log: Round And Cache Model)."""

    def test_zero_window_stores_nothing(self):
        """C = 0 retains no messages; within-step propagation is processor state, not cache."""
        cache = MessageCache(capacity_rounds=0)
        entry = cache.insert(torch.tensor([0.0]), emitted_step=0, emitted_round=0)
        self.assertIsInstance(entry, CacheEntry)  # still returned, e.g. for tracing
        self.assertEqual(len(cache), 0)
        cache.advance_round(0, 0)
        self.assertEqual(cache.entries(), [])

    def test_delivery_precedes_eviction(self):
        """A full window delivers the new round first, then evicts; it never rejects."""
        cache = MessageCache(capacity_rounds=1)
        _push_round(cache, [0.0], step=0, round_index=0)
        self.assertEqual(_entry_values(cache.entries()), [0.0])
        evicted = _push_round(cache, [1.0], step=1, round_index=0)
        self.assertEqual(_entry_values(evicted), [0.0])
        self.assertEqual(_entry_values(cache.entries()), [1.0])

    def test_window_evicts_oldest_round_first(self):
        """Exceeding the window drops the oldest round's messages, newest stay."""
        cache = MessageCache(capacity_rounds=2)
        _push_round(cache, [0.0], step=0, round_index=0)
        _push_round(cache, [1.0], step=1, round_index=0)
        self.assertEqual(sorted(_entry_values(cache.entries())), [0.0, 1.0])
        evicted = _push_round(cache, [2.0], step=2, round_index=0)
        self.assertEqual(_entry_values(evicted), [0.0])
        self.assertEqual(sorted(_entry_values(cache.entries())), [1.0, 2.0])

    def test_window_counts_rounds_not_steps(self):
        """C is measured in rounds: with R = 2, both step-0 rounds age past a C = 2 window."""
        cache = MessageCache(capacity_rounds=2)
        _push_round(cache, [0.0], step=0, round_index=0)
        _push_round(cache, [1.0], step=0, round_index=1)
        evicted = _push_round(cache, [2.0], step=1, round_index=0)
        self.assertEqual(_entry_values(evicted), [0.0])
        self.assertEqual(sorted(_entry_values(cache.entries())), [1.0, 2.0])

    def test_equal_age_ties_evicted_by_delivery_sequence(self):
        """Equal-observation-step messages are evicted in delivery-sequence order."""
        cache = MessageCache(capacity_rounds=1)
        common = {"emitted_step": 0, "emitted_round": 0, "observation_step": 0}
        cache.insert(torch.tensor([0.0]), delivery_sequence=2, **common)
        cache.insert(torch.tensor([1.0]), delivery_sequence=0, **common)
        cache.insert(torch.tensor([2.0]), delivery_sequence=1, **common)
        self.assertEqual(cache.advance_round(0, 0), [])
        self.assertEqual(_entry_values(cache.entries()), [1.0, 2.0, 0.0])
        evicted = cache.advance_round(1, 0)
        self.assertEqual(_entry_values(evicted), [1.0, 2.0, 0.0])
        self.assertEqual({entry.observation_step for entry in evicted}, {0})
        # Note: the decision log allows the round window to be "optionally bounded
        # further by a per-round message cap". That cap is an explicitly optional
        # extension, deliberately absent from the Phase 0 MessageCache (a pure C-round
        # window), so round-window eviction above is the only path exercising the
        # delivery-sequence tie-break; a later phase adding the cap must reuse the
        # same oldest-observation-first, then-delivery-sequence eviction order.

    def test_age_is_derived_at_read_without_mutating_payloads(self):
        """Age is current_step - observation_step at read time; stored contents never change."""
        cache = MessageCache(capacity_rounds=8)
        payload = torch.tensor([7.0, 8.0])
        snapshot = payload.clone()
        cache.insert(payload, emitted_step=2, emitted_round=0)
        cache.advance_round(2, 0)
        entry = cache.entries()[0]
        self.assertEqual(entry.observation_step, 2)  # defaults to the emission step
        for current_step, expected_age in ((2, 0), (5, 3), (9, 7)):
            self.assertEqual(entry.age(current_step), expected_age)
            self.assertEqual(current_step - entry.observation_step, expected_age)
            self.assertTrue(torch.equal(entry.message, snapshot))
        self.assertIs(entry.message, payload)  # stored object, never copied or rewritten
        cache.advance_round(3, 0)  # time advances with no deliveries
        entry_after = cache.entries()[0]
        self.assertEqual(entry_after.observation_step, 2)
        self.assertTrue(torch.equal(entry_after.message, snapshot))
        self.assertEqual([age for _, age in cache.iter_with_age(5)], [3])

    def test_window_spans_movement_step_boundary(self):
        """With R = 1 and C = 2, a step-t message is still readable at t + 1, gone at t + 2."""
        cache = MessageCache(capacity_rounds=2)
        _push_round(cache, [0.0], step=0, round_index=0)
        _push_round(cache, [1.0], step=1, round_index=0)
        carried = [e for e in cache.entries() if float(e.message[0]) == 0.0]
        self.assertEqual(len(carried), 1)  # step-0 message readable during step 1
        self.assertEqual(carried[0].age(1), 1)
        evicted = cache.advance_round(2, 0)  # step 2 completes with no deliveries
        self.assertEqual(_entry_values(evicted), [0.0])
        self.assertEqual(_entry_values(cache.entries()), [1.0])

    def test_open_round_deliveries_remain_distinguishable(self):
        """Round-local deliveries are separable from the older stored window."""
        cache = MessageCache(capacity_rounds=4)
        _push_round(cache, [0.0], step=0, round_index=0)
        cache.insert(torch.tensor([1.0]), emitted_step=1, emitted_round=0)  # round still open
        self.assertEqual(_entry_values(cache.entries()), [0.0, 1.0])
        self.assertEqual(_entry_values(cache.window_entries()), [0.0])

    def test_insert_ahead_of_open_round_is_rejected(self):
        """Skipping past a still-open round fails at insert time and leaves the cache usable."""
        cache = MessageCache(capacity_rounds=4)
        cache.insert(torch.tensor([0.0]), emitted_step=0, emitted_round=0)  # round (0, 0) open
        with self.assertRaises(ValueError) as ctx:
            cache.insert(torch.tensor([1.0]), emitted_step=0, emitted_round=2)
        self.assertIn("(0, 0)", str(ctx.exception))
        self.assertIn("(0, 2)", str(ctx.exception))
        cache.advance_round(0, 0)  # the open round still completes normally
        self.assertEqual(_entry_values(cache.entries()), [0.0])

    def test_relay_floor_c_must_cover_ttl(self):
        """The relay-correctness floor is C >= TTL (compiler-level rejection is config's job)."""
        self.assertTrue(MessageCache(capacity_rounds=3).supports_ttl(3))
        self.assertFalse(MessageCache(capacity_rounds=2).supports_ttl(3))

    def test_reset_clears_all_message_memory(self):
        """Episode reset empties the cache entirely."""
        cache = MessageCache(capacity_rounds=4)
        _push_round(cache, [0.0, 1.0], step=0, round_index=0)
        self.assertEqual(len(cache), 2)
        cache.clear()
        self.assertEqual(len(cache), 0)
        self.assertEqual(cache.entries(), [])


# ---------------------------------------------------------------------------
# (5) PacketBatch protocol/payload separation
# ---------------------------------------------------------------------------


class TestPacketBatchProtocolSeparation(unittest.TestCase):
    """Protocol metadata must stay outside the payload tensor (plan: PacketBatch)."""

    def test_payload_and_protocol_fields_are_separate_tensors(self):
        """Payload keeps its constructed float [P, D] shape; protocol fields are long [P]."""
        batch = _packet_batch(count=2, dimension=4)
        self.assertEqual(batch.payload.shape, (2, 4))
        self.assertTrue(batch.payload.is_floating_point())
        for name in _PACKET_PROTOCOL_FIELDS:
            field_tensor = getattr(batch, name)
            self.assertEqual(field_tensor.shape, (2,), msg=name)
            self.assertEqual(field_tensor.dtype, torch.long, msg=name)

    def test_mutating_protocol_metadata_leaves_payload_unchanged(self):
        """Hop bookkeeping (TTL decrement, previous-hop update) cannot alter payloads."""
        batch = _packet_batch(count=3, dimension=2)
        payload_before = batch.payload.clone()
        batch.ttl.sub_(1)  # TTL is decremented once per hop
        batch.previous_hop_index.fill_(1)
        self.assertTrue(torch.equal(batch.payload, payload_before))
        self.assertTrue(torch.equal(batch.ttl, torch.full((3,), 2, dtype=torch.long)))


if __name__ == "__main__":
    unittest.main()
