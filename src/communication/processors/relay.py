"""First-seen unchanged multi-hop relay processor (Phase 3).

Implements the ``multihop_relay`` semantics of
``docs/communication_implementation_plan.md`` ("Unchanged Relay Processing")
over the *existing* fixed-round runtime and slotted broadcast transport:

- Message identity survives hops: a forwarded :class:`Frame` carries the
  ``message_id`` of the received copy, which the transport preserves when
  pre-set (the transport's relay hook).
- A per-receiver duplicate cache keyed by ``message_id`` drops copies BEFORE
  application delivery and forwarding, with a traced drop reason. The origin
  of a message is seeded into its own duplicate cache on first transmission,
  so no cycle can deliver a message back to its source.
- First-seen copies are delivered to the receiver's preserved inbox view and
  enqueued for forwarding in the NEXT round: the runtime double-buffers
  ``emit_round`` emissions, so information moves at most one graph hop per
  round.
- Payloads are never transformed. A forwarded frame re-uses the delivered
  payload row and the original ``origin``/``created_step``/``created_round``/
  ``observation_step``; only ``hop_count`` advances. Every receiver therefore
  sees the origin's exact bytes, with ``origin_index`` and ``sender_index``
  distinguishing the true source from the immediate sender.
- ``ttl`` is the total hop budget of a message. A copy delivered after hop
  ``h`` has ``ttl - h`` hops left; copies with no budget left are delivered
  but never re-emitted (``dropped_ttl``).

Previous-hop exclusion under the broadcast transport:

The slotted transport is broadcast-only — an emitted frame is copied to
*every* valid out-edge of its sender. The relay therefore excludes the
previous hop in two layers: when the previous hop is the only out-neighbour
(e.g. an isolated ``A<->B`` pair), the frame is not emitted at all (traced as
``dropped_no_target``), so no ping-pong transmission ever occurs; when other
targets exist, the frame is emitted (and traced as ``forwarded`` per intended
target), and the broadcast copy physically reaching the previous hop is
suppressed at delivery as a guaranteed duplicate (the previous hop has, by
construction, already seen the message). Suppressed copies never reach the
application inbox, the per-agent message caches, or the step's
``delivered_messages`` log (see :meth:`RelayProcessor.accepted_rows`); they
remain visible only as transport-level ``communication_delivery`` records
paired with ``dropped_duplicate`` relay records.

Cross-step continuation and the cache window:

Frames enqueued in the LAST round of movement step ``t`` have their "next
round" at round 0 of step ``t + 1``. Because agents may leave the graph
between steps, a carried frame whose forwarder OR origin is no longer a node
of the new step's graph is consumed with a traced ``dropped_off_graph``
(the transport can encode neither). That queue and the duplicate cache live
in a :class:`RelayState` shared by the processor and the
:class:`RelayCarryoverTransport`, *beside* the per-agent
:class:`~src.communication.types.MessageCache`: the duplicate-suppression
memory horizon EQUALS the cache window ``C`` — an id recorded during round
``n`` (counting completed rounds across movement steps, exactly the cadence
of ``MessageCache.advance_round``) is remembered while ``n`` is within the
last ``C`` completed rounds, the same lifetime as the corresponding
``CachedDelivery`` entries. ``C >= TTL`` (validated at config parse and plan
build) therefore guarantees a message id outlives every live copy of its
message, which is what makes duplicate suppression correct across cycles and
movement-step boundaries.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

from src.communication.tracing import relay_decision_record
from src.communication.transport import Frame, SlottedRadiusTransport, concat_edge_message_batches
from src.communication.types import (
    CommunicationGraph,
    CommunicationResult,
    EdgeMessageBatch,
    InboxBatch,
)
from src.utils.logger import get_logger

logger = get_logger("acn.communication")

__all__ = ["RelayState", "RelayCarryoverTransport", "RelayProcessor"]


@dataclass(frozen=True, eq=False)
class _QueuedForward:
    """One first-seen copy queued for forwarding in the next round.

    Attributes:
        sender: Agent id that will re-emit the frame (the copy's receiver).
        frame: The unchanged frame to re-emit (``message_id`` pre-set,
            ``hop_count`` = hops already traversed).
        previous_hop: Agent id of the copy's immediate sender, excluded from
            the forward target set.
    """

    sender: str
    frame: Frame
    previous_hop: str


class RelayState:
    """Cross-step relay protocol state shared by processor and transport.

    Lives beside the per-agent message caches (see the module docstring for
    the horizon equality) and is cleared on episode reset through
    :meth:`RelayCarryoverTransport.reset_message_ids`.

    Attributes:
        seen: Per-agent duplicate cache: ``message_id`` -> completed-round
            counter at which the id was recorded, in recording order.
        rounds_completed: Monotone count of processed communication rounds
            across movement steps (the ``MessageCache.advance_round``
            cadence).
        carryover: Frames enqueued in the last round of a step, awaiting
            re-injection at round 0 of the next step.
        trace_records: Relay decision records accumulated since the last
            ``finalize`` drain.
    """

    def __init__(self) -> None:
        """Create empty relay protocol state."""
        self.seen: Dict[str, "OrderedDict[int, int]"] = {}
        self.rounds_completed: int = 0
        self.carryover: List[_QueuedForward] = []
        self.trace_records: List[Dict[str, object]] = []

    def clear(self) -> None:
        """Drop all protocol memory (episode reset)."""
        self.seen.clear()
        self.rounds_completed = 0
        self.carryover.clear()
        self.trace_records.clear()

    def trace_summary(self) -> Dict[str, object]:
        """Compact JSON-safe summary for transition trace rows."""
        return {
            "type": "RelayState",
            "rounds_completed": self.rounds_completed,
            "seen_ids": sum(len(ids) for ids in self.seen.values()),
            "carryover": len(self.carryover),
        }


def _context_int(context: Mapping[str, object], key: str, default: int) -> int:
    """Read an integer from the public runtime context with a fallback."""
    value = context.get(key, default)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return default


def _context_scheme(context: Mapping[str, object]) -> Optional[str]:
    """Read the scheme name from the public runtime context."""
    value = context.get("scheme")
    return value if isinstance(value, str) else None


def _out_neighbours(graph: CommunicationGraph) -> Dict[str, List[str]]:
    """Map each agent id to its out-neighbour ids in graph edge order."""
    neighbours: Dict[str, List[str]] = {}
    sources = graph.edge_index[0].tolist()
    receivers = graph.edge_index[1].tolist()
    for source, receiver in zip(sources, receivers):
        neighbours.setdefault(graph.agent_ids[source], []).append(graph.agent_ids[receiver])
    return neighbours


def _emit_queued(
    queued: Sequence[_QueuedForward],
    graph: CommunicationGraph,
    step: int,
    round_index: int,
    ttl: int,
    scheme: Optional[str],
    trace_sink: List[Dict[str, object]],
) -> Dict[str, List[Frame]]:
    """Build the outboxes that forward queued frames over ``graph``.

    Applies previous-hop exclusion against the *current* graph (forwarding
    that continues across a movement step uses the new step's edges) and
    appends one ``forwarded`` trace record per intended target.

    Args:
        queued: Queued forwards in enqueue order.
        graph: The frozen graph of the round in which the frames transmit.
        step: Movement step of the transmission.
        round_index: Round (within ``step``) in which the frames transmit.
        ttl: The scheme's total hop budget (for the records' remaining ttl).
        scheme: Scheme name recorded on trace records.
        trace_sink: Mutable list receiving the ``forwarded`` and drop records.

    Returns:
        Outboxes for the transport, keyed by forwarding agent id. Frames
        whose forwarder or origin left the graph (``dropped_off_graph``), or
        whose only out-neighbour is the previous hop (``dropped_no_target``),
        are consumed with a traced drop reason and never emitted.
    """
    on_graph = set(graph.agent_ids)
    neighbours = _out_neighbours(graph)
    outboxes: Dict[str, List[Frame]] = {}
    for item in queued:
        frame = item.frame
        drop_fields = dict(
            step=step,
            round_index=round_index,
            message_id=int(frame.message_id),  # type: ignore[arg-type]
            origin=frame.origin,
            sender=item.sender,
            receiver=None,
            previous_hop=item.previous_hop,
            # No transmission occurs on a drop: the budget the frame held.
            ttl=ttl - frame.hop_count,
            scheme=scheme,
        )
        if item.sender not in on_graph or frame.origin not in on_graph:
            # The transport cannot address an off-graph forwarder nor encode
            # an off-graph origin; the frame is consumed with a traced drop.
            departed = item.sender if item.sender not in on_graph else frame.origin
            logger.debug(
                "Relay forward of message {} dropped: '{}' left the graph",
                frame.message_id,
                departed,
            )
            trace_sink.append(relay_decision_record(decision="dropped_off_graph", **drop_fields))
            continue
        targets = [
            receiver
            for receiver in neighbours.get(item.sender, [])
            if receiver != item.previous_hop
        ]
        if not targets:
            # Previous-hop exclusion left nowhere to send (e.g. an isolated
            # bidirectional pair): the frame is consumed, never emitted.
            trace_sink.append(relay_decision_record(decision="dropped_no_target", **drop_fields))
            continue
        outboxes.setdefault(item.sender, []).append(item.frame)
        remaining = ttl - (item.frame.hop_count + 1)
        for target in targets:
            trace_sink.append(
                relay_decision_record(
                    decision="forwarded",
                    step=step,
                    round_index=round_index,
                    message_id=int(item.frame.message_id),  # type: ignore[arg-type]
                    origin=item.frame.origin,
                    sender=item.sender,
                    receiver=target,
                    previous_hop=item.previous_hop,
                    ttl=remaining,
                    scheme=scheme,
                )
            )
    return outboxes


class RelayCarryoverTransport(SlottedRadiusTransport):
    """Slotted broadcast transport that re-injects cross-step relay forwards.

    Frames enqueued in the last round of movement step ``t`` are transported
    in round 0 of step ``t + 1``: this transport merges them into the round-0
    source outboxes (after each sender's source frames), leaving the
    runtime's round loop untouched. Previous-hop filtering for carried
    frames runs against the *new* step's graph at re-injection time.
    """

    def __init__(self, state: RelayState, ttl: int) -> None:
        """Create a carryover-aware transport.

        Args:
            state: The relay protocol state shared with the processor.
            ttl: The scheme's total hop budget (for ``forwarded`` records).
        """
        super().__init__()
        self._state = state
        self._ttl = ttl

    def reset_message_ids(self) -> None:
        """Reset message ids AND all cross-step relay state (episode reset)."""
        super().reset_message_ids()
        self._state.clear()

    def transmit_round(
        self,
        outboxes: Mapping[str, Sequence[object]],
        graph: CommunicationGraph,
        transport_state: object,
        round_index: int,
        context: Mapping[str, object],
    ) -> EdgeMessageBatch:
        """Transmit one round, re-injecting carried-over forwards at round 0.

        Args:
            outboxes: The round's outboxes (round 0: the source outboxes).
            graph: The frozen communication graph for this movement step.
            transport_state: Ignored (stateless per step).
            round_index: Zero-based round index within the current step.
            context: Public runtime context (``step``/``scheme`` are read for
                the carried frames' ``forwarded`` trace records).

        Returns:
            The delivered :class:`EdgeMessageBatch` of the round.
        """
        if round_index == 0 and self._state.carryover:
            queued = self._state.carryover
            self._state.carryover = []
            step = _context_int(context, "step", graph.step)
            scheme = _context_scheme(context)
            carried = _emit_queued(
                queued, graph, step, 0, self._ttl, scheme, self._state.trace_records
            )
            if carried:
                merged: Dict[str, List[object]] = {
                    agent_id: list(frames) for agent_id, frames in outboxes.items()
                }
                for agent_id, frames in carried.items():
                    merged.setdefault(agent_id, []).extend(frames)
                outboxes = merged
        return super().transmit_round(outboxes, graph, transport_state, round_index, context)


class RelayProcessor:
    """First-seen unchanged-forwarding processor with duplicate suppression.

    Implements the same duck-typed surface as
    :class:`~src.communication.processors.identity.DirectProcessor`
    (``initialize`` / ``process_round`` / ``emit_round`` / ``finalize``),
    re-emits first-seen copies through the runtime's ``emit_round`` hook, and
    reports the first-seen row indices of each round through the runtime's
    ``accepted_rows`` hook, so duplicate-suppressed copies never enter the
    per-agent message caches or the step's ``delivered_messages`` log. Round
    state is the ordered list of *accepted* (first-seen) delivery batches;
    ``finalize`` returns the preserved, origin-distinct application inbox per
    agent, plus the step's relay decision records under
    ``diagnostics["processor_trace_records"]``.

    Cross-step protocol memory (duplicate cache, carryover queue) lives in
    the shared :class:`RelayState`; see the module docstring for its horizon
    equality with the message-cache window.
    """

    def __init__(self, ttl: int, cache_window: int, state: Optional[RelayState] = None) -> None:
        """Create a relay processor.

        Args:
            ttl: Total hop budget per message (``>= 1``).
            cache_window: The plan's ``C`` window in rounds; also the
                duplicate-suppression memory horizon. Must satisfy
                ``cache_window >= ttl``.
            state: Shared protocol state (created when ``None``); pass the
                same instance to the :class:`RelayCarryoverTransport`.

        Raises:
            ValueError: If ``ttl < 1`` or ``cache_window < ttl``.
        """
        if ttl < 1:
            raise ValueError("RelayProcessor ttl must be >= 1, got {}.".format(ttl))
        if cache_window < ttl:
            raise ValueError(
                "RelayProcessor requires cache_window >= ttl for correct duplicate "
                "suppression, got cache_window={} < ttl={}.".format(cache_window, ttl)
            )
        self._ttl = ttl
        self._cache_window = cache_window
        self._state = state if state is not None else RelayState()
        self._pending: List[_QueuedForward] = []
        self._last_accepted: List[int] = []

    @property
    def state(self) -> RelayState:
        """The shared cross-step relay protocol state."""
        return self._state

    def initialize(
        self,
        local_observations: Mapping[str, object],
        graph: CommunicationGraph,
        context: Mapping[str, object],
    ) -> object:
        """Create the round-zero state: an empty list of accepted batches.

        Cross-step protocol memory (duplicate cache, carryover queue) is
        deliberately NOT touched here; it survives movement steps and is
        cleared only on episode reset.

        Args:
            local_observations: Ignored; relay transforms nothing.
            graph: The frozen communication graph for this movement step.
            context: Public runtime context (unused).

        Returns:
            The initial round state (an empty list).
        """
        self._pending = []
        self._last_accepted = []
        return []

    def _prune_seen(self, current_round: int) -> None:
        """Forget message ids recorded before the ``C``-round memory horizon."""
        threshold = current_round - self._cache_window
        for ids in self._state.seen.values():
            while ids:
                oldest_id = next(iter(ids))
                if ids[oldest_id] >= threshold:
                    break
                del ids[oldest_id]

    def process_round(
        self,
        state: object,
        inbox: object,
        graph: CommunicationGraph,
        round_index: int,
        context: Mapping[str, object],
    ) -> object:
        """Apply duplicate suppression, app delivery, and forward enqueueing.

        For every delivered copy, in delivery order: an already-seen
        ``message_id`` is dropped before application delivery and forwarding
        (``dropped_duplicate``); a first-seen copy is recorded, delivered to
        the receiver's inbox view (``delivered``), and enqueued for
        forwarding in the next round unless its hop budget is exhausted
        (``dropped_ttl``).

        Args:
            state: The list of previously accepted batches.
            inbox: This round's delivered :class:`EdgeMessageBatch`.
            graph: The frozen communication graph for this movement step.
            round_index: Zero-based round index within the current step.
            context: Public runtime context (``step``/``scheme`` for traces).

        Returns:
            The state with this round's accepted (first-seen) rows appended.
        """
        batches: List[EdgeMessageBatch] = list(state) if isinstance(state, list) else []
        if isinstance(inbox, InboxBatch):
            inbox = inbox.messages
        current_round = self._state.rounds_completed
        self._prune_seen(current_round)
        self._pending = []
        self._last_accepted = []
        if isinstance(inbox, EdgeMessageBatch) and inbox.num_messages > 0:
            accepted = self._process_rows(inbox, graph, round_index, context, current_round)
            self._last_accepted = accepted
            if accepted:
                batches.append(inbox.select(accepted))
        self._state.rounds_completed = current_round + 1
        return batches

    def _process_rows(
        self,
        inbox: EdgeMessageBatch,
        graph: CommunicationGraph,
        round_index: int,
        context: Mapping[str, object],
        current_round: int,
    ) -> List[int]:
        """Process one round's delivered rows; return the accepted row indices."""
        step = _context_int(context, "step", graph.step)
        scheme = _context_scheme(context)
        senders = inbox.sender_index.tolist()
        receivers = inbox.receiver_index.tolist()
        origins = inbox.origin_index.tolist()
        message_ids = inbox.message_id.tolist()
        hops = inbox.metadata["hop_count"].tolist()
        created_steps = inbox.metadata["created_step"].tolist()
        created_rounds = inbox.metadata["created_round"].tolist()
        observation_steps = inbox.metadata["observation_step"].tolist()

        accepted: List[int] = []
        for row in range(inbox.num_messages):
            sender_id = graph.agent_ids[senders[row]]
            receiver_id = graph.agent_ids[receivers[row]]
            origin_id = graph.agent_ids[origins[row]]
            message_id = int(message_ids[row])
            hop = int(hops[row])
            remaining = self._ttl - hop
            if hop == 1:
                # First transmission: the origin has "seen" its own message.
                self._state.seen.setdefault(origin_id, OrderedDict()).setdefault(
                    message_id, current_round
                )
            seen = self._state.seen.setdefault(receiver_id, OrderedDict())
            record_fields = dict(
                step=step,
                round_index=round_index,
                message_id=message_id,
                origin=origin_id,
                sender=sender_id,
                receiver=receiver_id,
                previous_hop=sender_id,
                ttl=remaining,
                scheme=scheme,
            )
            trace = self._state.trace_records
            if message_id in seen:
                trace.append(relay_decision_record(decision="dropped_duplicate", **record_fields))
                continue
            seen[message_id] = current_round
            accepted.append(row)
            trace.append(relay_decision_record(decision="delivered", **record_fields))
            if remaining <= 0:
                trace.append(relay_decision_record(decision="dropped_ttl", **record_fields))
                continue
            self._pending.append(
                _QueuedForward(
                    sender=receiver_id,
                    frame=Frame(
                        payload=inbox.payload[row],
                        origin=origin_id,
                        created_step=int(created_steps[row]),
                        created_round=int(created_rounds[row]),
                        observation_step=int(observation_steps[row]),
                        message_id=message_id,
                        hop_count=hop,
                    ),
                    previous_hop=sender_id,
                )
            )
        return accepted

    def accepted_rows(
        self,
        delivered: EdgeMessageBatch,
        graph: CommunicationGraph,
        round_index: int,
        context: Mapping[str, object],
    ) -> List[int]:
        """Return the first-seen row indices of the round just processed.

        Runtime hook (see :mod:`src.communication.runtime`): called once per
        round, immediately after :meth:`process_round` with the same
        delivered batch. Only these rows enter the receivers' message caches
        and the step's ``delivered_messages`` log, so duplicate-suppressed
        copies (e.g. the broadcast echo back to a previous hop) never reach
        policy-visible delivery surfaces.

        Args:
            delivered: The delivered batch of the round just processed
                (unused; the indices were computed by :meth:`process_round`).
            graph: The frozen communication graph (unused).
            round_index: Zero-based round index (unused).
            context: Public runtime context (unused).

        Returns:
            Accepted row indices of the last processed round, in row order.
        """
        return list(self._last_accepted)

    def emit_round(
        self,
        state: object,
        graph: CommunicationGraph,
        round_index: int,
        context: Mapping[str, object],
    ) -> Dict[str, List[Frame]]:
        """Emit the round's queued forwards for transmission in the next round.

        In the last round of the step, the "next round" belongs to the next
        movement step: the queue is stashed in the shared state and the
        :class:`RelayCarryoverTransport` re-injects it at round 0 of the next
        step (target filtering then uses the next step's graph).

        Args:
            state: Ignored (the queue is per-round processor state).
            graph: The frozen communication graph for this movement step.
            round_index: Zero-based round index within the current step.
            context: Public runtime context (``rounds``/``step``/``scheme``).

        Returns:
            Outboxes for the next round (empty in the last round).
        """
        pending, self._pending = self._pending, []
        if not pending:
            return {}
        rounds = _context_int(context, "rounds", round_index + 1)
        if round_index >= rounds - 1:
            self._state.carryover.extend(pending)
            return {}
        step = _context_int(context, "step", graph.step)
        return _emit_queued(
            pending,
            graph,
            step,
            round_index + 1,
            self._ttl,
            _context_scheme(context),
            self._state.trace_records,
        )

    def finalize(
        self,
        state: object,
        inbox: object,
        graph: CommunicationGraph,
        context: Mapping[str, object],
    ) -> CommunicationResult:
        """Produce the preserved, origin-distinct inbox result for the step.

        Args:
            state: The list of accepted batches from :meth:`process_round`.
            inbox: The final round's inbox (already processed; ignored).
            graph: The frozen communication graph for this movement step.
            context: Public runtime context (unused).

        Returns:
            A :class:`CommunicationResult` with one (possibly empty)
            first-seen :class:`EdgeMessageBatch` per graph agent id in
            ``agent_output`` (the ``preserved-inbox`` contract: distinct
            origin-preserving payloads), the whole-step accepted
            :class:`InboxBatch` under ``diagnostics["inbox"]``, and the
            step's relay decision records under
            ``diagnostics["processor_trace_records"]``.
        """
        batches: List[EdgeMessageBatch] = list(state) if isinstance(state, list) else []
        delivered = concat_edge_message_batches(batches)
        inbox_view = InboxBatch(messages=delivered)

        agent_output: Dict[str, object] = {}
        for index, agent_id in enumerate(graph.agent_ids):
            agent_output[agent_id] = inbox_view.messages_for(index)

        records = self._state.trace_records
        self._state.trace_records = []

        logger.debug(
            "RelayProcessor finalized step {}: {} first-seen delivery(ies), {} carryover",
            graph.step,
            delivered.num_messages,
            len(self._state.carryover),
        )
        return CommunicationResult(
            agent_output=agent_output,
            final_state=None,
            delivered_messages=delivered,
            protocol_state=self._state,
            diagnostics={"inbox": inbox_view, "processor_trace_records": records},
        )
