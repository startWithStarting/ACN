"""Fixed-round communication scheduler (Phase 1).

Runs the ``R`` synchronous communication rounds of one movement step over a
FROZEN graph, per ``docs/communication_implementation_plan.md`` ("Runtime
Model") and ``docs/communication_decision_log.md`` ("Round And Cache Model"):

- The graph is built (or supplied) once per movement step and never changes
  across the step's rounds.
- Rounds are double-buffered: round ``r`` transports the outboxes emitted at
  the end of round ``r - 1`` (round 0 transports the source outboxes derived
  from local observations). A message emitted in round ``r`` is therefore not
  readable — and not re-emittable — until round ``r + 1``, so information
  moves at most one graph hop per round.
- Each agent owns a :class:`~src.communication.types.MessageCache` with a
  ``C``-round window that survives movement-step boundaries.
  ``advance_round`` is called exactly once per round for every agent on the
  graph, including rounds that delivered nothing.

Processors may optionally implement ``emit_round(state, graph, round_index,
context) -> Mapping[str, Sequence[Frame]]`` to emit frames for the *next*
round (the relay extension point). Processors without it — like the direct
processor — emit nothing after round 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import torch

from src.communication.plans import CommunicationPlan
from src.communication.tracing import (
    communication_delivery_records,
    communication_graph_record,
)
from src.communication.transport import Frame, concat_edge_message_batches
from src.communication.types import (
    CommunicationGraph,
    CommunicationResult,
    EdgeMessageBatch,
    MessageCache,
)
from src.utils.logger import get_logger

logger = get_logger("acn.communication")

__all__ = ["CachedDelivery", "CommunicationRuntime"]


@dataclass(frozen=True, eq=False)
class CachedDelivery:
    """Immutable per-delivery record stored in a receiver's message cache.

    Attributes:
        payload: The delivered payload row (1-D tensor, unchanged).
        message_id: Stable message identity assigned by the transport.
        origin: Agent id of the message's original source.
        sender: Agent id of the immediate sender of this copy.
        receiver: Agent id of the receiving agent (the cache owner).
        hop_count: Graph hops traversed by this copy.
        created_step: Movement step of message creation.
        created_round: Communication round of message creation.
        observation_step: Step of the underlying observation (age anchor).
        delivered_round: Round index within the step in which this copy
            arrived.
    """

    payload: torch.Tensor
    message_id: int
    origin: str
    sender: str
    receiver: str
    hop_count: int
    created_step: int
    created_round: int
    observation_step: int
    delivered_round: int


class CommunicationRuntime:
    """Per-environment scheduler owning the per-agent message caches.

    One runtime instance lives as long as its environment. Movement steps are
    executed with :meth:`run_step`; :meth:`reset` clears all message caches
    (episode reset). Caches are created lazily with the plan's
    ``cache_window`` and persist across movement steps, implementing the
    cross-step ``C``-round window of the decision log.
    """

    def __init__(
        self,
        plan: CommunicationPlan,
        scheme_name: Optional[str] = None,
        collect_traces: bool = True,
    ) -> None:
        """Create a runtime for one compiled communication plan.

        Args:
            plan: The compiled scheme (topology, transport, processor,
                rounds ``R``, cache window ``C``).
            scheme_name: Scheme name recorded on trace records (the plan
                itself is anonymous).
            collect_traces: Whether :meth:`run_step` builds trace records
                into ``diagnostics["trace_records"]``.
        """
        self._plan = plan
        self._scheme_name = scheme_name
        self._collect_traces = collect_traces
        self._caches: Dict[str, MessageCache] = {}

    @property
    def plan(self) -> CommunicationPlan:
        """The compiled communication plan this runtime executes."""
        return self._plan

    @property
    def caches(self) -> Mapping[str, MessageCache]:
        """Read-only view of the per-agent message caches created so far."""
        return dict(self._caches)

    def cache_for(self, agent_id: str) -> MessageCache:
        """Return (creating lazily) the message cache of one agent.

        Args:
            agent_id: The agent's id.

        Returns:
            The agent's :class:`MessageCache` with the plan's ``C`` window.
        """
        cache = self._caches.get(agent_id)
        if cache is None:
            cache = MessageCache(self._plan.cache_window)
            self._caches[agent_id] = cache
        return cache

    def reset(self) -> None:
        """Clear all message caches and transport identity state (episode reset)."""
        for cache in self._caches.values():
            cache.clear()
        self._caches.clear()
        reset_ids = getattr(self._plan.transport, "reset_message_ids", None)
        if callable(reset_ids):
            reset_ids()
        logger.debug("CommunicationRuntime reset: caches cleared")

    def _insert_deliveries(
        self,
        delivered: EdgeMessageBatch,
        graph: CommunicationGraph,
        step: int,
        round_index: int,
    ) -> None:
        """Insert one round's deliveries into the receivers' caches, in order."""
        if delivered.num_messages == 0:
            return
        receivers = delivered.receiver_index.tolist()
        senders = delivered.sender_index.tolist()
        origins = delivered.origin_index.tolist()
        message_ids = delivered.message_id.tolist()
        metadata = {key: value.tolist() for key, value in delivered.metadata.items()}

        def _meta(key: str, row: int, default: int) -> int:
            values = metadata.get(key)
            return int(values[row]) if values is not None else default

        for row in range(delivered.num_messages):
            receiver_id = graph.agent_ids[receivers[row]]
            observation_step = _meta("observation_step", row, step)
            entry = CachedDelivery(
                payload=delivered.payload[row],
                message_id=int(message_ids[row]),
                origin=graph.agent_ids[origins[row]],
                sender=graph.agent_ids[senders[row]],
                receiver=receiver_id,
                hop_count=_meta("hop_count", row, 1),
                created_step=_meta("created_step", row, step),
                created_round=_meta("created_round", row, round_index),
                observation_step=observation_step,
                delivered_round=round_index,
            )
            self.cache_for(receiver_id).insert(
                message=entry,
                emitted_step=step,
                emitted_round=round_index,
                observation_step=observation_step,
            )

    def run_step(
        self,
        local_observations: Mapping[str, object],
        outboxes: Mapping[str, Sequence[Frame]],
        step: int,
        agents: Optional[Sequence[object]] = None,
        graph: Optional[CommunicationGraph] = None,
        context: Optional[Mapping[str, object]] = None,
    ) -> CommunicationResult:
        """Run the ``R`` communication rounds of one movement step.

        Args:
            local_observations: Policy-visible local observation per agent id
                (passed to the processor's ``initialize``).
            outboxes: Source outboxes for round 0, typically produced by a
                :class:`~src.communication.sources.MessageSource` from the
                same ``local_observations`` (``u_i(t) = g_comm(o_i(t))``).
            step: Current movement-step index.
            agents: Agent objects used to build the frozen graph via the
                plan's topology. Required when ``graph`` is ``None``.
            graph: Optional pre-built frozen graph (skips topology).
            context: Optional extra public context merged into the round
                context (reserved keys ``step``, ``rounds``, ``cache_window``,
                and ``scheme`` are set by the runtime).

        Returns:
            The :class:`CommunicationResult` produced by the processor's
            ``finalize``. The runtime overwrites ``delivered_messages`` with
            its authoritative concatenation of every round's deliveries and
            adds ``diagnostics`` entries: ``caches`` (per-agent
            :class:`MessageCache` handles), ``graph``, ``rounds``, and — when
            trace collection is enabled — ``trace_records``.

        Raises:
            ValueError: If neither ``agents`` nor ``graph`` is provided.
        """
        plan = self._plan
        if graph is None:
            if agents is None:
                raise ValueError(
                    "run_step needs either a pre-built graph or the agents to build one from."
                )
            graph = plan.topology.build(agents, step)

        run_context: Dict[str, object] = dict(context or {})
        run_context["step"] = step
        run_context["rounds"] = plan.rounds
        run_context["cache_window"] = plan.cache_window
        run_context["scheme"] = self._scheme_name

        transport_state = plan.transport.reset(graph, run_context)
        state = plan.processor.initialize(local_observations, graph, run_context)

        trace_records: List[Dict[str, object]] = []
        if self._collect_traces:
            trace_records.append(communication_graph_record(graph, self._scheme_name))

        current_outboxes: Mapping[str, Sequence[Frame]] = outboxes
        delivered_batches: List[EdgeMessageBatch] = []
        last_delivered: EdgeMessageBatch = EdgeMessageBatch.empty()
        for round_index in range(plan.rounds):
            delivered = plan.transport.transmit_round(
                current_outboxes, graph, transport_state, round_index, run_context
            )
            delivered_batches.append(delivered)
            last_delivered = delivered

            # Round-local deliveries enter the caches before the round is
            # marked complete below.
            self._insert_deliveries(delivered, graph, step, round_index)

            state = plan.processor.process_round(
                state, delivered, graph, round_index, run_context
            )

            # Double buffering: emissions computed from post-round state are
            # transported in round r + 1, never within round r.
            emit_round = getattr(plan.processor, "emit_round", None)
            if callable(emit_round):
                current_outboxes = emit_round(state, graph, round_index, run_context) or {}
            else:
                current_outboxes = {}

            # Every agent's cache window advances exactly once per round,
            # whether or not the round delivered anything to that agent.
            for agent_id in graph.agent_ids:
                self.cache_for(agent_id).advance_round(step, round_index)

            if self._collect_traces:
                trace_records.extend(
                    communication_delivery_records(delivered, graph, step, self._scheme_name)
                )

        result = plan.processor.finalize(state, last_delivered, graph, run_context)

        # The runtime's transport log is authoritative for the step's
        # delivered messages, independent of processor bookkeeping.
        result.delivered_messages = concat_edge_message_batches(delivered_batches)

        result.diagnostics["caches"] = {
            agent_id: self.cache_for(agent_id) for agent_id in graph.agent_ids
        }
        result.diagnostics["graph"] = graph
        result.diagnostics["rounds"] = plan.rounds
        if self._collect_traces:
            result.diagnostics["trace_records"] = trace_records

        logger.debug(
            "Communication step {} finished: {} round(s), {} delivered message(s)",
            step,
            plan.rounds,
            result.delivered_messages.num_messages,
        )
        return result
