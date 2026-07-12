"""Inbox-preserving direct processor (Phase 1).

Implements the ``one_hop_direct`` receiver semantics from
``docs/communication_implementation_plan.md`` ("Direct Processing" /
"Inbox-Preserving Processing"): a delivered frame is stored into the
receiver's view and never re-emitted (no relay). Distinct messages stay
distinct — the processor performs no sum, mean, or any other aggregation;
any size-consistent encoding happens in the receiving policy after delivery.
"""

from __future__ import annotations

from typing import Dict, List, Mapping

from src.communication.transport import Frame, concat_edge_message_batches
from src.communication.types import (
    CommunicationGraph,
    CommunicationResult,
    EdgeMessageBatch,
    InboxBatch,
)
from src.utils.logger import get_logger

logger = get_logger("acn.communication")

__all__ = ["DirectProcessor"]


class DirectProcessor:
    """Store-without-forwarding processor with a preserved distinct inbox.

    Round state is the ordered list of the step's delivered
    :class:`EdgeMessageBatch` objects. ``finalize`` returns a
    :class:`CommunicationResult` whose ``agent_output`` maps *every* agent id
    on the graph to that receiver's own :class:`EdgeMessageBatch` (possibly
    empty) in delivery order — distinct incoming payloads plus sender
    identity, per the ``preserved-inbox`` output contract. The whole-step
    :class:`InboxBatch` view is exposed under ``diagnostics["inbox"]``.
    """

    def initialize(
        self,
        local_observations: Mapping[str, object],
        graph: CommunicationGraph,
        context: Mapping[str, object],
    ) -> object:
        """Create the round-zero state: an empty list of delivered batches.

        Args:
            local_observations: Ignored; the direct processor transforms
                nothing and reads no observations.
            graph: The frozen communication graph for this movement step.
            context: Public runtime context (unused).

        Returns:
            The initial round state (an empty list).
        """
        return []

    def process_round(
        self,
        state: object,
        inbox: object,
        graph: CommunicationGraph,
        round_index: int,
        context: Mapping[str, object],
    ) -> object:
        """Store the round's delivered messages into the state, unchanged.

        Args:
            state: The list of previously delivered batches.
            inbox: This round's delivered :class:`EdgeMessageBatch`.
            graph: The frozen communication graph for this movement step.
            round_index: Zero-based round index (unused).
            context: Public runtime context (unused).

        Returns:
            The state with this round's batch appended.
        """
        batches: List[EdgeMessageBatch] = list(state) if state is not None else []
        if isinstance(inbox, InboxBatch):
            inbox = inbox.messages
        if isinstance(inbox, EdgeMessageBatch) and inbox.num_messages > 0:
            batches.append(inbox)
        return batches

    def emit_round(
        self,
        state: object,
        graph: CommunicationGraph,
        round_index: int,
        context: Mapping[str, object],
    ) -> Dict[str, List[Frame]]:
        """Return no outboxes: the direct processor never re-emits (no relay).

        Args:
            state: Ignored.
            graph: Ignored.
            round_index: Ignored.
            context: Ignored.

        Returns:
            An empty mapping.
        """
        return {}

    def finalize(
        self,
        state: object,
        inbox: object,
        graph: CommunicationGraph,
        context: Mapping[str, object],
    ) -> CommunicationResult:
        """Produce the preserved-inbox result after the final round.

        Args:
            state: The list of delivered batches accumulated by
                :meth:`process_round`.
            inbox: The final round's inbox (already stored; ignored).
            graph: The frozen communication graph for this movement step.
            context: Public runtime context (unused).

        Returns:
            A :class:`CommunicationResult` with one (possibly empty)
            :class:`EdgeMessageBatch` per graph agent id in ``agent_output``,
            the concatenated step deliveries in ``delivered_messages``, and
            the whole-step :class:`InboxBatch` in ``diagnostics["inbox"]``.
        """
        batches: List[EdgeMessageBatch] = list(state) if state is not None else []
        delivered = concat_edge_message_batches(batches)
        inbox_view = InboxBatch(messages=delivered)

        agent_output: Dict[str, object] = {}
        for index, agent_id in enumerate(graph.agent_ids):
            agent_output[agent_id] = inbox_view.messages_for(index)

        logger.debug(
            "DirectProcessor finalized step {}: {} delivered message(s) across {} agent(s)",
            graph.step,
            delivered.num_messages,
            graph.num_agents,
        )
        return CommunicationResult(
            agent_output=agent_output,
            final_state=None,
            delivered_messages=delivered,
            protocol_state=None,
            diagnostics={"inbox": inbox_view},
        )
