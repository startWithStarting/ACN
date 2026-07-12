"""Common synchronous slotted radius transport (Phase 1).

Implements the ``RoundTransport`` protocol from
``docs/communication_implementation_plan.md`` ("Common Transport Protocol"):
opaque frames move at most one hop per communication round over the valid
edges of the frozen graph, with broadcast delivery and no loss, queues,
fragmentation, retransmission, or cost. Transport metadata (message id,
origin, immediate sender, creation step/round, hop count) is kept separate
from the application payload at all times.

Determinism contract:

- Message ids are allocated from a monotone per-instance counter, iterating
  senders in graph node-index order and each sender's outbox in sequence
  order, so identical runs allocate identical ids. The counter is *not* reset
  by :meth:`SlottedRadiusTransport.reset` (which is per movement step) so ids
  stay unique across steps; call :meth:`SlottedRadiusTransport.reset_message_ids`
  on episode reset.
- Delivered rows are ordered by (graph edge order, outbox frame order), and
  graph edges are themselves sorted by ``(source, receiver)`` by the topology
  builder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from src.communication.types import CommunicationGraph, EdgeMessageBatch
from src.utils.logger import get_logger

logger = get_logger("acn.communication")

__all__ = ["Frame", "SlottedRadiusTransport", "concat_edge_message_batches"]


def concat_edge_message_batches(batches: Sequence[EdgeMessageBatch]) -> EdgeMessageBatch:
    """Concatenate message batches, preserving batch order and row order.

    Empty batches are skipped, so a mix of empty and non-empty rounds
    concatenates cleanly regardless of the empty batches' payload dimension.

    Args:
        batches: Batches in delivery order (e.g. one per communication round).

    Returns:
        One :class:`EdgeMessageBatch` with all rows in the given order. When
        every batch is empty, an empty batch (payload dimension of the first
        input, or 0) is returned.

    Raises:
        ValueError: If non-empty batches disagree on payload dimension or
            metadata keys.
    """
    non_empty = [batch for batch in batches if batch.num_messages > 0]
    if not non_empty:
        payload_dim = batches[0].payload.size(1) if batches and batches[0].payload.dim() > 1 else 0
        return EdgeMessageBatch.empty(payload_dim=payload_dim)
    if len(non_empty) == 1:
        return non_empty[0]

    first_keys = set(non_empty[0].metadata.keys())
    for batch in non_empty[1:]:
        if set(batch.metadata.keys()) != first_keys:
            raise ValueError(
                "Cannot concatenate message batches with different metadata keys: "
                "{} vs {}.".format(sorted(first_keys), sorted(batch.metadata.keys()))
            )
    return EdgeMessageBatch(
        payload=torch.cat([batch.payload for batch in non_empty], dim=0),
        sender_index=torch.cat([batch.sender_index for batch in non_empty], dim=0),
        receiver_index=torch.cat([batch.receiver_index for batch in non_empty], dim=0),
        origin_index=torch.cat([batch.origin_index for batch in non_empty], dim=0),
        message_id=torch.cat([batch.message_id for batch in non_empty], dim=0),
        round_index=torch.cat([batch.round_index for batch in non_empty], dim=0),
        metadata={
            key: torch.cat([batch.metadata[key] for batch in non_empty], dim=0)
            for key in sorted(first_keys)
        },
    )


@dataclass(frozen=True, eq=False)
class Frame:
    """One opaque outgoing frame handed to the transport.

    The payload is the only application-visible content; everything else is
    transport metadata. ``privileged`` may carry simulator-only evaluation
    data (e.g. the ground-truth opponent id behind a contact report) and must
    never be exposed to policies.

    Equality/hashing are identity-based (``eq=False``) because ``payload`` is
    a tensor (see the ``src.communication.types`` module docstring).

    Attributes:
        payload: 1-D payload tensor (lists/tuples are coerced to ``float32``).
        origin: Agent id of the original source of the message.
        created_step: Movement step at which the message was created.
        created_round: Communication round at which the message was created.
        observation_step: Step of the underlying observation; defaults to
            ``created_step`` (source message) and stays fixed under relay.
        message_id: Stable message identity. ``None`` means "not yet
            assigned"; the transport allocates a fresh id on first
            transmission. Re-emitted (relayed) frames keep their id.
        hop_count: Graph hops already traversed by this frame (``0`` for a
            source frame). Deliveries report ``hop_count + 1``.
        privileged: Simulator-only metadata for tracing/evaluation; never
            policy-visible and never transported inside the payload.
    """

    payload: torch.Tensor
    origin: str
    created_step: int
    created_round: int = 0
    observation_step: Optional[int] = None
    message_id: Optional[int] = None
    hop_count: int = 0
    privileged: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        payload = self.payload
        if not isinstance(payload, torch.Tensor):
            payload = torch.as_tensor(payload, dtype=torch.float32)
            object.__setattr__(self, "payload", payload)
        if payload.dim() != 1:
            raise ValueError(
                "Frame payload must be a 1-D tensor, got shape {}.".format(tuple(payload.shape))
            )
        if not isinstance(self.origin, str) or not self.origin:
            raise ValueError("Frame origin must be a non-empty agent id string.")
        if self.created_step < 0 or self.created_round < 0 or self.hop_count < 0:
            raise ValueError(
                "Frame created_step, created_round, and hop_count must be >= 0 "
                "(got {}, {}, {}).".format(self.created_step, self.created_round, self.hop_count)
            )
        if self.observation_step is None:
            object.__setattr__(self, "observation_step", self.created_step)
        if self.message_id is not None and self.message_id < 0:
            raise ValueError("Frame message_id must be >= 0 when set.")


class SlottedRadiusTransport:
    """Synchronous slotted broadcast transport over the frozen radius graph.

    Each round, every frame in a sender's outbox is copied to *every* valid
    outgoing edge of that sender (broadcast delivery) and delivered within the
    same round. Frames never travel more than one hop per round; multi-hop
    propagation requires a processor that re-emits in a later round. There is
    no loss, queueing, bandwidth limit, or cost.
    """

    def __init__(self) -> None:
        """Create a transport with a fresh message-id counter."""
        self._next_message_id = 0

    def reset_message_ids(self) -> None:
        """Reset the monotone message-id counter (episode reset only)."""
        self._next_message_id = 0

    def reset(self, graph: CommunicationGraph, context: Mapping[str, object]) -> object:
        """Reset per-step transport state for a freshly frozen graph.

        The slotted transport keeps no queues or per-step state, and message
        identity allocation is owned by the instance so ids remain unique
        across movement steps.

        Args:
            graph: The frozen communication graph for this movement step.
            context: Public runtime context (unused).

        Returns:
            ``None`` (no per-step state).
        """
        return None

    def transmit_round(
        self,
        outboxes: Mapping[str, Sequence[object]],
        graph: CommunicationGraph,
        transport_state: object,
        round_index: int,
        context: Mapping[str, object],
    ) -> EdgeMessageBatch:
        """Deliver one synchronous round of frames over the frozen graph.

        Args:
            outboxes: Zero or more outgoing :class:`Frame` objects per sending
                agent id. Every key must name a node of ``graph``.
            graph: The frozen communication graph for this movement step.
            transport_state: Ignored (the slotted transport is stateless
                per step).
            round_index: Zero-based round index within the current step.
            context: Public runtime context (unused).

        Returns:
            An :class:`EdgeMessageBatch` with one row per (frame, edge)
            delivery. Payloads are transported unchanged and remain distinct;
            transport metadata is carried in the batch's index fields and in
            the ``created_step``, ``created_round``, ``observation_step``,
            and ``hop_count`` metadata tensors.

        Raises:
            ValueError: If an outbox key or a frame origin is not on the
                graph, a frame is not a :class:`Frame`, or payload dimensions
                are inconsistent within the round.
        """
        if round_index < 0:
            raise ValueError("round_index must be >= 0, got {}.".format(round_index))
        index_of = {agent_id: i for i, agent_id in enumerate(graph.agent_ids)}

        unknown = sorted(set(outboxes.keys()) - set(index_of.keys()))
        if unknown:
            raise ValueError(
                "Outbox sender(s) {} are not nodes of the communication graph "
                "(agents: {}). Inactive agents are excluded from the graph and "
                "cannot transmit.".format(unknown, list(graph.agent_ids))
            )

        # Deterministic id allocation: senders in node-index order, frames in
        # outbox sequence order, independent of the outbox mapping order.
        assigned: Dict[int, List[Tuple[Frame, int]]] = {}
        payload_dim: Optional[int] = None
        for agent_id in graph.agent_ids:
            frames = outboxes.get(agent_id)
            if not frames:
                continue
            sender_idx = index_of[agent_id]
            rows: List[Tuple[Frame, int]] = []
            for frame in frames:
                if not isinstance(frame, Frame):
                    raise ValueError(
                        "Outbox of '{}' contains a non-Frame object of type {}.".format(
                            agent_id, type(frame).__name__
                        )
                    )
                if frame.origin not in index_of:
                    raise ValueError(
                        "Frame origin '{}' (sent by '{}') is not on the communication "
                        "graph.".format(frame.origin, agent_id)
                    )
                dim = frame.payload.size(0)
                if payload_dim is None:
                    payload_dim = dim
                elif dim != payload_dim:
                    raise ValueError(
                        "Inconsistent payload dimensions within one round: got {} and {}. "
                        "All frames of a round must share one payload dimension.".format(
                            payload_dim, dim
                        )
                    )
                if frame.message_id is not None:
                    message_id = frame.message_id
                else:
                    message_id = self._next_message_id
                    self._next_message_id += 1
                rows.append((frame, message_id))
            assigned[sender_idx] = rows

        if not assigned or graph.num_edges == 0:
            return EdgeMessageBatch.empty(payload_dim=payload_dim or 0)

        payloads: List[torch.Tensor] = []
        sender_index: List[int] = []
        receiver_index: List[int] = []
        origin_index: List[int] = []
        message_ids: List[int] = []
        created_step: List[int] = []
        created_round: List[int] = []
        observation_step: List[int] = []
        hop_count: List[int] = []

        edge_sources = graph.edge_index[0].tolist()
        edge_receivers = graph.edge_index[1].tolist()
        for source, receiver in zip(edge_sources, edge_receivers):
            for frame, message_id in assigned.get(source, ()):
                payloads.append(frame.payload)
                sender_index.append(source)
                receiver_index.append(receiver)
                origin_index.append(index_of[frame.origin])
                message_ids.append(message_id)
                created_step.append(frame.created_step)
                created_round.append(frame.created_round)
                # observation_step is filled in __post_init__, never None here.
                observation_step.append(int(frame.observation_step))  # type: ignore[arg-type]
                hop_count.append(frame.hop_count + 1)

        if not payloads:
            return EdgeMessageBatch.empty(payload_dim=payload_dim or 0)

        def _long(values: List[int]) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.long)

        delivered = EdgeMessageBatch(
            payload=torch.stack(payloads, dim=0),
            sender_index=_long(sender_index),
            receiver_index=_long(receiver_index),
            origin_index=_long(origin_index),
            message_id=_long(message_ids),
            round_index=torch.full((len(payloads),), round_index, dtype=torch.long),
            metadata={
                "created_step": _long(created_step),
                "created_round": _long(created_round),
                "observation_step": _long(observation_step),
                "hop_count": _long(hop_count),
            },
        )
        logger.debug(
            "SlottedRadiusTransport delivered {} frame(s) in round {} at step {}",
            delivered.num_messages,
            round_index,
            graph.step,
        )
        return delivered
