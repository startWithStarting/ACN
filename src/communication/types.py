"""Core data types for the ACN communication module (Phase 0 contract).

This module defines the passive data contracts shared by the communication
topology, transport, processor, and runtime layers described in
``docs/communication_implementation_plan.md`` ("Core Data Types") and
``docs/communication_decision_log.md`` ("Round And Cache Model"). It contains
no runtime behavior: environments, agents, and entry points do not use it yet.

Step and round semantics (summary of the agreed model):

- Each movement step ``t`` contains a fixed number ``R`` of synchronous
  communication rounds executed over a same-team radius graph that is frozen
  for the whole step. One round permits at most one physical graph hop.
  Movement is selected after the ``R`` rounds, from post-communication state.
- ``C`` is the message cache: a sliding window over the last ``C`` completed
  rounds. The window spans movement-step boundaries, so a round may read
  messages emitted in previous steps. When the window advances, the oldest
  round is evicted first; equal-age entries are evicted in delivery-sequence
  order. The cache is distinct from a processor's intrinsic round-to-round
  hidden state, which exists even at ``C = 0``.
- Relay correctness requires ``C >= TTL``; the scheme compiler must reject
  relay configurations that violate it (see :meth:`MessageCache.supports_ttl`).
- Message payloads are immutable. A message's policy-visible age is derived at
  read time as ``current_step - observation_step``; stored messages are never
  rewritten.

Identity boundary:

- ``CommunicationGraph.agent_ids`` are privileged simulator identities kept
  for tracing and evaluation. They are runtime metadata, not automatically
  model features. Policy-visible identity is configured separately; opponent
  IDs must never silently enter policy-visible fields.

Determinism contract:

- Graph builders must produce edges in a deterministic order for a fixed
  simulation state and seed. ``CommunicationGraph`` preserves the edge order
  it was constructed with and never reorders edges; all derived structures
  (inbox grouping, dense conversion, cache eviction) are order-stable.

Equality semantics:

- Every tensor-bearing dataclass in this module declares ``eq=False``:
  ``==`` is identity comparison and hashing is identity-based. Field-wise
  dataclass equality would compare tensors with ``==`` inside a tuple and
  raise instead of returning a bool, so value comparison must be done
  explicitly per field (e.g. with ``torch.equal``).

All tensor fields tolerate empty (0-length) tensors and support ``.to(device)``
without hidden NumPy conversion. Empty and disconnected graphs are valid.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch

from src.utils.logger import get_logger

logger = get_logger("acn.communication")

__all__ = [
    "BROADCAST_DESTINATION",
    "NO_PREVIOUS_HOP",
    "CommunicationGraph",
    "EdgeMessageBatch",
    "InboxBatch",
    "PacketBatch",
    "CommunicationResult",
    "CacheEntry",
    "MessageCache",
]

Device = Union[str, torch.device]

#: Sentinel value in :attr:`PacketBatch.destination_index` marking a broadcast packet.
BROADCAST_DESTINATION: int = -1

#: Sentinel value in :attr:`PacketBatch.previous_hop_index` for packets that have not
#: yet traversed any edge (origin-emitted, first transmission pending).
NO_PREVIOUS_HOP: int = -1


def _check_tensor(name: str, value: object) -> torch.Tensor:
    """Assert that ``value`` is a torch.Tensor and return it."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    return value


def _check_long_vector(
    name: str,
    tensor: object,
    length: int,
    minimum: Optional[int] = None,
) -> None:
    """Validate a 1-D torch.long tensor of a given length with an optional minimum value."""
    checked = _check_tensor(name, tensor)
    if checked.dtype != torch.long:
        raise ValueError(f"{name} must have dtype torch.long, got {checked.dtype}")
    if checked.dim() != 1:
        raise ValueError(f"{name} must be 1-D, got shape {tuple(checked.shape)}")
    if checked.size(0) != length:
        raise ValueError(f"{name} must have length {length}, got {checked.size(0)}")
    if minimum is not None and checked.numel() > 0 and int(checked.min().item()) < minimum:
        raise ValueError(f"{name} must have values >= {minimum}, got min {int(checked.min())}")


def _empty_long(device: Optional[Device] = None) -> torch.Tensor:
    """Return an empty 1-D torch.long tensor on ``device``."""
    return torch.zeros(0, dtype=torch.long, device=device)


def _as_index_tensor(index: Union[torch.Tensor, Sequence[int]], device: torch.device) -> torch.Tensor:
    """Coerce ``index`` to a 1-D torch.long tensor on ``device``.

    Boolean masks and floating-point tensors are rejected rather than silently
    coerced: a mask cast to 0/1 indices selects the wrong rows and a float
    index would be truncated.
    """
    if isinstance(index, torch.Tensor):
        if index.dtype == torch.bool:
            raise TypeError(
                "index must be an integer tensor, got dtype torch.bool; convert a mask "
                "to indices with mask.nonzero(as_tuple=False).flatten()"
            )
        if index.is_floating_point():
            raise TypeError(f"index must be an integer tensor, got dtype {index.dtype}")
    tensor = torch.as_tensor(index, dtype=torch.long, device=device)
    if tensor.dim() == 0:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 1:
        raise ValueError(f"index must be 1-D, got shape {tuple(tensor.shape)}")
    return tensor


@dataclass(frozen=True, eq=False)
class CommunicationGraph:
    """Frozen snapshot of the same-team communication topology for one movement step.

    The graph is built once per movement step and frozen across all ``R``
    communication rounds of that step. Only same-team, radius-valid directed
    edges appear; self-edges are configurable by the builder and disabled by
    default for transport.

    Determinism: the edge order given at construction is preserved verbatim.
    Builders must emit edges deterministically for a fixed state and seed;
    this class never reorders them.

    Attributes:
        agent_ids: Privileged simulator identities, index-aligned with node
            indices ``0..N-1``. Runtime metadata for tracing/evaluation, not
            automatically model features.
        edge_index: ``torch.long`` tensor of shape ``[2, E]``; row 0 is the
            source (sender) node index, row 1 the receiver node index.
        edge_distance: Tensor of shape ``[E]`` with the sender-receiver
            distance for each edge.
        edge_features: Optional tensor of shape ``[E, F_e]`` with additional
            per-edge features, or ``None``.
        team_index: ``torch.long`` tensor of shape ``[N]`` with each agent's
            team index.
        step: Movement step at which the snapshot was frozen.
    """

    agent_ids: Tuple[str, ...]
    edge_index: torch.Tensor
    edge_distance: torch.Tensor
    edge_features: Optional[torch.Tensor]
    team_index: torch.Tensor
    step: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_ids", tuple(self.agent_ids))
        num_agents = len(self.agent_ids)

        edge_index = _check_tensor("edge_index", self.edge_index)
        if edge_index.dtype != torch.long:
            raise ValueError(f"edge_index must have dtype torch.long, got {edge_index.dtype}")
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")
        num_edges = edge_index.size(1)

        edge_distance = _check_tensor("edge_distance", self.edge_distance)
        if edge_distance.dim() != 1 or edge_distance.size(0) != num_edges:
            raise ValueError(
                f"edge_distance must have shape [{num_edges}], got {tuple(edge_distance.shape)}"
            )
        if num_edges > 0 and float(edge_distance.min().item()) < 0.0:
            raise ValueError("edge_distance must be non-negative")

        if self.edge_features is not None:
            edge_features = _check_tensor("edge_features", self.edge_features)
            if edge_features.dim() != 2 or edge_features.size(0) != num_edges:
                raise ValueError(
                    f"edge_features must have shape [{num_edges}, F_e], "
                    f"got {tuple(edge_features.shape)}"
                )

        _check_long_vector("team_index", self.team_index, num_agents, minimum=0)

        if num_edges > 0:
            lo = int(edge_index.min().item())
            hi = int(edge_index.max().item())
            if lo < 0 or hi >= num_agents:
                raise ValueError(
                    f"edge_index values must lie in [0, {num_agents}), got range [{lo}, {hi}]"
                )

        if self.step < 0:
            raise ValueError(f"step must be >= 0, got {self.step}")

    @property
    def num_agents(self) -> int:
        """Number of nodes ``N`` in the snapshot."""
        return len(self.agent_ids)

    @property
    def num_edges(self) -> int:
        """Number of directed edges ``E`` in the snapshot."""
        return self.edge_index.size(1)

    def to(self, device: Device) -> CommunicationGraph:
        """Return a copy of the graph with all tensor fields moved to ``device``."""
        return CommunicationGraph(
            agent_ids=self.agent_ids,
            edge_index=self.edge_index.to(device),
            edge_distance=self.edge_distance.to(device),
            edge_features=None if self.edge_features is None else self.edge_features.to(device),
            team_index=self.team_index.to(device),
            step=self.step,
        )

    @classmethod
    def empty(
        cls,
        agent_ids: Sequence[str] = (),
        team_index: Optional[Union[torch.Tensor, Sequence[int]]] = None,
        step: int = 0,
        device: Optional[Device] = None,
    ) -> CommunicationGraph:
        """Build a valid graph with zero edges (disconnected or agentless).

        Args:
            agent_ids: Simulator identities for the ``N`` nodes; may be empty.
            team_index: Per-agent team indices; defaults to all zeros.
            step: Movement step of the snapshot.
            device: Device for the tensor fields.

        Returns:
            A ``CommunicationGraph`` with ``E == 0``.
        """
        num_agents = len(agent_ids)
        if team_index is None:
            team = torch.zeros(num_agents, dtype=torch.long, device=device)
        else:
            team = torch.as_tensor(team_index, dtype=torch.long, device=device)
        return cls(
            agent_ids=tuple(agent_ids),
            edge_index=torch.zeros((2, 0), dtype=torch.long, device=device),
            edge_distance=torch.zeros(0, device=device),
            edge_features=None,
            team_index=team,
            step=step,
        )


@dataclass(eq=False)
class EdgeMessageBatch:
    """Batch of distinct delivered messages, one row per (message, edge) delivery.

    This is the core delivery representation. Messages remain distinct until a
    processor explicitly aggregates them; the transport layer never merges or
    averages payloads.

    Attributes:
        payload: Tensor of shape ``[M, D]`` (or a structured payload whose
            leading dimension is ``M``). Payloads are immutable by contract.
        sender_index: ``torch.long`` ``[M]``; immediate sender node index.
        receiver_index: ``torch.long`` ``[M]``; receiver node index.
        origin_index: ``torch.long`` ``[M]``; original source node index
            (differs from ``sender_index`` under relay).
        message_id: ``torch.long`` ``[M]``; stable message identity.
        round_index: ``torch.long`` ``[M]``; communication round of delivery.
        metadata: Extra per-message tensors, each with leading dimension ``M``.
            Protocol metadata here is privileged unless a scheme explicitly
            selects it as policy-visible.
    """

    payload: torch.Tensor
    sender_index: torch.Tensor
    receiver_index: torch.Tensor
    origin_index: torch.Tensor
    message_id: torch.Tensor
    round_index: torch.Tensor
    metadata: Dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        payload = _check_tensor("payload", self.payload)
        if payload.dim() < 1:
            raise ValueError("payload must have at least 1 dimension [M, ...]")
        num_messages = payload.size(0)

        _check_long_vector("sender_index", self.sender_index, num_messages, minimum=0)
        _check_long_vector("receiver_index", self.receiver_index, num_messages, minimum=0)
        _check_long_vector("origin_index", self.origin_index, num_messages, minimum=0)
        _check_long_vector("message_id", self.message_id, num_messages, minimum=0)
        _check_long_vector("round_index", self.round_index, num_messages, minimum=0)

        for key, value in self.metadata.items():
            checked = _check_tensor(f"metadata[{key!r}]", value)
            if checked.dim() < 1 or checked.size(0) != num_messages:
                raise ValueError(
                    f"metadata[{key!r}] must have leading dimension {num_messages}, "
                    f"got shape {tuple(checked.shape)}"
                )

    @property
    def num_messages(self) -> int:
        """Number of message rows ``M`` in the batch."""
        return self.payload.size(0)

    def select(self, index: Union[torch.Tensor, Sequence[int]]) -> EdgeMessageBatch:
        """Return a new batch containing the rows selected by ``index``.

        Row order follows ``index`` order, so selection is deterministic and
        preserves delivery order when ``index`` is sorted.

        Args:
            index: 1-D integer tensor or sequence of row indices.

        Returns:
            A new ``EdgeMessageBatch`` with the selected rows (metadata included).
        """
        idx = _as_index_tensor(index, self.payload.device)
        return EdgeMessageBatch(
            payload=self.payload[idx],
            sender_index=self.sender_index[idx],
            receiver_index=self.receiver_index[idx],
            origin_index=self.origin_index[idx],
            message_id=self.message_id[idx],
            round_index=self.round_index[idx],
            metadata={key: value[idx] for key, value in self.metadata.items()},
        )

    def to(self, device: Device) -> EdgeMessageBatch:
        """Return a copy of the batch with all tensor fields moved to ``device``."""
        return EdgeMessageBatch(
            payload=self.payload.to(device),
            sender_index=self.sender_index.to(device),
            receiver_index=self.receiver_index.to(device),
            origin_index=self.origin_index.to(device),
            message_id=self.message_id.to(device),
            round_index=self.round_index.to(device),
            metadata={key: value.to(device) for key, value in self.metadata.items()},
        )

    @classmethod
    def empty(
        cls,
        payload_dim: int = 0,
        device: Optional[Device] = None,
        payload_dtype: torch.dtype = torch.float32,
    ) -> EdgeMessageBatch:
        """Build a valid zero-message batch with payload shape ``[0, payload_dim]``."""
        return cls(
            payload=torch.zeros((0, payload_dim), dtype=payload_dtype, device=device),
            sender_index=_empty_long(device),
            receiver_index=_empty_long(device),
            origin_index=_empty_long(device),
            message_id=_empty_long(device),
            round_index=_empty_long(device),
            metadata={},
        )


@dataclass(eq=False)
class InboxBatch:
    """Receiver-oriented view over an :class:`EdgeMessageBatch`.

    The variable-size inbox is the canonical delivery result. Fixed-size
    conversion (:meth:`to_dense`) is performed by the receiving policy or its
    model adapter, not by the transport layer. Grouping never changes message
    identity or order: within each receiver, messages keep their batch order.

    Attributes:
        messages: The underlying distinct-message batch.
        receiver_ptr: Optional CSR-style ``torch.long`` pointer of shape
            ``[num_receivers + 1]``. When present, ``messages`` must be sorted
            and grouped by receiver so rows ``receiver_ptr[i]:receiver_ptr[i+1]``
            are exactly receiver ``i``'s inbox. When ``None``, grouping is
            derived from ``messages.receiver_index``.
    """

    messages: EdgeMessageBatch
    receiver_ptr: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if self.receiver_ptr is None:
            return
        ptr = _check_tensor("receiver_ptr", self.receiver_ptr)
        if ptr.dtype != torch.long:
            raise ValueError(f"receiver_ptr must have dtype torch.long, got {ptr.dtype}")
        if ptr.dim() != 1 or ptr.numel() < 1:
            raise ValueError("receiver_ptr must be 1-D with at least one element")
        num_messages = self.messages.num_messages
        if int(ptr[0].item()) != 0 or int(ptr[-1].item()) != num_messages:
            raise ValueError(
                f"receiver_ptr must start at 0 and end at num_messages={num_messages}"
            )
        counts = ptr[1:] - ptr[:-1]
        if counts.numel() > 0 and int(counts.min().item()) < 0:
            raise ValueError("receiver_ptr must be non-decreasing")
        expected = torch.repeat_interleave(
            torch.arange(counts.numel(), dtype=torch.long, device=ptr.device), counts
        )
        if not torch.equal(expected, self.messages.receiver_index.to(ptr.device)):
            raise ValueError("receiver_ptr is inconsistent with messages.receiver_index")

    @property
    def num_messages(self) -> int:
        """Total number of delivered messages in the inbox view."""
        return self.messages.num_messages

    def resolve_num_receivers(self, num_receivers: Optional[int] = None) -> int:
        """Resolve the number of receivers this view covers.

        Resolution precedence: ``receiver_ptr`` (authoritative when present),
        then the explicit ``num_receivers`` argument, then inference as
        ``max(receiver_index) + 1`` (``0`` for an empty inbox). Pass
        ``num_receivers`` explicitly (typically ``graph.num_agents``) whenever
        trailing zero-message receivers must be represented.

        Args:
            num_receivers: Explicit receiver count, or ``None`` to infer.

        Returns:
            The resolved receiver count.

        Raises:
            ValueError: If an explicit count contradicts ``receiver_ptr`` or is
                smaller than ``max(receiver_index) + 1``.
        """
        if self.receiver_ptr is not None:
            from_ptr = self.receiver_ptr.numel() - 1
            if num_receivers is not None and num_receivers != from_ptr:
                raise ValueError(
                    f"num_receivers={num_receivers} contradicts receiver_ptr ({from_ptr})"
                )
            return from_ptr
        min_required = 0
        if self.messages.num_messages > 0:
            min_required = int(self.messages.receiver_index.max().item()) + 1
        if num_receivers is None:
            return min_required
        if num_receivers < min_required:
            raise ValueError(
                f"num_receivers={num_receivers} is smaller than max(receiver_index)+1="
                f"{min_required}"
            )
        return num_receivers

    def messages_for(self, receiver: int) -> EdgeMessageBatch:
        """Return the (possibly empty) message batch delivered to one receiver.

        Args:
            receiver: Receiver node index (``>= 0``). Without ``receiver_ptr``,
                any index beyond the observed receivers yields an empty batch;
                with ``receiver_ptr``, the index must be within its range.

        Returns:
            An ``EdgeMessageBatch`` with that receiver's messages in delivery order.
        """
        if receiver < 0:
            raise ValueError(f"receiver must be >= 0, got {receiver}")
        if self.receiver_ptr is not None:
            num_receivers = self.receiver_ptr.numel() - 1
            if receiver >= num_receivers:
                raise ValueError(
                    f"receiver {receiver} out of range for receiver_ptr ({num_receivers})"
                )
            start = int(self.receiver_ptr[receiver].item())
            end = int(self.receiver_ptr[receiver + 1].item())
            index: torch.Tensor = torch.arange(
                start, end, dtype=torch.long, device=self.messages.payload.device
            )
        else:
            index = (self.messages.receiver_index == receiver).nonzero(as_tuple=False).flatten()
        return self.messages.select(index)

    def iter_receivers(
        self,
        num_receivers: Optional[int] = None,
        include_empty: bool = True,
    ) -> Iterator[Tuple[int, EdgeMessageBatch]]:
        """Iterate ``(receiver, inbox)`` pairs sparsely, in receiver order.

        Args:
            num_receivers: Explicit receiver count (see :meth:`resolve_num_receivers`).
            include_empty: Whether to yield receivers with zero messages.

        Yields:
            Tuples of receiver index and that receiver's ``EdgeMessageBatch``.
        """
        total = self.resolve_num_receivers(num_receivers)
        for receiver in range(total):
            batch = self.messages_for(receiver)
            if include_empty or batch.num_messages > 0:
                yield receiver, batch

    def to_dense(
        self,
        num_receivers: Optional[int] = None,
        pad_value: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert to a dense per-receiver payload tensor plus validity mask.

        Message order within each receiver is preserved (stable). Zero-message
        receivers produce fully masked rows; a fully empty inbox produces a
        ``[N, 0, ...]`` payload and ``[N, 0]`` mask.

        Args:
            num_receivers: Explicit receiver count (see :meth:`resolve_num_receivers`).
            pad_value: Fill value for padded payload slots.

        Returns:
            A tuple ``(dense, mask)`` where ``dense`` has shape
            ``[N, K, *payload.shape[1:]]`` with ``K`` the maximum inbox size,
            and ``mask`` is a boolean tensor of shape ``[N, K]``, ``True``
            where a real message is present.
        """
        total = self.resolve_num_receivers(num_receivers)
        payload = self.messages.payload
        receiver = self.messages.receiver_index
        num_messages = self.messages.num_messages

        counts = torch.bincount(receiver, minlength=total)
        max_inbox = int(counts.max().item()) if counts.numel() > 0 else 0

        dense_shape = (total, max_inbox) + tuple(payload.shape[1:])
        dense = payload.new_full(dense_shape, pad_value)
        mask = torch.zeros((total, max_inbox), dtype=torch.bool, device=payload.device)

        if num_messages > 0 and max_inbox > 0:
            order = torch.argsort(receiver, stable=True)
            sorted_receiver = receiver[order]
            group_start = torch.cumsum(counts, dim=0) - counts
            positions = (
                torch.arange(num_messages, dtype=torch.long, device=payload.device)
                - group_start[sorted_receiver]
            )
            dense[sorted_receiver, positions] = payload[order]
            mask[sorted_receiver, positions] = True
        return dense, mask

    def to(self, device: Device) -> InboxBatch:
        """Return a copy of the inbox view with all tensor fields moved to ``device``."""
        return InboxBatch(
            messages=self.messages.to(device),
            receiver_ptr=None if self.receiver_ptr is None else self.receiver_ptr.to(device),
        )


@dataclass(eq=False)
class PacketBatch:
    """Batch of protocol packets with transport metadata kept apart from payload.

    Protocol metadata (IDs, hops, TTL, fragments) must not enter learned
    payload features unless a scheme explicitly selects them as policy-visible.

    Attributes:
        payload: Tensor with leading dimension ``P`` (opaque application payload).
        packet_id: ``torch.long`` ``[P]``; unique per-packet identity.
        message_id: ``torch.long`` ``[P]``; application message the packet carries.
        origin_index: ``torch.long`` ``[P]``; original source node index.
        previous_hop_index: ``torch.long`` ``[P]``; immediate previous hop, or
            :data:`NO_PREVIOUS_HOP` for a not-yet-transmitted origin packet.
        destination_index: ``torch.long`` ``[P]``; destination node index, or
            :data:`BROADCAST_DESTINATION` for broadcast.
        sequence_number: ``torch.long`` ``[P]``; per-origin send sequence.
        ttl: ``torch.long`` ``[P]``; remaining hop budget (``>= 0``).
        fragment_index: ``torch.long`` ``[P]``; fragment position within the message.
        fragment_count: ``torch.long`` ``[P]``; total fragments of the message (``>= 1``).
        created_step: ``torch.long`` ``[P]``; movement step of packet creation.
        created_round: ``torch.long`` ``[P]``; communication round of packet creation.
    """

    payload: torch.Tensor
    packet_id: torch.Tensor
    message_id: torch.Tensor
    origin_index: torch.Tensor
    previous_hop_index: torch.Tensor
    destination_index: torch.Tensor
    sequence_number: torch.Tensor
    ttl: torch.Tensor
    fragment_index: torch.Tensor
    fragment_count: torch.Tensor
    created_step: torch.Tensor
    created_round: torch.Tensor

    def __post_init__(self) -> None:
        payload = _check_tensor("payload", self.payload)
        if payload.dim() < 1:
            raise ValueError("payload must have at least 1 dimension [P, ...]")
        num_packets = payload.size(0)

        _check_long_vector("packet_id", self.packet_id, num_packets, minimum=0)
        _check_long_vector("message_id", self.message_id, num_packets, minimum=0)
        _check_long_vector("origin_index", self.origin_index, num_packets, minimum=0)
        _check_long_vector(
            "previous_hop_index", self.previous_hop_index, num_packets, minimum=NO_PREVIOUS_HOP
        )
        _check_long_vector(
            "destination_index", self.destination_index, num_packets, minimum=BROADCAST_DESTINATION
        )
        _check_long_vector("sequence_number", self.sequence_number, num_packets, minimum=0)
        _check_long_vector("ttl", self.ttl, num_packets, minimum=0)
        _check_long_vector("fragment_index", self.fragment_index, num_packets, minimum=0)
        _check_long_vector("fragment_count", self.fragment_count, num_packets, minimum=1)
        _check_long_vector("created_step", self.created_step, num_packets, minimum=0)
        _check_long_vector("created_round", self.created_round, num_packets, minimum=0)

        if num_packets > 0 and bool((self.fragment_index >= self.fragment_count).any()):
            raise ValueError("fragment_index must be < fragment_count for every packet")

    @property
    def num_packets(self) -> int:
        """Number of packets ``P`` in the batch."""
        return self.payload.size(0)

    @property
    def is_broadcast(self) -> torch.Tensor:
        """Boolean tensor ``[P]``, ``True`` where the packet is a broadcast."""
        return self.destination_index == BROADCAST_DESTINATION

    def select(self, index: Union[torch.Tensor, Sequence[int]]) -> PacketBatch:
        """Return a new batch containing the packets selected by ``index`` (in order)."""
        idx = _as_index_tensor(index, self.payload.device)
        return PacketBatch(
            payload=self.payload[idx],
            packet_id=self.packet_id[idx],
            message_id=self.message_id[idx],
            origin_index=self.origin_index[idx],
            previous_hop_index=self.previous_hop_index[idx],
            destination_index=self.destination_index[idx],
            sequence_number=self.sequence_number[idx],
            ttl=self.ttl[idx],
            fragment_index=self.fragment_index[idx],
            fragment_count=self.fragment_count[idx],
            created_step=self.created_step[idx],
            created_round=self.created_round[idx],
        )

    def to(self, device: Device) -> PacketBatch:
        """Return a copy of the batch with all tensor fields moved to ``device``."""
        return PacketBatch(
            payload=self.payload.to(device),
            packet_id=self.packet_id.to(device),
            message_id=self.message_id.to(device),
            origin_index=self.origin_index.to(device),
            previous_hop_index=self.previous_hop_index.to(device),
            destination_index=self.destination_index.to(device),
            sequence_number=self.sequence_number.to(device),
            ttl=self.ttl.to(device),
            fragment_index=self.fragment_index.to(device),
            fragment_count=self.fragment_count.to(device),
            created_step=self.created_step.to(device),
            created_round=self.created_round.to(device),
        )

    @classmethod
    def empty(
        cls,
        payload_dim: int = 0,
        device: Optional[Device] = None,
        payload_dtype: torch.dtype = torch.float32,
    ) -> PacketBatch:
        """Build a valid zero-packet batch with payload shape ``[0, payload_dim]``."""
        return cls(
            payload=torch.zeros((0, payload_dim), dtype=payload_dtype, device=device),
            packet_id=_empty_long(device),
            message_id=_empty_long(device),
            origin_index=_empty_long(device),
            previous_hop_index=_empty_long(device),
            destination_index=_empty_long(device),
            sequence_number=_empty_long(device),
            ttl=_empty_long(device),
            fragment_index=_empty_long(device),
            fragment_count=_empty_long(device),
            created_step=_empty_long(device),
            created_round=_empty_long(device),
        )


@dataclass(eq=False)
class CommunicationResult:
    """Outcome of running the ``R`` communication rounds of one movement step.

    Attributes:
        agent_output: Policy-facing output per agent id. Depending on the
            scheme this may be a tensor embedding, a preserved inbox,
            structured reports, or ``None`` for a no-communication baseline.
        final_state: Optional stacked post-communication state tensor.
        delivered_messages: All messages delivered during the step, distinct
            and in delivery order.
        protocol_state: Opaque protocol state carried to the next step
            (e.g. relay queues), or ``None``.
        diagnostics: Free-form diagnostics for tracing and metrics. May
            contain privileged data; never fed to policies.
    """

    agent_output: Dict[str, object]
    final_state: Optional[torch.Tensor]
    delivered_messages: EdgeMessageBatch
    protocol_state: Optional[object] = None
    diagnostics: Dict[str, object] = field(default_factory=dict)

    def to(self, device: Device) -> CommunicationResult:
        """Return a copy with tensor fields moved to ``device``.

        ``final_state`` and ``delivered_messages`` are always moved. Values in
        ``agent_output`` are moved when they are tensors or communication batch
        types (:class:`EdgeMessageBatch`, :class:`InboxBatch`,
        :class:`PacketBatch`, :class:`CommunicationGraph`), covering the
        documented per-scheme outputs such as a preserved inbox; truly opaque
        objects (including ``protocol_state`` and ``diagnostics``) are passed
        through untouched.
        """
        movable = (torch.Tensor, EdgeMessageBatch, InboxBatch, PacketBatch, CommunicationGraph)
        moved_output: Dict[str, object] = {}
        for agent_id, value in self.agent_output.items():
            if isinstance(value, movable):
                moved_output[agent_id] = value.to(device)
            else:
                moved_output[agent_id] = value
        return CommunicationResult(
            agent_output=moved_output,
            final_state=None if self.final_state is None else self.final_state.to(device),
            delivered_messages=self.delivered_messages.to(device),
            protocol_state=self.protocol_state,
            diagnostics=self.diagnostics,
        )

    @classmethod
    def empty(
        cls,
        agent_ids: Sequence[str] = (),
        payload_dim: int = 0,
        device: Optional[Device] = None,
    ) -> CommunicationResult:
        """Build the explicit empty result used by the ``none`` scheme.

        A no-communication baseline receives this instead of a different
        movement interface: every agent maps to an explicit empty view (an
        empty tuple, so lookups and ``len()`` succeed) and the delivered
        batch is empty.
        """
        return cls(
            agent_output={agent_id: () for agent_id in agent_ids},
            final_state=None,
            delivered_messages=EdgeMessageBatch.empty(payload_dim=payload_dim, device=device),
            protocol_state=None,
            diagnostics={},
        )


@dataclass(frozen=True, eq=False)
class CacheEntry:
    """One stored message in a :class:`MessageCache`.

    Entries are immutable: age is derived at read time and never written back.
    ``message`` is typically a tensor, so equality/hashing are identity-based
    (``eq=False``; see the module docstring).

    Attributes:
        message: Opaque, immutable message payload object.
        emitted_step: Movement step during which the message entered the cache.
        emitted_round: Communication round (within ``emitted_step``) of entry.
        delivery_sequence: Global delivery order; breaks equal-age eviction ties.
        observation_step: Step of the underlying observation; equals
            ``emitted_step`` for source messages and stays fixed under relay.
    """

    message: Any
    emitted_step: int
    emitted_round: int
    delivery_sequence: int
    observation_step: int

    def age(self, current_step: int) -> int:
        """Derive the policy-visible age ``current_step - observation_step``."""
        return current_step - self.observation_step


class MessageCache:
    """Per-agent sliding window over the last ``C`` completed communication rounds.

    The cache is the protocol memory used for relaying, duplicate suppression,
    temporal filtering, and persistence. Its window is counted in rounds and
    spans movement-step boundaries, so a round may read messages emitted in
    previous steps. It is distinct from a processor's intrinsic round-to-round
    hidden state, which exists even at ``C = 0``.

    Lifecycle per communication round ``(step, round_index)``:

    1. Deliveries are stored with :meth:`insert` (round-local deliveries stay
       distinguishable from older entries via their emitted labels and by the
       :meth:`window_entries` / :meth:`entries` split).
    2. The runtime calls :meth:`advance_round` exactly once when the round
       completes; entries older than the last ``C`` completed rounds are
       evicted, oldest round first, ties by delivery sequence.

    ``capacity_rounds == 0`` means no storage: :meth:`insert` still returns the
    created entry (for tracing) but nothing is retained. Episode reset must
    call :meth:`clear`.

    Relay correctness requires ``C >= TTL`` (:meth:`supports_ttl`); the scheme
    compiler rejects relay configurations that violate it.
    """

    def __init__(self, capacity_rounds: int) -> None:
        """Create an empty cache.

        Args:
            capacity_rounds: Window size ``C`` in completed rounds; ``0``
                disables storage entirely.
        """
        if capacity_rounds < 0:
            raise ValueError(f"capacity_rounds must be >= 0, got {capacity_rounds}")
        self._capacity_rounds = capacity_rounds
        self._buckets: OrderedDict[Tuple[int, int], List[CacheEntry]] = OrderedDict()
        self._window: Deque[Tuple[int, int]] = deque()
        self._last_completed: Optional[Tuple[int, int]] = None
        self._next_sequence = 0

    @property
    def capacity_rounds(self) -> int:
        """The configured window size ``C``."""
        return self._capacity_rounds

    @property
    def last_completed_round(self) -> Optional[Tuple[int, int]]:
        """Label ``(step, round_index)`` of the most recently completed round."""
        return self._last_completed

    @property
    def completed_window(self) -> Tuple[Tuple[int, int], ...]:
        """Labels of the completed rounds currently inside the window, oldest first."""
        return tuple(self._window)

    @property
    def stored_rounds(self) -> Tuple[Tuple[int, int], ...]:
        """Labels of rounds with retained entries (completed and open), oldest first."""
        return tuple(self._buckets.keys())

    def __len__(self) -> int:
        """Total number of retained entries."""
        return sum(len(bucket) for bucket in self._buckets.values())

    def supports_ttl(self, ttl: int) -> bool:
        """Return whether this window satisfies the relay-correctness floor ``C >= TTL``."""
        return self._capacity_rounds >= ttl

    def insert(
        self,
        message: Any,
        emitted_step: int,
        emitted_round: int,
        delivery_sequence: Optional[int] = None,
        observation_step: Optional[int] = None,
    ) -> CacheEntry:
        """Store a delivered message under round ``(emitted_step, emitted_round)``.

        The round label must be later than the last completed round (no
        insertion into the past). While a round is open (has stored entries but
        was not completed with :meth:`advance_round`), inserts must target that
        open round: skipping ahead to a later label is rejected here rather
        than surfacing as a confusing :meth:`advance_round` error later.

        Args:
            message: Opaque immutable payload; never copied or mutated.
            emitted_step: Movement step of emission/delivery.
            emitted_round: Communication round of emission/delivery.
            delivery_sequence: Explicit global delivery order; auto-assigned
                from an internal monotone counter when ``None``.
            observation_step: Step of the underlying observation; defaults to
                ``emitted_step`` (source message).

        Returns:
            The created :class:`CacheEntry`. With ``capacity_rounds == 0`` the
            entry is returned but not stored.
        """
        label = (emitted_step, emitted_round)
        if self._last_completed is not None and label <= self._last_completed:
            raise ValueError(
                f"cannot insert into round {label}: round {self._last_completed} "
                "already completed"
            )
        if delivery_sequence is None:
            delivery_sequence = self._next_sequence
        self._next_sequence = max(self._next_sequence, delivery_sequence + 1)
        entry = CacheEntry(
            message=message,
            emitted_step=emitted_step,
            emitted_round=emitted_round,
            delivery_sequence=delivery_sequence,
            observation_step=emitted_step if observation_step is None else observation_step,
        )
        if self._capacity_rounds == 0:
            return entry
        if self._buckets:
            newest = next(reversed(self._buckets))
            if label < newest:
                raise ValueError(
                    f"cannot insert into round {label}: later round {newest} already stored"
                )
            newest_is_open = self._last_completed is None or newest > self._last_completed
            if newest_is_open and label > newest:
                raise ValueError(
                    f"cannot insert into round {label}: round {newest} is still open; "
                    "complete it with advance_round first"
                )
        self._buckets.setdefault(label, []).append(entry)
        return entry

    def advance_round(self, step: int, round_index: int) -> List[CacheEntry]:
        """Mark round ``(step, round_index)`` completed and advance the window.

        Must be called exactly once per communication round, in order, whether
        or not the round delivered messages (empty rounds still age the window).

        Args:
            step: Movement step of the completed round.
            round_index: Round index within the step.

        Returns:
            Entries evicted from the window, oldest round first; ties within a
            round are ordered by delivery sequence.
        """
        label = (step, round_index)
        if self._last_completed is not None and label <= self._last_completed:
            raise ValueError(
                f"rounds must complete in order: {label} is not after {self._last_completed}"
            )
        for bucket_label in self._buckets:
            if self._last_completed is not None and bucket_label <= self._last_completed:
                continue  # Already-completed round still inside the window.
            if bucket_label < label:
                raise ValueError(
                    f"cannot complete round {label}: stored round {bucket_label} "
                    "was never completed"
                )
            if bucket_label > label:
                raise ValueError(
                    f"cannot complete round {label}: round {bucket_label} holds "
                    "messages from a later round"
                )
        self._last_completed = label
        self._window.append(label)

        evicted: List[CacheEntry] = []
        while len(self._window) > self._capacity_rounds:
            expired = self._window.popleft()
            bucket = self._buckets.pop(expired, None)
            if bucket:
                evicted.extend(sorted(bucket, key=lambda entry: entry.delivery_sequence))
        if evicted:
            logger.debug(
                "MessageCache evicted {} entry(ies) outside the {}-round window at round {}",
                len(evicted),
                self._capacity_rounds,
                label,
            )
        return evicted

    def entries(self) -> List[CacheEntry]:
        """Return every retained entry, oldest round first, delivery order within a round.

        Includes entries of the currently open (not yet completed) round, which
        remain distinguishable from older ones via their emitted labels; use
        :meth:`window_entries` for the completed ``C``-round window only.
        Reading never mutates the cache.
        """
        collected: List[CacheEntry] = []
        for bucket in self._buckets.values():
            collected.extend(sorted(bucket, key=lambda entry: entry.delivery_sequence))
        return collected

    def window_entries(self) -> List[CacheEntry]:
        """Return entries of completed rounds inside the window (excludes the open round)."""
        collected: List[CacheEntry] = []
        for label in self._window:
            bucket = self._buckets.get(label)
            if bucket:
                collected.extend(sorted(bucket, key=lambda entry: entry.delivery_sequence))
        return collected

    def iter_with_age(self, current_step: int) -> Iterator[Tuple[CacheEntry, int]]:
        """Yield ``(entry, age)`` pairs with age derived as in :meth:`CacheEntry.age`."""
        for entry in self.entries():
            yield entry, entry.age(current_step)

    def clear(self) -> None:
        """Drop all entries and reset the round window (episode reset)."""
        dropped = len(self)
        self._buckets.clear()
        self._window.clear()
        self._last_completed = None
        self._next_sequence = 0
        if dropped:
            logger.debug("MessageCache cleared {} entry(ies) on reset", dropped)
