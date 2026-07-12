"""PyG-backed aggregation processor (Phase 2).

Implements the ``one_hop_mean`` (and related) receiver semantics from
``docs/communication_implementation_plan.md`` ("PyG Aggregation Processing"):
each round's delivered :class:`EdgeMessageBatch` is accumulated unchanged, and
``finalize`` reduces the step's inbox to ONE aggregated vector per agent with
a public PyG aggregation module::

    aggregated = aggregation(messages.payload, index=messages.receiver_index,
                             dim_size=num_agents)

Boundary rules (``docs/communication_decision_log.md``, "PyG And Protocol
Boundary"):

- PyG supplies only the aggregation module; ACN owns topology, round timing,
  and delivery. The processor never re-emits (``emit_round`` returns ``{}``),
  so the shared slotted transport behaves exactly as under ``one_hop_direct``.
- The payload path stays torch end-to-end: no ``.numpy()``, no ``.detach()``.
  A later learned scheme must be able to backpropagate through this exact
  aggregation call.
- Zero-message receivers get an explicit zeros vector and a delivered count
  of 0 — never NaN (max aggregation over an empty group is masked to zeros
  explicitly rather than trusting the backend's fill value).
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Mapping, Tuple

import torch

from src.communication.transport import Frame, concat_edge_message_batches
from src.communication.types import (
    CommunicationGraph,
    CommunicationResult,
    EdgeMessageBatch,
    InboxBatch,
)
from src.utils.logger import get_logger

logger = get_logger("acn.communication")

__all__ = ["SUPPORTED_AGGREGATIONS", "AggregatedVectorView", "PyGAggregationProcessor"]

#: Aggregation names accepted by :class:`PyGAggregationProcessor`, mapped to
#: the public ``torch_geometric.nn.aggr`` module they select. The registry's
#: known-name superset (softmax, attention, ...) stays config-tolerated but is
#: rejected here until a later phase enables it.
_AGGREGATION_MODULES: Tuple[str, ...] = ("max", "mean", "sum")

SUPPORTED_AGGREGATIONS: Tuple[str, ...] = _AGGREGATION_MODULES


def _build_aggregation_module(name: str) -> torch.nn.Module:
    """Instantiate the public PyG aggregation module selected by ``name``.

    torch_geometric is imported lazily so that simulations which never build
    an aggregating plan do not pay its import cost.

    Args:
        name: One of :data:`SUPPORTED_AGGREGATIONS`.

    Returns:
        The instantiated ``torch_geometric.nn.aggr`` module.

    Raises:
        ValueError: If ``name`` is not a supported aggregation.
    """
    from torch_geometric.nn.aggr import MaxAggregation, MeanAggregation, SumAggregation

    factories = {
        "max": MaxAggregation,
        "mean": MeanAggregation,
        "sum": SumAggregation,
    }
    factory = factories.get(name)
    if factory is None:
        raise ValueError(
            "Unsupported communication aggregation {!r}. Supported aggregations: "
            "{}.".format(name, ", ".join(SUPPORTED_AGGREGATIONS))
        )
    return factory()


class AggregatedVectorView:
    """Policy-facing per-agent output of an aggregating scheme.

    A read-only, dict-like view with exactly two keys: ``"vector"`` (the
    ``[D]`` aggregated payload tensor, kept torch end-to-end so gradients can
    flow through it) and ``"count"`` (how many messages were delivered to the
    agent this step; ``0`` implies a zeros vector). It intentionally does NOT
    subclass ``Mapping``: the trace serializer (``src.utils.history
    .to_jsonable``) recurses into real mappings, which would embed the live
    tensor in JSONL rows, whereas this view is summarized via
    :meth:`trace_summary` instead.

    Attributes:
        vector: The aggregated payload tensor of shape ``[D]``.
        count: Number of messages delivered to this agent during the step.
    """

    _KEYS: Tuple[str, ...] = ("vector", "count")

    __slots__ = ("vector", "count")

    def __init__(self, vector: torch.Tensor, count: int) -> None:
        """Create the view.

        Args:
            vector: Aggregated 1-D payload tensor.
            count: Delivered-message count behind the aggregation (``>= 0``).
        """
        if not isinstance(vector, torch.Tensor) or vector.dim() != 1:
            raise ValueError(
                "AggregatedVectorView vector must be a 1-D torch.Tensor, got {!r}.".format(vector)
            )
        if count < 0:
            raise ValueError("AggregatedVectorView count must be >= 0, got {}.".format(count))
        self.vector = vector
        self.count = int(count)

    def __getitem__(self, key: str) -> object:
        if key == "vector":
            return self.vector
        if key == "count":
            return self.count
        raise KeyError(key)

    def get(self, key: str, default: object = None) -> object:
        """Return ``self[key]``, or ``default`` for an unknown key."""
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> Tuple[str, ...]:
        """Return the fixed key set ``('vector', 'count')``."""
        return self._KEYS

    def __iter__(self) -> Iterator[str]:
        return iter(self._KEYS)

    def __len__(self) -> int:
        return len(self._KEYS)

    def __contains__(self, key: object) -> bool:
        return key in self._KEYS

    def __repr__(self) -> str:
        return "AggregatedVectorView(vector={}, count={})".format(
            self.vector.tolist(), self.count
        )

    def trace_summary(self) -> Dict[str, Any]:
        """Compact JSON-safe summary for transition trace rows.

        The live tensor never enters JSONL rows; the aggregated values are
        exported as plain floats (``Tensor.tolist`` works on autograd tensors
        without detaching the graph the policy still holds).
        """
        return {
            "type": "AggregatedVector",
            "count": self.count,
            "dim": int(self.vector.numel()),
            "vector": [float(value) for value in self.vector.tolist()],
        }


class PyGAggregationProcessor:
    """Aggregate the step's delivered inbox to one vector per agent via PyG.

    Round state is the ordered list of the step's delivered
    :class:`EdgeMessageBatch` objects (identical bookkeeping to the direct
    processor, so both scheme families share the delivery runtime unchanged).
    The processor never re-emits — ``emit_round`` returns ``{}`` — and only
    ``finalize`` aggregates, producing the ``aggregated-vector`` output
    contract: ``agent_output[agent_id]`` is an :class:`AggregatedVectorView`
    with the agent's ``[D]`` vector and delivered count.
    """

    def __init__(self, aggregation: str, payload_dim: int) -> None:
        """Create the processor.

        Args:
            aggregation: Aggregation name; one of
                :data:`SUPPORTED_AGGREGATIONS` (``max``, ``mean``, ``sum``).
            payload_dim: Payload vector dimension ``D``, used to shape the
                zeros output of a step that delivered nothing at all.

        Raises:
            ValueError: If the aggregation name is unsupported or
                ``payload_dim`` is not a positive integer.
        """
        if not isinstance(payload_dim, int) or isinstance(payload_dim, bool) or payload_dim < 1:
            raise ValueError(
                "PyGAggregationProcessor payload_dim must be a positive integer, "
                "got {!r}.".format(payload_dim)
            )
        self._aggregation_name = aggregation
        self._aggregation = _build_aggregation_module(aggregation)
        self._payload_dim = payload_dim

    @property
    def aggregation_name(self) -> str:
        """The configured aggregation name (``max``, ``mean``, or ``sum``)."""
        return self._aggregation_name

    def initialize(
        self,
        local_observations: Mapping[str, object],
        graph: CommunicationGraph,
        context: Mapping[str, object],
    ) -> object:
        """Create the round-zero state: an empty list of delivered batches.

        Args:
            local_observations: Ignored; aggregation reads only deliveries.
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

        Aggregation is deferred to :meth:`finalize`; per-round the batches are
        kept distinct exactly like the direct processor keeps them.

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
        """Return no outboxes: the aggregating processor never re-emits.

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
        """Aggregate the step's deliveries to one vector per agent.

        Args:
            state: The list of delivered batches accumulated by
                :meth:`process_round`.
            inbox: The final round's inbox (already stored; ignored).
            graph: The frozen communication graph for this movement step.
            context: Public runtime context (unused).

        Returns:
            A :class:`CommunicationResult` whose ``agent_output`` maps every
            graph agent id to an :class:`AggregatedVectorView`, whose
            ``final_state`` is the stacked ``[N, D]`` aggregated tensor, and
            whose ``delivered_messages`` is the step's concatenated inbox.
        """
        batches: List[EdgeMessageBatch] = list(state) if state is not None else []
        delivered = concat_edge_message_batches(batches)
        num_agents = graph.num_agents

        counts = torch.bincount(delivered.receiver_index, minlength=num_agents)
        if delivered.num_messages == 0:
            # Nothing arrived anywhere: explicit zeros for every agent, at the
            # configured payload dimension (the empty batch's dim may be 0).
            aggregated = delivered.payload.new_zeros((num_agents, self._payload_dim))
        else:
            aggregated = self._aggregation(
                delivered.payload,
                index=delivered.receiver_index,
                dim_size=num_agents,
            )
            zero_inbox = counts == 0
            if bool(zero_inbox.any()):
                # Explicit zeros (and never NaN) for zero-message receivers.
                # torch.where keeps the graph differentiable for the rows that
                # did receive messages; no in-place writes, no detach.
                aggregated = torch.where(
                    zero_inbox.unsqueeze(-1), torch.zeros_like(aggregated), aggregated
                )

        counts_list = counts.tolist()
        agent_output: Dict[str, object] = {}
        for index, agent_id in enumerate(graph.agent_ids):
            agent_output[agent_id] = AggregatedVectorView(
                vector=aggregated[index], count=int(counts_list[index])
            )

        logger.debug(
            "PyGAggregationProcessor finalized step {}: '{}' over {} message(s) "
            "across {} agent(s)",
            graph.step,
            self._aggregation_name,
            delivered.num_messages,
            num_agents,
        )
        return CommunicationResult(
            agent_output=agent_output,
            final_state=aggregated,
            delivered_messages=delivered,
            protocol_state=None,
            diagnostics={"aggregation": self._aggregation_name},
        )
