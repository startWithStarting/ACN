"""Communication plan composition and the no-communication baseline components.

This module defines the Phase 0 contract for compiled communication schemes
(see ``docs/communication_implementation_plan.md``, section "Scheme Registry
And Compiler", and ``docs/communication_decision_log.md``, section "Round And
Cache Model").

A :class:`CommunicationPlan` bundles the three runtime components of a scheme
(topology, transport, processor) with the plan-level parameters that the
compiler in :mod:`src.communication.registry` validates. Phase 0 ships only
the typing protocols and the explicit no-op ``none`` scheme; real component
implementations arrive in Phase 1 and later.

Round and cache semantics fixed by the decision log:

- ``rounds`` is ``R``, the number of synchronous communication rounds per
  movement step. The same-team graph is frozen across all ``R`` rounds, one
  round permits at most one graph hop, and movement is selected only after
  the rounds complete (post-communication).
- ``cache_window`` is ``C``, a sliding window over the last ``C`` rounds of
  delivered messages. The window spans movement-step boundaries; eviction is
  oldest round first, with equal-age ties broken by delivery sequence. The
  cache is distinct from a processor's intrinsic round-to-round hidden state,
  which exists even when ``C == 0``. Relay correctness requires ``C >= TTL``,
  which the compiler enforces.
- Message payloads are immutable. Age is derived at read time as
  ``current_step - observation_step``; it is never stored on the payload.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence, runtime_checkable

from src.communication.types import CommunicationGraph, CommunicationResult, EdgeMessageBatch
from src.utils.logger import get_logger

logger = get_logger("acn.communication")


@runtime_checkable
class CommunicationTopology(Protocol):
    """Builds the communication graph used for one movement step.

    The baseline runtime calls ``build`` once per movement step; the returned
    graph is frozen across all ``R`` communication rounds of that step.
    Real implementations (e.g. the same-team radius topology) arrive in
    Phase 1.
    """

    def build(self, agents: Sequence[object], step: int) -> CommunicationGraph:
        """Build the frozen communication graph for the given movement step.

        Args:
            agents: Active agent objects (or identifiers) at the step start.
            step: Current movement-step index.

        Returns:
            A :class:`CommunicationGraph` snapshot. Empty and disconnected
            graphs are valid.
        """
        ...


@runtime_checkable
class RoundTransport(Protocol):
    """Synchronous slotted transport moving opaque frames one hop per round.

    The transport delivers each frame over at most one valid graph edge per
    round and keeps transport metadata separate from application payloads.
    Real implementations arrive in Phase 1.
    """

    def reset(self, graph: CommunicationGraph, context: Mapping[str, object]) -> object:
        """Reset per-step transport state for a freshly frozen graph.

        Args:
            graph: The frozen communication graph for this movement step.
            context: Public runtime context (e.g. step index, configuration).

        Returns:
            An opaque transport state passed back into ``transmit_round``.
        """
        ...

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
            outboxes: Zero or more outgoing frames per sending agent.
            graph: The frozen communication graph for this movement step.
            transport_state: Opaque state produced by ``reset`` or the
                previous round.
            round_index: Zero-based round index within the current step.
            context: Public runtime context.

        Returns:
            The batch of frames delivered during this round. Messages remain
            distinct; aggregation is a processor decision, never a transport
            one.
        """
        ...


@runtime_checkable
class CommunicationProcessor(Protocol):
    """Per-agent transformation of local state and delivered payloads.

    A processor may consume only its local observation, its own recurrent or
    protocol state, messages delivered by the communication runtime, and
    public configuration such as the round number. Privileged simulator
    identity must never silently enter its inputs. Real implementations
    arrive in Phase 1.
    """

    def initialize(
        self,
        local_observations: Mapping[str, object],
        graph: CommunicationGraph,
        context: Mapping[str, object],
    ) -> object:
        """Create the round-zero communication state from local observations.

        Args:
            local_observations: Policy-visible local observation per agent.
            graph: The frozen communication graph for this movement step.
            context: Public runtime context.

        Returns:
            The initial round state threaded through ``process_round``.
        """
        ...

    def process_round(
        self,
        state: object,
        inbox: object,
        graph: CommunicationGraph,
        round_index: int,
        context: Mapping[str, object],
    ) -> object:
        """Advance the communication state by one synchronous round.

        Args:
            state: Round state from ``initialize`` or the previous round.
            inbox: Messages delivered to each agent during this round.
            graph: The frozen communication graph for this movement step.
            round_index: Zero-based round index within the current step.
            context: Public runtime context.

        Returns:
            The updated round state for the next round.
        """
        ...

    def finalize(
        self,
        state: object,
        inbox: object,
        graph: CommunicationGraph,
        context: Mapping[str, object],
    ) -> CommunicationResult:
        """Produce the policy-facing result after the final round.

        Args:
            state: Round state after the last ``process_round`` call.
            inbox: Messages delivered during the final round.
            graph: The frozen communication graph for this movement step.
            context: Public runtime context.

        Returns:
            The :class:`CommunicationResult` consumed by movement selection.
        """
        ...


@dataclass
class CommunicationPlan:
    """A compiled communication scheme ready for the runtime.

    Instances are produced by scheme builders registered in
    :mod:`src.communication.registry` and validated by
    ``create_communication_plan``.

    Attributes:
        topology: Builder of the per-step frozen communication graph.
        transport: Synchronous slotted one-hop-per-round frame transport.
        processor: Per-agent round processing (rule-based or learned).
        rounds: ``R``, synchronous communication rounds per movement step.
            Must be ``>= 1`` for every enabled non-``none`` scheme; the
            ``none`` scheme uses ``0``.
        cache_window: ``C``, the sliding message-cache window in rounds. The
            window spans movement-step boundaries and must satisfy
            ``cache_window >= ttl`` for relay schemes.
        graph_update: Graph rebuild policy. Phase 0 supports only
            ``"frozen"`` (one snapshot per movement step); dynamic per-round
            rebuilding is a later configurable extension.
        differentiable: Whether the scheme's communication computation must
            stay inside an autograd-capable forward pass.
        output_contract: Declared shape/semantics of the policy-facing
            output (e.g. ``"empty"`` for the no-communication baseline).
    """

    topology: CommunicationTopology
    transport: RoundTransport
    processor: CommunicationProcessor
    rounds: int
    cache_window: int
    graph_update: str
    differentiable: bool
    output_contract: str


def empty_communication_graph(step: int = 0) -> CommunicationGraph:
    """Build the canonical empty communication graph.

    Delegates to :meth:`CommunicationGraph.empty` so "empty" has exactly one
    definition per type.

    Args:
        step: Movement-step index recorded on the snapshot.

    Returns:
        A :class:`CommunicationGraph` with no agents and no edges. All tensor
        fields are empty tensors, which remain valid under device movement.
    """
    return CommunicationGraph.empty(step=step)


def empty_edge_message_batch() -> EdgeMessageBatch:
    """Build an :class:`EdgeMessageBatch` containing zero messages.

    Delegates to :meth:`EdgeMessageBatch.empty`.

    Returns:
        An empty batch whose tensor fields all have a zero-length message
        dimension.
    """
    return EdgeMessageBatch.empty()


def empty_communication_result(agent_ids: Sequence[str] = ()) -> CommunicationResult:
    """Build the explicit empty communication result.

    A no-communication baseline receives an explicit empty view, never a
    missing one: every agent id maps to an empty tuple (zero delivered
    payloads), so lookups succeed and the movement interface stays identical
    across ablations. Delegates to :meth:`CommunicationResult.empty`.

    Args:
        agent_ids: Agent identifiers that must each receive an explicit
            empty view.

    Returns:
        A :class:`CommunicationResult` with per-agent empty outputs, no final
        state, an empty delivered-message batch, no protocol state, and empty
        diagnostics.
    """
    return CommunicationResult.empty(agent_ids=agent_ids)


class NoneTopology:
    """Explicit no-op topology for the ``none`` scheme.

    It does not inspect the agents and always returns the canonical empty
    graph: the no-communication baseline has no communication edges by
    definition.
    """

    def build(self, agents: Sequence[object], step: int) -> CommunicationGraph:
        """Return the empty communication graph for the given step.

        Args:
            agents: Ignored; the none scheme never builds edges.
            step: Movement-step index recorded on the snapshot.

        Returns:
            The canonical empty :class:`CommunicationGraph`.
        """
        return empty_communication_graph(step=step)


class NoneTransport:
    """Explicit no-op transport for the ``none`` scheme.

    The ``none`` plan runs zero rounds, so these methods are never invoked in
    normal operation; if called anyway they carry no state and deliver
    nothing.
    """

    def reset(self, graph: CommunicationGraph, context: Mapping[str, object]) -> object:
        """Return ``None``: the none transport keeps no state."""
        return None

    def transmit_round(
        self,
        outboxes: Mapping[str, Sequence[object]],
        graph: CommunicationGraph,
        transport_state: object,
        round_index: int,
        context: Mapping[str, object],
    ) -> EdgeMessageBatch:
        """Deliver nothing: always returns an empty message batch."""
        return empty_edge_message_batch()


class NoneProcessor:
    """Explicit no-op processor for the ``none`` scheme.

    It maintains no state and finalizes to the explicit empty
    :class:`CommunicationResult`, giving every agent on the graph an empty
    view rather than a missing one.
    """

    def initialize(
        self,
        local_observations: Mapping[str, object],
        graph: CommunicationGraph,
        context: Mapping[str, object],
    ) -> object:
        """Return ``None``: the none processor keeps no round state."""
        return None

    def process_round(
        self,
        state: object,
        inbox: object,
        graph: CommunicationGraph,
        round_index: int,
        context: Mapping[str, object],
    ) -> object:
        """Return the state unchanged; nothing is processed."""
        return state

    def finalize(
        self,
        state: object,
        inbox: object,
        graph: CommunicationGraph,
        context: Mapping[str, object],
    ) -> CommunicationResult:
        """Return the explicit empty result for the agents on the graph.

        Args:
            state: Ignored.
            inbox: Ignored.
            graph: Graph whose ``agent_ids`` receive explicit empty views.
            context: Ignored.

        Returns:
            ``empty_communication_result(graph.agent_ids)``.
        """
        agent_ids = tuple(getattr(graph, "agent_ids", ()) or ())
        return empty_communication_result(agent_ids=agent_ids)


def build_none_plan(config: object = None) -> CommunicationPlan:
    """Build the ``none`` (no-communication baseline) plan.

    The plan runs zero rounds over an empty frozen graph, keeps no message
    cache, and declares the ``"empty"`` output contract: its result is the
    explicit empty :class:`CommunicationResult` produced by
    :func:`empty_communication_result`.

    Args:
        config: Accepted for scheme-builder signature uniformity; ignored.

    Returns:
        The compiled no-communication :class:`CommunicationPlan`.
    """
    logger.debug("Building 'none' communication plan (rounds=0, cache_window=0)")
    return CommunicationPlan(
        topology=NoneTopology(),
        transport=NoneTransport(),
        processor=NoneProcessor(),
        rounds=0,
        cache_window=0,
        graph_update="frozen",
        differentiable=False,
        output_contract="empty",
    )
