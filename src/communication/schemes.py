"""Built-in communication scheme builders (Phases 1-3).

Registers real named schemes with the registry in
:mod:`src.communication.registry`. Phase 1 delivers ``one_hop_direct``:
same-team radius topology, synchronous slotted broadcast transport, and the
inbox-preserving direct processor, running exactly one communication round
per movement step. Phase 2 delivers ``one_hop_mean``: the same topology and
transport, with the PyG aggregation processor reducing each agent's delivered
inbox to one vector (``mean``, ``sum``, or ``max``).

This module is imported by the registry's ``_auto_register`` (mirroring how
``src.agents.registry`` auto-imports strategy modules), so the decorators run
at import time.
"""

from __future__ import annotations

from collections.abc import Mapping

from src.communication.plans import CommunicationPlan
from src.communication.processors.identity import DirectProcessor
from src.communication.processors.relay import (
    RelayCarryoverTransport,
    RelayProcessor,
    RelayState,
)
from src.communication.registry import register_communication_scheme
from src.communication.topology import RadiusTopology
from src.communication.transport import SlottedRadiusTransport
from src.utils.logger import get_logger

logger = get_logger("acn.communication")

__all__ = ["build_one_hop_direct", "build_one_hop_mean", "build_multihop_relay"]


def _field(config: object, name: str, default: object = None) -> object:
    """Read a field from a typed config object or a mapping (None-safe)."""
    if config is None:
        return default
    if isinstance(config, Mapping):
        value = config.get(name, default)
    else:
        value = getattr(config, name, default)
    return default if value is None else value


def _reject(scheme: str, message: str) -> ValueError:
    """Build a consistent, actionable scheme-compilation error."""
    return ValueError("Cannot compile communication scheme '{}': {}".format(scheme, message))


@register_communication_scheme("one_hop_direct")
def build_one_hop_direct(config: object) -> CommunicationPlan:
    """Build the ``one_hop_direct`` plan (Phase 1 baseline).

    Semantics per the implementation plan: same-team radius topology
    (radius-ownership rule configurable, default ``sender``), slotted radius
    broadcast transport, inbox-preserving direct processor, exactly one
    communication round per movement step, output contract
    ``preserved-inbox`` (distinct incoming payloads plus sender identity).

    Args:
        config: The ``environment.communication`` section — a typed
            :class:`~src.communication.config.CommunicationConfig` or a
            mapping using the documented schema field names.

    Returns:
        The compiled :class:`CommunicationPlan`.

    Raises:
        ValueError: On logically inconsistent combinations: more than one
            round, a non-radius or non-frozen topology, a non-broadcast or
            non-slotted transport, any aggregation (which would break the
            preserved-inbox contract), relay settings, or an engineered
            payload whose dimension is not 4.
    """
    scheme = "one_hop_direct"

    rounds = _field(config, "rounds_per_step", 1)
    if rounds != 1:
        raise _reject(
            scheme,
            "rounds_per_step={!r}, but one-hop direct delivery runs exactly one "
            "communication round per movement step. Use a multihop scheme for more "
            "rounds.".format(rounds),
        )

    cache_window = _field(config, "cache_window", 0)
    if not isinstance(cache_window, int) or isinstance(cache_window, bool) or cache_window < 0:
        raise _reject(
            scheme, "cache_window must be a non-negative integer, got {!r}.".format(cache_window)
        )

    topology_cfg = _field(config, "topology")
    topology_type = _field(topology_cfg, "type", "radius")
    if topology_type != "radius":
        raise _reject(
            scheme, "topology.type must be 'radius', got {!r}.".format(topology_type)
        )
    if _field(topology_cfg, "freeze_within_step", True) is not True:
        raise _reject(
            scheme,
            "topology.freeze_within_step must be true; dynamic per-round graph "
            "rebuilding is not supported by this runtime.",
        )
    radius_rule = _field(topology_cfg, "radius_rule", "sender")
    include_self_edges = _field(topology_cfg, "include_self_edges", False)

    transport_cfg = _field(config, "transport")
    transport_type = _field(transport_cfg, "type", "slotted_radius")
    if transport_type != "slotted_radius":
        raise _reject(
            scheme, "transport.type must be 'slotted_radius', got {!r}.".format(transport_type)
        )
    delivery = _field(transport_cfg, "delivery", "broadcast")
    if delivery != "broadcast":
        raise _reject(
            scheme,
            "transport.delivery must be 'broadcast', got {!r}; addressed delivery is a "
            "later protocol feature.".format(delivery),
        )

    processor_cfg = _field(config, "processor")
    aggregation = _field(processor_cfg, "aggregation", "none")
    if aggregation != "none":
        raise _reject(
            scheme,
            "processor.aggregation={!r} would break the preserved-inbox contract; "
            "one_hop_direct never aggregates. Use 'one_hop_mean' for aggregated "
            "delivery.".format(aggregation),
        )
    update = _field(processor_cfg, "update", "identity")
    if update != "identity":
        raise _reject(
            scheme, "processor.update must be 'identity', got {!r}.".format(update)
        )
    if (
        _field(processor_cfg, "ttl") is not None
        or _field(processor_cfg, "forwarding") is not None
        or bool(_field(processor_cfg, "packet_relay", False))
    ):
        raise _reject(
            scheme,
            "relay settings (ttl/forwarding/packet_relay) are set, but the direct "
            "processor never re-emits. Use 'multihop_relay' for packet relay.",
        )

    payload_cfg = _field(config, "payload")
    payload_type = _field(payload_cfg, "type", "engineered_vector")
    payload_dimension = _field(payload_cfg, "dimension", 4)
    if payload_type == "engineered_vector" and payload_dimension != 4:
        raise _reject(
            scheme,
            "payload.dimension={!r}, but the engineered bearing report is exactly "
            "4-dimensional: [observer_x, observer_y, direction_x, "
            "direction_y].".format(payload_dimension),
        )

    logger.debug(
        "Building 'one_hop_direct' plan (radius_rule={}, include_self_edges={}, "
        "cache_window={})",
        radius_rule,
        include_self_edges,
        cache_window,
    )
    return CommunicationPlan(
        topology=RadiusTopology(
            radius_rule=str(radius_rule),
            include_self_edges=bool(include_self_edges),
        ),
        transport=SlottedRadiusTransport(),
        processor=DirectProcessor(),
        rounds=1,
        cache_window=cache_window,
        graph_update="frozen",
        differentiable=False,
        output_contract="preserved-inbox",
    )


@register_communication_scheme("one_hop_mean")
def build_one_hop_mean(config: object) -> CommunicationPlan:
    """Build the ``one_hop_mean`` plan (Phase 2 aggregating baseline).

    Semantics per the implementation plan: the same same-team radius topology
    and slotted radius broadcast transport as ``one_hop_direct``, exactly one
    communication round per movement step, and the PyG aggregation processor
    reducing each agent's delivered inbox to ONE vector. Output contract
    ``aggregated-vector``: ``agent_output[agent]`` carries ``{"vector",
    "count"}`` (see
    :class:`~src.communication.processors.pyg.AggregatedVectorView`).

    Args:
        config: The ``environment.communication`` section — a typed
            :class:`~src.communication.config.CommunicationConfig` or a
            mapping using the documented schema field names.

    Returns:
        The compiled :class:`CommunicationPlan`.

    Raises:
        ValueError: On logically inconsistent combinations: more than one
            round, a non-radius or non-frozen topology, a non-broadcast or
            non-slotted transport, a missing or unsupported aggregation
            (``mean``, ``sum``, and ``max`` are enabled in this phase), relay
            settings, or an engineered payload whose dimension is not 4.
    """
    # Imported lazily (mirrored inside PyGAggregationProcessor) so that plans
    # which never aggregate do not pay the torch_geometric import cost.
    from src.communication.processors.pyg import (
        SUPPORTED_AGGREGATIONS,
        PyGAggregationProcessor,
    )

    scheme = "one_hop_mean"

    rounds = _field(config, "rounds_per_step", 1)
    if rounds != 1:
        raise _reject(
            scheme,
            "rounds_per_step={!r}, but one-hop aggregated delivery runs exactly one "
            "communication round per movement step. Use a multihop scheme for more "
            "rounds.".format(rounds),
        )

    cache_window = _field(config, "cache_window", 0)
    if not isinstance(cache_window, int) or isinstance(cache_window, bool) or cache_window < 0:
        raise _reject(
            scheme, "cache_window must be a non-negative integer, got {!r}.".format(cache_window)
        )

    topology_cfg = _field(config, "topology")
    topology_type = _field(topology_cfg, "type", "radius")
    if topology_type != "radius":
        raise _reject(
            scheme, "topology.type must be 'radius', got {!r}.".format(topology_type)
        )
    if _field(topology_cfg, "freeze_within_step", True) is not True:
        raise _reject(
            scheme,
            "topology.freeze_within_step must be true; dynamic per-round graph "
            "rebuilding is not supported by this runtime.",
        )
    radius_rule = _field(topology_cfg, "radius_rule", "sender")
    include_self_edges = _field(topology_cfg, "include_self_edges", False)

    transport_cfg = _field(config, "transport")
    transport_type = _field(transport_cfg, "type", "slotted_radius")
    if transport_type != "slotted_radius":
        raise _reject(
            scheme, "transport.type must be 'slotted_radius', got {!r}.".format(transport_type)
        )
    delivery = _field(transport_cfg, "delivery", "broadcast")
    if delivery != "broadcast":
        raise _reject(
            scheme,
            "transport.delivery must be 'broadcast', got {!r}; addressed delivery is a "
            "later protocol feature.".format(delivery),
        )

    processor_cfg = _field(config, "processor")
    aggregation = _field(processor_cfg, "aggregation", "none")
    if aggregation not in SUPPORTED_AGGREGATIONS:
        raise _reject(
            scheme,
            "processor.aggregation={!r}, but an aggregating scheme requires one of: {}. "
            "Set processor.aggregation explicitly (use 'one_hop_direct' for a preserved "
            "inbox instead).".format(aggregation, ", ".join(SUPPORTED_AGGREGATIONS)),
        )
    update = _field(processor_cfg, "update", "identity")
    if update != "identity":
        raise _reject(
            scheme, "processor.update must be 'identity', got {!r}.".format(update)
        )
    if (
        _field(processor_cfg, "ttl") is not None
        or _field(processor_cfg, "forwarding") is not None
        or bool(_field(processor_cfg, "packet_relay", False))
    ):
        raise _reject(
            scheme,
            "relay settings (ttl/forwarding/packet_relay) are set, but the aggregating "
            "processor never re-emits. Use 'multihop_relay' for packet relay.",
        )

    payload_cfg = _field(config, "payload")
    payload_type = _field(payload_cfg, "type", "engineered_vector")
    payload_dimension = _field(payload_cfg, "dimension", 4)
    if payload_type == "engineered_vector" and payload_dimension != 4:
        raise _reject(
            scheme,
            "payload.dimension={!r}, but the engineered bearing report is exactly "
            "4-dimensional: [observer_x, observer_y, direction_x, "
            "direction_y].".format(payload_dimension),
        )

    logger.debug(
        "Building 'one_hop_mean' plan (aggregation={}, radius_rule={}, "
        "include_self_edges={}, cache_window={})",
        aggregation,
        radius_rule,
        include_self_edges,
        cache_window,
    )
    return CommunicationPlan(
        topology=RadiusTopology(
            radius_rule=str(radius_rule),
            include_self_edges=bool(include_self_edges),
        ),
        transport=SlottedRadiusTransport(),
        processor=PyGAggregationProcessor(
            aggregation=str(aggregation),
            payload_dim=int(payload_dimension),
        ),
        rounds=1,
        cache_window=cache_window,
        graph_update="frozen",
        differentiable=False,
        output_contract="aggregated-vector",
    )


@register_communication_scheme("multihop_relay")
def build_multihop_relay(config: object) -> CommunicationPlan:
    """Build the ``multihop_relay`` plan (Phase 3).

    Semantics per the implementation plan ("Unchanged Relay Processing"):
    same-team radius topology, slotted radius broadcast transport (with
    cross-step carryover re-injection), and the first-seen unchanged-relay
    processor with duplicate suppression and TTL. Output contract
    ``preserved-inbox``: distinct origin-preserving payloads, byte-identical
    from origin to every receiver.

    Args:
        config: The ``environment.communication`` section — a typed
            :class:`~src.communication.config.CommunicationConfig` or a
            mapping using the documented schema field names.

    Returns:
        The compiled :class:`CommunicationPlan`.

    Raises:
        ValueError: On logically inconsistent combinations: ``rounds < 1``, a
            non-radius or non-frozen topology, a non-broadcast or non-slotted
            transport, any aggregation (which would break the preserved-inbox
            contract), a forwarding rule other than ``first_seen_unchanged``,
            disabled duplicate suppression, a missing or non-positive ``ttl``,
            a cache window below ``ttl`` (the ``C >= TTL`` relay-correctness
            floor), or an engineered payload whose dimension is not 4.
    """
    scheme = "multihop_relay"

    rounds = _field(config, "rounds_per_step", 1)
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 1:
        raise _reject(
            scheme,
            "rounds_per_step must be an integer >= 1, got {!r}. Relay moves a packet at "
            "most one graph hop per round; R caps within-step reachability.".format(rounds),
        )

    cache_window = _field(config, "cache_window", 0)
    if not isinstance(cache_window, int) or isinstance(cache_window, bool) or cache_window < 0:
        raise _reject(
            scheme, "cache_window must be a non-negative integer, got {!r}.".format(cache_window)
        )

    topology_cfg = _field(config, "topology")
    topology_type = _field(topology_cfg, "type", "radius")
    if topology_type != "radius":
        raise _reject(
            scheme, "topology.type must be 'radius', got {!r}.".format(topology_type)
        )
    if _field(topology_cfg, "freeze_within_step", True) is not True:
        raise _reject(
            scheme,
            "topology.freeze_within_step must be true; dynamic per-round graph "
            "rebuilding is not supported by this runtime.",
        )
    radius_rule = _field(topology_cfg, "radius_rule", "sender")
    include_self_edges = _field(topology_cfg, "include_self_edges", False)

    transport_cfg = _field(config, "transport")
    transport_type = _field(transport_cfg, "type", "slotted_radius")
    if transport_type != "slotted_radius":
        raise _reject(
            scheme, "transport.type must be 'slotted_radius', got {!r}.".format(transport_type)
        )
    delivery = _field(transport_cfg, "delivery", "broadcast")
    if delivery != "broadcast":
        raise _reject(
            scheme,
            "transport.delivery must be 'broadcast', got {!r}; addressed delivery is a "
            "later protocol feature.".format(delivery),
        )

    processor_cfg = _field(config, "processor")
    aggregation = _field(processor_cfg, "aggregation", "none")
    if aggregation != "none":
        raise _reject(
            scheme,
            "processor.aggregation={!r} would break the preserved-inbox contract; relayed "
            "payloads are forwarded byte-identical and never merged. Use an aggregating "
            "scheme for aggregated delivery.".format(aggregation),
        )
    update = _field(processor_cfg, "update", "identity")
    if update != "identity":
        raise _reject(
            scheme, "processor.update must be 'identity', got {!r}.".format(update)
        )
    forwarding = _field(processor_cfg, "forwarding", "first_seen_unchanged")
    if forwarding != "first_seen_unchanged":
        raise _reject(
            scheme,
            "processor.forwarding must be 'first_seen_unchanged' (the only relay rule), "
            "got {!r}.".format(forwarding),
        )
    duplicate_suppression = _field(processor_cfg, "duplicate_suppression", True)
    if duplicate_suppression is not True:
        raise _reject(
            scheme,
            "processor.duplicate_suppression={!r}, but relay correctness requires it: "
            "without duplicate suppression, cycles re-deliver and re-forward packets "
            "without bound. Remove the key or set it to true.".format(duplicate_suppression),
        )
    ttl = _field(processor_cfg, "ttl")
    if ttl is None:
        raise _reject(
            scheme,
            "processor.ttl is required: it is the packet hop budget (one hop per round). "
            "Set processor.ttl >= 1 and ensure cache_window >= ttl.",
        )
    if not isinstance(ttl, int) or isinstance(ttl, bool) or ttl < 1:
        raise _reject(scheme, "processor.ttl must be an integer >= 1, got {!r}.".format(ttl))
    if cache_window < ttl:
        raise _reject(
            scheme,
            "cache_window={} is below processor.ttl={}, violating the relay-correctness "
            "floor C >= TTL: duplicate suppression must remember a packet id for its whole "
            "TTL lifetime, or a still-live copy is re-accepted as new and cycles duplicate "
            "without bound. Increase cache_window to at least {} or reduce ttl. "
            "(TTL > rounds_per_step is fine: the cache window spans movement "
            "steps.)".format(cache_window, ttl, ttl),
        )

    payload_cfg = _field(config, "payload")
    payload_type = _field(payload_cfg, "type", "engineered_vector")
    payload_dimension = _field(payload_cfg, "dimension", 4)
    if payload_type == "engineered_vector" and payload_dimension != 4:
        raise _reject(
            scheme,
            "payload.dimension={!r}, but the engineered bearing report is exactly "
            "4-dimensional: [observer_x, observer_y, direction_x, "
            "direction_y].".format(payload_dimension),
        )

    logger.debug(
        "Building 'multihop_relay' plan (rounds={}, ttl={}, cache_window={}, radius_rule={})",
        rounds,
        ttl,
        cache_window,
        radius_rule,
    )
    relay_state = RelayState()
    return CommunicationPlan(
        topology=RadiusTopology(
            radius_rule=str(radius_rule),
            include_self_edges=bool(include_self_edges),
        ),
        transport=RelayCarryoverTransport(relay_state, ttl),
        processor=RelayProcessor(ttl=ttl, cache_window=cache_window, state=relay_state),
        rounds=rounds,
        cache_window=cache_window,
        graph_update="frozen",
        differentiable=False,
        output_contract="preserved-inbox",
    )
