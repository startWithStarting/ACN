"""ACN communication package.

Phase 0 of ``docs/communication_implementation_plan.md``: typed configuration
(:mod:`src.communication.config`), core data types
(:mod:`src.communication.types`), plan composition and the ``none`` baseline
(:mod:`src.communication.plans`), and the named-scheme registry/compiler
(:mod:`src.communication.registry`).

Phase 1 (Radius Graph And One-Hop Direct Delivery): the same-team radius
topology (:mod:`src.communication.topology`), the common slotted radius
transport (:mod:`src.communication.transport`), the fixed-round scheduler
(:mod:`src.communication.runtime`), the engineered bearing-report source
(:mod:`src.communication.sources`), the inbox-preserving direct processor
(:mod:`src.communication.processors`), communication trace records
(:mod:`src.communication.tracing`), and the registered ``one_hop_direct``
scheme (:mod:`src.communication.schemes`).

Environment integration: the parallel environment
(:mod:`src.env.parallel_env`) compiles the configured plan at construction and
runs the communication rounds inside ``step(actions)`` before movement; the
AEC environment reports enabled communication as unsupported.

The legacy :class:`CommunicationModel` hierarchy from
:mod:`src.communication.models` remains re-exported for backward
compatibility; it will be deprecated once compatibility adapters exist.
"""

# Legacy models (kept for backward compatibility; external code imports these).
from .models import CommunicationModel, GNNCommunicationModel, NoCommunicationModel

# Phase 0: core data types.
from .types import (
    BROADCAST_DESTINATION,
    NO_PREVIOUS_HOP,
    CacheEntry,
    CommunicationGraph,
    CommunicationResult,
    EdgeMessageBatch,
    InboxBatch,
    MessageCache,
    PacketBatch,
)

# Phase 0: typed configuration.
from .config import (
    VALID_RADIUS_RULES,
    VALID_SCHEMES,
    CommunicationConfig,
    ConfigError,
    PayloadConfig,
    ProcessorConfig,
    TopologyConfig,
    TransportConfig,
    parse_communication_config,
)

# Phase 0: plan composition and the `none` baseline components.
from .plans import (
    CommunicationPlan,
    CommunicationProcessor,
    CommunicationTopology,
    NoneProcessor,
    NoneTopology,
    NoneTransport,
    RoundTransport,
    build_none_plan,
    empty_communication_graph,
    empty_communication_result,
    empty_edge_message_batch,
)

# Phase 0: named-scheme registry and compiler.
from .registry import (
    SchemeBuilder,
    create_communication_plan,
    list_communication_schemes,
    register_communication_scheme,
)

# Phase 1: topology, transport, runtime, sources, processors, tracing, schemes.
from .topology import RadiusTopology
from .transport import Frame, SlottedRadiusTransport, concat_edge_message_batches
from .runtime import CachedDelivery, CommunicationRuntime
from .sources import EngineeredBearingSource, MessageSource, create_message_source
from .processors import DirectProcessor
from .tracing import (
    DELIVERY_RECORD_TYPE,
    GRAPH_RECORD_TYPE,
    communication_delivery_records,
    communication_graph_record,
)
from .schemes import build_one_hop_direct

__all__ = [
    # Legacy models (backward compatibility).
    "CommunicationModel",
    "GNNCommunicationModel",
    "NoCommunicationModel",
    # Core data types.
    "BROADCAST_DESTINATION",
    "NO_PREVIOUS_HOP",
    "CacheEntry",
    "CommunicationGraph",
    "CommunicationResult",
    "EdgeMessageBatch",
    "InboxBatch",
    "MessageCache",
    "PacketBatch",
    # Typed configuration.
    "VALID_RADIUS_RULES",
    "VALID_SCHEMES",
    "CommunicationConfig",
    "ConfigError",
    "PayloadConfig",
    "ProcessorConfig",
    "TopologyConfig",
    "TransportConfig",
    "parse_communication_config",
    # Plans and `none` baseline.
    "CommunicationPlan",
    "CommunicationProcessor",
    "CommunicationTopology",
    "NoneProcessor",
    "NoneTopology",
    "NoneTransport",
    "RoundTransport",
    "build_none_plan",
    "empty_communication_graph",
    "empty_communication_result",
    "empty_edge_message_batch",
    # Registry and compiler.
    "SchemeBuilder",
    "create_communication_plan",
    "list_communication_schemes",
    "register_communication_scheme",
    # Phase 1 runtime components.
    "RadiusTopology",
    "Frame",
    "SlottedRadiusTransport",
    "concat_edge_message_batches",
    "CachedDelivery",
    "CommunicationRuntime",
    "EngineeredBearingSource",
    "MessageSource",
    "create_message_source",
    "DirectProcessor",
    "DELIVERY_RECORD_TYPE",
    "GRAPH_RECORD_TYPE",
    "communication_delivery_records",
    "communication_graph_record",
    "build_one_hop_direct",
]
