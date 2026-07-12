"""ACN communication package.

Phase 0 of ``docs/communication_implementation_plan.md``: typed configuration
(:mod:`src.communication.config`), core data types
(:mod:`src.communication.types`), plan composition and the ``none`` baseline
(:mod:`src.communication.plans`), and the named-scheme registry/compiler
(:mod:`src.communication.registry`). No runtime integration yet: environments,
agents, and entry points do not consume this package.

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
]
