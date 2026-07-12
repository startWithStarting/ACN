"""Typed parsing and validation for the ``environment.communication`` config block.

This module converts the plain dict produced by YAML loading into immutable,
validated dataclasses. It performs no file I/O: dict in, validated config out.

The communication model is organized on two independent axes (see
``docs/communication_decision_log.md``, "Round And Cache Model"):

- ``rounds_per_step`` (R): the number of synchronous communication rounds executed
  per movement step over the graph frozen at the start of the step. Movement is
  selected after the R rounds, so same-step communication affects same-step movement.
- ``cache_window`` (C): a sliding window, measured in rounds, of retained past
  messages. The window spans movement-step boundaries; eviction is oldest round
  first with equal-age ties broken by delivery sequence. The cache is distinct
  from a processor's intrinsic round-to-round hidden state, which exists even at
  ``C = 0``.

Relay correctness requires ``C >= TTL``: duplicate suppression must remember a
packet identity for its whole TTL lifetime, so relay configurations with
``ttl > cache_window`` are rejected here. ``TTL > R`` is allowed because the
cache window spans movement steps.

An absent ``communication`` block, ``enabled: false``, or ``scheme: "none"`` all
normalize to the same canonical disabled configuration, so existing scenarios
without a communication block are unaffected.
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Mapping, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger("acn.communication")

__all__ = [
    "ConfigError",
    "VALID_SCHEMES",
    "VALID_RADIUS_RULES",
    "TopologyConfig",
    "PayloadConfig",
    "ProcessorConfig",
    "TransportConfig",
    "CommunicationConfig",
    "parse_communication_config",
]


class ConfigError(ValueError):
    """Raised when a communication configuration is invalid or inconsistent."""


#: Named communication schemes accepted by the scheme registry/compiler.
VALID_SCHEMES: Tuple[str, ...] = (
    "none",
    "one_hop_direct",
    "one_hop_mean",
    "multihop_relay",
    "multihop_gnn",
)

#: Radius-ownership rules deciding which agent's radius validates an edge.
VALID_RADIUS_RULES: Tuple[str, ...] = ("mutual", "sender", "receiver", "minimum")

# Warn-only known-value sets. Unknown values are tolerated here so that later
# phases can add components; the scheme compiler enforces the supported subset.
_KNOWN_TOPOLOGY_TYPES: Tuple[str, ...] = ("radius", "none")
_KNOWN_PAYLOAD_TYPES: Tuple[str, ...] = ("engineered_vector", "learned")
_KNOWN_TRANSPORT_TYPES: Tuple[str, ...] = ("slotted_radius", "none")
_KNOWN_DELIVERY_MODES: Tuple[str, ...] = ("broadcast", "addressed")
_KNOWN_AGGREGATIONS: Tuple[str, ...] = (
    "none",
    "sum",
    "mean",
    "max",
    "min",
    "softmax",
    "attention",
    "multi",
    "deepsets",
    "set_transformer",
    "custom",
)


# ---------------------------------------------------------------------------
# Raw-value helpers
# ---------------------------------------------------------------------------


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    """Return ``value`` if it is a mapping, otherwise raise a ConfigError."""
    if not isinstance(value, MappingABC):
        raise ConfigError(
            "{} must be a mapping of keys to values, got {} ({!r}). "
            "Check the YAML indentation of this block.".format(
                context, type(value).__name__, value
            )
        )
    return value


def _warn_unknown_keys(context: str, data: Mapping[str, Any], known: Tuple[str, ...]) -> None:
    unknown = sorted(set(data.keys()) - set(known))
    if unknown:
        logger.warning(
            "Ignoring unknown keys in {}: {}. Known keys: {}",
            context,
            ", ".join(unknown),
            ", ".join(known),
        )


def _get_bool(data: Mapping[str, Any], key: str, default: bool, context: str) -> bool:
    value = data.get(key, default)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ConfigError(
            "{}.{} must be a boolean (true/false), got {} ({!r}).".format(
                context, key, type(value).__name__, value
            )
        )
    return value


def _get_int(data: Mapping[str, Any], key: str, default: int, context: str) -> int:
    value = data.get(key, default)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(
            "{}.{} must be an integer, got {} ({!r}).".format(
                context, key, type(value).__name__, value
            )
        )
    return value


def _get_str(data: Mapping[str, Any], key: str, default: str, context: str) -> str:
    value = data.get(key, default)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ConfigError(
            "{}.{} must be a string, got {} ({!r}).".format(
                context, key, type(value).__name__, value
            )
        )
    return value


def _get_optional_int(data: Mapping[str, Any], key: str, context: str) -> Optional[int]:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(
            "{}.{} must be an integer, got {} ({!r}).".format(
                context, key, type(value).__name__, value
            )
        )
    return value


def _get_optional_bool(data: Mapping[str, Any], key: str, context: str) -> Optional[bool]:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ConfigError(
            "{}.{} must be a boolean (true/false), got {} ({!r}).".format(
                context, key, type(value).__name__, value
            )
        )
    return value


def _get_optional_str(data: Mapping[str, Any], key: str, context: str) -> Optional[str]:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(
            "{}.{} must be a string, got {} ({!r}).".format(
                context, key, type(value).__name__, value
            )
        )
    return value


def _warn_unknown_value(context: str, value: str, known: Tuple[str, ...]) -> None:
    if value not in known:
        logger.warning(
            "{} is '{}', which is not one of the known values ({}). "
            "The scheme compiler will reject it if it is unsupported.",
            context,
            value,
            ", ".join(known),
        )


# ---------------------------------------------------------------------------
# Section dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TopologyConfig:
    """Communication graph construction settings.

    Attributes:
        type: Topology builder name. The initial builder is ``"radius"``.
        radius_rule: Which agent's radius validates a directed edge. One of
            ``mutual``, ``sender``, ``receiver``, ``minimum``.
        include_self_edges: Whether agents receive their own transmissions.
        freeze_within_step: Whether the graph is built once per movement step
            and frozen across all R rounds (the baseline behavior).
    """

    type: str = "radius"
    radius_rule: str = "sender"
    include_self_edges: bool = False
    freeze_within_step: bool = True

    def __post_init__(self) -> None:
        if self.radius_rule not in VALID_RADIUS_RULES:
            raise ConfigError(
                "topology.radius_rule must be one of {} (got '{}').".format(
                    ", ".join(VALID_RADIUS_RULES), self.radius_rule
                )
            )

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "TopologyConfig":
        """Build a TopologyConfig from the ``topology`` sub-block (None means defaults)."""
        if data is None:
            return cls()
        data = _require_mapping(data, "communication.topology")
        context = "communication.topology"
        _warn_unknown_keys(
            context, data, ("type", "radius_rule", "include_self_edges", "freeze_within_step")
        )
        topology_type = _get_str(data, "type", "radius", context)
        _warn_unknown_value("communication.topology.type", topology_type, _KNOWN_TOPOLOGY_TYPES)
        return cls(
            type=topology_type,
            radius_rule=_get_str(data, "radius_rule", "sender", context),
            include_self_edges=_get_bool(data, "include_self_edges", False, context),
            freeze_within_step=_get_bool(data, "freeze_within_step", True, context),
        )


@dataclass(frozen=True)
class PayloadConfig:
    """Message payload settings.

    Attributes:
        type: Payload source kind, e.g. ``"engineered_vector"`` or ``"learned"``.
        dimension: Payload vector dimension; must be a positive integer.
        coordinate_frame: Frame of any spatial payload fields (``"global"`` for
            the engineered bearing-report baseline).
    """

    type: str = "engineered_vector"
    dimension: int = 4
    coordinate_frame: str = "global"

    def __post_init__(self) -> None:
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int):
            raise ConfigError(
                "payload.dimension must be an integer, got {} ({!r}).".format(
                    type(self.dimension).__name__, self.dimension
                )
            )
        if self.dimension <= 0:
            raise ConfigError(
                "payload.dimension must be a positive integer (got {}). It is the "
                "size of each message vector.".format(self.dimension)
            )

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "PayloadConfig":
        """Build a PayloadConfig from the ``payload`` sub-block (None means defaults)."""
        if data is None:
            return cls()
        data = _require_mapping(data, "communication.payload")
        context = "communication.payload"
        _warn_unknown_keys(context, data, ("type", "dimension", "coordinate_frame"))
        payload_type = _get_str(data, "type", "engineered_vector", context)
        _warn_unknown_value("communication.payload.type", payload_type, _KNOWN_PAYLOAD_TYPES)
        return cls(
            type=payload_type,
            dimension=_get_int(data, "dimension", 4, context),
            coordinate_frame=_get_str(data, "coordinate_frame", "global", context),
        )


@dataclass(frozen=True)
class ProcessorConfig:
    """Communication processor settings.

    Required fields have generic defaults matching the ``one_hop_direct``
    baseline; the optional fields stay ``None`` unless a scheme uses them.

    Attributes:
        backend: Processor implementation family (``"pyg"``, ``"torch"``,
            ``"identity"``).
        aggregation: Aggregation name; ``"none"`` preserves distinct messages.
        update: Node update rule name (``"identity"`` for rule-based schemes).
        model: Optional learned model name (e.g. ``"graph_sage"``).
        layers: Optional learned model depth. One configured layer corresponds
            to one communication round unless a scheme documents otherwise.
        hidden_dimension: Optional learned hidden state width.
        shared_weights: Optional flag for learned schemes: ``true`` shares one
            set of round weights across all R rounds; ``false``/absent keeps
            separate weights per round (the ``multihop_gnn`` default).
        packet_relay: Optional flag for packet-preserving relay state.
        forwarding: Optional relay forwarding rule (e.g. ``"first_seen_unchanged"``).
        ttl: Optional packet time-to-live in hops (one hop per round).
        duplicate_suppression: Optional relay duplicate-drop flag.
    """

    backend: str = "pyg"
    aggregation: str = "none"
    update: str = "identity"
    model: Optional[str] = None
    layers: Optional[int] = None
    hidden_dimension: Optional[int] = None
    shared_weights: Optional[bool] = None
    packet_relay: Optional[bool] = None
    forwarding: Optional[str] = None
    ttl: Optional[int] = None
    duplicate_suppression: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.ttl is not None and self.ttl < 1:
            raise ConfigError(
                "processor.ttl must be >= 1 when set (got {}). TTL counts graph hops, "
                "one hop per communication round.".format(self.ttl)
            )
        if self.layers is not None and self.layers < 1:
            raise ConfigError(
                "processor.layers must be >= 1 when set (got {}).".format(self.layers)
            )
        if self.hidden_dimension is not None and self.hidden_dimension < 1:
            raise ConfigError(
                "processor.hidden_dimension must be >= 1 when set (got {}).".format(
                    self.hidden_dimension
                )
            )

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "ProcessorConfig":
        """Build a ProcessorConfig from the ``processor`` sub-block (None means defaults)."""
        if data is None:
            return cls()
        data = _require_mapping(data, "communication.processor")
        context = "communication.processor"
        _warn_unknown_keys(
            context,
            data,
            (
                "backend",
                "aggregation",
                "update",
                "model",
                "layers",
                "hidden_dimension",
                "shared_weights",
                "packet_relay",
                "forwarding",
                "ttl",
                "duplicate_suppression",
            ),
        )
        aggregation = _get_str(data, "aggregation", "none", context)
        _warn_unknown_value("communication.processor.aggregation", aggregation, _KNOWN_AGGREGATIONS)
        return cls(
            backend=_get_str(data, "backend", "pyg", context),
            aggregation=aggregation,
            update=_get_str(data, "update", "identity", context),
            model=_get_optional_str(data, "model", context),
            layers=_get_optional_int(data, "layers", context),
            hidden_dimension=_get_optional_int(data, "hidden_dimension", context),
            shared_weights=_get_optional_bool(data, "shared_weights", context),
            packet_relay=_get_optional_bool(data, "packet_relay", context),
            forwarding=_get_optional_str(data, "forwarding", context),
            ttl=_get_optional_int(data, "ttl", context),
            duplicate_suppression=_get_optional_bool(data, "duplicate_suppression", context),
        )


@dataclass(frozen=True)
class TransportConfig:
    """Transport settings for the common synchronous slotted radius transport.

    Attributes:
        type: Transport name; the initial transport is ``"slotted_radius"``.
        delivery: ``"broadcast"`` or ``"addressed"``.
        free: Whether communication carries no reward cost (the initial model).
    """

    type: str = "slotted_radius"
    delivery: str = "broadcast"
    free: bool = True

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "TransportConfig":
        """Build a TransportConfig from the ``transport`` sub-block (None means defaults)."""
        if data is None:
            return cls()
        data = _require_mapping(data, "communication.transport")
        context = "communication.transport"
        _warn_unknown_keys(context, data, ("type", "delivery", "free"))
        transport_type = _get_str(data, "type", "slotted_radius", context)
        _warn_unknown_value("communication.transport.type", transport_type, _KNOWN_TRANSPORT_TYPES)
        delivery = _get_str(data, "delivery", "broadcast", context)
        _warn_unknown_value("communication.transport.delivery", delivery, _KNOWN_DELIVERY_MODES)
        return cls(
            type=transport_type,
            delivery=delivery,
            free=_get_bool(data, "free", True, context),
        )


# ---------------------------------------------------------------------------
# Top-level communication config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CommunicationConfig:
    """Validated ``environment.communication`` configuration.

    The default-constructed instance is the canonical disabled configuration.
    Use :meth:`from_dict` to parse a loaded YAML block; ``None`` (absent block)
    parses to the disabled configuration.

    Attributes:
        enabled: Whether communication runs at all.
        scheme: Named scheme; one of :data:`VALID_SCHEMES`.
        rounds_per_step: R, synchronous communication rounds per movement step.
        cache_window: C, sliding window of retained past rounds; spans movement
            step boundaries. Must satisfy ``cache_window >= processor.ttl`` for
            relay schemes.
        topology: Graph construction settings.
        payload: Message payload settings.
        processor: Processing settings.
        transport: Delivery transport settings.
    """

    enabled: bool = False
    scheme: str = "none"
    rounds_per_step: int = 0
    cache_window: int = 0
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    payload: PayloadConfig = field(default_factory=PayloadConfig)
    processor: ProcessorConfig = field(default_factory=ProcessorConfig)
    transport: TransportConfig = field(default_factory=TransportConfig)

    def __post_init__(self) -> None:
        self._validate()

    @property
    def is_active(self) -> bool:
        """True when communication actually runs (enabled and not scheme 'none')."""
        return self.enabled and self.scheme != "none"

    @property
    def uses_relay(self) -> bool:
        """True when the configuration implies packet-preserving relay semantics."""
        return (
            self.scheme == "multihop_relay"
            or bool(self.processor.packet_relay)
            or self.processor.forwarding is not None
        )

    def _validate(self) -> None:
        """Check cross-field invariants; raise ConfigError with actionable messages."""
        if self.scheme not in VALID_SCHEMES:
            raise ConfigError(
                "Unknown communication scheme '{}'. Valid schemes: {}.".format(
                    self.scheme, ", ".join(VALID_SCHEMES)
                )
            )
        if isinstance(self.rounds_per_step, bool) or not isinstance(self.rounds_per_step, int):
            raise ConfigError(
                "communication.rounds_per_step must be an integer, got {} ({!r}).".format(
                    type(self.rounds_per_step).__name__, self.rounds_per_step
                )
            )
        if isinstance(self.cache_window, bool) or not isinstance(self.cache_window, int):
            raise ConfigError(
                "communication.cache_window must be an integer, got {} ({!r}).".format(
                    type(self.cache_window).__name__, self.cache_window
                )
            )
        if self.rounds_per_step < 0:
            raise ConfigError(
                "communication.rounds_per_step must be >= 0 (got {}).".format(
                    self.rounds_per_step
                )
            )
        if self.cache_window < 0:
            raise ConfigError(
                "communication.cache_window must be >= 0 (got {}). It is the number of "
                "past communication rounds retained in the message cache.".format(
                    self.cache_window
                )
            )
        if not self.is_active:
            return
        if self.rounds_per_step < 1:
            raise ConfigError(
                "communication.rounds_per_step must be >= 1 when communication is enabled "
                "(got {} for scheme '{}'). Each movement step runs R synchronous rounds; "
                "set enabled: false or scheme: none to disable communication.".format(
                    self.rounds_per_step, self.scheme
                )
            )
        if self.uses_relay:
            ttl = self.processor.ttl
            if ttl is None:
                raise ConfigError(
                    "Relay communication (scheme '{}') requires processor.ttl, the packet "
                    "hop lifetime in rounds. Set processor.ttl and ensure "
                    "cache_window >= ttl.".format(self.scheme)
                )
            if ttl > self.cache_window:
                raise ConfigError(
                    "Relay configuration violates the C >= TTL invariant: processor.ttl={} "
                    "exceeds cache_window={}. Duplicate suppression must remember a packet "
                    "for its whole TTL lifetime or cycles re-deliver evicted packets as "
                    "new; increase cache_window to at least {} or reduce ttl. "
                    "(TTL > rounds_per_step is fine: the cache window spans movement "
                    "steps.)".format(ttl, self.cache_window, ttl)
                )

    @classmethod
    def disabled(cls) -> "CommunicationConfig":
        """Return the canonical disabled configuration (equivalent to scheme 'none')."""
        return cls()

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "CommunicationConfig":
        """Parse the ``environment.communication`` block of a loaded YAML config.

        Args:
            data: The plain-dict communication block, or None when the block is
                absent. An absent block, ``enabled: false``, or ``scheme: "none"``
                all yield the canonical disabled configuration.

        Returns:
            A validated, immutable CommunicationConfig.

        Raises:
            ConfigError: If the block is malformed or violates an invariant
                (unknown scheme, rounds < 1 while enabled, negative cache_window,
                non-positive payload dimension, or a relay scheme with
                ttl > cache_window).
        """
        if data is None:
            return cls.disabled()
        data = _require_mapping(data, "environment.communication")
        context = "communication"
        _warn_unknown_keys(
            context,
            data,
            (
                "enabled",
                "scheme",
                "rounds_per_step",
                "cache_window",
                "topology",
                "payload",
                "processor",
                "transport",
            ),
        )
        enabled = _get_bool(data, "enabled", False, context)
        scheme = _get_optional_str(data, "scheme", context)
        if scheme is not None and scheme not in VALID_SCHEMES:
            raise ConfigError(
                "Unknown communication scheme '{}'. Valid schemes: {}.".format(
                    scheme, ", ".join(VALID_SCHEMES)
                )
            )
        if enabled and scheme is None:
            raise ConfigError(
                "communication.enabled is true but no scheme is set. Choose one of: {} "
                "(or set enabled: false).".format(", ".join(VALID_SCHEMES))
            )
        if not enabled or scheme is None or scheme == "none":
            if enabled:
                logger.debug("Communication scheme 'none' selected; treating as disabled.")
            elif scheme is not None or len(data) > 1:
                logger.debug(
                    "Communication block present but disabled; ignoring its other settings."
                )
            return cls.disabled()
        config = cls(
            enabled=True,
            scheme=scheme,
            rounds_per_step=_get_int(data, "rounds_per_step", 1, context),
            cache_window=_get_int(data, "cache_window", 0, context),
            topology=TopologyConfig.from_dict(data.get("topology")),
            payload=PayloadConfig.from_dict(data.get("payload")),
            processor=ProcessorConfig.from_dict(data.get("processor")),
            transport=TransportConfig.from_dict(data.get("transport")),
        )
        logger.debug(
            "Parsed communication config: scheme={} rounds_per_step={} cache_window={}",
            config.scheme,
            config.rounds_per_step,
            config.cache_window,
        )
        return config

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain nested dict mirroring the YAML schema (for tracing/hashing)."""
        return asdict(self)


def parse_communication_config(data: Optional[Mapping[str, Any]]) -> CommunicationConfig:
    """Parse an ``environment.communication`` block; None means disabled.

    Convenience wrapper around :meth:`CommunicationConfig.from_dict`.

    Args:
        data: The plain-dict communication block from the loaded YAML, or None.

    Returns:
        A validated, immutable CommunicationConfig.

    Raises:
        ConfigError: If the block is malformed or violates an invariant.
    """
    return CommunicationConfig.from_dict(data)
