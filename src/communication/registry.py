"""Named communication scheme registry and plan compiler.

This module mirrors the decorator-based registration pattern of
:mod:`src.agents.registry`: scheme builders register under a string name and
``create_communication_plan`` compiles a configuration into a validated
:class:`~src.communication.plans.CommunicationPlan`.

Example:
    Register a new scheme builder::

        @register_communication_scheme("one_hop_direct")
        def build_one_hop_direct(config) -> CommunicationPlan:
            ...

    Compile a plan from configuration::

        plan = create_communication_plan(config)

Phase 0 registers only the ``none`` baseline. The other named schemes
(``one_hop_direct``, ``one_hop_mean``, ``multihop_relay``, ``multihop_gnn``)
are reserved: compiling them raises ``NotImplementedError`` naming the
delivery phase that adds them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Callable, Dict, List

from src.communication.plans import (
    CommunicationPlan,
    CommunicationProcessor,
    CommunicationTopology,
    RoundTransport,
    build_none_plan,
)
from src.utils.logger import get_logger

logger = get_logger("acn.communication")

# A scheme builder turns the (typed or mapping) communication config section
# into a compiled CommunicationPlan.
SchemeBuilder = Callable[[object], CommunicationPlan]

# Global scheme registry.
_SCHEME_REGISTRY: Dict[str, SchemeBuilder] = {}

# Named schemes defined by the implementation plan but not implemented yet,
# mapped to the delivery phase that adds them. They must NOT be registered
# until that phase lands.
_PLANNED_SCHEME_PHASES: Dict[str, str] = {
    "one_hop_direct": "Phase 1 (Radius Graph And One-Hop Direct Delivery)",
    "one_hop_mean": "Phase 2 (PyG Aggregations And Rule-Based Processing)",
    "multihop_relay": "Phase 3 (Multi-Hop Unchanged Relay)",
    "multihop_gnn": "Phase 4 (Learned PyG Communication)",
}


def register_communication_scheme(name: str) -> Callable[[SchemeBuilder], SchemeBuilder]:
    """Decorator to register a communication scheme builder.

    Args:
        name: The string identifier for this scheme (used as the ``scheme``
            value in the ``environment.communication`` config section).

    Returns:
        A decorator that registers the builder and returns it unchanged.

    Example:
        Register a direct one-hop scheme::

            @register_communication_scheme("one_hop_direct")
            def build_one_hop_direct(config) -> CommunicationPlan:
                ...
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"Communication scheme name must be a non-empty string, got {name!r}")

    def decorator(builder: SchemeBuilder) -> SchemeBuilder:
        if name in _SCHEME_REGISTRY:
            logger.warning("Overwriting existing communication scheme '{}' in registry", name)
        _SCHEME_REGISTRY[name] = builder
        logger.debug("Registered communication scheme: {}", name)
        return builder

    return decorator


def list_communication_schemes() -> List[str]:
    """Return the list of registered communication scheme names."""
    return list(_SCHEME_REGISTRY.keys())


def _get_field(config: object, name: str, default: object = None) -> object:
    """Read a field from a typed config object or a mapping.

    Args:
        config: Typed config object, mapping, or ``None``.
        name: Field name per the documented configuration schema.
        default: Value returned when the field (or config) is absent.

    Returns:
        The field value or ``default``.
    """
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _validate_plan(scheme: str, plan: CommunicationPlan, config: object) -> None:
    """Re-validate plan-level invariants after a builder has run.

    Args:
        scheme: The scheme name the plan was compiled from.
        plan: The compiled plan.
        config: The communication config section the builder consumed.

    Raises:
        TypeError: If the builder returned a non-plan or a component that
            does not satisfy its protocol.
        ValueError: If a round/cache/TTL invariant is violated.
    """
    if not isinstance(plan, CommunicationPlan):
        raise TypeError(
            f"Builder for communication scheme '{scheme}' must return a CommunicationPlan, "
            f"got {type(plan).__name__}"
        )

    protocol_expectations = (
        ("topology", CommunicationTopology),
        ("transport", RoundTransport),
        ("processor", CommunicationProcessor),
    )
    for attr_name, protocol in protocol_expectations:
        component = getattr(plan, attr_name)
        if not isinstance(component, protocol):
            raise TypeError(
                f"Communication scheme '{scheme}' compiled a {attr_name} component of type "
                f"{type(component).__name__} that does not satisfy the {protocol.__name__} "
                "protocol"
            )

    for attr_name in ("rounds", "cache_window"):
        value = getattr(plan, attr_name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(
                f"Communication scheme '{scheme}' compiled {attr_name}={value!r}; "
                f"{attr_name} must be a non-negative integer"
            )

    if not isinstance(plan.graph_update, str) or not plan.graph_update:
        raise ValueError(
            f"Communication scheme '{scheme}' compiled graph_update={plan.graph_update!r}; "
            "graph_update must be a non-empty string (Phase 0 supports only 'frozen')"
        )
    if not isinstance(plan.output_contract, str) or not plan.output_contract:
        raise ValueError(
            f"Communication scheme '{scheme}' compiled output_contract="
            f"{plan.output_contract!r}; every plan must declare its policy-facing output "
            "contract as a non-empty string"
        )
    if not isinstance(plan.differentiable, bool):
        raise ValueError(
            f"Communication scheme '{scheme}' compiled differentiable="
            f"{plan.differentiable!r}; differentiable must be a bool"
        )

    if scheme == "none":
        if plan.rounds != 0 or plan.cache_window != 0 or plan.output_contract != "empty":
            raise ValueError(
                "The 'none' scheme must compile to rounds=0, cache_window=0, and "
                f"output_contract='empty'; got rounds={plan.rounds}, "
                f"cache_window={plan.cache_window}, output_contract={plan.output_contract!r}"
            )
        return

    if plan.rounds < 1:
        raise ValueError(
            f"Communication scheme '{scheme}' is enabled but compiled with "
            f"rounds={plan.rounds}; every enabled non-none scheme must run at least one "
            "synchronous communication round per movement step (rounds >= 1)"
        )

    ttl = _get_field(_get_field(config, "processor"), "ttl")
    if ttl is not None:
        try:
            ttl_value = int(ttl)
        except (TypeError, ValueError):
            raise ValueError(
                f"Communication scheme '{scheme}' configured processor ttl={ttl!r}; "
                "ttl must be an integer number of hops/rounds"
            )
        if ttl_value > plan.cache_window:
            raise ValueError(
                f"Relay-correctness invariant violated for scheme '{scheme}': processor "
                f"ttl={ttl_value} exceeds cache_window={plan.cache_window}. Duplicate "
                "suppression must remember a packet for its whole lifetime, and a copy can "
                "still arrive up to TTL rounds after creation, so the message-cache window C "
                "must satisfy C >= TTL. TTL > R is allowed because the cache window spans "
                "movement-step boundaries."
            )


def create_communication_plan(config: object) -> CommunicationPlan:
    """Compile a communication configuration into a validated plan.

    Args:
        config: The ``environment.communication`` configuration section — the
            typed object from :mod:`src.communication.config` or any mapping
            using the documented schema field names (``enabled``, ``scheme``,
            ``rounds_per_step``, ``cache_window``, ``topology``, ``payload``,
            ``processor``, ``transport``). ``None``, an absent ``enabled``
            field, or ``enabled: false`` compiles the ``none`` baseline so a
            no-communication run still receives an explicit plan (and an
            explicit empty view), never a missing one. Communication defaults
            to disabled, matching
            :func:`src.communication.config.parse_communication_config`.

    Returns:
        The compiled and validated :class:`CommunicationPlan`.

    Raises:
        NotImplementedError: If the scheme is defined by the implementation
            plan but its delivery phase has not landed yet.
        ValueError: If the scheme is unknown or a plan-level invariant fails.
        TypeError: If the builder returns an invalid plan or component.
    """
    # Default matches config.parse_communication_config: communication is
    # disabled unless the config explicitly enables it (migration rule).
    enabled = bool(_get_field(config, "enabled", False))
    scheme = _get_field(config, "scheme", "none")
    if scheme is None:
        scheme = "none"
    if not isinstance(scheme, str):
        raise ValueError(
            f"Communication scheme name must be a string, got {type(scheme).__name__}: "
            f"{scheme!r}"
        )

    if not enabled and scheme != "none":
        logger.debug("Communication disabled; compiling 'none' instead of scheme '{}'", scheme)
        scheme = "none"

    builder = _SCHEME_REGISTRY.get(scheme)
    if builder is None:
        if scheme in _PLANNED_SCHEME_PHASES:
            raise NotImplementedError(
                f"Communication scheme '{scheme}' is defined by "
                "docs/communication_implementation_plan.md but is not implemented yet; it is "
                f"delivered in {_PLANNED_SCHEME_PHASES[scheme]}. Registered schemes: "
                f"{sorted(_SCHEME_REGISTRY)}"
            )
        raise ValueError(
            f"Unknown communication scheme: '{scheme}'. Registered schemes: "
            f"{sorted(_SCHEME_REGISTRY)}. Planned but not yet implemented: "
            f"{sorted(_PLANNED_SCHEME_PHASES)}"
        )

    plan = builder(config)
    _validate_plan(scheme, plan, config)
    logger.debug(
        "Compiled communication scheme '{}' (rounds={}, cache_window={}, output_contract={})",
        scheme,
        plan.rounds,
        plan.cache_window,
        plan.output_contract,
    )
    return plan


def _auto_register() -> None:
    """Register built-in communication schemes at import time."""
    register_communication_scheme("none")(build_none_plan)
    logger.debug("Auto-registered communication schemes: {}", list_communication_schemes())


# Trigger auto-registration when the module is imported.
_auto_register()
