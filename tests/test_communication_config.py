"""Phase 0 acceptance tests for the ``environment.communication`` config schema.

These tests pin the documented behavior of the typed communication config
(docs/communication_implementation_plan.md, "Configuration Schema") and the
round/cache invariants fixed by docs/communication_decision_log.md
("Round And Cache Model"):

- the baseline ``one_hop_direct`` YAML block from the plan document parses;
- an absent communication block is the canonical disabled ("none") config;
- ``rounds_per_step < 1`` while enabled is rejected;
- relay with ``ttl > cache_window`` is rejected (relay correctness is C >= TTL,
  while ``TTL > R`` stays legal because the cache window spans movement steps);
- non-positive payload dimensions are rejected;
- unknown schemes are rejected with an actionable message.
"""

import unittest
from typing import Any, Dict, Optional

import yaml

from src.communication.config import (
    VALID_SCHEMES,
    CommunicationConfig,
    ConfigError,
    parse_communication_config,
)

# The proposed baseline configuration, copied verbatim from
# docs/communication_implementation_plan.md, section "Configuration Schema".
BASELINE_ONE_HOP_DIRECT_YAML = """
environment:
  communication:
    enabled: true
    scheme: "one_hop_direct"
    rounds_per_step: 1

    topology:
      type: "radius"
      radius_rule: "sender"
      include_self_edges: false
      freeze_within_step: true

    payload:
      type: "engineered_vector"
      dimension: 4
      coordinate_frame: "global"

    processor:
      backend: "pyg"
      aggregation: "none"
      update: "identity"

    transport:
      type: "slotted_radius"
      delivery: "broadcast"
      free: true
"""


def _baseline_block() -> Dict[str, Any]:
    """Return the plan document's baseline communication block as a plain dict."""
    loaded = yaml.safe_load(BASELINE_ONE_HOP_DIRECT_YAML)
    return loaded["environment"]["communication"]


def _relay_block(cache_window: Optional[int], ttl: int, rounds: int = 3) -> Dict[str, Any]:
    """Build a ``multihop_relay`` block modeled on the plan document's relay example."""
    block: Dict[str, Any] = {
        "enabled": True,
        "scheme": "multihop_relay",
        "rounds_per_step": rounds,
        "topology": {"type": "radius", "radius_rule": "sender", "freeze_within_step": True},
        "payload": {"type": "engineered_vector", "dimension": 4},
        "processor": {
            "backend": "identity",
            "aggregation": "none",
            "forwarding": "first_seen_unchanged",
            "ttl": ttl,
            "duplicate_suppression": True,
        },
        "transport": {"type": "slotted_radius", "delivery": "broadcast", "free": True},
    }
    if cache_window is not None:
        block["cache_window"] = cache_window
    return block


class TestBaselineBlockParses(unittest.TestCase):
    """The plan document's baseline one_hop_direct YAML block must parse verbatim."""

    def test_baseline_yaml_block_parses(self):
        """Every documented field of the baseline block lands on the typed config."""
        config = parse_communication_config(_baseline_block())

        self.assertTrue(config.enabled)
        self.assertTrue(config.is_active)
        self.assertEqual(config.scheme, "one_hop_direct")
        self.assertEqual(config.rounds_per_step, 1)

        self.assertEqual(config.topology.type, "radius")
        self.assertEqual(config.topology.radius_rule, "sender")
        self.assertFalse(config.topology.include_self_edges)
        self.assertTrue(config.topology.freeze_within_step)

        self.assertEqual(config.payload.type, "engineered_vector")
        self.assertEqual(config.payload.dimension, 4)
        self.assertEqual(config.payload.coordinate_frame, "global")

        self.assertEqual(config.processor.backend, "pyg")
        self.assertEqual(config.processor.aggregation, "none")
        self.assertEqual(config.processor.update, "identity")

        self.assertEqual(config.transport.type, "slotted_radius")
        self.assertEqual(config.transport.delivery, "broadcast")
        self.assertTrue(config.transport.free)

    def test_baseline_block_is_not_relay(self):
        """one_hop_direct must not trip the relay-only C >= TTL validation path."""
        config = parse_communication_config(_baseline_block())
        self.assertFalse(config.uses_relay)


class TestDisabledNormalization(unittest.TestCase):
    """Absent block, enabled: false, and scheme 'none' all mean disabled."""

    def test_absent_block_is_disabled_none(self):
        """No communication block at all parses to the canonical disabled config."""
        config = parse_communication_config(None)
        self.assertFalse(config.enabled)
        self.assertFalse(config.is_active)
        self.assertEqual(config.scheme, "none")
        self.assertEqual(config, CommunicationConfig.disabled())

    def test_enabled_false_is_disabled(self):
        """enabled: false yields the same canonical disabled config as an absent block."""
        block = _baseline_block()
        block["enabled"] = False
        config = parse_communication_config(block)
        self.assertFalse(config.is_active)
        self.assertEqual(config, parse_communication_config(None))

    def test_scheme_none_is_disabled(self):
        """scheme 'none' is the explicit spelling of the disabled configuration."""
        config = parse_communication_config({"enabled": True, "scheme": "none"})
        self.assertFalse(config.is_active)
        self.assertEqual(config, CommunicationConfig.disabled())


class TestRoundsValidation(unittest.TestCase):
    """R must be >= 1 whenever communication is actually enabled."""

    def test_zero_rounds_while_enabled_rejected(self):
        """rounds_per_step: 0 with an enabled scheme fails with an actionable error."""
        block = _baseline_block()
        block["rounds_per_step"] = 0
        with self.assertRaises(ConfigError) as ctx:
            parse_communication_config(block)
        self.assertIn("rounds_per_step", str(ctx.exception))

    def test_negative_rounds_while_enabled_rejected(self):
        """Negative round counts are rejected outright."""
        block = _baseline_block()
        block["rounds_per_step"] = -3
        with self.assertRaises(ConfigError):
            parse_communication_config(block)


class TestRelayCacheInvariant(unittest.TestCase):
    """Relay correctness requires C >= TTL; the config must reject C < TTL."""

    def test_ttl_exceeding_cache_window_rejected(self):
        """A relay whose ttl exceeds cache_window breaks duplicate suppression."""
        with self.assertRaises(ConfigError) as ctx:
            parse_communication_config(_relay_block(cache_window=2, ttl=3))
        message = str(ctx.exception)
        self.assertIn("ttl", message.lower())
        self.assertIn("cache_window", message)

    def test_missing_cache_window_with_ttl_rejected(self):
        """The default cache_window of 0 cannot satisfy C >= TTL for any relay."""
        with self.assertRaises(ConfigError):
            parse_communication_config(_relay_block(cache_window=None, ttl=3))

    def test_cache_window_equal_to_ttl_accepted(self):
        """C == TTL is the exact relay-correctness floor and must be accepted."""
        config = parse_communication_config(_relay_block(cache_window=3, ttl=3))
        self.assertTrue(config.uses_relay)
        self.assertEqual(config.cache_window, 3)
        self.assertEqual(config.processor.ttl, 3)

    def test_ttl_greater_than_rounds_accepted_when_cache_covers_it(self):
        """TTL > R is legal: the cache window spans movement-step boundaries."""
        config = parse_communication_config(_relay_block(cache_window=5, ttl=5, rounds=3))
        self.assertEqual(config.rounds_per_step, 3)
        self.assertEqual(config.processor.ttl, 5)
        self.assertGreaterEqual(config.cache_window, config.processor.ttl)

    def test_cache_window_may_exceed_ttl(self):
        """C > TTL is allowed for schemes wanting a longer temporal window."""
        config = parse_communication_config(_relay_block(cache_window=8, ttl=3))
        self.assertEqual(config.cache_window, 8)


class TestPayloadValidation(unittest.TestCase):
    """Payload dimension must be a positive integer."""

    def test_zero_dimension_rejected(self):
        """dimension: 0 is rejected."""
        block = _baseline_block()
        block["payload"]["dimension"] = 0
        with self.assertRaises(ConfigError) as ctx:
            parse_communication_config(block)
        self.assertIn("dimension", str(ctx.exception))

    def test_negative_dimension_rejected(self):
        """Negative dimensions are rejected."""
        block = _baseline_block()
        block["payload"]["dimension"] = -4
        with self.assertRaises(ConfigError):
            parse_communication_config(block)


class TestUnknownScheme(unittest.TestCase):
    """Unknown schemes must fail early with a message naming the valid options."""

    def test_unknown_scheme_rejected_with_actionable_message(self):
        """The error names the bad scheme and lists every valid scheme name."""
        block = _baseline_block()
        block["scheme"] = "smoke_signals"
        with self.assertRaises(ConfigError) as ctx:
            parse_communication_config(block)
        message = str(ctx.exception)
        self.assertIn("smoke_signals", message)
        for scheme in VALID_SCHEMES:
            self.assertIn(scheme, message)

    def test_unknown_scheme_rejected_even_when_disabled(self):
        """A typo'd scheme fails early instead of being silently ignored."""
        with self.assertRaises(ConfigError):
            parse_communication_config({"enabled": False, "scheme": "carrier_pigeon"})


class TestNamedSchemes(unittest.TestCase):
    """The five named schemes from the decision log are exactly the valid set."""

    def test_valid_schemes_match_decision_log(self):
        """VALID_SCHEMES is the documented set of named schemes."""
        self.assertEqual(
            set(VALID_SCHEMES),
            {"none", "one_hop_direct", "one_hop_mean", "multihop_relay", "multihop_gnn"},
        )


if __name__ == "__main__":
    unittest.main()
