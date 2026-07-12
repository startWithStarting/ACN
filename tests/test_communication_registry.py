"""Phase 0 acceptance tests for the communication scheme registry and compiler.

These tests pin the documented behavior of the scheme registry
(docs/communication_implementation_plan.md, "Scheme Registry And Compiler",
"Initial Named Schemes", and "Delivery Phases: Phase 0"):

- ``none`` compiles to a plan whose processor yields an explicit empty
  :class:`CommunicationResult` (an empty view, never a missing one);
- an unknown scheme fails with an error naming the available schemes;
- the four named-but-not-yet-implemented schemes raise ``NotImplementedError``
  in Phase 0;
- config round-trips are deterministic: parsing the same dict twice yields
  equal configs.
"""

import unittest

import torch

from src.communication.config import parse_communication_config
from src.communication.plans import CommunicationPlan
from src.communication.registry import (
    create_communication_plan,
    list_communication_schemes,
)
from src.communication.types import CommunicationGraph, CommunicationResult

# Named schemes defined by the implementation plan whose delivery phases have
# not landed yet. Compiling them must raise NotImplementedError.
# (one_hop_direct landed in Phase 1 and is now a registered scheme.)
UNIMPLEMENTED_SCHEMES = ("one_hop_mean", "multihop_gnn")


def _two_agent_graph(step: int = 0) -> CommunicationGraph:
    """Build a minimal two-agent, zero-edge graph for exercising processors."""
    return CommunicationGraph(
        agent_ids=("blue_0", "blue_1"),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_distance=torch.zeros(0),
        edge_features=None,
        team_index=torch.zeros(2, dtype=torch.long),
        step=step,
    )


def _run_plan_pipeline(plan: CommunicationPlan, graph: CommunicationGraph) -> CommunicationResult:
    """Run a plan's processor over a frozen graph: initialize -> rounds -> finalize."""
    context = {"step": graph.step}
    state = plan.processor.initialize({}, graph, context)
    inbox = None
    for round_index in range(plan.rounds):
        state = plan.processor.process_round(state, inbox, graph, round_index, context)
    return plan.processor.finalize(state, inbox, graph, context)


class TestNoneScheme(unittest.TestCase):
    """The 'none' baseline compiles and yields an explicit empty result."""

    def test_none_is_registered(self):
        """Phase 0 registers the 'none' scheme."""
        self.assertIn("none", list_communication_schemes())

    def test_absent_config_compiles_none_plan(self):
        """No config at all still compiles an explicit no-communication plan."""
        plan = create_communication_plan(None)
        self.assertIsInstance(plan, CommunicationPlan)
        self.assertEqual(plan.rounds, 0)
        self.assertEqual(plan.cache_window, 0)
        self.assertEqual(plan.graph_update, "frozen")
        self.assertFalse(plan.differentiable)
        self.assertEqual(plan.output_contract, "empty")

    def test_disabled_config_compiles_none_plan(self):
        """enabled: false compiles the 'none' baseline even if a scheme is named."""
        plan = create_communication_plan({"enabled": False, "scheme": "one_hop_direct"})
        self.assertEqual(plan.rounds, 0)
        self.assertEqual(plan.output_contract, "empty")

    def test_typed_disabled_config_compiles_none_plan(self):
        """The typed disabled config from the parser compiles the 'none' baseline."""
        config = parse_communication_config(None)
        plan = create_communication_plan(config)
        self.assertEqual(plan.rounds, 0)
        self.assertEqual(plan.output_contract, "empty")

    def test_missing_enabled_field_defaults_to_disabled(self):
        """A raw mapping without 'enabled' compiles 'none', matching the config parser."""
        raw = {"scheme": "one_hop_direct"}
        self.assertFalse(parse_communication_config(raw).is_active)
        plan = create_communication_plan(raw)
        self.assertEqual(plan.rounds, 0)
        self.assertEqual(plan.output_contract, "empty")

    def test_none_processor_yields_explicit_empty_result(self):
        """Every agent receives an explicit empty view, never a missing one."""
        plan = create_communication_plan({"enabled": True, "scheme": "none"})
        graph = _two_agent_graph()
        result = _run_plan_pipeline(plan, graph)

        self.assertIsInstance(result, CommunicationResult)
        # Explicit per-agent entries: lookups succeed and contents are empty.
        self.assertEqual(set(result.agent_output.keys()), {"blue_0", "blue_1"})
        for agent_id, output in result.agent_output.items():
            with self.subTest(agent=agent_id):
                self.assertEqual(len(output), 0)
        self.assertIsNone(result.final_state)
        self.assertIsNone(result.protocol_state)
        self.assertEqual(result.delivered_messages.payload.numel(), 0)
        self.assertEqual(result.delivered_messages.sender_index.numel(), 0)
        self.assertEqual(result.delivered_messages.receiver_index.numel(), 0)

    def test_none_topology_builds_empty_graph(self):
        """The 'none' topology builds a valid empty graph (empty graphs are legal)."""
        plan = create_communication_plan(None)
        graph = plan.topology.build([], step=5)
        self.assertEqual(len(graph.agent_ids), 0)
        self.assertEqual(graph.edge_index.shape, (2, 0))
        self.assertEqual(graph.step, 5)

    def test_none_transport_delivers_nothing(self):
        """If invoked at all, the 'none' transport delivers an empty message batch."""
        plan = create_communication_plan(None)
        graph = plan.topology.build([], step=0)
        state = plan.transport.reset(graph, {})
        delivered = plan.transport.transmit_round({}, graph, state, 0, {})
        self.assertEqual(delivered.payload.numel(), 0)


class TestUnknownScheme(unittest.TestCase):
    """Unknown schemes fail early with an error naming the available schemes."""

    def test_unknown_scheme_error_names_available_schemes(self):
        """The error message names the bad scheme and the registered schemes."""
        with self.assertRaises(ValueError) as ctx:
            create_communication_plan({"enabled": True, "scheme": "carrier_pigeon"})
        message = str(ctx.exception)
        self.assertIn("carrier_pigeon", message)
        self.assertIn("none", message)

    def test_unknown_scheme_is_not_a_not_implemented_error(self):
        """A truly unknown scheme is a ValueError, not NotImplementedError."""
        try:
            create_communication_plan({"enabled": True, "scheme": "carrier_pigeon"})
        except NotImplementedError:  # pragma: no cover - defect signal
            self.fail("Unknown scheme raised NotImplementedError instead of ValueError")
        except ValueError:
            pass
        else:  # pragma: no cover - defect signal
            self.fail("Unknown scheme did not raise")


class TestPlannedSchemes(unittest.TestCase):
    """Named schemes without a landed delivery phase raise NotImplementedError."""

    def test_unimplemented_named_schemes_raise_not_implemented(self):
        """one_hop_mean/multihop_relay/multihop_gnn are Phase 2+."""
        for scheme in UNIMPLEMENTED_SCHEMES:
            with self.subTest(scheme=scheme):
                with self.assertRaises(NotImplementedError) as ctx:
                    create_communication_plan(
                        {"enabled": True, "scheme": scheme, "rounds_per_step": 1}
                    )
                self.assertIn(scheme, str(ctx.exception))

    def test_unimplemented_schemes_are_not_registered(self):
        """Planned schemes must not appear as registered until their phase lands."""
        registered = set(list_communication_schemes())
        for scheme in UNIMPLEMENTED_SCHEMES:
            with self.subTest(scheme=scheme):
                self.assertNotIn(scheme, registered)


class TestConfigRoundTrip(unittest.TestCase):
    """Config parsing is deterministic and round-trips through to_dict."""

    BASELINE_BLOCK = {
        "enabled": True,
        "scheme": "one_hop_direct",
        "rounds_per_step": 1,
        "topology": {
            "type": "radius",
            "radius_rule": "sender",
            "include_self_edges": False,
            "freeze_within_step": True,
        },
        "payload": {"type": "engineered_vector", "dimension": 4, "coordinate_frame": "global"},
        "processor": {"backend": "pyg", "aggregation": "none", "update": "identity"},
        "transport": {"type": "slotted_radius", "delivery": "broadcast", "free": True},
    }

    def test_same_dict_parses_to_equal_config_twice(self):
        """Parsing the same dict twice yields equal typed configs."""
        first = parse_communication_config(self.BASELINE_BLOCK)
        second = parse_communication_config(self.BASELINE_BLOCK)
        self.assertEqual(first, second)

    def test_to_dict_is_deterministic(self):
        """to_dict output is identical across independent parses of the same dict."""
        first = parse_communication_config(self.BASELINE_BLOCK).to_dict()
        second = parse_communication_config(self.BASELINE_BLOCK).to_dict()
        self.assertEqual(first, second)

    def test_to_dict_reparses_to_equal_config(self):
        """from_dict(to_dict(config)) reproduces the original typed config."""
        config = parse_communication_config(self.BASELINE_BLOCK)
        round_tripped = parse_communication_config(config.to_dict())
        self.assertEqual(config, round_tripped)

    def test_disabled_round_trip_is_deterministic(self):
        """The disabled config also parses and round-trips deterministically."""
        first = parse_communication_config(None)
        second = parse_communication_config(first.to_dict())
        self.assertEqual(first, second)

    def test_none_plan_compilation_is_deterministic(self):
        """Compiling the same disabled config twice yields identical plan parameters."""
        config = parse_communication_config(None)
        plan_a = create_communication_plan(config)
        plan_b = create_communication_plan(config)
        for attr in ("rounds", "cache_window", "graph_update", "differentiable",
                     "output_contract"):
            with self.subTest(attribute=attr):
                self.assertEqual(getattr(plan_a, attr), getattr(plan_b, attr))


if __name__ == "__main__":
    unittest.main()
