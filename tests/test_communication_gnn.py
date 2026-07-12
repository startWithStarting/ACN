"""Phase 4 tests: the ``multihop_gnn`` scheme and the GraphSAGE communicator.

Covers the scheme compiler (plan fields, resolved-hyperparameter marker,
rejections for every documented inconsistency), the environment boundary
(differentiable plans are never executed by the env: no runtime, no
``communication`` observation key, AEC still rejects), and the
:class:`GraphSAGECommunicator` module itself (R-hop propagation limit by
construction, empty/disconnected graphs, per-round vs shared weights).
"""

import unittest

import torch

from src.agents.factory import create_agents_from_config
from src.communication.config import parse_communication_config
from src.communication.processors.gnn import GraphSAGECommunicator
from src.communication.registry import (
    create_communication_plan,
    list_communication_schemes,
)
from src.communication.schemes import DifferentiableCommunicationMarker
from src.communication.topology import RadiusTopology
from src.env.aec_env import AECGameEnv
from src.env.parallel_env import ParallelGameEnv


def gnn_block(**overrides):
    """Build a valid multihop_gnn communication block; overrides merge on top."""
    block = {
        "enabled": True,
        "scheme": "multihop_gnn",
        "rounds_per_step": 3,
        "cache_window": 0,
        "payload": {"type": "learned", "dimension": 16},
        "processor": {
            "backend": "pyg",
            "model": "graph_sage",
            "layers": 3,
            "aggregation": "sum",
            "hidden_dimension": 32,
        },
    }
    block.update(overrides)
    return block


def line_graph_edges():
    """Bidirectional 4-node line graph A-B-C-D (A is exactly 3 hops from D)."""
    return torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)


class TestMultihopGnnCompiles(unittest.TestCase):
    """The scheme compiles to the documented differentiable plan."""

    def test_scheme_is_registered(self):
        self.assertIn("multihop_gnn", list_communication_schemes())

    def test_plan_fields_match_the_decision_log(self):
        plan = create_communication_plan(gnn_block())
        self.assertEqual(plan.rounds, 3)
        self.assertEqual(plan.cache_window, 0)
        self.assertEqual(plan.graph_update, "frozen")
        self.assertTrue(plan.differentiable)
        self.assertEqual(plan.output_contract, "learned-embedding")
        self.assertIsInstance(plan.topology, RadiusTopology)

    def test_typed_config_compiles_identically(self):
        typed = parse_communication_config(gnn_block())
        plan = create_communication_plan(typed)
        self.assertEqual(plan.rounds, 3)
        self.assertTrue(plan.differentiable)

    def test_marker_carries_resolved_hyperparameters(self):
        plan = create_communication_plan(gnn_block())
        marker = plan.processor
        self.assertIsInstance(marker, DifferentiableCommunicationMarker)
        self.assertIs(plan.transport, marker)
        self.assertEqual(marker.rounds, 3)
        self.assertEqual(marker.aggregation, "sum")
        self.assertEqual(marker.hidden_dimension, 32)
        self.assertEqual(marker.message_dimension, 16)
        self.assertFalse(marker.shared_weights)

    def test_defaults_are_three_sum_rounds(self):
        """Omitted layers/rounds/aggregation resolve to the reference model."""
        block = gnn_block()
        del block["rounds_per_step"]
        block["processor"] = {}
        plan = create_communication_plan(block)
        self.assertEqual(plan.rounds, 3)
        self.assertEqual(plan.processor.aggregation, "sum")
        self.assertEqual(plan.processor.hidden_dimension, 64)

    def test_mean_and_max_aggregations_are_selectable(self):
        for aggregation in ("mean", "max"):
            with self.subTest(aggregation=aggregation):
                block = gnn_block()
                block["processor"]["aggregation"] = aggregation
                plan = create_communication_plan(block)
                self.assertEqual(plan.processor.aggregation, aggregation)

    def test_shared_weights_option_is_recorded(self):
        block = gnn_block()
        block["processor"]["shared_weights"] = True
        plan = create_communication_plan(block)
        self.assertTrue(plan.processor.shared_weights)

    def test_marker_refuses_env_side_execution(self):
        """Every processor/transport entry point raises with a clear message."""
        plan = create_communication_plan(gnn_block())
        marker = plan.processor
        calls = (
            lambda: marker.initialize({}, None, {}),
            lambda: marker.process_round(None, None, None, 0, {}),
            lambda: marker.finalize(None, None, None, {}),
            lambda: marker.reset(None, {}),
            lambda: marker.transmit_round({}, None, None, 0, {}),
        )
        for index, call in enumerate(calls):
            with self.subTest(entry_point=index):
                with self.assertRaisesRegex(RuntimeError, "differentiable"):
                    call()


class TestMultihopGnnRejections(unittest.TestCase):
    """The compiler rejects every documented inconsistency, actionably."""

    def assert_rejects(self, block, pattern):
        with self.assertRaisesRegex(ValueError, pattern):
            create_communication_plan(block)

    def test_rounds_not_equal_layers_is_rejected(self):
        self.assert_rejects(gnn_block(rounds_per_step=2), "one graph hop per round")

    def test_layers_without_rounds_must_still_match(self):
        block = gnn_block(rounds_per_step=3)
        block["processor"]["layers"] = 4
        self.assert_rejects(block, "rounds_per_step")

    def test_nonzero_cache_window_is_rejected(self):
        self.assert_rejects(gnn_block(cache_window=1), "C = 0")

    def test_relay_fields_are_rejected(self):
        for key, value in (
            ("ttl", 3),
            ("forwarding", "first_seen_unchanged"),
            ("packet_relay", True),
            ("duplicate_suppression", True),
        ):
            with self.subTest(field=key):
                block = gnn_block()
                block["processor"][key] = value
                self.assert_rejects(block, "relay")

    def test_engineered_payload_is_rejected(self):
        self.assert_rejects(
            gnn_block(payload={"type": "engineered_vector", "dimension": 4}), "learned"
        )

    def test_missing_payload_dimension_is_rejected(self):
        self.assert_rejects(gnn_block(payload={"type": "learned"}), "dimension")

    def test_unsupported_aggregation_is_rejected(self):
        block = gnn_block()
        block["processor"]["aggregation"] = "attention"
        self.assert_rejects(block, "sum")

    def test_non_graph_sage_model_is_rejected(self):
        block = gnn_block()
        block["processor"]["model"] = "gat"
        self.assert_rejects(block, "graph_sage")

    def test_invalid_layers_is_rejected(self):
        block = gnn_block(rounds_per_step=0)
        block["processor"]["layers"] = 0
        self.assert_rejects(block, "layers")


class TestEnvSkipsDifferentiableExecution(unittest.TestCase):
    """The env never executes a differentiable plan's rounds."""

    def make_env(self):
        env_config = {
            "width": 60,
            "height": 60,
            "max_cycles": 5,
            "save_episode_gifs": False,
            "communication": gnn_block(),
        }
        agents = create_agents_from_config(
            {"blue_agents": [{"count": 2}], "red_agents": [{"count": 1}]}, env_config
        )
        return ParallelGameEnv(agents, **env_config), env_config

    @staticmethod
    def stay_actions(env):
        return {name: {"direction": (0.0, 0.0), "speed": 0.0} for name in env.agents}

    def test_runtime_is_none_and_plan_is_kept(self):
        env, _ = self.make_env()
        self.assertIsNone(env.communication_runtime)
        self.assertIsNotNone(env.communication_plan)
        self.assertTrue(env.communication_plan.differentiable)
        env.close()

    def test_observations_carry_no_communication_key(self):
        env, _ = self.make_env()
        observations, infos = env.reset(seed=3)
        for name, observation in observations.items():
            self.assertNotIn("communication", observation, name)
        observations, _, _, _, infos = env.step(self.stay_actions(env))
        for name, observation in observations.items():
            self.assertNotIn("communication", observation, name)
        for agent_infos in infos.values():
            self.assertNotIn("communication", agent_infos)
        self.assertEqual(env.last_communication_trace_records, [])
        env.close()

    def test_aec_still_rejects_active_communication(self):
        _, env_config = self.make_env()
        agents = create_agents_from_config(
            {"blue_agents": [{"count": 1}], "red_agents": [{"count": 1}]}, env_config
        )
        with self.assertRaises(NotImplementedError):
            AECGameEnv(agents, **env_config)


class TestGraphSAGECommunicator(unittest.TestCase):
    """The learned communicator: shapes, hop limit, degenerate graphs."""

    def setUp(self):
        torch.manual_seed(11)

    def build(self, rounds, **kwargs):
        return GraphSAGECommunicator(
            in_dim=5, hidden_dim=8, message_dim=4, rounds=rounds, **kwargs
        )

    def test_output_shape(self):
        module = self.build(rounds=3)
        out = module(torch.randn(4, 5), line_graph_edges())
        self.assertEqual(tuple(out.shape), (4, 4))

    def test_empty_edge_graph_is_valid_and_self_dependent(self):
        """Isolated nodes get their own encoding through the update path."""
        module = self.build(rounds=3)
        x = torch.randn(3, 5)
        empty = torch.zeros((2, 0), dtype=torch.long)
        out = module(x, empty)
        self.assertEqual(tuple(out.shape), (3, 4))
        self.assertTrue(torch.isfinite(out).all())
        perturbed = x.clone()
        perturbed[1] += 5.0
        out2 = module(perturbed, empty)
        self.assertFalse(torch.allclose(out[1], out2[1]))
        # Other isolated nodes are untouched: no edges, no influence.
        self.assertTrue(torch.allclose(out[0], out2[0]))
        self.assertTrue(torch.allclose(out[2], out2[2]))

    def test_information_propagates_at_most_r_hops(self):
        """On the A-B-C-D line, A is exactly 3 hops from D: R=2 cannot reach."""
        edges = line_graph_edges()
        x = torch.randn(4, 5)
        perturbed = x.clone()
        perturbed[0] += 10.0

        two_rounds = self.build(rounds=2)
        base = two_rounds(x, edges)
        moved = two_rounds(perturbed, edges)
        self.assertTrue(torch.allclose(base[3], moved[3]), "R=2 leaked beyond 2 hops")
        # ... while C (2 hops away) is reachable at R=2.
        self.assertFalse(torch.allclose(base[2], moved[2]))

        three_rounds = self.build(rounds=3)
        base = three_rounds(x, edges)
        moved = three_rounds(perturbed, edges)
        self.assertFalse(
            torch.allclose(base[3], moved[3]), "R=3 must reach exactly 3 hops"
        )

    def test_separate_weights_per_round_is_the_default(self):
        module = self.build(rounds=3)
        self.assertEqual(len(module.convs), 3)
        shared = self.build(rounds=3, shared_weights=True)
        self.assertEqual(len(shared.convs), 1)
        # Shared weights still run all R rounds (hop limit stays R).
        edges = line_graph_edges()
        x = torch.randn(4, 5)
        perturbed = x.clone()
        perturbed[0] += 10.0
        self.assertFalse(
            torch.allclose(shared(x, edges)[3], shared(perturbed, edges)[3])
        )

    def test_disconnected_components_do_not_mix(self):
        """Two 2-node components: perturbing one never affects the other."""
        module = self.build(rounds=3)
        edges = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
        x = torch.randn(4, 5)
        perturbed = x.clone()
        perturbed[0] += 10.0
        base = module(x, edges)
        moved = module(perturbed, edges)
        self.assertTrue(torch.allclose(base[2], moved[2]))
        self.assertTrue(torch.allclose(base[3], moved[3]))
        self.assertFalse(torch.allclose(base[1], moved[1]))

    def test_invalid_construction_is_rejected(self):
        with self.assertRaises(ValueError):
            GraphSAGECommunicator(0, 8, 4)
        with self.assertRaises(ValueError):
            GraphSAGECommunicator(5, 8, 4, rounds=0)
        with self.assertRaises(ValueError):
            GraphSAGECommunicator(5, 8, 4, aggregation="attention")


if __name__ == "__main__":
    unittest.main()
