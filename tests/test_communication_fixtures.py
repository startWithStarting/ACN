"""Tests for the Phase 0 communication graph fixtures."""

import unittest

import torch

from src.communication.types import CommunicationGraph
from tests.communication_fixtures import (
    DISCONNECTED_COMPONENTS,
    TWO_TEAM_POSITIONS,
    TWO_TEAM_RADIUS,
    chain_graph,
    disconnected_graph,
    empty_graph,
    two_team_graph,
)

ALL_FIXTURES = (chain_graph, disconnected_graph, empty_graph, two_team_graph)


def _edge_pairs(graph: CommunicationGraph):
    """Return the graph's directed edges as a list of (source, receiver) index tuples."""
    return [
        (int(src), int(rcv))
        for src, rcv in zip(graph.edge_index[0].tolist(), graph.edge_index[1].tolist())
    ]


class TestFixtureShapes(unittest.TestCase):
    """Structural invariants shared by every fixture."""

    def test_edge_counts(self):
        """Each fixture has exactly its documented number of directed edges."""
        self.assertEqual(chain_graph().edge_index.shape[1], 4)
        self.assertEqual(disconnected_graph().edge_index.shape[1], 4)
        self.assertEqual(empty_graph().edge_index.shape[1], 0)
        self.assertEqual(two_team_graph().edge_index.shape[1], 4)

    def test_tensor_shapes_and_dtypes(self):
        """edge_index is [2, E] long, edge_distance is [E] float, team_index is [N] long."""
        for fixture in ALL_FIXTURES:
            graph = fixture()
            with self.subTest(fixture=fixture.__name__):
                self.assertEqual(graph.edge_index.dim(), 2)
                self.assertEqual(graph.edge_index.shape[0], 2)
                self.assertEqual(graph.edge_index.dtype, torch.long)
                self.assertEqual(graph.edge_distance.shape, (graph.edge_index.shape[1],))
                self.assertTrue(graph.edge_distance.is_floating_point())
                self.assertEqual(graph.team_index.shape, (len(graph.agent_ids),))
                self.assertEqual(graph.team_index.dtype, torch.long)
                self.assertIsNone(graph.edge_features)
                self.assertEqual(graph.step, 0)

    def test_edge_indices_in_range_and_no_self_edges(self):
        """Every edge references a valid agent index and never loops back to its source."""
        for fixture in ALL_FIXTURES:
            graph = fixture()
            num_agents = len(graph.agent_ids)
            with self.subTest(fixture=fixture.__name__):
                for src, rcv in _edge_pairs(graph):
                    self.assertTrue(0 <= src < num_agents)
                    self.assertTrue(0 <= rcv < num_agents)
                    self.assertNotEqual(src, rcv)

    def test_edges_sorted_by_source_then_receiver(self):
        """Edge ordering follows the documented (source, receiver) sort."""
        for fixture in ALL_FIXTURES:
            graph = fixture()
            pairs = _edge_pairs(graph)
            with self.subTest(fixture=fixture.__name__):
                self.assertEqual(pairs, sorted(pairs))

    def test_step_parameter_is_recorded(self):
        """Fixtures stamp the requested movement step onto the snapshot."""
        for fixture in ALL_FIXTURES:
            with self.subTest(fixture=fixture.__name__):
                self.assertEqual(fixture(step=7).step, 7)


class TestSameTeamOnly(unittest.TestCase):
    """No fixture may contain a cross-team edge."""

    def test_all_edges_are_same_team(self):
        """Sender and receiver team labels match on every edge of every fixture."""
        for fixture in ALL_FIXTURES:
            graph = fixture()
            with self.subTest(fixture=fixture.__name__):
                for src, rcv in _edge_pairs(graph):
                    self.assertEqual(
                        int(graph.team_index[src]),
                        int(graph.team_index[rcv]),
                        msg="cross-team edge {}->{}".format(
                            graph.agent_ids[src], graph.agent_ids[rcv]
                        ),
                    )

    def test_two_team_graph_has_edges_on_both_teams(self):
        """Blue and red each keep their own intra-team edges."""
        graph = two_team_graph()
        edge_teams = {int(graph.team_index[src]) for src, _ in _edge_pairs(graph)}
        self.assertEqual(edge_teams, {0, 1})

    def test_two_team_proximity_does_not_create_cross_team_edges(self):
        """A cross-team pair inside the radius still gets no edge (team gate, not distance)."""
        bx, by = TWO_TEAM_POSITIONS["blue_0"]
        rx, ry = TWO_TEAM_POSITIONS["red_0"]
        cross_distance = ((bx - rx) ** 2 + (by - ry) ** 2) ** 0.5
        self.assertLessEqual(cross_distance, TWO_TEAM_RADIUS)
        graph = two_team_graph()
        blue_0 = graph.agent_ids.index("blue_0")
        red_0 = graph.agent_ids.index("red_0")
        pairs = _edge_pairs(graph)
        self.assertNotIn((blue_0, red_0), pairs)
        self.assertNotIn((red_0, blue_0), pairs)


class TestChainGraph(unittest.TestCase):
    """The A-B-C chain used by one-hop versus multi-hop delivery tests."""

    def test_exact_edge_set(self):
        """The chain contains exactly A->B, B->A, B->C, C->B in sorted order."""
        graph = chain_graph()
        self.assertEqual(graph.agent_ids, ("A", "B", "C"))
        expected = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        self.assertTrue(torch.equal(graph.edge_index, expected))

    def test_no_edge_between_a_and_c(self):
        """A and C are two hops apart: neither A->C nor C->A exists."""
        pairs = _edge_pairs(chain_graph())
        self.assertNotIn((0, 2), pairs)
        self.assertNotIn((2, 0), pairs)

    def test_adjacent_distances(self):
        """Adjacent chain agents sit exactly 1.0 apart."""
        graph = chain_graph()
        for value in graph.edge_distance.tolist():
            self.assertAlmostEqual(value, 1.0, places=6)


class TestDisconnectedGraph(unittest.TestCase):
    """Two components with no path between them."""

    def test_no_cross_component_edges(self):
        """Every edge stays inside its documented component."""
        graph = disconnected_graph()
        component_of = {}
        for component_index, members in enumerate(DISCONNECTED_COMPONENTS):
            for agent_id in members:
                component_of[agent_id] = component_index
        for src, rcv in _edge_pairs(graph):
            self.assertEqual(
                component_of[graph.agent_ids[src]],
                component_of[graph.agent_ids[rcv]],
                msg="cross-component edge {}->{}".format(
                    graph.agent_ids[src], graph.agent_ids[rcv]
                ),
            )

    def test_both_components_have_edges(self):
        """Each component keeps its own bidirectional pair."""
        pairs = _edge_pairs(disconnected_graph())
        self.assertEqual(pairs, [(0, 1), (1, 0), (2, 3), (3, 2)])


class TestEmptyGraph(unittest.TestCase):
    """Agents present, zero edges: valid zero-length tensors."""

    def test_zero_length_tensors_are_valid(self):
        """Empty graphs still construct with well-formed [2, 0] and [0] tensors."""
        graph = empty_graph()
        self.assertEqual(len(graph.agent_ids), 3)
        self.assertEqual(graph.edge_index.shape, (2, 0))
        self.assertEqual(graph.edge_index.dtype, torch.long)
        self.assertEqual(graph.edge_distance.shape, (0,))
        self.assertEqual(graph.edge_index.numel(), 0)
        self.assertEqual(graph.edge_distance.numel(), 0)
        self.assertEqual(graph.team_index.shape, (3,))


class TestDeterminism(unittest.TestCase):
    """Repeated construction must be bit-for-bit identical."""

    def test_repeated_construction_is_identical(self):
        """Two calls to the same fixture return equal ids, edges, distances, and teams."""
        for fixture in ALL_FIXTURES:
            first = fixture()
            second = fixture()
            with self.subTest(fixture=fixture.__name__):
                self.assertEqual(first.agent_ids, second.agent_ids)
                self.assertTrue(torch.equal(first.edge_index, second.edge_index))
                self.assertTrue(torch.equal(first.edge_distance, second.edge_distance))
                self.assertTrue(torch.equal(first.team_index, second.team_index))
                self.assertEqual(first.step, second.step)


if __name__ == "__main__":
    unittest.main()
