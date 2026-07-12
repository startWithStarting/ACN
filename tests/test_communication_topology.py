"""Tests for the Phase 1 same-team radius topology builder."""

import unittest

import torch

from src.agents.base_agent import AgentType, BaseAgent
from src.communication.topology import RadiusTopology
from src.communication.types import CommunicationGraph
from tests.communication_fixtures import (
    CHAIN_POSITIONS,
    CHAIN_RADIUS,
    DISCONNECTED_POSITIONS,
    DISCONNECTED_RADIUS,
    EMPTY_POSITIONS,
    EMPTY_RADIUS,
    TWO_TEAM_POSITIONS,
    TWO_TEAM_RADIUS,
    TWO_TEAM_TEAMS,
    chain_graph,
    disconnected_graph,
    empty_graph,
    two_team_graph,
)


def make_agent(name, x, y, team="blue", radius=None, active=True):
    """Build a real BaseAgent with a resolved communication radius attribute."""
    agent = BaseAgent(
        name=name,
        agent_type=AgentType.BLUE if team == "blue" else AgentType.RED,
        communication_bandwidth=1,
        processing_capability=1,
        x=x,
        y=y,
    )
    agent.is_active = active
    if radius is not None:
        agent.communication_radius = radius
    return agent


def agents_from_positions(positions, radius, teams=None):
    """Build agents (in lexicographic id order) from a fixture position table."""
    teams = teams or {}
    return [
        make_agent(name, x, y, team=teams.get(name, "blue"), radius=radius)
        for name, (x, y) in sorted(positions.items())
    ]


def edge_pairs(graph):
    """Return the graph's directed edges as (source, receiver) index tuples."""
    return [
        (int(src), int(rcv))
        for src, rcv in zip(graph.edge_index[0].tolist(), graph.edge_index[1].tolist())
    ]


def assert_graphs_equal(test, built, fixture):
    """Assert a built graph matches a Phase 0 fixture oracle exactly."""
    test.assertEqual(built.agent_ids, fixture.agent_ids)
    test.assertTrue(torch.equal(built.edge_index, fixture.edge_index))
    test.assertTrue(torch.allclose(built.edge_distance, fixture.edge_distance))
    test.assertTrue(torch.equal(built.team_index, fixture.team_index))
    test.assertEqual(built.step, fixture.step)


class TestRadiusTopologyAgainstFixtures(unittest.TestCase):
    """The builder must reproduce the hand-written Phase 0 fixture graphs."""

    def test_chain_graph_matches_fixture(self):
        agents = agents_from_positions(CHAIN_POSITIONS, CHAIN_RADIUS)
        built = RadiusTopology().build(agents, step=0)
        assert_graphs_equal(self, built, chain_graph())

    def test_disconnected_graph_matches_fixture(self):
        agents = agents_from_positions(DISCONNECTED_POSITIONS, DISCONNECTED_RADIUS)
        built = RadiusTopology().build(agents, step=0)
        assert_graphs_equal(self, built, disconnected_graph())

    def test_empty_graph_matches_fixture(self):
        agents = agents_from_positions(EMPTY_POSITIONS, EMPTY_RADIUS)
        built = RadiusTopology().build(agents, step=0)
        assert_graphs_equal(self, built, empty_graph())
        self.assertEqual(built.num_edges, 0)

    def test_two_team_graph_matches_fixture(self):
        agents = agents_from_positions(
            TWO_TEAM_POSITIONS, TWO_TEAM_RADIUS, teams={"red_0": "red", "red_1": "red"}
        )
        built = RadiusTopology().build(agents, step=0)
        assert_graphs_equal(self, built, two_team_graph())

    def test_same_team_edges_only(self):
        """Cross-team pairs inside the radius still never produce edges."""
        agents = agents_from_positions(
            TWO_TEAM_POSITIONS, TWO_TEAM_RADIUS, teams={"red_0": "red", "red_1": "red"}
        )
        built = RadiusTopology().build(agents, step=0)
        team = {name: TWO_TEAM_TEAMS[name] for name in built.agent_ids}
        for source, receiver in edge_pairs(built):
            self.assertEqual(
                team[built.agent_ids[source]],
                team[built.agent_ids[receiver]],
                "cross-team edge must never exist",
            )


class TestRadiusRules(unittest.TestCase):
    """Radius-ownership rules, including asymmetric configurations."""

    def _two_agents(self, radius_a, radius_b):
        return [
            make_agent("A", 0.0, 0.0, radius=radius_a),
            make_agent("B", 1.0, 0.0, radius=radius_b),
        ]

    def test_default_rule_is_sender(self):
        self.assertEqual(RadiusTopology().radius_rule, "sender")

    def test_sender_rule_asymmetric_edges(self):
        """Big sender radius on A only: A->B exists, B->A does not."""
        agents = self._two_agents(2.0, 0.5)
        built = RadiusTopology(radius_rule="sender").build(agents, step=0)
        self.assertEqual(edge_pairs(built), [(0, 1)])

    def test_receiver_rule_asymmetric_edges(self):
        """Receiver rule flips the asymmetry: only B->A survives."""
        agents = self._two_agents(2.0, 0.5)
        built = RadiusTopology(radius_rule="receiver").build(agents, step=0)
        self.assertEqual(edge_pairs(built), [(1, 0)])

    def test_mutual_rule_requires_both_ranges(self):
        agents = self._two_agents(2.0, 0.5)
        built = RadiusTopology(radius_rule="mutual").build(agents, step=0)
        self.assertEqual(edge_pairs(built), [])

    def test_minimum_rule_uses_smaller_radius(self):
        agents = self._two_agents(2.0, 0.5)
        built = RadiusTopology(radius_rule="minimum").build(agents, step=0)
        self.assertEqual(edge_pairs(built), [])

    def test_symmetric_radii_give_bidirectional_edges_under_every_rule(self):
        for rule in ("mutual", "sender", "receiver", "minimum"):
            agents = self._two_agents(1.5, 1.5)
            built = RadiusTopology(radius_rule=rule).build(agents, step=0)
            self.assertEqual(edge_pairs(built), [(0, 1), (1, 0)], "rule={}".format(rule))

    def test_exact_boundary_distance_is_included(self):
        """distance == radius produces an edge (<=, not <)."""
        agents = self._two_agents(1.0, 1.0)
        built = RadiusTopology(radius_rule="sender").build(agents, step=0)
        self.assertEqual(edge_pairs(built), [(0, 1), (1, 0)])

    def test_unknown_rule_rejected(self):
        with self.assertRaises(ValueError):
            RadiusTopology(radius_rule="strongest")


class TestSelfEdgesAndActivity(unittest.TestCase):
    """Self-edge configuration and inactive-agent exclusion."""

    def test_self_edges_disabled_by_default(self):
        agents = agents_from_positions(CHAIN_POSITIONS, CHAIN_RADIUS)
        built = RadiusTopology().build(agents, step=0)
        for source, receiver in edge_pairs(built):
            self.assertNotEqual(source, receiver)

    def test_self_edges_enabled(self):
        agents = agents_from_positions(CHAIN_POSITIONS, CHAIN_RADIUS)
        built = RadiusTopology(include_self_edges=True).build(agents, step=0)
        pairs = edge_pairs(built)
        for i in range(len(agents)):
            self.assertIn((i, i), pairs)
        distances = dict(zip(pairs, built.edge_distance.tolist()))
        self.assertEqual(distances[(0, 0)], 0.0)

    def test_inactive_agents_excluded(self):
        """An inactive relay node vanishes from the node set and cuts the chain."""
        agents = agents_from_positions(CHAIN_POSITIONS, CHAIN_RADIUS)
        agents[1].is_active = False  # B
        built = RadiusTopology().build(agents, step=0)
        self.assertEqual(built.agent_ids, ("A", "C"))
        self.assertEqual(built.num_edges, 0)


class TestDeterminismAndValidation(unittest.TestCase):
    """Deterministic edge ordering and input validation."""

    def test_edges_sorted_by_source_then_receiver(self):
        agents = agents_from_positions(DISCONNECTED_POSITIONS, DISCONNECTED_RADIUS)
        built = RadiusTopology().build(agents, step=0)
        pairs = edge_pairs(built)
        self.assertEqual(pairs, sorted(pairs))

    def test_two_builds_are_identical(self):
        agents = agents_from_positions(TWO_TEAM_POSITIONS, TWO_TEAM_RADIUS)
        first = RadiusTopology().build(agents, step=3)
        second = RadiusTopology().build(agents, step=3)
        self.assertTrue(torch.equal(first.edge_index, second.edge_index))
        self.assertTrue(torch.equal(first.edge_distance, second.edge_distance))
        self.assertEqual(first.agent_ids, second.agent_ids)

    def test_radius_mapping_overrides_attribute(self):
        """An explicit radius mapping wins over the agent attribute."""
        agents = self._chain_with_attr_radius(CHAIN_RADIUS)
        built = RadiusTopology(radius_by_agent={"A": 0.0, "B": 0.0, "C": 0.0}).build(
            agents, step=0
        )
        self.assertEqual(built.num_edges, 0)

    def test_radius_mapping_must_cover_every_agent(self):
        agents = self._chain_with_attr_radius(CHAIN_RADIUS)
        with self.assertRaises(ValueError):
            RadiusTopology(radius_by_agent={"A": 1.5}).build(agents, step=0)

    def test_missing_radius_attribute_rejected(self):
        agent = make_agent("A", 0.0, 0.0, radius=None)
        with self.assertRaises(ValueError):
            RadiusTopology().build([agent], step=0)

    def test_negative_radius_rejected(self):
        agent = make_agent("A", 0.0, 0.0, radius=-1.0)
        with self.assertRaises(ValueError):
            RadiusTopology().build([agent], step=0)

    def test_duplicate_names_rejected(self):
        agents = [make_agent("A", 0.0, 0.0, radius=1.0), make_agent("A", 1.0, 0.0, radius=1.0)]
        with self.assertRaises(ValueError):
            RadiusTopology().build(agents, step=0)

    def test_missing_position_rejected(self):
        agent = make_agent("A", None, None, radius=1.0)
        with self.assertRaises(ValueError):
            RadiusTopology().build([agent], step=0)

    def test_no_agents_is_a_valid_empty_graph(self):
        built = RadiusTopology().build([], step=5)
        self.assertIsInstance(built, CommunicationGraph)
        self.assertEqual(built.num_agents, 0)
        self.assertEqual(built.num_edges, 0)
        self.assertEqual(built.step, 5)

    def _chain_with_attr_radius(self, radius):
        return agents_from_positions(CHAIN_POSITIONS, radius)


if __name__ == "__main__":
    unittest.main()
