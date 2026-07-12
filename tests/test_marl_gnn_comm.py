"""Trainer tests for differentiable multihop_gnn communication.

Covers the Phase 4 acceptance criteria on the training side:

- gradients from the PPO loss reach the local encoder, EVERY round's SAGE
  weights, and the action head (named-parameter grad-norm assertions);
- the R-hop limit holds through the ACTION LOGITS (structural decentralized
  execution, not trusted batching): on the A-B-C-D line graph a perturbation
  of A cannot move D's logits at R=2 and must move them at R=3;
- per-agent and batched evaluation are equivalent (the plan's
  decentralized-batching requirement);
- rollout storage keeps RAW node features + edge_index, so recomputing the
  communication forward pass from the assembled rollout reproduces the
  collection-time embeddings before the first update;
- privileged fields stay unreachable through the graph path: the actor reads
  no privileged key and the node features are exactly the policy-visible
  local features;
- a tiny 2-update end-to-end training run with the shipped
  benchmark_blue_mappo_gnn.yaml stays finite.
"""

import tempfile
import unittest
from pathlib import Path

import torch
import yaml
from tensordict import TensorDict

from src.communication.types import CommunicationGraph
from src.training.marl.adapters import (
    COMMUNICATION_INPUT_KEYS,
    PRIVILEGED_STATE_KEY,
    attach_communication_inputs,
    max_team_edges,
)
from src.training.marl.config import TrainingSettings
from src.training.marl.contract import TeamTransition
from src.training.marl.ppo import GnnCommSettings, TeamPPO, _GnnActorNet
from src.training.marl.runner import MarlTrainingRunner

REPO_ROOT = Path(__file__).resolve().parents[1]

TEAM = ["blue_0", "blue_1", "blue_2"]
FEATURES = 6
SLOTS = 2
PRIV = 10
ACTIONS = 5
MAX_EDGES = max_team_edges(len(TEAM))


def gnn_settings(rounds=2):
    return GnnCommSettings(message_dim=4, hidden_dim=8, rounds=rounds, aggregation="sum")


def make_settings(seed=3, rollout=6):
    return TrainingSettings(
        trainable_team="blue",
        actor="shared",
        critic="global",
        rollout_length=rollout,
        updates=1,
        epochs=2,
        minibatches=2,
        lr=1e-2,
        seed=seed,
        hidden_size=16,
        entropy_coef=0.01,
    )


def make_method(seed=3, rollout=6, rounds=2):
    torch.manual_seed(seed)
    return TeamPPO(
        make_settings(seed=seed, rollout=rollout),
        team_names=TEAM,
        feature_dim=FEATURES,
        privileged_dim=PRIV,
        num_actions=ACTIONS,
        comm=gnn_settings(rounds=rounds),
    )


def team_graph(num_agents=len(TEAM), step=0):
    """A connected line graph over the team (so SAGE neighbour weights train)."""
    sources, receivers = [], []
    for i in range(num_agents - 1):
        sources += [i, i + 1]
        receivers += [i + 1, i]
    return CommunicationGraph(
        agent_ids=tuple(TEAM[:num_agents]),
        edge_index=torch.tensor([sources, receivers], dtype=torch.long),
        edge_distance=torch.zeros(len(sources)),
        edge_features=None,
        team_index=torch.zeros(num_agents, dtype=torch.long),
        step=step,
    )


def synthetic_inputs(generator, step=0):
    """Encoded policy inputs plus the attached raw communication graph inputs."""
    td = TensorDict(
        {
            "features": torch.randn(len(TEAM), FEATURES, generator=generator),
            "contacts_mask": torch.zeros(len(TEAM), SLOTS, dtype=torch.bool),
        },
        batch_size=[len(TEAM)],
    )
    return attach_communication_inputs(td, team_graph(step=step), MAX_EDGES)


def collect_rollout(method, data_seed=17, rollout=6):
    """Drive the method through one synthetic rollout via the real contract."""
    generator = torch.Generator().manual_seed(data_seed)
    inputs = synthetic_inputs(generator, step=0)
    privileged = torch.randn(PRIV, generator=generator)
    for step in range(rollout):
        selection = method.select_actions(inputs, privileged)
        next_inputs = synthetic_inputs(generator, step=step + 1)
        next_privileged = torch.randn(PRIV, generator=generator)
        method.ingest_transition(
            TeamTransition(
                step=step,
                policy_inputs=inputs,
                privileged_state=privileged,
                actions=selection.actions,
                extras=selection.extras,
                rewards=torch.randn(len(TEAM), generator=generator),
                dones=torch.zeros(len(TEAM), dtype=torch.bool),
                terminateds=torch.zeros(len(TEAM), dtype=torch.bool),
                next_policy_inputs=next_inputs,
                next_privileged_state=next_privileged,
            )
        )
        inputs = next_inputs
        privileged = next_privileged


def line_batch(net_inputs_features, num_agents=4):
    """Build a [num_agents] sample batch sharing one 4-node line graph."""
    sources, receivers = [], []
    for i in range(num_agents - 1):
        sources += [i, i + 1]
        receivers += [i + 1, i]
    edges = torch.zeros((2, num_agents * (num_agents - 1)), dtype=torch.long)
    edge_index = torch.tensor([sources, receivers], dtype=torch.long)
    edges[:, : edge_index.size(1)] = edge_index
    return {
        "features": net_inputs_features,
        "comm_nodes": net_inputs_features.unsqueeze(0).expand(
            num_agents, num_agents, net_inputs_features.size(1)
        ),
        "comm_edges": edges.unsqueeze(0).expand(num_agents, 2, edges.size(1)),
        "comm_num_edges": torch.full((num_agents,), edge_index.size(1), dtype=torch.long),
        "comm_node_index": torch.arange(num_agents, dtype=torch.long),
    }


class TestGradientFlow(unittest.TestCase):
    """PPO loss gradients reach encoder, every round's SAGE weights, and head."""

    def test_named_parameter_grad_norms_are_positive(self):
        rounds = 3
        method = make_method(seed=9, rounds=rounds)
        collect_rollout(method)
        learner = method._learners[0]
        rollout = method._buffer.as_tensordict()
        sub = rollout[torch.as_tensor(learner.rows, dtype=torch.long)]
        with torch.no_grad():
            learner.gae(sub)
        loss_td = learner.loss(sub.reshape(-1))
        total = sum(
            loss_td[key]
            for key in ("loss_objective", "loss_critic", "loss_entropy")
            if key in loss_td.keys()
        )
        learner.optimizer.zero_grad(set_to_none=True)
        total.backward()

        net = method._actor_nets["__shared__"]
        groups = {"communicator.encoder": 0.0, "communicator.output": 0.0, "head": 0.0}
        for round_index in range(rounds):
            groups["communicator.convs.{}".format(round_index)] = 0.0
        for name, parameter in net.named_parameters():
            if parameter.grad is None:
                continue
            for group in groups:
                if name.startswith(group):
                    groups[group] += float(parameter.grad.norm())
        for group, norm in groups.items():
            with self.subTest(parameter_group=group):
                self.assertGreater(norm, 0.0, "no gradient reached {}".format(group))

    def test_update_trains_communicator_and_head_jointly(self):
        method = make_method(seed=11, rounds=2)
        net = method._actor_nets["__shared__"]
        before = {
            name: parameter.detach().clone() for name, parameter in net.named_parameters()
        }
        collect_rollout(method)
        metrics = method.update()
        for key in ("loss_objective", "loss_critic", "loss_entropy"):
            self.assertTrue(torch.isfinite(torch.tensor(metrics[key])), key)
        changed_groups = set()
        for name, parameter in net.named_parameters():
            if not torch.equal(before[name], parameter.detach()):
                changed_groups.add(name.split(".")[0])
        self.assertIn("communicator", changed_groups)
        self.assertIn("head", changed_groups)


class TestRHopActionLimit(unittest.TestCase):
    """Agent D's logits are R-hop-limited by construction (A is 3 hops away)."""

    def build_net(self, rounds, seed=5):
        torch.manual_seed(seed)
        return _GnnActorNet(
            feature_dim=FEATURES,
            num_actions=ACTIONS,
            hidden_size=8,
            comm=gnn_settings(rounds=rounds),
        )

    def logits(self, net, features):
        batch = line_batch(features)
        with torch.no_grad():
            return net(
                batch["features"],
                batch["comm_nodes"],
                batch["comm_edges"],
                batch["comm_num_edges"],
                batch["comm_node_index"],
            )

    def test_r2_invariant_r3_sensitive_beyond_hops(self):
        torch.manual_seed(7)
        features = torch.randn(4, FEATURES)
        perturbed = features.clone()
        perturbed[0] += 10.0  # move agent A far away in feature space

        two_round_net = self.build_net(rounds=2)
        base = self.logits(two_round_net, features)
        moved = self.logits(two_round_net, perturbed)
        self.assertTrue(
            torch.allclose(base[3], moved[3], atol=1e-6),
            "R=2: agent D's logits must be invariant to A (3 hops away)",
        )
        # C (2 hops from A) must be reachable at R=2 — the graph is live.
        self.assertFalse(torch.allclose(base[2], moved[2]))

        three_round_net = self.build_net(rounds=3)
        base = self.logits(three_round_net, features)
        moved = self.logits(three_round_net, perturbed)
        self.assertFalse(
            torch.allclose(base[3], moved[3]),
            "R=3: agent D's logits must be sensitive to A (exactly 3 hops away)",
        )


class TestPerAgentVsBatchedEquivalence(unittest.TestCase):
    """Batched evaluation returns exactly the agent-by-agent outputs."""

    def test_batched_equals_per_sample(self):
        torch.manual_seed(13)
        net = _GnnActorNet(
            feature_dim=FEATURES, num_actions=ACTIONS, hidden_size=8, comm=gnn_settings(3)
        )
        features = torch.randn(4, FEATURES)
        batch = line_batch(features)
        with torch.no_grad():
            batched = net(
                batch["features"],
                batch["comm_nodes"],
                batch["comm_edges"],
                batch["comm_num_edges"],
                batch["comm_node_index"],
            )
            for row in range(4):
                single = net(
                    batch["features"][row : row + 1],
                    batch["comm_nodes"][row : row + 1],
                    batch["comm_edges"][row : row + 1],
                    batch["comm_num_edges"][row : row + 1],
                    batch["comm_node_index"][row : row + 1],
                )
                self.assertTrue(
                    torch.allclose(batched[row], single[0], atol=1e-6), "row {}".format(row)
                )


class TestRolloutReconstruction(unittest.TestCase):
    """Stored raw features/edge_index reproduce collection-time embeddings."""

    def test_recomputed_embeddings_match_collection_time(self):
        method = make_method(seed=21, rounds=2)
        net = method._actor_nets["__shared__"]
        generator = torch.Generator().manual_seed(33)

        collected = []
        inputs = synthetic_inputs(generator, step=0)
        privileged = torch.randn(PRIV, generator=generator)
        for step in range(4):
            with torch.no_grad():
                collected.append(
                    net.communicate(
                        inputs["comm_nodes"],
                        inputs["comm_edges"],
                        inputs["comm_num_edges"],
                        inputs["comm_node_index"],
                    )
                )
            selection = method.select_actions(inputs, privileged)
            next_inputs = synthetic_inputs(generator, step=step + 1)
            method.ingest_transition(
                TeamTransition(
                    step=step,
                    policy_inputs=inputs,
                    privileged_state=privileged,
                    actions=selection.actions,
                    extras=selection.extras,
                    rewards=torch.zeros(len(TEAM)),
                    dones=torch.zeros(len(TEAM), dtype=torch.bool),
                    terminateds=torch.zeros(len(TEAM), dtype=torch.bool),
                    next_policy_inputs=next_inputs,
                    next_privileged_state=privileged,
                )
            )
            inputs = next_inputs

        rollout = method._buffer.as_tensordict()  # [N, T]
        for key in COMMUNICATION_INPUT_KEYS:
            self.assertIn(key, set(rollout.keys()))
        for step in range(4):
            step_td = rollout[:, step]
            with torch.no_grad():
                recomputed = net.communicate(
                    step_td["comm_nodes"],
                    step_td["comm_edges"],
                    step_td["comm_num_edges"],
                    step_td["comm_node_index"],
                )
            self.assertTrue(
                torch.allclose(collected[step], recomputed, atol=1e-6),
                "step {}: rollout reconstruction diverged".format(step),
            )


class TestPrivilegedBoundary(unittest.TestCase):
    """No privileged field can reach the actor through the graph path."""

    def test_actor_in_keys_carry_no_privileged_key(self):
        method = make_method(seed=3)
        actor = method._actors["__shared__"]
        module = actor.module[0]  # the TensorDictModule wrapping the net
        self.assertNotIn(PRIVILEGED_STATE_KEY, module.in_keys)
        self.assertIn("features", module.in_keys)
        for key in COMMUNICATION_INPUT_KEYS:
            self.assertIn(key, module.in_keys)

    def test_node_features_are_exactly_the_policy_features(self):
        generator = torch.Generator().manual_seed(5)
        inputs = synthetic_inputs(generator)
        for row in range(len(TEAM)):
            self.assertTrue(torch.equal(inputs["comm_nodes"][row], inputs["features"]))

    def test_graph_team_mismatch_is_rejected(self):
        generator = torch.Generator().manual_seed(5)
        td = TensorDict(
            {
                "features": torch.randn(len(TEAM), FEATURES, generator=generator),
                "contacts_mask": torch.zeros(len(TEAM), SLOTS, dtype=torch.bool),
            },
            batch_size=[len(TEAM)],
        )
        with self.assertRaisesRegex(ValueError, "node"):
            attach_communication_inputs(td, team_graph(num_agents=2), MAX_EDGES)


class TestTrainingSmoke(unittest.TestCase):
    """Two tiny updates of the shipped GNN benchmark config stay finite."""

    def test_two_updates_with_multihop_gnn(self):
        with open(REPO_ROOT / "config" / "benchmark_blue_mappo_gnn.yaml") as handle:
            config = yaml.safe_load(handle)
        config["training"].update(
            {"updates": 2, "rollout_length": 8, "epochs": 1, "minibatches": 2,
             "hidden_size": 16, "checkpoint_every": 10}
        )
        config["environment"]["max_cycles"] = 8
        with tempfile.TemporaryDirectory() as tmp:
            runner = MarlTrainingRunner(config, results_dir=tmp)
            self.assertTrue(runner._differentiable_comm)
            self.assertEqual(runner.encoder.comm_dim, 0)
            metrics = runner.run()
        self.assertEqual(metrics["update"], 2.0)
        for key in ("mean_reward", "loss_objective", "loss_critic", "loss_entropy"):
            value = torch.tensor(metrics[key])
            self.assertTrue(torch.isfinite(value), "{} is not finite".format(key))


if __name__ == "__main__":
    unittest.main()
