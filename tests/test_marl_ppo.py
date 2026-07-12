"""TeamPPO tests: determinism, gradient flow, and the four actor/critic combos.

These tests run TeamPPO on synthetic transitions (no environment) so they are
fast and fully controlled: features and rewards come from a dedicated seeded
generator, network init and action sampling from ``torch.manual_seed``.
"""

import unittest

import torch
from tensordict import TensorDict

from src.training.marl.config import TrainingSettings
from src.training.marl.contract import TeamTransition
from src.training.marl.ppo import TeamPPO

TEAM = ["red_0", "red_1"]
FEATURES = 8
SLOTS = 2
PRIV = 12
ACTIONS = 5


def make_settings(seed=3, actor="shared", critic="global", rollout=6):
    return TrainingSettings(
        trainable_team="red",
        actor=actor,
        critic=critic,
        rollout_length=rollout,
        updates=1,
        epochs=2,
        minibatches=2,
        lr=1e-2,
        seed=seed,
        hidden_size=16,
        entropy_coef=0.01,
    )


def make_method(seed=3, actor="shared", critic="global", rollout=6):
    torch.manual_seed(seed)
    settings = make_settings(seed=seed, actor=actor, critic=critic, rollout=rollout)
    return TeamPPO(
        settings,
        team_names=TEAM,
        feature_dim=FEATURES,
        privileged_dim=PRIV,
        num_actions=ACTIONS,
    )


def synthetic_inputs(generator):
    return TensorDict(
        {
            "features": torch.randn(len(TEAM), FEATURES, generator=generator),
            "contacts_mask": torch.zeros(len(TEAM), SLOTS, dtype=torch.bool),
        },
        batch_size=[len(TEAM)],
    )


def collect_rollout(method, data_seed=17, rollout=6):
    """Drive the method through one synthetic rollout via the real contract."""
    generator = torch.Generator().manual_seed(data_seed)
    inputs = synthetic_inputs(generator)
    privileged = torch.randn(PRIV, generator=generator)
    for step in range(rollout):
        selection = method.select_actions(inputs, privileged)
        next_inputs = synthetic_inputs(generator)
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


class TestDeterminism(unittest.TestCase):
    def first_update_metrics(self, seed):
        method = make_method(seed=seed)
        collect_rollout(method)
        self.assertTrue(method.ready_to_update())
        return method.update()

    def test_same_seed_identical_first_update_losses(self):
        first = self.first_update_metrics(seed=5)
        second = self.first_update_metrics(seed=5)
        for key in ("loss_objective", "loss_critic", "loss_entropy", "mean_reward"):
            self.assertEqual(first[key], second[key], key)

    def test_different_seeds_differ(self):
        first = self.first_update_metrics(seed=5)
        second = self.first_update_metrics(seed=6)
        self.assertNotEqual(
            (first["loss_objective"], first["loss_critic"]),
            (second["loss_objective"], second["loss_critic"]),
        )


class TestGradientFlow(unittest.TestCase):
    def assert_params_change(self, actor, critic):
        method = make_method(seed=9, actor=actor, critic=critic)
        actor_before = {
            key: [p.detach().clone() for p in net.parameters()]
            for key, net in method._actor_nets.items()
        }
        critic_before = {
            key: [p.detach().clone() for p in net.parameters()]
            for key, net in method._critic_nets.items()
        }
        collect_rollout(method)
        method.update()

        def changed(before, nets):
            return any(
                any(not torch.equal(old, new.detach()) for old, new in zip(before[key], net.parameters()))
                for key, net in nets.items()
            )

        self.assertTrue(changed(actor_before, method._actor_nets), "actor did not train")
        self.assertTrue(changed(critic_before, method._critic_nets), "critic did not train")

    def test_gradients_reach_actor_and_critic_in_all_combos(self):
        for actor in ("shared", "separate"):
            for critic in ("local", "global"):
                with self.subTest(actor=actor, critic=critic):
                    self.assert_params_change(actor, critic)

    def test_shared_central_critic_is_one_network_under_separate_actors(self):
        method = make_method(seed=9, actor="separate", critic="global")
        critic_ids = {id(learner.critic) for learner in method._learners}
        self.assertEqual(len(critic_ids), 1)
        # Per-agent actors are distinct copies.
        actor_ids = {id(learner.actor) for learner in method._learners}
        self.assertEqual(len(actor_ids), len(TEAM))


class TestFourComboSmoke(unittest.TestCase):
    def test_one_tiny_update_per_combo(self):
        for actor in ("shared", "separate"):
            for critic in ("local", "global"):
                with self.subTest(actor=actor, critic=critic):
                    method = make_method(seed=21, actor=actor, critic=critic)
                    collect_rollout(method, data_seed=23)
                    metrics = method.update()
                    self.assertEqual(metrics["update"], 1.0)
                    for key in ("loss_objective", "loss_critic", "loss_entropy"):
                        self.assertTrue(
                            torch.isfinite(torch.tensor(metrics[key])), key
                        )
                    # The buffer was consumed.
                    self.assertFalse(method.ready_to_update())


class TestCheckpointRoundTrip(unittest.TestCase):
    def test_state_dict_restores_updates_and_parameters(self):
        method = make_method(seed=31)
        collect_rollout(method)
        method.update()
        state = method.state_dict()

        torch.manual_seed(31)
        clone = TeamPPO(
            make_settings(seed=31),
            team_names=TEAM,
            feature_dim=FEATURES,
            privileged_dim=PRIV,
            num_actions=ACTIONS,
        )
        clone.load_state_dict(state)
        self.assertEqual(clone.updates_done, 1)
        for key, net in method._actor_nets.items():
            for old, new in zip(net.parameters(), clone._actor_nets[key].parameters()):
                self.assertTrue(torch.equal(old, new))


if __name__ == "__main__":
    unittest.main()
