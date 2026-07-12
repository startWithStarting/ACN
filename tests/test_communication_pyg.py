"""Phase 2 tests: PyG aggregation processor and the ``one_hop_mean`` scheme.

Pins the acceptance criteria of docs/communication_implementation_plan.md
("Delivery Phases: Phase 2"):

- PyG outputs match hand calculations (mean/sum/max on the A-B-C chain);
- zero-inbox receivers get an explicit zeros vector and count 0, never NaN;
- inbox-preserving and aggregating schemes share the same delivery runtime
  (identical topology and transport trace records for identical geometry);
- processor configuration is validated at plan build;
- the payload path stays torch end-to-end (gradients flow, no NumPy);
- the parallel environment carries the {"vector", "count"} view into the next
  observation and it survives trace serialization, including a full
  ``run.py --mode parallel`` run with trace recording enabled.
"""

import copy
import glob
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from src.agents.factory import create_agents_from_config
from src.communication.processors.pyg import (
    SUPPORTED_AGGREGATIONS,
    AggregatedVectorView,
    PyGAggregationProcessor,
)
from src.communication.registry import create_communication_plan, list_communication_schemes
from src.communication.runtime import CommunicationRuntime
from src.communication.transport import Frame
from src.communication.types import EdgeMessageBatch
from src.env.parallel_env import ParallelGameEnv
from src.utils.history import to_jsonable
from tests.communication_fixtures import CHAIN_POSITIONS, CHAIN_RADIUS
from tests.test_communication_runtime import agents_from_positions

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Distinct 4-d payloads (signed components make max/mean visibly different).
PAYLOAD_A = (1.0, -2.0, 0.5, 0.0)
PAYLOAD_B = (0.0, 1.0, 1.0, -1.0)
PAYLOAD_C = (3.0, -4.0, -0.5, 2.0)

# Hand-computed aggregation of B's inbox {PAYLOAD_A, PAYLOAD_C} on the chain
# (A->B and C->B are the only edges into B; every value is float32-exact).
EXPECTED_B = {
    "mean": (2.0, -3.0, 0.0, 1.0),
    "sum": (4.0, -6.0, 0.0, 2.0),
    "max": (3.0, -2.0, 0.5, 2.0),
}


def scheme_config(scheme, aggregation=None, **overrides):
    """Build an enabled communication config mapping for one scheme."""
    config = {
        "enabled": True,
        "scheme": scheme,
        "rounds_per_step": 1,
        "cache_window": 0,
    }
    if aggregation is not None:
        config["processor"] = {"aggregation": aggregation}
    config.update(overrides)
    return config


def source_frame(payload, origin):
    """Build a round-0 source frame (tuples are coerced to float32 tensors)."""
    if not isinstance(payload, torch.Tensor):
        payload = torch.tensor(payload, dtype=torch.float32)
    return Frame(payload=payload, origin=origin, created_step=0)


def run_chain_step(plan, scheme_name, outbox_payloads):
    """Run one movement step of a plan over the A-B-C chain fixture geometry."""
    runtime = CommunicationRuntime(plan, scheme_name=scheme_name)
    outboxes = {
        agent_id: [source_frame(payload, agent_id)]
        for agent_id, payload in outbox_payloads.items()
    }
    return runtime.run_step(
        local_observations={},
        outboxes=outboxes,
        step=0,
        agents=agents_from_positions(CHAIN_POSITIONS, CHAIN_RADIUS),
    )


class TestHandComputedAggregations(unittest.TestCase):
    """PyG mean/sum/max outputs match hand calculations on the A-B-C chain."""

    def _run(self, aggregation):
        plan = create_communication_plan(scheme_config("one_hop_mean", aggregation))
        return run_chain_step(
            plan,
            "one_hop_mean",
            {"A": PAYLOAD_A, "B": PAYLOAD_B, "C": PAYLOAD_C},
        )

    def test_aggregations_match_hand_computed_values(self):
        for aggregation in SUPPORTED_AGGREGATIONS:
            with self.subTest(aggregation=aggregation):
                result = self._run(aggregation)
                # B hears A and C; the aggregate is the hand-computed value.
                out_b = result.agent_output["B"]
                self.assertEqual(out_b["count"], 2)
                self.assertTrue(
                    torch.equal(
                        out_b["vector"],
                        torch.tensor(EXPECTED_B[aggregation], dtype=torch.float32),
                    )
                )
                # A and C each hear only B: every reduction returns B's payload.
                for agent_id in ("A", "C"):
                    out = result.agent_output[agent_id]
                    self.assertEqual(out["count"], 1)
                    self.assertTrue(
                        torch.equal(
                            out["vector"], torch.tensor(PAYLOAD_B, dtype=torch.float32)
                        )
                    )

    def test_final_state_stacks_per_agent_vectors(self):
        result = self._run("mean")
        self.assertEqual(tuple(result.final_state.shape), (3, 4))
        for index, agent_id in enumerate(("A", "B", "C")):
            self.assertTrue(
                torch.equal(result.final_state[index], result.agent_output[agent_id]["vector"])
            )


class TestZeroInboxReceivers(unittest.TestCase):
    """Zero-message receivers get zeros and count 0 — never NaN."""

    def test_zero_inbox_yields_zeros_for_every_aggregation(self):
        for aggregation in SUPPORTED_AGGREGATIONS:
            with self.subTest(aggregation=aggregation):
                plan = create_communication_plan(scheme_config("one_hop_mean", aggregation))
                # Only A emits: B receives one message; A and C receive nothing
                # (C is not A's neighbour on the chain).
                result = run_chain_step(plan, "one_hop_mean", {"A": PAYLOAD_A})
                for agent_id in ("A", "C"):
                    out = result.agent_output[agent_id]
                    self.assertEqual(out["count"], 0)
                    self.assertFalse(bool(torch.isnan(out["vector"]).any()))
                    self.assertTrue(torch.equal(out["vector"], torch.zeros(4)))
                self.assertEqual(result.agent_output["B"]["count"], 1)

    def test_step_with_no_messages_at_all_yields_zeros_at_payload_dim(self):
        plan = create_communication_plan(scheme_config("one_hop_mean", "max"))
        result = run_chain_step(plan, "one_hop_mean", {})
        for agent_id in ("A", "B", "C"):
            out = result.agent_output[agent_id]
            self.assertEqual(out["count"], 0)
            self.assertEqual(tuple(out["vector"].shape), (4,))
            self.assertTrue(torch.equal(out["vector"], torch.zeros(4)))
            self.assertFalse(bool(torch.isnan(out["vector"]).any()))


class TestSharedDeliveryRuntime(unittest.TestCase):
    """one_hop_mean and one_hop_direct share the delivery runtime unchanged."""

    OUTBOXES = {"A": PAYLOAD_A, "B": PAYLOAD_B, "C": PAYLOAD_C}

    def _records_without_scheme(self, result):
        records = copy.deepcopy(result.diagnostics["trace_records"])
        for record in records:
            record.pop("scheme", None)
        return records

    def test_topology_and_transport_traces_match_for_identical_geometry(self):
        direct = run_chain_step(
            create_communication_plan(scheme_config("one_hop_direct")),
            "one_hop_direct",
            self.OUTBOXES,
        )
        mean = run_chain_step(
            create_communication_plan(scheme_config("one_hop_mean", "mean")),
            "one_hop_mean",
            self.OUTBOXES,
        )
        # Identical graph and delivery records (modulo the scheme name):
        # topology and transport are shared, only processing differs.
        self.assertEqual(
            self._records_without_scheme(direct), self._records_without_scheme(mean)
        )
        for field in ("message_id", "sender_index", "receiver_index", "origin_index"):
            self.assertTrue(
                torch.equal(
                    getattr(direct.delivered_messages, field),
                    getattr(mean.delivered_messages, field),
                )
            )
        self.assertTrue(
            torch.equal(
                direct.delivered_messages.metadata["hop_count"],
                mean.delivered_messages.metadata["hop_count"],
            )
        )
        # The processors differ exactly in their output contract.
        self.assertIsInstance(direct.agent_output["B"], EdgeMessageBatch)
        self.assertIsInstance(mean.agent_output["B"], AggregatedVectorView)

    def test_aggregating_processor_never_reemits(self):
        plan = create_communication_plan(scheme_config("one_hop_mean", "mean"))
        self.assertEqual(plan.processor.emit_round(None, None, 0, {}), {})
        # Even with extra rounds the chain's second hop is never reached.
        plan.rounds = 2
        result = run_chain_step(plan, "one_hop_mean", {"A": PAYLOAD_A})
        self.assertEqual(result.agent_output["C"]["count"], 0)
        self.assertEqual(result.delivered_messages.num_messages, 1)


class TestTorchEndToEndPayloadPath(unittest.TestCase):
    """No .numpy()/.detach(): gradients flow through the exact aggregation call."""

    def test_gradients_flow_from_output_to_source_payloads(self):
        payload_a = torch.tensor(PAYLOAD_A, requires_grad=True)
        payload_c = torch.tensor(PAYLOAD_C, requires_grad=True)
        plan = create_communication_plan(scheme_config("one_hop_mean", "mean"))
        result = run_chain_step(plan, "one_hop_mean", {"A": payload_a, "C": payload_c})

        vector_b = result.agent_output["B"]["vector"]
        self.assertTrue(vector_b.requires_grad)
        vector_b.sum().backward()
        # Mean over two messages: d(out)/d(payload) is 1/2 elementwise.
        self.assertTrue(torch.equal(payload_a.grad, torch.full((4,), 0.5)))
        self.assertTrue(torch.equal(payload_c.grad, torch.full((4,), 0.5)))

    def test_outputs_are_torch_never_numpy(self):
        plan = create_communication_plan(scheme_config("one_hop_mean", "sum"))
        result = run_chain_step(plan, "one_hop_mean", {"A": PAYLOAD_A, "B": PAYLOAD_B})
        self.assertIsInstance(result.final_state, torch.Tensor)
        for agent_id in ("A", "B", "C"):
            out = result.agent_output[agent_id]
            self.assertIsInstance(out["vector"], torch.Tensor)
            self.assertNotIsInstance(out["vector"], np.ndarray)
            self.assertIsInstance(out["count"], int)


class TestPlanValidation(unittest.TestCase):
    """Processor configuration is validated at plan build (startup)."""

    def test_one_hop_mean_is_registered(self):
        self.assertIn("one_hop_mean", list_communication_schemes())

    def test_unsupported_aggregation_rejected_at_build(self):
        with self.assertRaises(ValueError) as ctx:
            create_communication_plan(scheme_config("one_hop_mean", "softmax"))
        message = str(ctx.exception)
        self.assertIn("softmax", message)
        for supported in SUPPORTED_AGGREGATIONS:
            self.assertIn(supported, message)

    def test_missing_aggregation_rejected_at_build(self):
        # The processor default 'none' would break the aggregated contract.
        with self.assertRaises(ValueError):
            create_communication_plan(scheme_config("one_hop_mean"))

    def test_multiple_rounds_rejected(self):
        with self.assertRaises(ValueError):
            create_communication_plan(
                scheme_config("one_hop_mean", "mean", rounds_per_step=2)
            )

    def test_relay_settings_rejected(self):
        with self.assertRaises(ValueError):
            create_communication_plan(
                {
                    "enabled": True,
                    "scheme": "one_hop_mean",
                    "rounds_per_step": 1,
                    "cache_window": 3,
                    "processor": {"aggregation": "mean", "ttl": 2},
                }
            )

    def test_wrong_engineered_payload_dimension_rejected(self):
        with self.assertRaises(ValueError):
            create_communication_plan(
                scheme_config("one_hop_mean", "mean", payload={"dimension": 6})
            )

    def test_processor_constructor_validates_directly(self):
        with self.assertRaises(ValueError):
            PyGAggregationProcessor(aggregation="attention", payload_dim=4)
        with self.assertRaises(ValueError):
            PyGAggregationProcessor(aggregation="mean", payload_dim=0)


def make_mean_env(**communication_overrides):
    """Build a 3-blue/2-red parallel env with one_hop_mean enabled."""
    env_config = {
        "width": 100,
        "height": 80,
        "max_cycles": 20,
        "save_episode_gifs": False,
        "communication_radius": 1000.0,
        "communication": dict(
            scheme_config("one_hop_mean", "mean"), **communication_overrides
        ),
    }
    agents = create_agents_from_config(
        {
            "blue_agents": [{"count": 3, "detection_radius": 1000.0}],
            "red_agents": [{"count": 2, "detection_radius": 1000.0}],
        },
        env_config=env_config,
    )
    return ParallelGameEnv(agents, **env_config)


def stay_actions(env):
    """Deterministic zero-movement actions for every current agent."""
    return {name: {"direction": (0.0, 0.0), "speed": 0.0} for name in env.agents}


class TestParallelEnvIntegration(unittest.TestCase):
    """The next observation's communication view carries {"vector", "count"}."""

    def test_reset_carries_explicit_empty_view(self):
        env = make_mean_env()
        observations, _ = env.reset(seed=11)
        for observation in observations.values():
            view = observation["communication"]
            self.assertEqual(view["scheme"], "one_hop_mean")
            self.assertEqual(len(view["inbox"]), 0)
        env.close()

    def test_step_view_carries_vector_and_count(self):
        env = make_mean_env()
        env.reset(seed=11)
        observations, _, _, _, infos = env.step(stay_actions(env))
        for name, expected_count in (
            ("blue_0", 4), ("blue_1", 4), ("blue_2", 4), ("red_3", 3), ("red_4", 3),
        ):
            view = observations[name]["communication"]
            self.assertEqual(view["scheme"], "one_hop_mean")
            inbox = view["inbox"]
            self.assertEqual(set(inbox.keys()), {"vector", "count"})
            self.assertEqual(inbox["count"], expected_count)
            self.assertIsInstance(inbox["vector"], torch.Tensor)
            self.assertEqual(tuple(inbox["vector"].shape), (4,))
            self.assertFalse(bool(torch.isnan(inbox["vector"]).any()))
            self.assertEqual(
                infos[name]["communication"]["messages_delivered"], expected_count
            )
        env.close()

    def test_mean_vector_equals_mean_of_direct_inbox(self):
        """Same seed and geometry: the mean view averages the direct inbox."""
        env_direct = make_mean_env(scheme="one_hop_direct", processor=None)
        env_mean = make_mean_env()
        try:
            env_direct.reset(seed=11)
            env_mean.reset(seed=11)
            direct_obs, _, _, _, _ = env_direct.step(stay_actions(env_direct))
            mean_obs, _, _, _, _ = env_mean.step(stay_actions(env_mean))
            for name in ("blue_0", "blue_1", "blue_2", "red_3", "red_4"):
                direct_inbox = direct_obs[name]["communication"]["inbox"]
                mean_view = mean_obs[name]["communication"]["inbox"]
                self.assertTrue(
                    torch.allclose(
                        direct_inbox.payload.mean(dim=0), mean_view["vector"]
                    )
                )
                self.assertEqual(direct_inbox.num_messages, mean_view["count"])
        finally:
            env_direct.close()
            env_mean.close()

    def test_step_observation_with_view_is_json_serializable(self):
        env = make_mean_env()
        env.reset(seed=11)
        observations, _, _, _, infos = env.step(stay_actions(env))
        row = json.dumps(to_jsonable(observations["blue_0"]))
        summary = json.loads(row)["communication"]["inbox"]
        self.assertEqual(summary["type"], "AggregatedVector")
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["dim"], 4)
        self.assertEqual(len(summary["vector"]), 4)
        json.dumps(to_jsonable(infos["blue_0"]))
        env.close()


class TestRunPyEndToEnd(unittest.TestCase):
    """run.py --mode parallel with tracing exits 0 and writes summarized rows."""

    CONFIG_TEMPLATE = """\
experiment_name: "one_hop_mean_e2e"
results_base_dir: "{results_dir}"

agents:
  blue_agents:
    - count: 2
      detection_radius: 1000.0
  red_agents:
    - count: 1
      detection_radius: 1000.0

environment:
  width: 40
  height: 30
  max_cycles: 3
  save_episode_gifs: false
  communication_radius: 1000.0
  communication:
    enabled: true
    scheme: "one_hop_mean"
    rounds_per_step: 1
    cache_window: 0
    processor:
      backend: "pyg"
      aggregation: "mean"
      update: "identity"

analysis:
  trace:
    enabled: true
  plots:
    generate_after_run: false

logging:
  level: "WARNING"
"""

    def test_run_py_parallel_with_tracing_exits_zero(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            results_dir = os.path.join(tmp_dir, "results")
            config_path = os.path.join(tmp_dir, "one_hop_mean_e2e.yaml")
            with open(config_path, "w") as config_file:
                config_file.write(self.CONFIG_TEMPLATE.format(results_dir=results_dir))

            env = dict(os.environ)
            env.setdefault("SDL_VIDEODRIVER", "dummy")
            env.setdefault("MPLBACKEND", "Agg")
            completed = subprocess.run(
                [sys.executable, "run.py", "--mode", "parallel", "--config", config_path],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=300,
            )
            self.assertEqual(
                completed.returncode,
                0,
                "run.py failed:\nstdout:\n{}\nstderr:\n{}".format(
                    completed.stdout, completed.stderr
                ),
            )

            transition_files = glob.glob(
                os.path.join(results_dir, "**", "agent_transitions.jsonl"), recursive=True
            )
            self.assertEqual(len(transition_files), 1)
            summaries = []
            with open(transition_files[0]) as trace_file:
                for line in trace_file:
                    row = json.loads(line)  # every row must be valid JSON
                    view = (row.get("next_observation") or {}).get("communication")
                    if view is not None:
                        self.assertEqual(view["scheme"], "one_hop_mean")
                        inbox = view["inbox"]
                        if isinstance(inbox, dict):
                            summaries.append(inbox)
            self.assertTrue(summaries, "no aggregated views reached the trace rows")
            for summary in summaries:
                self.assertEqual(summary["type"], "AggregatedVector")
                self.assertEqual(summary["dim"], 4)
                self.assertEqual(len(summary["vector"]), 4)
                self.assertGreaterEqual(summary["count"], 0)


if __name__ == "__main__":
    unittest.main()
