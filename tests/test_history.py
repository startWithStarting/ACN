import os
import tempfile
import unittest
import uuid

import numpy as np

from src.agents.blue_agent import BlueAgent
from src.agents.red_agent import RedAgent
from src.analysis.blue_history import generate_plots, reconstruct_blue_history, summarize_blue_history
from src.utils.experiment import (
    build_file_run_id,
    build_persisted_run_id,
    should_generate_prediction_plots,
    should_record_trace,
)
from src.utils.history import RunHistoryRecorder, create_history_recorder


class TestRunHistoryRecorder(unittest.TestCase):
    def test_analysis_defaults_record_trace_without_eager_plots(self):
        self.assertTrue(should_record_trace({}))
        self.assertFalse(should_generate_prediction_plots({}))

    def test_file_and_persisted_run_id_shapes(self):
        file_run_id = build_file_run_id("experiment", mode_suffix="_parallel")
        self.assertRegex(file_run_id, r"^experiment_\d{8}_\d{6}_parallel$")
        uuid.UUID(build_persisted_run_id())

    def test_file_recorder_manifest_stores_run_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = create_history_recorder(
                persist=False,
                results_dir=tmpdir,
                config={"experiment_name": "history_test"},
                mode="parallel",
                config_path="config/test.yaml",
                run_id="history_test_20260505_120000_parallel",
            )
            recorder.finish(duration_seconds=0.0, num_steps=0)

            manifest = os.path.join(tmpdir, "trace", "manifest.json")
            with open(manifest, "r", encoding="utf-8") as f:
                self.assertIn('"run_id": "history_test_20260505_120000_parallel"', f.read())

    def test_records_and_reconstructs_blue_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            blue = BlueAgent(
                "blue_0",
                communication_bandwidth=5,
                processing_capability=3,
                detection_radius=10.0,
            )
            red = RedAgent(
                "red_0",
                communication_bandwidth=1,
                processing_capability=1,
            )
            blue.x = 0.0
            blue.y = 0.0
            blue.direction = (1.0, 0.0)
            blue.speed = 0.0
            red.x = 3.0
            red.y = 4.0
            red.direction = (1.0, 0.0)
            red.speed = 0.0

            agent_objects = {"blue_0": blue, "red_0": red}
            recorder = RunHistoryRecorder(
                results_dir=tmpdir,
                config={"experiment_name": "history_test"},
                mode="parallel",
                config_path="config/test.yaml",
            )
            recorder.register_agents(agent_objects.values())

            observations = {
                "blue_0": {
                    "position": np.array([0.0, 0.0], dtype=np.float32),
                    "timestamp": 0,
                    "red_agents": {"red_0": {"position": (3.0, 4.0)}},
                },
                "red_0": {"position": np.array([3.0, 4.0], dtype=np.float32), "timestamp": 0},
            }
            blue.predicted_positions = {"red_0": [(4.0, 4.0)]}
            blue.current_target_position = np.array([4.0, 4.0], dtype=np.float32)
            state_before = recorder.snapshot_agents(agent_objects)
            red.x = 4.0
            red.y = 4.0
            state_after = recorder.snapshot_agents(agent_objects)

            recorder.record_blue_events(
                episode=1,
                step=0,
                observations=observations,
                agent_objects=agent_objects,
            )
            recorder.record_agent_transitions(
                episode=1,
                step=0,
                observations=observations,
                actions={
                    "blue_0": {"direction": np.array([1.0, 0.0]), "speed": np.array([1.0])},
                    "red_0": {"direction": np.array([1.0, 0.0]), "speed": np.array([1.0])},
                },
                next_observations=observations,
                rewards={"blue_0": 0.0, "red_0": 0.0},
                terminations={"blue_0": False, "red_0": False},
                truncations={"blue_0": False, "red_0": False},
                infos={"blue_0": {}, "red_0": {}},
                state_before=state_before,
                state_after=state_after,
                agent_objects=agent_objects,
            )

            observations_step_1 = {
                "blue_0": {
                    "position": np.array([0.0, 0.0], dtype=np.float32),
                    "timestamp": 1,
                    "red_agents": {"red_0": {"position": (4.0, 4.0)}},
                },
                "red_0": {"position": np.array([4.0, 4.0], dtype=np.float32), "timestamp": 1},
            }
            state_before_step_1 = recorder.snapshot_agents(agent_objects)
            state_after_step_1 = recorder.snapshot_agents(agent_objects)
            recorder.record_agent_transitions(
                episode=1,
                step=1,
                observations=observations_step_1,
                actions={"blue_0": None, "red_0": None},
                next_observations=observations_step_1,
                rewards={"blue_0": 0.0, "red_0": 0.0},
                terminations={"blue_0": False, "red_0": False},
                truncations={"blue_0": False, "red_0": False},
                infos={"blue_0": {}, "red_0": {}},
                state_before=state_before_step_1,
                state_after=state_after_step_1,
                agent_objects=agent_objects,
            )

            recorder.finish(duration_seconds=0.1, num_steps=2)

            history = reconstruct_blue_history(tmpdir, "blue_0")
            summary = summarize_blue_history(history)

            self.assertEqual(summary["blue_agent_id"], "blue_0")
            self.assertEqual(summary["targets_observed"], {"red_0": 1})
            self.assertEqual(summary["targets_predicted"], {"red_0": 1})
            self.assertEqual(summary["prediction_error"]["red_0"]["count"], 1)
            self.assertAlmostEqual(summary["prediction_error"]["red_0"]["mean"], 0.0)

            plot_paths = generate_plots(
                history,
                target_id="red_0",
                output_dir=os.path.join(tmpdir, "analysis"),
                plot_type="all",
            )
            self.assertTrue(plot_paths)
            for path in plot_paths:
                self.assertTrue(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
