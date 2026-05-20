"""Shared experiment utilities for setup, plotting, and result export."""

import os
import uuid
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("acn.utils.experiment")


def get_analysis_config(config):
    """Return normalized analysis config with conservative defaults."""
    analysis_config = config.get("analysis", {}) if config else {}
    trace_config = analysis_config.get("trace", {})
    plots_config = analysis_config.get("plots", {})
    return {
        "trace": {
            "enabled": trace_config.get("enabled", True),
        },
        "plots": {
            "generate_after_run": plots_config.get("generate_after_run", False),
        },
    }


def should_record_trace(config):
    """Return whether local run trace data should be written."""
    return bool(get_analysis_config(config)["trace"]["enabled"])


def should_generate_prediction_plots(config):
    """Return whether expensive per-blue/per-red plots should run after simulation."""
    return bool(get_analysis_config(config)["plots"]["generate_after_run"])


def build_file_run_id(experiment_name, mode_suffix=""):
    """Create a timestamped run id for local file-backed runs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{experiment_name}_{timestamp}{mode_suffix}"


def build_persisted_run_id():
    """Create a UUID run id for DB-backed persisted runs."""
    return str(uuid.uuid4())


def setup_experiment_results_dir(
    base_results_dir,
    experiment_name,
    config_path,
    mode_suffix="",
    run_id=None,
):
    """Creates the results directory for a local file-backed experiment run."""
    subfolder = "default"
    if config_path:
        filename = os.path.basename(config_path)
        base_name = os.path.splitext(filename)[0]
        if base_name.endswith("_config"):
            subfolder = base_name.replace("_config", "")
        else:
            subfolder = base_name

    dirname = run_id or build_file_run_id(experiment_name, mode_suffix)
    results_dir = os.path.join(base_results_dir, subfolder, dirname)
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    logger.info("Results will be saved in: {}", results_dir)
    return results_dir


def save_prediction_plots(blue_agents, results_dir):
    """Generate coordinate, distance, and error plots for each blue/red pair."""
    logger.info("Generating prediction plots for {} blue agents", len(blue_agents))
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    for agent in blue_agents:
        if not hasattr(agent, "prediction_history") or not agent.prediction_history:
            continue

        for red_name, predictions in agent.prediction_history.items():
            has_actual = (
                hasattr(agent, "actual_position_history")
                and red_name in agent.actual_position_history
            )
            if not has_actual or len(agent.actual_position_history[red_name]) < 2:
                continue

            actual_positions = agent.actual_position_history[red_name]
            pred_positions = agent.prediction_history[red_name]

            actual_timestamps = [t for _, t in actual_positions]
            actual_x = [pos[0] for pos, _ in actual_positions]
            actual_y = [pos[1] for pos, _ in actual_positions]

            pred_timestamps = [t for _, t in pred_positions]
            pred_x = [pos[0] for pos, _ in pred_positions]
            pred_y = [pos[1] for pos, _ in pred_positions]

            # 1. X/Y coordinates
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

            ax1.plot(actual_timestamps, actual_x, "ro-", label="Actual X")
            ax1.plot(pred_timestamps, pred_x, "bo-", label="Predicted X")
            ax1.set_title(f"X Coord: {agent.name} observing {red_name}")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.plot(actual_timestamps, actual_y, "ro-", label="Actual Y")
            ax2.plot(pred_timestamps, pred_y, "bo-", label="Predicted Y")
            ax2.set_title(f"Y Coord: {agent.name} observing {red_name}")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"{agent.name}_observing_{red_name}_coordinates.png"))
            plt.close(fig)

            # 2. Euclidean distance from grid center
            fig, ax = plt.subplots(figsize=(10, 6))
            center_x, center_y = 50, 50
            actual_dist = [
                np.sqrt((pos[0] - center_x) ** 2 + (pos[1] - center_y) ** 2)
                for pos, _ in actual_positions
            ]
            pred_dist = [
                np.sqrt((pos[0] - center_x) ** 2 + (pos[1] - center_y) ** 2)
                for pos, _ in pred_positions
            ]
            ax.plot(actual_timestamps, actual_dist, "ro-", label="Actual Distance")
            ax.plot(pred_timestamps, pred_dist, "bo-", label="Predicted Distance")
            ax.set_title(f"Distance from Center: {agent.name} observing {red_name}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"{agent.name}_observing_{red_name}_distance.png"))
            plt.close(fig)

            # 3. Prediction error
            common = set(actual_timestamps) & set(pred_timestamps)
            if common:
                actual_by_ts = dict(actual_positions)
                matching_ts = []
                matching_errors = []
                for pred_pos, t in pred_positions:
                    if t in actual_by_ts:
                        actual_pos = actual_by_ts[t]
                        error = np.sqrt(
                            (pred_pos[0] - actual_pos[0]) ** 2 + (pred_pos[1] - actual_pos[1]) ** 2
                        )
                        matching_ts.append(t)
                        matching_errors.append(error)

                if matching_ts:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(matching_ts, matching_errors, "go-", label="Prediction Error")
                    ax.set_title(f"Prediction Error: {agent.name} observing {red_name}")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(plots_dir, f"{agent.name}_observing_{red_name}_error.png"))
                    plt.close(fig)


def save_timing_stats(results_dir, duration, max_cycles, total_steps=None, num_agents=None, filename="timing_stats.txt"):
    """Write timing summary to a text file in the results directory."""
    path = os.path.join(results_dir, filename)
    with open(path, "w") as f:
        f.write(f"Duration: {duration:.4f}\n")
        f.write(f"Cycles: {max_cycles}\n")
        if total_steps is not None:
            f.write(f"Total Steps: {total_steps}\n")
            if duration > 0:
                f.write(f"FPS: {total_steps / duration:.4f}\n")
        if num_agents is not None:
            f.write(f"Agents: {num_agents}\n")
    logger.debug("Timing stats written to {}", path)
