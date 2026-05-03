import argparse
import os
import warnings
from dotenv import load_dotenv
import time

# Deprecation warning
warnings.warn(
    "main_parallel.py is deprecated. Use 'python run.py --mode parallel' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Load environment variables
load_dotenv()

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.experiment import setup_experiment_results_dir, save_prediction_plots, save_timing_stats
from src.agents.factory import create_agents_from_config
from src.env.parallel_env import ParallelGameEnv

logger = get_logger("acn.main.parallel")

def main(config_path):
    """
    Main function to run the Parallel MADRL experiment.
    """
    config = load_config(config_path)
    if not config:
        logger.error("Failed to load configuration. Exiting.")
        return

    experiment_name = config.get("experiment_name", "default_experiment")
    base_results_dir = config.get("results_base_dir", "results")
    results_dir = setup_experiment_results_dir(base_results_dir, experiment_name, config_path, mode_suffix="_parallel")

    logger.info("Starting PARALLEL experiment: {}", experiment_name)

    # Load env_config first to pass grid dimensions to agents
    env_config = config.get("environment", {})

    # 1. Create agents
    agents_config = config.get("agents", {})
    all_agents = create_agents_from_config(agents_config, env_config)

    if not all_agents:
        logger.error("No agents defined. Exiting.")
        return

    logger.info("Created {} agents.", len(all_agents))

    # 2. Set up Parallel Environment
    env_config["experiment_results_dir"] = results_dir

    env = ParallelGameEnv(agents=all_agents, **env_config)
    env.env_config["agents"] = config.get("agents", {})

    logger.info("Parallel Game environment created.")

    # 3. Simulate
    observations, infos = env.reset()
    max_cycles = env_config.get("max_cycles", 100)

    start_time = time.time()

    for cycle in range(max_cycles):
        cycle_start = time.time()

        if not env.agents:
            logger.info("All agents done.")
            break

        logger.debug("Cycle {}/{} | Active agents: {}", cycle + 1, max_cycles, len(env.agents))

        # Collect actions for all active agents simultaneously
        actions = {}
        for agent_name in env.agents:
            if agent_name in observations:
                obs = observations[agent_name]
                agent_obj = env.agent_objects[agent_name]
                action = agent_obj.choose_action(obs)
                actions[agent_name] = action

        observations, rewards, terminations, truncations, infos = env.step(actions)

        # Check for user interrupt (rendering)
        if hasattr(env, 'should_quit') and env.should_quit:
            logger.info("Simulation stopped by user.")
            break

        cycle_dur = time.time() - cycle_start
        logger.debug("Cycle took {:.4f}s", cycle_dur)

    duration = time.time() - start_time
    logger.info("Total time: {:.2f}s", duration)

    save_timing_stats(results_dir, duration, max_cycles, filename="timing_stats_parallel.txt")

    blue_agents = [a for a in all_agents if hasattr(a, 'agent_type') and a.agent_type.value == 'blue']
    env.close()

    save_prediction_plots(blue_agents, results_dir)

    # Save raw position data for analysis if in debug mode
    if env_config.get("debug_mode", False):
        import json
        data_export = {}
        for agent in blue_agents:
            agent_data = {
                "prediction_history": {},
                "actual_history": {}
            }
            if hasattr(agent, 'prediction_history'):
                for red_name, history in agent.prediction_history.items():
                    agent_data["prediction_history"][red_name] = [
                        ([float(pos[0]), float(pos[1])], float(t)) for pos, t in history
                    ]
            if hasattr(agent, 'actual_position_history'):
                for red_name, history in agent.actual_position_history.items():
                    agent_data["actual_history"][red_name] = [
                        ([float(pos[0]), float(pos[1])], float(t)) for pos, t in history
                    ]
            data_export[agent.name] = agent_data

        json_path = os.path.join(results_dir, "position_records.json")
        with open(json_path, "w") as f:
            json.dump(data_export, f, indent=2)
        logger.info("Position records saved to: {}", json_path)

    logger.info("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Parallel MADRL experiments.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.getenv("ACN_CONFIG_PATH", "config/avoidant_config.yaml"),
        help="Path to config file."
    )
    args = parser.parse_args()
    main(args.config)
