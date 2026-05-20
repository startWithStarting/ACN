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
from src.utils.experiment import (
    build_file_run_id,
    build_persisted_run_id,
    setup_experiment_results_dir,
    save_prediction_plots,
    save_timing_stats,
    should_generate_prediction_plots,
    should_record_trace,
)
from src.utils.history import create_history_recorder
from src.agents.factory import create_agents_from_config
from src.env.parallel_env import ParallelGameEnv

logger = get_logger("acn.main.parallel")

def main(config_path, persist=False, database_url=None):
    """
    Main function to run the Parallel MADRL experiment.
    """
    config = load_config(config_path)
    if not config:
        logger.error("Failed to load configuration. Exiting.")
        return

    experiment_name = config.get("experiment_name", "default_experiment")
    base_results_dir = config.get("results_base_dir", "results")
    run_id = build_persisted_run_id() if persist else build_file_run_id(
        experiment_name,
        mode_suffix="_parallel",
    )
    results_dir = None if persist else setup_experiment_results_dir(
        base_results_dir,
        experiment_name,
        config_path,
        mode_suffix="_parallel",
        run_id=run_id,
    )

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

    recorder = create_history_recorder(
        persist=persist,
        results_dir=results_dir,
        config=config,
        mode="parallel",
        config_path=config_path,
        enabled=persist or should_record_trace(config),
        run_id=run_id,
        database_url=database_url,
    )
    recorder.register_agents(all_agents)

    # 2. Set up Parallel Environment
    env_config["experiment_results_dir"] = results_dir

    env = ParallelGameEnv(agents=all_agents, **env_config)
    env.env_config["agents"] = config.get("agents", {})

    logger.info("Parallel Game environment created.")

    # 3. Simulate
    observations, infos = env.reset()
    max_cycles = env_config.get("max_cycles", 100)

    start_time = time.time()
    completed_steps = 0

    for cycle in range(max_cycles):
        cycle_start = time.time()

        if not env.agents:
            logger.info("All agents done.")
            break

        logger.debug("Cycle {}/{} | Active agents: {}", cycle + 1, max_cycles, len(env.agents))

        # Collect actions for all active agents simultaneously
        state_before = recorder.snapshot_agents(env.agent_objects)
        actions = {}
        for agent_name in env.agents:
            if agent_name in observations:
                obs = observations[agent_name]
                agent_obj = env.agent_objects[agent_name]
                action = agent_obj.choose_action(obs)
                actions[agent_name] = action

        recorder.record_blue_events(
            episode=env.current_episode_number,
            step=cycle,
            observations=observations,
            agent_objects=env.agent_objects,
        )

        next_observations, rewards, terminations, truncations, infos = env.step(actions)
        state_after = recorder.snapshot_agents(env.agent_objects)
        recorder.record_agent_transitions(
            episode=env.current_episode_number,
            step=cycle,
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            rewards=rewards,
            terminations=terminations,
            truncations=truncations,
            infos=infos,
            state_before=state_before,
            state_after=state_after,
            agent_objects=env.agent_objects,
        )
        observations = next_observations
        completed_steps += 1

        # Check for user interrupt (rendering)
        if hasattr(env, 'should_quit') and env.should_quit:
            logger.info("Simulation stopped by user.")
            break

        cycle_dur = time.time() - cycle_start
        logger.debug("Cycle took {:.4f}s", cycle_dur)

    duration = time.time() - start_time
    logger.info("Total time: {:.2f}s", duration)

    recorder.finish(duration_seconds=duration, num_steps=completed_steps)
    if results_dir is not None:
        save_timing_stats(results_dir, duration, max_cycles, filename="timing_stats_parallel.txt")

    blue_agents = [a for a in all_agents if hasattr(a, 'agent_type') and a.agent_type.value == 'blue']
    env.close()

    if should_generate_prediction_plots(config) and results_dir is not None:
        save_prediction_plots(blue_agents, results_dir)
    elif should_generate_prediction_plots(config):
        logger.info("Skipping post-run prediction plots for DB-persisted run. Use the trace API.")
    else:
        if persist:
            logger.info("Skipping post-run prediction plots. Use the trace API.")
        else:
            logger.info("Skipping post-run prediction plots. Use src.analysis.blue_history on the trace.")

    # Save raw position data for analysis if in debug mode
    if env_config.get("debug_mode", False) and results_dir is not None:
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
    elif env_config.get("debug_mode", False):
        logger.info("Skipping debug JSON export for DB-persisted run.")

    if persist:
        logger.info("Done. Run persisted in DB as run_id={}.", run_id)
        print(f"Persisted run_id: {run_id}")
    else:
        logger.info("Done. Results saved in {}.", results_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Parallel MADRL experiments.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.getenv("ACN_CONFIG_PATH", "config/avoidant_config.yaml"),
        help="Path to config file."
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Persist run history directly to Postgres instead of writing JSON trace files."
    )
    parser.add_argument(
        "--database-url",
        help="Postgres URL for --persist. Defaults to ACN_DATABASE_URL."
    )
    args = parser.parse_args()
    main(args.config, persist=args.persist, database_url=args.database_url)
