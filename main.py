import argparse
import os
import warnings
from dotenv import load_dotenv

# Deprecation warning
warnings.warn(
    "main.py is deprecated. Use 'python run.py --mode aec' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Load environment variables from .env file
load_dotenv()
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.experiment import setup_experiment_results_dir, save_prediction_plots, save_timing_stats
from src.agents.factory import create_agents_from_config
from src.env.aec_env import AECGameEnv

logger = get_logger("acn.main.aec")

def main(config_path, train_mode=False):
    """
    Main function to run the MADRL experiment.
    """
    config = load_config(config_path)
    if not config:
        logger.error("Failed to load configuration. Exiting.")
        return

    experiment_name = config.get("experiment_name", "default_experiment")
    base_results_dir = config.get("results_base_dir", "results")
    results_dir = setup_experiment_results_dir(base_results_dir, experiment_name, config_path)

    logger.info("Starting experiment: {}", experiment_name)

    # 1. Create agents based on config
    agents_config = config.get("agents", {})
    all_agents = create_agents_from_config(agents_config)

    if not all_agents:
        logger.error("No agents defined in the configuration. Exiting.")
        return

    logger.info("Created {} agents.", len(all_agents))
    for agent in all_agents:
        logger.debug("  - {}: Type={}, Comm={}, Proc={}",
                     agent.name, agent.agent_type,
                     agent.communication_bandwidth, agent.processing_capability)

    if train_mode:
        from src.training.trainer import Trainer
        logger.info("Entering training mode")
        trainer = Trainer(agents=all_agents, config=config, results_dir=results_dir)
        trainer.train()
        return

    # 2. Set up the game environment
    env_config = config.get("environment", {})
    env_config["experiment_results_dir"] = results_dir
    env = AECGameEnv(agents=all_agents, **env_config)
    logger.info("Game environment created.")

    # 3. Simulate environment steps
    env.reset()
    max_cycles = env_config.get("max_cycles", 100)

    logger.info("Starting simulation cycles...")
    import time
    start_time = time.time()
    total_steps = 0

    for cycle in range(max_cycles):
        cycle_start = time.time()
        logger.debug("Cycle {}/{}", cycle + 1, max_cycles)
        # Loop until all agents have taken their turn in this cycle
        agents_in_cycle = set(env.agents)
        while agents_in_cycle:
            # Get the current agent from the environment's agent selector
            agent_name = env.agent_selection
            
            # Get observation from the environment
            observation = env.observe(agent_name)
            terminated = env.terminations[agent_name]
            truncated = env.truncations[agent_name]
            
            if terminated or truncated:
                action = None # No action if agent is done
            else:
                # Get the agent object and use its choose_action method
                agent_obj = env.agent_objects[agent_name]
                action = agent_obj.choose_action(observation)
                
                # Debug information for Blue agents (use logging)
                pass

            # Take the step which will automatically update agent_selection to the next agent
            env.step(action)
            
            # Remove the agent we just processed from our tracking set
            agents_in_cycle.remove(agent_name)
            
            # If the environment has reset (e.g., episode ended), break out of the inner loop
            if len(agents_in_cycle) > 0 and agent_name not in env.agents:
                break
            
        # Check if all agents are done
        if all(env.terminations.values()) or all(env.truncations.values()):
            logger.info("All agents are done. Breaking out of simulation loop.")
            break

        # Check if user closed the pygame window
        if hasattr(env, 'should_quit') and env.should_quit:
            logger.info("PyGame window closed. Stopping simulation.")
            break

        cycle_end = time.time()
        logger.debug("Cycle {} took {:.4f}s", cycle + 1, cycle_end - cycle_start)

    end_time = time.time()
    duration = end_time - start_time
    logger.info("Total simulation time: {:.2f}s", duration)
    total_steps = max_cycles * len(all_agents)
    logger.info("Average FPS: {:.2f}", total_steps / duration)

    save_timing_stats(results_dir, duration, max_cycles, total_steps=total_steps, num_agents=len(all_agents))

    # Before closing the environment, extract the BlueAgent objects for plotting
    blue_agents = [agent for agent in all_agents if hasattr(agent, 'agent_type') and agent.agent_type.value == 'blue']
    
    env.close()
    logger.info("Environment simulation finished.")

    # 4. Save prediction plots
    save_prediction_plots(blue_agents, results_dir)

    # 5. Save results
    logger.info("Results for experiment '{}' saved in '{}'.", experiment_name, results_dir)
    logger.info("Experiment finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MADRL experiments.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.getenv("ACN_CONFIG_PATH", "config/avoidant_config.yaml"),
        help="Path to the experiment configuration file (YAML or JSON)."
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run in training mode (PPO) instead of simulation mode."
    )
    args = parser.parse_args()
    main(args.config, train_mode=args.train)

