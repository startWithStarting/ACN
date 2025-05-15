import argparse
import os
import yaml # Or json, if preferred later
from datetime import datetime

from src.utils.config_loader import load_config
from src.env.aec_env import AECGameEnv

from src.agents.red_agent import RedAgent
from src.agents.blue_agent import BlueAgent
# from src.training.trainer import Trainer # To be uncommented later

def setup_experiment_results_dir(base_results_dir, experiment_name):
    """Sets up the directory for saving experiment results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(base_results_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved in: {results_dir}")
    return results_dir

def main(config_path):
    """
    Main function to run the MADRL experiment.
    """
    config = load_config(config_path)
    if not config:
        print("Failed to load configuration. Exiting.")
        return

    experiment_name = config.get("experiment_name", "default_experiment")
    base_results_dir = config.get("results_base_dir", "results")
    results_dir = setup_experiment_results_dir(base_results_dir, experiment_name)

    print(f"Starting experiment: {experiment_name}")

    # 1. Create agents based on config
    agents_config = config.get("agents", {})
    blue_agents_specs = agents_config.get("blue_agents", [])
    red_agents_specs = agents_config.get("red_agents", [])

    all_agents = []
    agent_id_counter = 0

    for spec in blue_agents_specs:
        count = spec.get("count", 0)
        communication_bandwidth = spec.get("communication_bandwidth", 0)
        processing_capability = spec.get("processing_capability", 0)
        for _ in range(count):
            agent_name = f"blue_{agent_id_counter}"
            all_agents.append(BlueAgent(agent_name, communication_bandwidth, processing_capability))
            agent_id_counter += 1

    for spec in red_agents_specs:
        count = spec.get("count", 0)
        communication_bandwidth = spec.get("communication_bandwidth", 0)
        processing_capability = spec.get("processing_capability", 0)
        for _ in range(count):
            agent_name = f"red_{agent_id_counter}"
            all_agents.append(RedAgent(agent_name, communication_bandwidth, processing_capability))
            agent_id_counter += 1
    
    if not all_agents:
        print("No agents defined in the configuration. Exiting.")
        return

    print(f"Created {len(all_agents)} agents.")
    for agent in all_agents:
        print(f"  - {agent.name}: Type={agent.agent_type}, Comm={agent.communication_bandwidth}, Proc={agent.processing_capability}")

    # 2. Set up the game environment
    env_config = config.get("environment", {})
    # Pass agent instances to the environment
    env_config["experiment_results_dir"] = results_dir # Pass the full results path for this run
    env = AECGameEnv(agents=all_agents, **env_config)
    print("Game environment created.")

    # 3. Simulate environment steps
    print("\nSimulating environment steps:")
    env.reset()
    max_cycles = env_config.get("max_cycles", 100)
    
    # Manually iterate through agents to ensure proper cycling
    for cycle in range(max_cycles):
        for agent_name in env.agents:
            # Set the current agent selection
            env.agent_selection = agent_name
            
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

            env.step(action)
            
        # Check if all agents are done
        if all(env.terminations.values()) or all(env.truncations.values()):
            print("All agents are done. Breaking out of simulation loop.")
            break

    env.close()
    print("\nEnvironment simulation finished.")

    # 4. Save results
    print(f"Results for experiment '{experiment_name}' saved in '{results_dir}'.")

    print("Experiment finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MADRL experiments.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/experiment_config.yaml",
        help="Path to the experiment configuration file (YAML or JSON)."
    )
    args = parser.parse_args()
    main(args.config)
