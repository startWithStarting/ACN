import argparse
import os
import yaml # Or json, if preferred later
from datetime import datetime

from src.utils.config_loader import load_config
from src.env.aec_env import AECGameEnv
from src.agents.base_agent import AgentType
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
    env = AECGameEnv(agents=all_agents, **env_config)
    print("Game environment created.")

    # 3. Initiate the training process (Placeholder)
    # training_config = config.get("training", {})
    # trainer = Trainer(env, all_agents, training_config, results_dir)
    # print("Initializing training...")
    # trainer.train() # This will be the main training loop

    # For now, let's just simulate some environment steps
    print("\nSimulating environment steps (PettingZoo AEC style):")
    env.reset()
    max_cycles = env_config.get("max_cycles", 100) # Get max_cycles from config or default
    for agent_name in env.agent_iter(max_iter=max_cycles * len(all_agents)): # Iterate through agents
        observation, reward, terminated, truncated, info = env.last()

        if terminated or truncated:
            print(f"Agent {agent_name} finished. Terminated: {terminated}, Truncated: {truncated}")
            action = None # No action if agent is done
        else:
            # Here, the RL agent would choose an action based on the observation
            # For now, sample a random action if action space is available
            if env.action_space(agent_name):
                 action = env.action_space(agent_name).sample()
                 print(f"Agent {agent_name} taking action: {action}")
            else: # Should not happen if agent is not done
                action = None
                print(f"Agent {agent_name} has no action space but is not done. This is unexpected.")


        if action is None and not (terminated or truncated):
            # This handles the case where an agent might be "skipped" if it's not its turn or some other logic
            # In AEC, env.last() gives the current agent that needs to act.
            # If an agent is done, it won't be selected by agent_iter for action.
            # However, if an agent is selected but cannot act (e.g. action space is None),
            # we might need to pass a None action or handle it specifically.
            # PettingZoo's step(None) is often used to signify no action or to pass control.
            print(f"Agent {agent_name} takes no action (or is done).")


        env.step(action)

    env.close()
    print("\nEnvironment simulation finished.")

    # 4. Save results (Placeholder)
    print(f"Results for experiment '{experiment_name}' would be saved in '{results_dir}'.")
    # Example: trainer.save_results()

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
