import argparse
import os
from dotenv import load_dotenv
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# Load environment variables
load_dotenv()

from src.utils.config_loader import load_config
from src.env.parallel_env import ParallelGameEnv

from src.agents.red_agent import RedAgent
from src.agents.blue_agent import BlueAgent

def setup_experiment_results_dir(base_results_dir, experiment_name, config_path):
    """Sets up the directory for saving experiment results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract subfolder name from config filename (e.g., 'avoidant_config.yaml' -> 'avoidant')
    subfolder = "default"
    if config_path:
        filename = os.path.basename(config_path)
        base_name = os.path.splitext(filename)[0]
        if base_name.endswith("_config"):
            subfolder = base_name.replace("_config", "")
        else:
            subfolder = base_name

    # Create nested directory: results/subfolder/experiment_name_timestamp_parallel
    results_dir = os.path.join(base_results_dir, subfolder, f"{experiment_name}_{timestamp}_parallel")
    os.makedirs(results_dir, exist_ok=True)
    
    # Create a plots directory within the results directory
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"Results will be saved in: {results_dir}")
    return results_dir

def save_prediction_plots(blue_agents, results_dir):
    """
    Generate and save plots comparing actual vs predicted positions.
    Reused logic from main.py
    """
    print(f"Starting save_prediction_plots with {len(blue_agents)} blue agents")
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    for i, agent in enumerate(blue_agents):
        if not hasattr(agent, 'prediction_history') or not agent.prediction_history:
            continue
            
        for red_name, predictions in agent.prediction_history.items():
            has_actual = hasattr(agent, 'actual_position_history')
            has_red_actual = has_actual and red_name in agent.actual_position_history
            
            if not has_actual or not has_red_actual:
                continue
                
            actual_positions = agent.actual_position_history[red_name]
            pred_positions = agent.prediction_history[red_name]
            
            if len(actual_positions) < 2:
                continue
            
            # --- Plotting Logic (Simplified for brevity, can match main.py more closely if needed) ---
            # 1. X/Y Coordinates
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
            
            actual_timestamps = [t for _, t in actual_positions]
            actual_x = [pos[0] for pos, _ in actual_positions]
            actual_y = [pos[1] for pos, _ in actual_positions]
            
            pred_timestamps = [t for _, t in pred_positions]
            pred_x = [pos[0] for pos, _ in pred_positions]
            pred_y = [pos[1] for pos, _ in pred_positions]
            
            ax1.plot(actual_timestamps, actual_x, 'ro-', label='Actual X')
            ax1.plot(pred_timestamps, pred_x, 'bo-', label='Predicted X')
            ax1.set_title(f'X Coord: {agent.name} observing {red_name}')
            ax1.legend()
            ax1.grid(True)
            
            ax2.plot(actual_timestamps, actual_y, 'ro-', label='Actual Y')
            ax2.plot(pred_timestamps, pred_y, 'bo-', label='Predicted Y')
            ax2.set_title(f'Y Coord: {agent.name} observing {red_name}')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'{agent.name}_observing_{red_name}_coordinates.png'))
            plt.close(fig)

def main(config_path):
    """
    Main function to run the Parallel MADRL experiment.
    """
    config = load_config(config_path)
    if not config:
        print("Failed to load configuration. Exiting.")
        return

    experiment_name = config.get("experiment_name", "default_experiment")
    base_results_dir = config.get("results_base_dir", "results")
    results_dir = setup_experiment_results_dir(base_results_dir, experiment_name, config_path)

    print(f"Starting PARALLEL experiment: {experiment_name}")

    # Load env_config first to pass grid dimensions to agents
    env_config = config.get("environment", {})
    
    # 1. Create agents
    agents_config = config.get("agents", {})
    blue_agents_specs = agents_config.get("blue_agents", [])
    red_agents_specs = agents_config.get("red_agents", [])

    all_agents = []
    agent_id_counter = 0

    for spec in blue_agents_specs:
        count = spec.get("count", 0)
        # ... retrieve other params ...
        for _ in range(count):
            agent_name = f"blue_{agent_id_counter}"
            # Reusing same agent class for now
            all_agents.append(BlueAgent(agent_name, 
                                      spec.get("communication_bandwidth", 0),
                                      spec.get("processing_capability", 0),
                                      spec.get("detection_radius", 20.0),
                                      spec.get("strategy_type", "pursuit"),
                                      spec.get("prediction_timeout", 50),
                                      spec.get("observation_window_size", 5),
                                      spec.get("prediction_interval", 1),
                                      grid_size=(float(env_config.get("width", 100)), 
                                                 float(env_config.get("height", 100))),
                                      debug_mode=env_config.get("debug_mode", False)))
            agent_id_counter += 1

    for spec in red_agents_specs:
        count = spec.get("count", 0)
        for _ in range(count):
            agent_name = f"red_{agent_id_counter}"
            all_agents.append(RedAgent(agent_name,
                                     spec.get("communication_bandwidth", 0),
                                     spec.get("processing_capability", 0),
                                     spec.get("detection_radius", 15.0),
                                     spec.get("strategy_type", "center")))
            agent_id_counter += 1
    
    if not all_agents:
        print("No agents defined. Exiting.")
        return

    print(f"Created {len(all_agents)} agents.")

    # 2. Set up Parallel Environment
    # env_config already loaded
    env_config["experiment_results_dir"] = results_dir

    # Initialize Parallel Env
    env = ParallelGameEnv(agents=all_agents, **env_config)
    
    # CRITICAL FIX: Pass 'agents' config to env so it can access parameters (e.g. max_force)
    # We do this AFTER init because 'agents' is also a keyword arg for the constructor (the list of objects)
    env.env_config["agents"] = config.get("agents", {})
    
    print("Parallel Game environment created.")

    # 3. Simulate
    print("\nSimulating (Parallel)...")
    observations, infos = env.reset()
    
    max_cycles = env_config.get("max_cycles", 100)
    
    start_time = time.time()
    
    for cycle in range(max_cycles):
        cycle_start = time.time()
        
        if not env.agents:
            print("All agents done.")
            break
            
        print(f"Cycle {cycle+1}/{max_cycles} | Active agents: {len(env.agents)}")
        
        # Collect actions for all active agents simultaneously
        actions = {}
        for agent_name in env.agents:
            if agent_name in observations:
                obs = observations[agent_name]
                agent_obj = env.agent_objects[agent_name]
                action = agent_obj.choose_action(obs)
                actions[agent_name] = action
        
        # Step the environment once with all actions
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Check for user interrupt (rendering)
        if hasattr(env, 'should_quit') and env.should_quit:
            print("Simulation stopped by user.")
            break
            
        cycle_dur = time.time() - cycle_start
        # print(f"Cycle took {cycle_dur:.4f}s")

    duration = time.time() - start_time
    print(f"Total time: {duration:.2f}s")
    
    with open("timing_stats_parallel.txt", "w") as f:
        f.write(f"Duration: {duration:.4f}\n")
        f.write(f"Cycles: {max_cycles}\n")

    blue_agents = [a for a in all_agents if hasattr(a, 'agent_type') and a.agent_type.value == 'blue']
    env.close()
    
    print("Generating plots...")
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
            # Convert tuples/numpy arrays to lists for JSON serialization
            if hasattr(agent, 'prediction_history'):
                for red_name, history in agent.prediction_history.items():
                    # History is list of ((x,y), t)
                    # Convert numpy types if present
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
        print(f"Position records saved to: {json_path}")
    print("Done.")

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
