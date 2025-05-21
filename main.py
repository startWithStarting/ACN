import argparse
import os
import yaml # Or json, if preferred later
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

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
    
    # Create a plots directory within the results directory
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"Results will be saved in: {results_dir}")
    return results_dir

def save_prediction_plots(blue_agents, results_dir):
    """
    Generate and save plots comparing actual vs predicted positions for each Blue agent
    that made detections and predictions during the simulation.
    
    Args:
        blue_agents (list): List of BlueAgent objects
        results_dir (str): Directory to save the plots in
    """
    print(f"Starting save_prediction_plots with {len(blue_agents)} blue agents")
    plots_dir = os.path.join(results_dir, "plots")
    print(f"Plots will be saved to: {plots_dir}")
    
    # Check if plots directory exists, create if not
    if not os.path.exists(plots_dir):
        print(f"Creating plots directory: {plots_dir}")
        os.makedirs(plots_dir, exist_ok=True)
    
    plot_count = 0
    
    for i, agent in enumerate(blue_agents):
        print(f"Processing agent {i+1}/{len(blue_agents)}: {agent.name}")
        
        # Debug: Print agent attributes
        print(f"  Agent has prediction_history: {hasattr(agent, 'prediction_history')}")
        if hasattr(agent, 'prediction_history'):
            print(f"  prediction_history has {len(agent.prediction_history)} entries")
        
        # Skip if the agent has no prediction history
        if not hasattr(agent, 'prediction_history') or not agent.prediction_history:
            print(f"  Skipping agent {agent.name} - no prediction history")
            continue
            
        for red_name, predictions in agent.prediction_history.items():
            print(f"  Processing predictions for {red_name} - {len(predictions)} predictions")
            
            # Debug: Print actual position history
            has_actual = hasattr(agent, 'actual_position_history')
            has_red_actual = has_actual and red_name in agent.actual_position_history
            actual_count = len(agent.actual_position_history.get(red_name, [])) if has_red_actual else 0
            print(f"  Has actual_position_history: {has_actual}, has data for {red_name}: {has_red_actual}, count: {actual_count}")
            
            # Skip if we don't have actual positions for this red agent
            if not has_actual or not has_red_actual or actual_count == 0:
                print(f"  Skipping {red_name} - no actual position history")
                continue
                
            # Get actual and predicted positions with timestamps
            actual_positions = agent.actual_position_history[red_name]
            pred_positions = agent.prediction_history[red_name]
            # Only plot if there are at least 2 actual observations
            if len(actual_positions) < 2:
                continue
            
            # Create two sets of plots: X/Y coordinates and Euclidean distance
            # 1. X and Y coordinates plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
            
            # Extract timestamps and positions
            actual_timestamps = [t for _, t in actual_positions]
            actual_x = [pos[0] for pos, _ in actual_positions]
            actual_y = [pos[1] for pos, _ in actual_positions]
            
            pred_timestamps = [t for _, t in pred_positions]
            pred_x = [pos[0] for pos, _ in pred_positions]
            pred_y = [pos[1] for pos, _ in pred_positions]
            
            # Plot X coordinates
            ax1.plot(actual_timestamps, actual_x, 'ro-', label='Actual X')
            ax1.plot(pred_timestamps, pred_x, 'bo-', label='Predicted X')
            ax1.set_xlabel('Timestamp')
            ax1.set_ylabel('X Coordinate')
            ax1.set_title(f'X Coordinate vs Time: {agent.name} observing {red_name}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot Y coordinates
            ax2.plot(actual_timestamps, actual_y, 'ro-', label='Actual Y')
            ax2.plot(pred_timestamps, pred_y, 'bo-', label='Predicted Y')
            ax2.set_xlabel('Timestamp')
            ax2.set_ylabel('Y Coordinate')
            ax2.set_title(f'Y Coordinate vs Time: {agent.name} observing {red_name}')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'{agent.name}_observing_{red_name}_coordinates.png'))
            plt.close(fig)
            
            # 2. Euclidean distance plot (distance from origin)
            fig, ax = plt.subplots(figsize=(10, 6))
            # Calculate Euclidean distances from grid center (50, 50)
            center_x, center_y = 50, 50
            actual_distances = [np.sqrt((x - center_x)**2 + (y - center_y)**2) for x, y in [(pos[0], pos[1]) for pos, _ in actual_positions]]
            pred_distances = [np.sqrt((x - center_x)**2 + (y - center_y)**2) for x, y in [(pos[0], pos[1]) for pos, _ in pred_positions]]
            # Plot distances
            ax.plot(actual_timestamps, actual_distances, 'ro-', label='Actual Distance')
            ax.plot(pred_timestamps, pred_distances, 'bo-', label='Predicted Distance')
            ax.set_xlabel('Timestamp')
            ax.set_ylabel('Distance from Center (√((x-50)²+(y-50)²))')
            ax.set_title(f'Distance from Center vs Time: {agent.name} observing {red_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'{agent.name}_observing_{red_name}_distance.png'))
            plt.close(fig)
            
            # 3. Prediction Error plot
            # We need to match predictions with actual positions at the same timesteps
            if set(actual_timestamps) & set(pred_timestamps):
                # Create a dictionary of actual positions by timestamp
                actual_by_timestamp = {t: pos for pos, t in actual_positions}
                
                # Get matching timestamps
                matching_timestamps = []
                matching_errors = []
                
                for pred_pos, t in pred_positions:
                    if t in actual_by_timestamp:
                        actual_pos = actual_by_timestamp[t]
                        error = np.sqrt((pred_pos[0] - actual_pos[0])**2 + (pred_pos[1] - actual_pos[1])**2)
                        matching_timestamps.append(t)
                        matching_errors.append(error)
                
                if matching_timestamps:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(matching_timestamps, matching_errors, 'go-', label='Prediction Error')
                    ax.set_xlabel('Timestamp')
                    ax.set_ylabel('Prediction Error (Euclidean Distance)')
                    ax.set_title(f'Prediction Error vs Time: {agent.name} observing {red_name}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(plots_dir, f'{agent.name}_observing_{red_name}_error.png'))
                    plt.close(fig)

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
    
    # Use PettingZoo's agent selector for proper agent cycling
    print("Starting simulation cycles...")
    for cycle in range(max_cycles):
        print(f"Cycle {cycle+1}/{max_cycles}")
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
                
                # Debug information for Blue agents
                if hasattr(agent_obj, 'agent_type') and agent_obj.agent_type.value == 'blue':
                    if hasattr(agent_obj, 'actual_position_history'):
                        for red_name, positions in agent_obj.actual_position_history.items():
                            print(f"  {agent_name} is tracking {red_name}: {len(positions)} actual positions")
                    
                    if hasattr(agent_obj, 'prediction_history'):
                        for red_name, predictions in agent_obj.prediction_history.items():
                            print(f"  {agent_name} has {len(predictions)} predictions for {red_name}")

            # Take the step which will automatically update agent_selection to the next agent
            env.step(action)
            
            # Remove the agent we just processed from our tracking set
            agents_in_cycle.remove(agent_name)
            
            # If the environment has reset (e.g., episode ended), break out of the inner loop
            if len(agents_in_cycle) > 0 and agent_name not in env.agents:
                break
            
        # Check if all agents are done
        if all(env.terminations.values()) or all(env.truncations.values()):
            print("All agents are done. Breaking out of simulation loop.")
            break

    # Before closing the environment, extract the BlueAgent objects for plotting
    blue_agents = [agent for agent in all_agents if hasattr(agent, 'agent_type') and agent.agent_type.value == 'blue']
    
    env.close()
    print("\nEnvironment simulation finished.")

    # 4. Save prediction plots
    print("Generating prediction comparison plots...")
    save_prediction_plots(blue_agents, results_dir)

    # 5. Save results
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
