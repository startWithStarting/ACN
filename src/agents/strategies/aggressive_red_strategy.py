import numpy as np
from typing import Dict, Any, Optional, Tuple, List

from src.utils.geometry import calculate_distance, is_within_detection_radius
from ..registry import register_strategy


@register_strategy("aggressive", side="red")
def aggressive_red_strategy(current_pos: Optional[Tuple[float, float]],
                           grid_center: Optional[Tuple[float, float]],
                           blue_agents: Dict[str, Dict[str, Any]],
                           detection_radius: float = 20.0,
                           center_weight: float = 0.3,
                           pursuit_weight: float = 0.7) -> Dict[str, Any]:
    """
    A strategy for red agents to detect and pursue blue agents.
    When no blue agents are detected, falls back to center-based behavior.

    Args:
        current_pos (Optional[Tuple[float, float]]): The agent's current position (x, y).
                                                     None if position is unknown.
        grid_center (Optional[Tuple[float, float]]): The center of the grid (x, y).
                                                     None if center is unknown.
        blue_agents (Dict[str, Dict[str, Any]]): Dictionary of blue agents with their positions.
        detection_radius (float): Radius within which the red agent can detect blue agents.
        center_weight (float): Weight given to the center-seeking behavior (0 to 1).
        pursuit_weight (float): Weight given to the blue-agent-pursuing behavior (0 to 1).

    Returns:
        Dict[str, Any]: A dictionary containing the 'direction' (normalized numpy array)
                         and 'speed' (integer).
    """
    default_direction = np.array([0.0, 0.0], dtype=np.float32)
    default_speed = 0  # Default is to stay still
    min_distance = 10.0  # Minimum distance to maintain from the center

    if current_pos is None or grid_center is None:
        return {'direction': default_direction, 'speed': 0}

    # Calculate center-seeking direction and speed (similar to center_based_movement_strategy)
    current_x, current_y = current_pos
    center_x, center_y = grid_center

    # Calculate vector towards the center
    center_direction = np.array([center_x - current_x, center_y - current_y])
    
    # Calculate distance to center
    distance_to_center = np.linalg.norm(center_direction)
    
    # Normalize center direction if not zero
    if distance_to_center > 1e-6:
        center_direction = center_direction / distance_to_center
    else:
        # If at center, choose a random direction
        random_direction = np.random.uniform(-1, 1, 2)
        center_direction = random_direction / np.linalg.norm(random_direction)
    
    # Determine center-based speed
    if distance_to_center < min_distance:
        # Reverse direction if too close to center
        center_direction = -center_direction
        center_speed = 5.0 * (1 + (min_distance - distance_to_center) / min_distance)
    else:
        # Cap the speed at 5 (the maximum for the action space)
        center_speed = min(5.0, distance_to_center / 10.0)

    # Detect blue agents within detection radius
    detected_blue_agents = []
    blue_distances = []
    for agent_name, agent_data in blue_agents.items():
        if 'position' in agent_data:
            blue_pos = agent_data['position']
            if is_within_detection_radius(current_pos, blue_pos, detection_radius):
                detected_blue_agents.append(blue_pos)
                blue_distances.append(calculate_distance(current_pos, blue_pos))

    # If no blue agents detected, just use center-seeking behavior
    if not detected_blue_agents:
        return {
            'direction': center_direction.astype(np.float32),
            'speed': center_speed
        }

    # Find the closest blue agent to pursue
    closest_idx = np.argmin(blue_distances)
    closest_blue_pos = detected_blue_agents[closest_idx]
    closest_distance = blue_distances[closest_idx]
    
    # Calculate pursuit direction (toward closest blue agent)
    pursuit_direction = np.array([closest_blue_pos[0] - current_x, 
                                closest_blue_pos[1] - current_y], dtype=np.float32)
    
    # Normalize pursuit direction
    pursuit_magnitude = np.linalg.norm(pursuit_direction)
    if pursuit_magnitude > 1e-6:
        pursuit_direction = pursuit_direction / pursuit_magnitude

    # Combine the two behaviors with weights
    combined_direction = (center_weight * center_direction + 
                          pursuit_weight * pursuit_direction)
    
    # Normalize the combined direction
    combined_magnitude = np.linalg.norm(combined_direction)
    if combined_magnitude > 1e-6:
        combined_direction = combined_direction / combined_magnitude

    # Set speed: higher when pursuing blue agents and faster the closer they are
    # Scale speed inversely with distance (closer blue agent = higher speed)
    pursuit_speed = min(5.0, 5.0 * (1 - closest_distance / detection_radius) + 2.0)

    # Final speed is a weighted combination
    speed = center_weight * center_speed + pursuit_weight * pursuit_speed
    # Ensure speed is between 1 and 5
    speed = max(1, min(5, speed))

    return {
        'direction': combined_direction.astype(np.float32),
        'speed': np.float32(speed)
    }
