import numpy as np
from typing import Dict, Any, Optional, Tuple, List

from src.utils.geometry import is_within_detection_radius
from ..registry import register_strategy


@register_strategy("avoidant", side="red")
def avoidant_red_strategy(current_pos: Optional[Tuple[float, float]],
                         grid_center: Optional[Tuple[float, float]],
                         blue_agents: Dict[str, Dict[str, Any]],
                         detection_radius: float = 15.0,
                         center_weight: float = 0.4,
                         avoidance_weight: float = 0.6) -> Dict[str, Any]:
    """
    A strategy for red agents to detect and avoid blue agents while still being drawn to the center.
    The agent will steer away from blue agents within its detection radius.

    Args:
        current_pos (Optional[Tuple[float, float]]): The agent's current position (x, y).
                                                     None if position is unknown.
        grid_center (Optional[Tuple[float, float]]): The center of the grid (x, y).
                                                     None if center is unknown.
        blue_agents (Dict[str, Dict[str, Any]]): Dictionary of blue agents with their positions.
        detection_radius (float): Radius within which the red agent can detect blue agents.
        center_weight (float): Weight given to the center-seeking behavior (0 to 1).
        avoidance_weight (float): Weight given to the blue-agent-avoiding behavior (0 to 1).

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
    for agent_name, agent_data in blue_agents.items():
        if 'position' in agent_data:
            blue_pos = agent_data['position']
            if is_within_detection_radius(current_pos, blue_pos, detection_radius):
                detected_blue_agents.append(blue_pos)

    # If no blue agents detected, just use center-seeking behavior
    if not detected_blue_agents:
        return {
            'direction': center_direction.astype(np.float32),
            'speed': center_speed
        }

    # Calculate avoidance direction (away from detected blue agents)
    avoidance_direction = np.zeros(2, dtype=np.float32)
    for blue_pos in detected_blue_agents:
        # Vector from blue agent to red agent
        avoid_vector = np.array([current_x - blue_pos[0], current_y - blue_pos[1]], dtype=np.float32)
        distance = np.linalg.norm(avoid_vector)
        
        # Stronger avoidance for closer blue agents
        if distance > 1e-6:
            # Weight inversely proportional to distance (closer agents have more influence)
            weight = 1.0 / (distance * distance)
            avoidance_direction += weight * avoid_vector / distance

    # Normalize avoidance direction if not zero
    avoidance_magnitude = np.linalg.norm(avoidance_direction)
    if avoidance_magnitude > 1e-6:
        avoidance_direction = avoidance_direction / avoidance_magnitude

    # Combine the two behaviors with weights
    combined_direction = (center_weight * center_direction + 
                          avoidance_weight * avoidance_direction)
    
    # Normalize the combined direction
    combined_magnitude = np.linalg.norm(combined_direction)
    if combined_magnitude > 1e-6:
        combined_direction = combined_direction / combined_magnitude

    # Set speed: higher when avoiding blue agents
    avoidance_speed = min(5.0, 3.0 + len(detected_blue_agents))  # More blue agents = higher speed, with max 5

    # Final speed is a weighted combination
    speed = center_weight * center_speed + avoidance_weight * avoidance_speed

    return {
        'direction': combined_direction.astype(np.float32),
        'speed': np.float32(speed)
    }
