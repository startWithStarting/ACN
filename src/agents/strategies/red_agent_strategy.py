import numpy as np
from typing import Dict, Any, Optional, Tuple

from ..registry import register_strategy

@register_strategy("center", side="red")
def center_based_movement_strategy(current_pos: Optional[Tuple[float, float]],
                                 grid_center: Optional[Tuple[float, float]]) -> Dict[str, Any]:
    """
    A strategy for red agents: move towards the center of the grid, but maintain a minimum
    distance of 10 units from the center. Speed is proportional to the distance from the center.
    When closer than 10 units, the agent is repelled away from the center.

    Args:
        current_pos (Optional[Tuple[float, float]]): The agent's current position (x, y). 
                                                     None if position is unknown.
        grid_center (Optional[Tuple[float, float]]): The center of the grid (x, y).
                                                     None if center is unknown.

    Returns:
        Dict[str, Any]: A dictionary containing the 'direction' (normalized numpy array)
                         and 'speed' (integer). Returns default action if info is missing.
    """
    default_direction = np.array([0.0, 0.0], dtype=np.float32)
    min_distance = 10.0  # Minimum distance to maintain from the center

    if current_pos is None or grid_center is None:
        # Handle missing information, return a default action (stay still)
        return {'direction': default_direction, 'speed': 0}

    current_x, current_y = current_pos
    center_x, center_y = grid_center

    # Calculate vector towards the center
    direction_vector = np.array([center_x - current_x, center_y - current_y])
    
    # Calculate distance to center
    distance_to_center = np.linalg.norm(direction_vector)
    
    if distance_to_center < 1e-6:  # If exactly at center (very unlikely)
        # Choose a random direction to move away
        random_direction = np.random.uniform(-1, 1, 2)
        normalized_direction = (random_direction / np.linalg.norm(random_direction)).astype(np.float32)
        speed = 5.0  # Move away with a moderate speed
    elif distance_to_center < min_distance:
        # Too close to center, reverse direction to move away
        normalized_direction = (-direction_vector / distance_to_center).astype(np.float32)
        # Speed proportional to how much closer than min_distance
        speed = 5.0 * (1 + (min_distance - distance_to_center) / min_distance)
    else:
        # Moving towards center with speed proportional to distance
        normalized_direction = (direction_vector / distance_to_center).astype(np.float32)
        # Cap the speed at 5 (the maximum for the action space)
        speed = min(5.0, distance_to_center / 10.0)

    # The action is a dictionary containing direction and speed
    action = {
        'direction': normalized_direction,
        'speed': np.float32(speed)
    }

    return action
