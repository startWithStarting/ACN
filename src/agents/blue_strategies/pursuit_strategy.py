import numpy as np
from typing import Dict, Any, Optional

from ...agents.registry import register_strategy


@register_strategy("pursuit", side="blue")
def pursuit_blue_strategy(current_pos: np.ndarray,
                          target_position: Optional[np.ndarray],
                          max_speed: float = 5.0,
                          speed_scale: float = 0.5) -> Dict[str, Any]:
    """
    A strategy where blue agents move toward a target position (with momentum/memory).
    
    Args:
        current_pos (np.ndarray): Current position of the blue agent (x, y)
        target_position (Optional[np.ndarray]): Target position to move toward (can be None)
        max_speed (float): Maximum speed limit
        speed_scale (float): Scale factor for speed calculation
        
    Returns:
        Dict[str, Any]: A dictionary containing the 'direction' (normalized numpy array)
                         and 'speed' (float).
    """
    # If no target position is available, stay still
    if target_position is None:
        return {
            'direction': np.array([0.0, 0.0], dtype=np.float32),
            'speed': np.array([0.0], dtype=np.float32)
        }
    
    # Calculate direction vector from current position to target position
    direction_vector = target_position - current_pos
    
    # Normalize the direction vector if it's not zero
    distance = np.linalg.norm(direction_vector)
    if distance > 0:
        direction_vector = direction_vector / distance
    
    # Set speed proportional to the distance
    speed = min(distance * speed_scale, max_speed)

    return {
        'direction': np.array(direction_vector, dtype=np.float32),
        'speed': np.array([speed], dtype=np.float32)
    }
