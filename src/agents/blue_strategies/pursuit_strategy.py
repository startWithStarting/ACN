import numpy as np
from typing import Dict, Any, Optional, Tuple, List

def pursuit_blue_strategy(current_pos: np.ndarray, 
                          detected_predictions: List[Tuple[float, float]],
                          max_speed: float = 5.0,
                          speed_scale: float = 0.5) -> Dict[str, Any]:
    """
    A strategy where blue agents move toward the average predicted position of detected red agents.
    
    Args:
        current_pos (np.ndarray): Current position of the blue agent (x, y)
        detected_predictions (List[Tuple[float, float]]): List of predicted positions for detected red agents
        max_speed (float): Maximum speed limit
        speed_scale (float): Scale factor for speed calculation
        
    Returns:
        Dict[str, Any]: A dictionary containing the 'direction' (normalized numpy array)
                         and 'speed' (float).
    """
    # If no predictions are available, stay still
    if not detected_predictions:
        return {
            'direction': np.array([0.0, 0.0], dtype=np.float32),
            'speed': 0
        }
    
    # Calculate average predicted position
    avg_predicted_pos = np.mean(detected_predictions, axis=0)
    
    # Calculate direction vector from current position to average predicted position
    direction_vector = avg_predicted_pos - current_pos
    
    # Normalize the direction vector if it's not zero
    distance = np.linalg.norm(direction_vector)
    if distance > 0:
        direction_vector = direction_vector / distance
    
    # Set speed proportional to the distance
    speed = min(distance * speed_scale, max_speed)
    
    return {
        'direction': np.array(direction_vector, dtype=np.float32),
        'speed': float(speed)
    }
