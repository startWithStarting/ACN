import numpy as np
from typing import Tuple


def calculate_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two positions."""
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def is_within_detection_radius(
    center_pos: Tuple[float, float],
    target_pos: Tuple[float, float],
    detection_radius: float,
) -> bool:
    """Check if target_pos is within detection_radius of center_pos."""
    if center_pos is None or target_pos is None:
        return False
    return calculate_distance(center_pos, target_pos) <= detection_radius
