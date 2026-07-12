import numpy as np
from typing import Dict, Any, Optional, Tuple

from src.utils.geometry import is_within_detection_radius
from ..registry import register_strategy


@register_strategy("team_based", side="red")
def team_based_red_strategy(current_pos: Optional[Tuple[float, float]],
                          grid_center: Optional[Tuple[float, float]],
                          red_teammates: Dict[str, Dict[str, Any]],
                          blue_agents: Dict[str, Dict[str, Any]],
                          detection_radius: float = 20.0,
                          center_weight: float = 0.3,
                          team_weight: float = 0.5,
                          avoidance_weight: float = 0.2) -> Dict[str, Any]:
    """
    A strategy for red agents that moves toward the average position of red teammates
    within detection radius, while still maintaining some center-based behavior and avoiding blue agents.

    Args:
        current_pos (Optional[Tuple[float, float]]): The agent's current position (x, y).
                                                     None if position is unknown.
        grid_center (Optional[Tuple[float, float]]): The center of the grid (x, y).
                                                     None if center is unknown.
        red_teammates (Dict[str, Dict[str, Any]]): Dictionary of red teammates with their positions.
        blue_agents (Dict[str, Dict[str, Any]]): Dictionary of blue agents with their positions.
        detection_radius (float): Radius within which the red agent can detect other agents.
        center_weight (float): Weight given to the center-seeking behavior (0 to 1).
        team_weight (float): Weight given to the team-following behavior (0 to 1).
        avoidance_weight (float): Weight given to blue-agent-avoiding behavior (0 to 1).

    Returns:
        Dict[str, Any]: A dictionary containing the 'direction' (normalized numpy array)
                         and 'speed' (integer).
    """
    default_direction = np.array([0.0, 0.0], dtype=np.float32)
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

    # Detect red teammates within detection radius
    detected_teammates = []
    for teammate_name, teammate_data in red_teammates.items():
        if 'position' in teammate_data:
            teammate_pos = teammate_data['position']
            if is_within_detection_radius(current_pos, teammate_pos, detection_radius):
                detected_teammates.append(teammate_pos)

    # If no red teammates detected, use more weight for center-seeking
    team_direction = np.zeros(2, dtype=np.float32)
    team_speed = 0
    if not detected_teammates:
        team_weight = 0.0  # No team weight if no teammates detected
        center_weight += team_weight  # Add team weight to center weight
    else:
        # Calculate average position of detected teammates
        avg_teammate_pos = np.mean(detected_teammates, axis=0)
        
        # Calculate direction vector from current position to average teammate position
        team_vector = np.array([avg_teammate_pos[0] - current_x, avg_teammate_pos[1] - current_y])
        team_distance = np.linalg.norm(team_vector)
        
        # Normalize team direction if not zero
        if team_distance > 1e-6:
            team_direction = team_vector / team_distance
            # Speed is higher when teammates are farther
            team_speed = min(5.0, team_distance / 5.0 + 2.0)
        else:
            # No clear direction if at same position as average teammate
            team_weight = 0.0  # No team weight if at same position
            center_weight += team_weight  # Add team weight to center weight

    # Detect blue agents within detection radius (for avoidance)
    detected_blue_agents = []
    for agent_name, agent_data in blue_agents.items():
        if 'position' in agent_data:
            blue_pos = agent_data['position']
            if is_within_detection_radius(current_pos, blue_pos, detection_radius):
                detected_blue_agents.append(blue_pos)

    # If no blue agents detected, no avoidance needed
    avoidance_direction = np.zeros(2, dtype=np.float32)
    avoidance_speed = 0
    if not detected_blue_agents:
        avoidance_weight = 0.0  # No avoidance weight if no blue agents
        # Distribute the avoidance weight between center and team
        if team_weight > 0:
            team_weight += avoidance_weight / 2
            center_weight += avoidance_weight / 2
        else:
            center_weight += avoidance_weight
    else:
        # Calculate avoidance direction (away from detected blue agents)
        for blue_pos in detected_blue_agents:
            # Vector from blue agent to red agent
            avoid_vector = np.array([current_x - blue_pos[0], current_y - blue_pos[1]], dtype=np.float32)
            avoid_distance = np.linalg.norm(avoid_vector)

            # Stronger avoidance for closer blue agents
            if avoid_distance > 1e-6:
                # Weight inversely proportional to distance
                weight = 1.0 / avoid_distance
                avoidance_direction += weight * avoid_vector / avoid_distance

        # Normalize avoidance direction if not zero
        avoidance_magnitude = np.linalg.norm(avoidance_direction)
        if avoidance_magnitude > 1e-6:
            avoidance_direction = avoidance_direction / avoidance_magnitude
            # Speed is higher when blue agents are closer
            avoidance_speed = min(5.0, 3.0 + len(detected_blue_agents))
        else:
            avoidance_weight = 0.0  # No avoidance if no clear direction
            # Distribute the avoidance weight between center and team
            if team_weight > 0:
                team_weight += avoidance_weight / 2
                center_weight += avoidance_weight / 2
            else:
                center_weight += avoidance_weight

    # Ensure weights sum to 1.0
    total_weight = center_weight + team_weight + avoidance_weight
    if total_weight > 0:
        center_weight /= total_weight
        team_weight /= total_weight
        avoidance_weight /= total_weight

    # Combine the behaviors with adjusted weights
    combined_direction = (center_weight * center_direction +
                          team_weight * team_direction +
                          avoidance_weight * avoidance_direction)

    # Normalize the combined direction
    combined_magnitude = np.linalg.norm(combined_direction)
    if combined_magnitude > 1e-6:
        combined_direction = combined_direction / combined_magnitude
    else:
        # No clear direction, use a small random movement
        random_direction = np.random.uniform(-1, 1, 2)
        combined_direction = random_direction / np.linalg.norm(random_direction)

    # Final speed is a weighted combination
    speed = (center_weight * center_speed +
             team_weight * team_speed +
             avoidance_weight * avoidance_speed)

    # Ensure speed is between 1 and 5
    speed = max(1, min(5, speed))

    return {
        'direction': combined_direction.astype(np.float32),
        'speed': np.float32(speed)
    }
