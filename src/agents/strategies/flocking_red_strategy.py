import numpy as np
from typing import Dict, Any, Optional, Tuple, List

def calculate_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two positions.

    Args:
        pos1 (Tuple[float, float]): First position (x, y)
        pos2 (Tuple[float, float]): Second position (x, y)

    Returns:
        float: Euclidean distance between the positions
    """
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def is_within_detection_radius(red_pos: Tuple[float, float], 
                             other_pos: Tuple[float, float], 
                             detection_radius: float) -> bool:
    """
    Check if another agent is within the red agent's detection radius.

    Args:
        red_pos (Tuple[float, float]): Position of the Red agent
        other_pos (Tuple[float, float]): Position of the other agent
        detection_radius (float): Detection radius of the Red agent

    Returns:
        bool: True if the other agent is within detection radius, False otherwise
    """
    if red_pos is None or other_pos is None:
        return False
    return calculate_distance(red_pos, other_pos) <= detection_radius

def flocking_red_strategy(current_pos: Optional[Tuple[float, float]], 
                        grid_center: Optional[Tuple[float, float]],
                        red_teammates: Dict[str, Dict[str, Any]],
                        blue_agents: Dict[str, Dict[str, Any]],
                        detection_radius: float = 20.0,
                        cohesion_weight: float = 0.33,
                        alignment_weight: float = 0.33,
                        separation_weight: float = 0.33,
                        separation_radius: float = 1.0,
                        timestamp: float = 0.0) -> Dict[str, Any]:
    """
    A flocking strategy for red agents based on classical flocking behaviors:
    1. Cohesion - moving toward the average position of red teammates
    2. Alignment - moving in the same direction as other red teammates
    3. Separation - moving away from red teammates that are too close

    Args:
        current_pos (Optional[Tuple[float, float]]): The agent's current position (x, y)
        grid_center (Optional[Tuple[float, float]]): The center of the grid (x, y)
        red_teammates (Dict[str, Dict[str, Any]]): Dictionary of red teammates with positions and timestamps
        blue_agents (Dict[str, Dict[str, Any]]): Dictionary of blue agents with positions
        detection_radius (float): Radius within which the red agent can detect other agents
        cohesion_weight (float): Weight given to the cohesion behavior (default: 0.33)
        alignment_weight (float): Weight given to the alignment behavior (default: 0.33)
        separation_weight (float): Weight given to the separation behavior (default: 0.33)
        separation_radius (float): Distance threshold for separation behavior (default: 5.0)

    Returns:
        Dict[str, Any]: A dictionary containing the 'direction' (normalized numpy array)
                         and 'speed' (integer).
    """
    default_direction = np.array([0.0, 0.0], dtype=np.float32)
    default_speed = 0  # Default is to stay still

    if current_pos is None:
        return {'direction': default_direction, 'speed': 0}
        
    # At the initial time step (timestamp near 0), move toward the center of the grid
    if timestamp <= 5.0:  # Using 5.0 as threshold to keep the initial behavior for longer
        # Direction vector toward the grid center
        if grid_center is not None:
            # Calculate vector from current position to grid center
            center_direction = np.array([grid_center[0] - current_pos[0], grid_center[1] - current_pos[1]])
            center_distance = np.linalg.norm(center_direction)
            
            if center_distance > 1e-6:
                # Normalize the direction vector
                direction = center_direction / center_distance
            else:
                # Already at center, choose a random direction
                random_direction = np.random.uniform(-1, 1, 2)
                direction = random_direction / np.linalg.norm(random_direction)
        else:
            # No grid center information, use random direction
            random_direction = np.random.uniform(-1, 1, 2)
            direction = random_direction / np.linalg.norm(random_direction)
        
        # Return maximum speed (5) toward the center
        return {
            'direction': direction.astype(np.float32),
            'speed': 5  # Maximum speed
        }

    # Filter teammates that are within detection radius and have position data
    detected_teammates = {}
    for teammate_name, teammate_data in red_teammates.items():
        if 'position' in teammate_data and 'timestamp' in teammate_data:
            if is_within_detection_radius(current_pos, teammate_data['position'], detection_radius):
                detected_teammates[teammate_name] = teammate_data

    # If no teammates detected, just stay still or move randomly
    if not detected_teammates:
        # Generate a small random movement if no information available
        random_direction = np.random.uniform(-1, 1, 2)
        norm = np.linalg.norm(random_direction)
        if norm > 1e-6:
            random_direction = random_direction / norm
        return {'direction': random_direction.astype(np.float32), 'speed': 1}

    # Create weighted vectors for each behavior rather than separate direction and speed
    cohesion_vector = np.array([0.0, 0.0], dtype=np.float32)
    alignment_vector = np.array([0.0, 0.0], dtype=np.float32)
    separation_vector = np.array([0.0, 0.0], dtype=np.float32)
    
    # 1. COHESION - Vector toward the average position of teammates
    if detected_teammates:
        # Calculate average position of teammates
        teammate_positions = [data['position'] for data in detected_teammates.values()]
        avg_position = np.mean(teammate_positions, axis=0)
        
        # Vector toward average position - magnitude represents strength
        cohesion_vector = np.array([avg_position[0] - current_pos[0], 
                                  avg_position[1] - current_pos[1]])
    
    # 2. ALIGNMENT - Vector in the average direction of teammates' movement
    # We need teammates that have previous positions to determine direction
    teammate_velocity_vectors = []
    for teammate_name, teammate_data in detected_teammates.items():
        if 'previous_positions' in teammate_data and len(teammate_data['previous_positions']) > 0:
            current_teammate_pos = teammate_data['position']
            prev_positions = teammate_data['previous_positions']
            if prev_positions:
                prev_teammate_pos = prev_positions[-1]
                
                # Calculate velocity vector (not normalized)
                velocity_vector = np.array([current_teammate_pos[0] - prev_teammate_pos[0],
                                           current_teammate_pos[1] - prev_teammate_pos[1]])
                
                # Only include if it's a significant movement
                if np.linalg.norm(velocity_vector) > 0.1:  # Threshold to filter out tiny movements
                    teammate_velocity_vectors.append(velocity_vector)
    
    # If we have teammate velocities, calculate the average
    if teammate_velocity_vectors:
        alignment_vector = np.mean(teammate_velocity_vectors, axis=0)
    else:
        # If we can't determine alignment, redistribute its weight
        alignment_weight = 0.0
        # Distribute the weight to other behaviors
        if cohesion_weight > 0 or separation_weight > 0:
            total = cohesion_weight + separation_weight
            cohesion_weight += (alignment_weight * cohesion_weight / total)
            separation_weight += (alignment_weight * separation_weight / total)
    
    # 3. SEPARATION - Composite vector away from all nearby teammates
    for teammate_name, teammate_data in detected_teammates.items():
        distance = calculate_distance(current_pos, teammate_data['position'])
        if distance < separation_radius and distance > 1e-6:  # Avoid division by zero
            # Vector pointing away from the teammate
            away_vector = np.array([current_pos[0] - teammate_data['position'][0],
                                  current_pos[1] - teammate_data['position'][1]])
            
            # Scale inversely with distance (closer = stronger repulsion)
            # This creates a magnitude that increases as agents get closer
            weight = (separation_radius - distance) / separation_radius
            
            # Add to separation vector without normalizing first
            separation_vector += weight * away_vector
    
    if np.linalg.norm(separation_vector) < 1e-6:
        # If no close teammates, redistribute separation weight
        separation_weight = 0.0
        # Distribute the weight to other behaviors
        if cohesion_weight > 0 or alignment_weight > 0:
            total = cohesion_weight + alignment_weight
            cohesion_weight += (separation_weight * cohesion_weight / total)
            alignment_weight += (separation_weight * alignment_weight / total)
    
    # Ensure weights sum to 1.0
    total_weight = cohesion_weight + alignment_weight + separation_weight
    if total_weight > 0:
        cohesion_weight /= total_weight
        alignment_weight /= total_weight
        separation_weight /= total_weight
    
    # Directly compute the weighted combined vector
    combined_vector = (cohesion_weight * cohesion_vector + 
                      alignment_weight * alignment_vector + 
                      separation_weight * separation_vector)
    
    # Calculate magnitude (speed) and normalize direction
    combined_magnitude = np.linalg.norm(combined_vector)
    
    if combined_magnitude > 1e-6:
        # Extract direction (normalized vector)
        direction = combined_vector / combined_magnitude
    else:
        # No clear movement, use a direction toward the center of the grid
        if grid_center is not None:
            center_direction = np.array([grid_center[0] - current_pos[0], grid_center[1] - current_pos[1]])
            center_distance = np.linalg.norm(center_direction)
            
            if center_distance > 1e-6:
                direction = center_direction / center_distance
            else:
                # Already at center, choose a random direction
                random_direction = np.random.uniform(-1, 1, 2)
                direction = random_direction / np.linalg.norm(random_direction)
        else:
            # No grid center information, use random direction
            random_direction = np.random.uniform(-1, 1, 2)
            direction = random_direction / np.linalg.norm(random_direction)
    
    # Always use maximum speed (5) to ensure significant movement
    return {
        'direction': direction.astype(np.float32),
        'speed': 10  # Maximum speed for all agents
    }
