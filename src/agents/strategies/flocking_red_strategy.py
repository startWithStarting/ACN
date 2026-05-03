import numpy as np
from typing import Dict, Any, Optional, Tuple, List

from src.utils.geometry import calculate_distance, is_within_detection_radius
from ..registry import register_strategy


def limit_magnitude(vector: np.ndarray, max_val: float) -> np.ndarray:
    """Limit the magnitude of a vector to max_val."""
    magnitude = np.linalg.norm(vector)
    if magnitude > max_val and magnitude > 1e-6:
        return (vector / magnitude) * max_val
    return vector


@register_strategy("flocking", side="red")
def flocking_red_strategy(current_pos: Optional[Tuple[float, float]],
                        grid_center: Optional[Tuple[float, float]],
                        red_teammates: Dict[str, Dict[str, Any]],
                        blue_agents: Dict[str, Dict[str, Any]],
                        detection_radius: float = 20.0,
                        cohesion_weight: float = 1.0,
                        alignment_weight: float = 1.0,
                        separation_weight: float = 1.0,
                        separation_radius: float = 4.0,
                        timestamp: float = 0.0,
                        observation: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Reynolds Flocking Implementation (Steering Behaviors).
    Uses Force = Desired - Current Velocity formulation.
    """
    if current_pos is None or observation is None:
        return {'direction': np.array([0.0, 0.0]), 'speed': np.array([0.0])}

    # Extract Physics Constants from observation (passed from config)
    MAX_SPEED = observation.get('max_speed', 5.0)
    MAX_FORCE = observation.get('max_force', 0.1)  # The "Inertia" factor (Lower = Smoother)

    current_p_vec = np.array(current_pos)
    
    # Get Current Velocity
    current_dir = observation.get('current_direction', (0.0, 0.0))
    current_spd = observation.get('current_speed', 0.0)
    
    # If starting from standstill, assume random small velocity to kickstart alignment
    if current_spd < 0.1:
        current_velocity = np.random.randn(2)
        current_velocity = (current_velocity / np.linalg.norm(current_velocity)) * (MAX_SPEED * 0.5)
    else:
        current_velocity = np.array(current_dir) * current_spd

    # --- 1. Calculate Desired Vectors (The "Want" State) ---
    
    # Cohesion: Steer towards center of mass
    center_of_mass = np.zeros(2)
    neighbor_count = 0
    
    # Alignment: Steer towards average heading
    avg_velocity = np.zeros(2)
    
    # Separation: Steer away from crowded neighbors
    separation_force = np.zeros(2)

    for name, data in red_teammates.items():
        # Get historical positions
        history = data  # list of (pos, ts)
        if not history:
            continue
            
        teammate_pos = np.array(history[-1][0]) # Use latest position
        distance = np.linalg.norm(current_p_vec - teammate_pos)
        
        if distance < detection_radius and distance > 1e-6:
            # Cohesion Accumulator
            center_of_mass += teammate_pos
            
            # Alignment Accumulator (Calculate teammate velocity)
            if len(history) > 1:
                oldest_pos = np.array(history[0][0])
                steps = len(history) - 1
                if steps > 0:
                    teammate_vel = (teammate_pos - oldest_pos) / steps
                    avg_velocity += teammate_vel
            
            # Separation Accumulator (Linear Separation: (Radius - Dist) / Radius)
            if distance < separation_radius:
                diff = current_p_vec - teammate_pos
                # Linear weight: 1.0 at dist=0, 0.0 at dist=radius
                weight = (separation_radius - distance) / separation_radius
                # Direction is diff / distance. Force vector = Direction * Weight
                # So: (diff / distance) * weight
                if distance > 1e-6:
                    separation_force += (diff / distance) * weight
                else:
                    # If on top of each other, random direction or just use diff if non-zero
                    separation_force += np.random.randn(2) # Panic separation
                
            neighbor_count += 1

    # --- 2. Calculate Steering Forces (The "Correction") ---
    steering = np.zeros(2)
    
    if neighbor_count > 0:
        # A. Cohesion Steering
        center_of_mass /= neighbor_count
        desired_cohesion = center_of_mass - current_p_vec # Vector to target
        # Normalize and scale to max speed (Desired Velocity)
        if np.linalg.norm(desired_cohesion) > 0:
            desired_cohesion = (desired_cohesion / np.linalg.norm(desired_cohesion)) * MAX_SPEED
        steer_cohesion = desired_cohesion - current_velocity
        steer_cohesion = limit_magnitude(steer_cohesion, MAX_FORCE) # Limit the turn rate
        
        # B. Alignment Steering
        avg_velocity /= neighbor_count
        if np.linalg.norm(avg_velocity) > 0:
            avg_velocity = (avg_velocity / np.linalg.norm(avg_velocity)) * MAX_SPEED
        steer_alignment = avg_velocity - current_velocity
        steer_alignment = limit_magnitude(steer_alignment, MAX_FORCE)
        
        # C. Separation Steering
        # Separation is usually a direct force, but we can treat it as desired velocity too
        if np.linalg.norm(separation_force) > 0:
            separation_force = (separation_force / np.linalg.norm(separation_force)) * MAX_SPEED
        steer_separation = separation_force - current_velocity
        steer_separation = limit_magnitude(steer_separation, MAX_FORCE * 1.5) # Allow stronger separation
        
        # D. Wall Avoidance Steering
        wall_weight = observation.get('wall_avoidance_weight', 5.0)
        wall_radius = observation.get('wall_detection_radius', 10.0)
        wall_force = np.zeros(2)
        
        # Grid dimensions (using 100x100 default as per config, but ideally should be passed)
        # We can infer from grid_center * 2 if available, or use defaults
        grid_w, grid_h = 100.0, 100.0 
        if grid_center is not None:
             grid_w, grid_h = grid_center[0] * 2, grid_center[1] * 2

        # Left Wall
        if current_p_vec[0] < wall_radius:
            dist = max(0.1, current_p_vec[0])
            wall_force[0] += 1.0 / (dist * dist)
        # Right Wall
        elif current_p_vec[0] > grid_w - wall_radius:
            dist = max(0.1, grid_w - current_p_vec[0])
            wall_force[0] -= 1.0 / (dist * dist)
        
        # Top Wall (0)
        if current_p_vec[1] < wall_radius:
            dist = max(0.1, current_p_vec[1])
            wall_force[1] += 1.0 / (dist * dist)
        # Bottom Wall
        elif current_p_vec[1] > grid_h - wall_radius:
            dist = max(0.1, grid_h - current_p_vec[1])
            wall_force[1] -= 1.0 / (dist * dist)
            
        if np.linalg.norm(wall_force) > 0:
             wall_force = (wall_force / np.linalg.norm(wall_force)) * MAX_SPEED
        
        steer_wall = wall_force - current_velocity
        steer_wall = limit_magnitude(steer_wall, MAX_FORCE * 2.0) # Allow sharper turns for walls

        # Apply Weights
        steering += steer_cohesion * cohesion_weight
        steering += steer_alignment * alignment_weight
        steering += steer_separation * separation_weight
        steering += steer_wall * wall_weight
        
    else:
        # No neighbors? Wander or seek center
        if grid_center is not None:
            desired = np.array(grid_center) - current_p_vec
            if np.linalg.norm(desired) > 0:
                desired = (desired / np.linalg.norm(desired)) * MAX_SPEED
            steer = desired - current_velocity
            steering += limit_magnitude(steer, MAX_FORCE)

    # --- 3. Apply Steering to Velocity ---
    # Clamp total steering force one last time to be safe
    steering = limit_magnitude(steering, MAX_FORCE)
    
    new_velocity = current_velocity + steering
    
    # --- 4. Limit Speed ---
    # Enforce Minimum Speed (Anti-Stagnation)
    MIN_SPEED = observation.get('min_speed', 2.0)
    
    current_speed_mag = np.linalg.norm(new_velocity)
    
    if current_speed_mag < MIN_SPEED and current_speed_mag > 1e-6:
        new_velocity = (new_velocity / current_speed_mag) * MIN_SPEED
    elif current_speed_mag > MAX_SPEED:
         new_velocity = (new_velocity / current_speed_mag) * MAX_SPEED
    
    # Calculate output direction and speed
    speed_mag = np.linalg.norm(new_velocity)
    if speed_mag > 1e-6:
        new_dir = new_velocity / speed_mag
    else:
        # Ensure fallback is a numpy array
        if current_dir is not None:
             new_dir = np.array(current_dir)
        else:
             new_dir = np.array([1.0, 0.0])
        speed_mag = 0.0

    return {
        'direction': new_dir.astype(np.float32),
        'speed': np.array([speed_mag], dtype=np.float32)
    }
