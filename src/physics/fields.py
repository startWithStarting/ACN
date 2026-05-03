"""Force fields for ACN physics.

This module provides force field types:
- ForceField (ABC)
- AttractorField (pull toward a point)
- RepulsorField (push away from a point)
- FlowField (constant-direction force like wind)

Usage:
    field = AttractorField(center=np.array([50.0, 50.0]), strength=0.5)
    force = field.force_at(agent_position)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from src.utils.logger import get_logger

logger = get_logger("acn.physics.fields")


class ForceField(ABC):
    """Abstract base class for force fields."""

    @abstractmethod
    def force_at(self, position: np.ndarray) -> np.ndarray:
        """Calculate the force vector at the given position."""
        pass


@dataclass
class AttractorField(ForceField):
    """Attractor field: pulls toward a center point."""
    center: np.ndarray
    strength: float = 0.5
    falloff: float = 0.0  # If > 0, force decreases with distance

    def __post_init__(self):
        self.center = np.array(self.center, dtype=np.float32)

    def force_at(self, position: np.ndarray) -> np.ndarray:
        """Calculate attraction force toward center."""
        pos = np.array(position, dtype=np.float32)
        delta = self.center - pos
        distance = np.linalg.norm(delta)

        if distance < 1e-6:
            return np.zeros(2, dtype=np.float32)

        direction = delta / distance

        # Calculate force magnitude
        if self.falloff > 0:
            # Inverse square falloff
            magnitude = self.strength / (1 + self.falloff * distance)
        else:
            magnitude = self.strength

        return direction * magnitude


@dataclass
class RepulsorField(ForceField):
    """Repulsor field: pushes away from a center point."""
    center: np.ndarray
    strength: float = 0.5
    radius: float = float('inf')  # Only apply force within this radius

    def __post_init__(self):
        self.center = np.array(self.center, dtype=np.float32)

    def force_at(self, position: np.ndarray) -> np.ndarray:
        """Calculate repulsion force away from center."""
        pos = np.array(position, dtype=np.float32)
        delta = pos - self.center
        distance = np.linalg.norm(delta)

        # Only apply within radius
        if distance > self.radius or distance < 1e-6:
            return np.zeros(2, dtype=np.float32)

        direction = delta / distance
        magnitude = self.strength * (1 - distance / self.radius)

        return direction * magnitude


@dataclass
class FlowField(ForceField):
    """Flow field: constant-direction force (like wind or current)."""
    direction: np.ndarray
    strength: float = 0.1

    def __post_init__(self):
        self.direction = np.array(self.direction, dtype=np.float32)
        # Normalize direction
        norm = np.linalg.norm(self.direction)
        if norm > 1e-6:
            self.direction = self.direction / norm

    def force_at(self, position: np.ndarray) -> np.ndarray:
        """Return constant force in the flow direction."""
        return self.direction * self.strength


@dataclass
class RadialFlowField(ForceField):
    """Radial flow: force points outward from center (opposite of attractor)."""
    center: np.ndarray
    strength: float = 0.5

    def __post_init__(self):
        self.center = np.array(self.center, dtype=np.float32)

    def force_at(self, position: np.ndarray) -> np.ndarray:
        """Calculate outward force from center."""
        pos = np.array(position, dtype=np.float32)
        delta = pos - self.center
        distance = np.linalg.norm(delta)

        if distance < 1e-6:
            return np.zeros(2, dtype=np.float32)

        direction = delta / distance
        return direction * self.strength


def create_force_field(config: dict) -> ForceField:
    """Factory function to create a force field from config."""
    field_type = config.get("type", "flow")

    if field_type == "attractor":
        return AttractorField(
            center=config.get("center", [50.0, 50.0]),
            strength=config.get("strength", 0.5),
            falloff=config.get("falloff", 0.0),
        )
    elif field_type == "repulsor":
        return RepulsorField(
            center=config.get("center", [50.0, 50.0]),
            strength=config.get("strength", 0.5),
            radius=config.get("radius", float('inf')),
        )
    elif field_type == "flow":
        return FlowField(
            direction=config.get("direction", [0.1, 0.0]),
            strength=config.get("strength", 0.1),
        )
    elif field_type == "radial":
        return RadialFlowField(
            center=config.get("center", [50.0, 50.0]),
            strength=config.get("strength", 0.5),
        )
    else:
        raise ValueError(f"Unknown force field type: {field_type}")
