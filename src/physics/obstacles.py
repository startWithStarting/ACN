"""Obstacle system for ACN physics.

This module provides obstacle types for the physics engine:
- Obstacle (ABC)
- RectObstacle (axis-aligned rectangle)
- CircleObstacle (circular obstacle)

Example:
    Use an obstacle directly::

        obstacle = RectObstacle(x=40, y=30, width=10, height=20)
        if obstacle.contains(point):
            ...
        nearest = obstacle.nearest_point(point)
        normal = obstacle.normal_at(point)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from src.utils.logger import get_logger

logger = get_logger("acn.physics.obstacles")


class Obstacle(ABC):
    """Abstract base class for obstacles."""

    @abstractmethod
    def contains(self, point: np.ndarray) -> bool:
        """Check if a point is inside the obstacle."""
        pass

    @abstractmethod
    def nearest_point(self, point: np.ndarray) -> np.ndarray:
        """Find the nearest point on the obstacle surface to the given point."""
        pass

    @abstractmethod
    def normal_at(self, point: np.ndarray) -> np.ndarray:
        """Get the surface normal at the nearest point."""
        pass

    @abstractmethod
    def distance_to(self, point: np.ndarray) -> float:
        """Calculate distance from point to obstacle surface."""
        pass


@dataclass
class RectObstacle(Obstacle):
    """Axis-aligned rectangular obstacle."""
    x: float
    y: float
    width: float
    height: float
    restitution: float = 0.8

    @property
    def left(self) -> float:
        return self.x

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def top(self) -> float:
        return self.y

    @property
    def bottom(self) -> float:
        return self.y + self.height

    def contains(self, point: np.ndarray) -> bool:
        """Check if point is inside the rectangle."""
        return (self.left <= point[0] <= self.right and
                self.top <= point[1] <= self.bottom)

    def nearest_point(self, point: np.ndarray) -> np.ndarray:
        """Find nearest point on rectangle boundary."""
        clamped_x = np.clip(point[0], self.left, self.right)
        clamped_y = np.clip(point[1], self.top, self.bottom)
        return np.array([clamped_x, clamped_y], dtype=np.float32)

    def normal_at(self, point: np.ndarray) -> np.ndarray:
        """Get normal pointing outward from the nearest face."""
        nearest = self.nearest_point(point)

        # Determine which face is closest
        if nearest[0] == self.left:
            return np.array([-1.0, 0.0], dtype=np.float32)
        elif nearest[0] == self.right:
            return np.array([1.0, 0.0], dtype=np.float32)
        elif nearest[1] == self.top:
            return np.array([0.0, -1.0], dtype=np.float32)
        else:
            return np.array([0.0, 1.0], dtype=np.float32)

    def distance_to(self, point: np.ndarray) -> float:
        """Distance from point to rectangle surface."""
        nearest = self.nearest_point(point)
        return np.linalg.norm(point - nearest)


@dataclass
class CircleObstacle(Obstacle):
    """Circular obstacle."""
    x: float
    y: float
    radius: float
    restitution: float = 0.8

    @property
    def center(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)

    def contains(self, point: np.ndarray) -> bool:
        """Check if point is inside the circle."""
        return np.linalg.norm(point - self.center) <= self.radius

    def nearest_point(self, point: np.ndarray) -> np.ndarray:
        """Find nearest point on circle boundary."""
        delta = point - self.center
        distance = np.linalg.norm(delta)
        if distance < 1e-6:
            # Point at center, return any boundary point
            return np.array([self.x + self.radius, self.y], dtype=np.float32)
        return self.center + (delta / distance) * self.radius

    def normal_at(self, point: np.ndarray) -> np.ndarray:
        """Get normal pointing outward."""
        delta = point - self.center
        distance = np.linalg.norm(delta)
        if distance < 1e-6:
            return np.array([1.0, 0.0], dtype=np.float32)
        return delta / distance

    def distance_to(self, point: np.ndarray) -> float:
        """Distance from point to circle surface."""
        return max(0, np.linalg.norm(point - self.center) - self.radius)


def create_obstacle(config: dict) -> Obstacle:
    """Factory function to create an obstacle from config."""
    obs_type = config.get("type", "rect")

    if obs_type == "rect":
        return RectObstacle(
            x=config["x"],
            y=config["y"],
            width=config["width"],
            height=config["height"],
            restitution=config.get("restitution", 0.8),
        )
    elif obs_type == "circle":
        return CircleObstacle(
            x=config["x"],
            y=config["y"],
            radius=config["radius"],
            restitution=config.get("restitution", 0.8),
        )
    else:
        raise ValueError(f"Unknown obstacle type: {obs_type}")
