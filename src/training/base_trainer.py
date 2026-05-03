"""Abstract trainer interface for ACN.

This module provides a trainer abstraction allowing different RL frameworks:
- BaseTrainer: Abstract interface
- SB3Trainer: Stable Baselines 3 implementation
- RLlibTrainer: Ray RLlib implementation (stub)

Usage:
    trainer = create_trainer("sb3", agents, config, results_dir)
    trainer.train(total_timesteps=100000)
    trainer.save("models/final_model")
    results = trainer.evaluate(env, num_episodes=5)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger("acn.training")


class BaseTrainer(ABC):
    """Abstract base class for trainers."""

    @abstractmethod
    def train(self, env: Any, total_timesteps: int, callbacks: Optional[Any] = None) -> None:
        """Train the agent(s)."""
        pass

    @abstractmethod
    def evaluate(self, env: Any, num_episodes: int = 5) -> List[Dict[str, Any]]:
        """
        Evaluate the trained agent(s).

        Returns:
            List of episode results (rewards, steps, etc.)
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the trained model."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load a trained model."""
        pass


class SB3Trainer(BaseTrainer):
    """Stable Baselines 3 trainer implementation."""

    def __init__(self, agents: List[Any], config: Dict[str, Any], results_dir: str):
        self.agents = agents
        self.config = config
        self.results_dir = results_dir
        self.model = None

        # Extract SB3-specific params
        self.learning_rate = config.get("learning_rate", 3e-4)
        self.n_steps = config.get("n_steps", 2048)
        self.batch_size = config.get("batch_size", 64)
        self.gamma = config.get("gamma", 0.99)
        self.ent_coef = config.get("ent_coef", 0.01)

    def train(self, env: Any, total_timesteps: int, callbacks: Optional[Any] = None) -> None:
        """Train using Stable Baselines 3 PPO."""
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback

        logger.info("Initializing SB3 PPO trainer. Timesteps: {}", total_timesteps)

        # Create checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=f"{self.results_dir}/models",
            name_prefix="agent_ppo"
        )

        # Create model
        self.model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            gamma=self.gamma,
            ent_coef=self.ent_coef,
            tensorboard_log=f"{self.results_dir}/logs"
        )

        # Train
        self.model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

        logger.info("Training complete")

    def evaluate(self, env: Any, num_episodes: int = 5) -> List[Dict[str, Any]]:
        """Evaluate the trained model."""
        if self.model is None:
            logger.warning("No trained model to evaluate")
            return []

        results = []

        for ep in range(num_episodes):
            obs, info = env.reset()
            done = False
            total_reward = 0
            steps = 0

            while not done:
                actions = {}
                for agent_name, agent_obs in obs.items():
                    action, _ = self.model.predict(agent_obs, deterministic=True)
                    actions[agent_name] = action

                obs, rewards, terminations, truncations, infos = env.step(actions)

                total_reward += sum(rewards.values())
                steps += 1

                if not obs or all(terminations.values()) or all(truncations.values()):
                    done = True

            results.append({
                "episode": ep + 1,
                "total_reward": total_reward,
                "steps": steps,
            })

            logger.info("Episode {}: reward={}, steps={}", ep + 1, total_reward, steps)

        return results

    def save(self, path: str) -> None:
        """Save the trained model."""
        if self.model:
            self.model.save(path)
            logger.info("Model saved to: {}", path)

    def load(self, path: str) -> None:
        """Load a trained model."""
        from stable_baselines3 import PPO

        self.model = PPO.load(path)
        logger.info("Model loaded from: {}", path)


class RLlibTrainer(BaseTrainer):
    """Ray RLlib trainer stub."""

    def __init__(self, agents: List[Any], config: Dict[str, Any], results_dir: str):
        self.agents = agents
        self.config = config
        self.results_dir = results_dir
        self.trainer = None
        logger.warning("RLlib trainer not fully implemented")

    def train(self, env: Any, total_timesteps: int, callbacks: Optional[Any] = None) -> None:
        raise NotImplementedError("RLlib trainer is a stub")

    def evaluate(self, env: Any, num_episodes: int = 5) -> List[Dict[str, Any]]:
        raise NotImplementedError("RLlib trainer is a stub")

    def save(self, path: str) -> None:
        raise NotImplementedError("RLlib trainer is a stub")

    def load(self, path: str) -> None:
        raise NotImplementedError("RLlib trainer is a stub")


def create_trainer(
    trainer_type: str,
    agents: List[Any],
    config: Dict[str, Any],
    results_dir: str
) -> BaseTrainer:
    """
    Factory function to create a trainer.

    Args:
        trainer_type: "sb3" or "rllib"
        agents: List of agent instances
        config: Configuration dict
        results_dir: Directory for results

    Returns:
        A BaseTrainer instance
    """
    if trainer_type == "sb3":
        return SB3Trainer(agents, config, results_dir)
    elif trainer_type == "rllib":
        return RLlibTrainer(agents, config, results_dir)
    else:
        raise ValueError(f"Unknown trainer type: {trainer_type}")