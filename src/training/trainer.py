import os
import numpy as np
import gymnasium
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
import supersuit as ss
from pettingzoo.utils.env import ParallelEnv

# Import the parallel environment
from src.env.parallel_env import ParallelGameEnv

class RedTeamWrapper(ParallelEnv):
    """
    Wraps the ParallelGameEnv to expose only Red agents to the learner.
    Blue agents are treated as part of the environment (stepped internally).
    """
    def __init__(self, env):
        self.env = env
        self.metadata = env.metadata
        self.render_mode = env.render_mode
        self.possible_agents = [a for a in env.possible_agents if "red" in a]
        self.blue_agents = [a for a in env.possible_agents if "blue" in a]
        self.agent_name_mapping = dict(zip(self.possible_agents, range(len(self.possible_agents))))
        
    @property
    def observation_spaces(self):
        return {agent: self.env.observation_space(agent) for agent in self.possible_agents}

    @property
    def action_spaces(self):
        return {agent: self.action_space(agent) for agent in self.possible_agents}

    def observation_space(self, agent):
        return self.env.observation_space(agent)

    def action_space(self, agent):
        # Flattened action space for PPO: [dir_x, dir_y, speed]
        # Direction elements are [-1, 1], Speed is [0, 10]
        return gymnasium.spaces.Box(
            low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 10.0], dtype=np.float32),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        self.agents = [a for a in self.env.agents if "red" in a]
        
        # Filter observations for red agents only
        red_obs = {k: v for k, v in obs.items() if k in self.agents}
        red_infos = {k: v for k, v in infos.items() if k in self.agents}
        return red_obs, red_infos

    def step(self, actions):
        # 1. Get actions for Blue agents (heuristic/environment managed)
        curr_obs = {agent: self.env._get_observation(agent, self.env.steps) for agent in self.blue_agents}
        blue_actions = {}
        for blue_name in self.blue_agents:
            if blue_name in self.env.agents: # Only if active
                blue_obj = self.env.agent_objects[blue_name]
                # Blue agents usually use 'choose_action' which returns a dict
                # We pass the observation if needed
                action = blue_obj.choose_action(curr_obs.get(blue_name))
                blue_actions[blue_name] = action
        
        # 1.1 Unflatten Red actions (from PPO Learner)
        red_actions = {}
        for agent, act in actions.items():
            if agent in self.possible_agents:
                 # act is [dx, dy, speed]
                 # Ensure types match what RedAgent expects (numpy arrays)
                 red_actions[agent] = {
                     'direction': np.array(act[:2], dtype=np.float32),
                     'speed': np.array([act[2]], dtype=np.float32)
                 }
            else:
                 red_actions[agent] = act
        
        # 2. Combine with Red actions
        full_actions = {**red_actions, **blue_actions}
        
        # 3. Step internal environment
        obs, rewards, terminations, truncations, infos = self.env.step(full_actions)
        
        # 4. Filter for Red agents
        self.agents = [a for a in self.env.agents if "red" in a]
        
        red_obs = {k: v for k, v in obs.items() if k in self.agents}
        red_rewards = {k: v for k, v in rewards.items() if k in self.agents}
        red_terminations = {k: v for k, v in terminations.items() if k in self.agents}
        red_truncations = {k: v for k, v in truncations.items() if k in self.agents}
        red_infos = {k: v for k, v in infos.items() if k in self.agents}
        
        # PettingZoo API requires returning dicts keyed by agent
        return red_obs, red_rewards, red_terminations, red_truncations, red_infos

    def render(self):
        return self.env.render()
    
    def close(self):
        self.env.close()

class Trainer:
    """
    Manages the MADRL training process using Stable Baselines 3.
    """
    def __init__(self, agents: list, config: dict, results_dir: str):
        """
        Initializes the Trainer.
        """
        self.agents = agents
        self.config = config
        self.results_dir = results_dir
        
        # Extract training params
        self.total_timesteps = config.get("total_timesteps", 100000)
        self.learning_rate = config.get("learning_rate", 3e-4)
        self.n_steps = config.get("n_steps", 2048)
        self.batch_size = config.get("batch_size", 64)
        self.gamma = config.get("gamma", 0.99)
        self.ent_coef = config.get("ent_coef", 0.01)
        
        # Env config
        self.env_config = config.get("environment", {})
        self.env_config["experiment_results_dir"] = results_dir
        
        print(f"Initializing PPO Trainer. Timesteps: {self.total_timesteps}")

    def train(self):
        """
        Runs the training using Stable Baselines 3 PPO.
        """
        # 1. Create the base Parallel Environment
        base_env = ParallelGameEnv(agents=self.agents, **self.env_config)
        
        # 2. Wrap it to expose only Red agents
        red_env = RedTeamWrapper(base_env)
        
        # 3. Vectorize with SuperSuit for SB3 compatibility
        # concat_vec_envs_v1 creates a single vectorized environment where agents are concatenated in the batch dimension
        # This allows parameter sharing (one policy for all red agents)
        env = ss.pettingzoo_env_to_vec_env_v1(red_env)
        env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class='stable_baselines3') # 1 vec env, parallelized
        env = VecMonitor(env)
        
        # 4. Initialize PPO
        # using 'MultiInputPolicy' because our observation is a Dict space
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            gamma=self.gamma,
            ent_coef=self.ent_coef,
            tensorboard_log=os.path.join(self.results_dir, "logs")
        )
        
        # 5. Train
        print("\nStarting PPO training...")
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=os.path.join(self.results_dir, "models"),
            name_prefix="red_agent_ppo"
        )
        
        model.learn(total_timesteps=self.total_timesteps, callback=checkpoint_callback)
        
        # 6. Save final model
        final_path = os.path.join(self.results_dir, "models", "final_model")
        model.save(final_path)
        print(f"Training finished. Model saved to {final_path}")
        
        # Close env
        env.close()
        
    def load_and_evaluate(self, model_path, num_episodes=5):
        """
        Loads a trained model and evaluates it.
        """
        print(f"Loading model from {model_path} for evaluation...")
        model = PPO.load(model_path)
        
        base_env = ParallelGameEnv(agents=self.agents, render_mode="human_pygame", **self.env_config)
        red_env = RedTeamWrapper(base_env)
        
        # We process step-by-step for evaluation to render properly
        for ep in range(num_episodes):
            obs, info = red_env.reset()
            done = False
            total_reward = 0
            
            while not done:
                # We need to construct actions for the valid agents
                # SB3 predict expects observation. 
                # Since we wrapped with supersuit vectorization during training, the model expects vectorized inputs?
                # Actually PPO.predict handles unvectorized input if we pass a single observation?
                # But here we have multiple agents.
                
                # Simple Manual Loop for Multi-Agent Inference with Shared Policy:
                actions = {}
                for agent_name, agent_obs in obs.items():
                    # Check if model expects vectorized input. PPO.predict usually handles it.
                    # But if trained on VecEnv, it might expect batch dim.
                    # Let's simple-wrap the obs or trust SB3.
                    action, _states = model.predict(agent_obs, deterministic=True)
                    actions[agent_name] = action
                
                obs, rewards, terminations, truncations, infos = red_env.step(actions)
                
                # Check if all done
                if not obs: # no agents left
                    done = True
                
                total_reward += sum(rewards.values())
                
                # termination/truncation check
                if all(terminations.values()) or all(truncations.values()):
                    done = True
                    
            print(f"Episode {ep+1} Reward: {total_reward}")
        
        red_env.close()

