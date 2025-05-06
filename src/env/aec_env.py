import functools
import random

import gymnasium
import numpy as np
from gymnasium.spaces import Discrete, Box

from pettingzoo.utils.agent_selector import AgentSelector
from pettingzoo.utils import wrappers # Keep wrappers import separate
from pettingzoo.utils.env import AECEnv

# Import agent types (adjust path if needed)
from src.agents.base_agent import BaseAgent 

def env(**kwargs):
    """
    The env function wraps the environment in wrappers PZ provides.
    """
    env = AECGameEnv(**kwargs)
    # Example wrapper: env = wrappers.AssertOutOfBoundsWrapper(env)
    # Example wrapper: env = wrappers.OrderEnforcingWrapper(env)
    return env

class AECGameEnv(AECEnv):
    """
    A basic PettingZoo AEC environment for the communicating agents game.

    In this simple version:
    - Observation: A dummy value (e.g., agent's index).
    - Action: A dummy discrete action (e.g., 0 or 1).
    - Reward: Simple reward (e.g., +1 for action 1, -1 for action 0).
    - Termination/Truncation: After a fixed number of steps per agent.
    """
    metadata = {
        "render_modes": ["human"],
        "name": "communicating_agents_v0",
        "is_parallelizable": False, # Usually True if step() doesn't depend on agent order
        "render_fps": 10,
    }

    def __init__(self, agents: list[BaseAgent], render_mode=None, max_cycles=100, **env_config):
        """
        Args:
            agents (list[BaseAgent]): List of agent objects participating.
            render_mode (str, optional): Rendering mode. Defaults to None.
            max_cycles (int): Max number of cycles (steps per agent) before truncation.
            env_config (dict): Additional environment configuration parameters.
        """
        super().__init__() # Initialize AECEnv base class

        if not agents:
            raise ValueError("Environment must be initialized with at least one agent.")

        self.max_cycles = max_cycles
        self.env_config = env_config
        print(f"Initializing AECGameEnv with max_cycles={max_cycles}, config={env_config}")

        # Store agent objects and create mapping from name to object
        self.agent_objects = {agent.name: agent for agent in agents}
        self.possible_agents = [agent.name for agent in agents] # List of agent names
        self.agent_name_mapping = {i: name for i, name in enumerate(self.possible_agents)}

        # PettingZoo API requirements
        self.agents = self.possible_agents[:] # Current list of active agents (names)
        self._agent_selector = AgentSelector(self.agents) # Cycles through agent names

        # Define spaces (MUST be defined after agents are known)
        # These are placeholders - replace with actual game spaces
        self._action_spaces = {name: Discrete(2) for name in self.possible_agents} # Example: 2 actions
        # Example observation: agent index and current cycle count
        self._observation_spaces = {
            name: Box(low=0, high=max(len(agents), max_cycles), shape=(2,), dtype=np.float32)
            for i, name in enumerate(self.possible_agents)
        }

        self.render_mode = render_mode

        # Internal state
        self.steps = 0
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.rewards = {agent: 0 for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0 for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}

        # Start the agent selection cycle
        self.agent_selection = self._agent_selector.reset() # Get the first agent

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]

    def observe(self, agent):
        """
        Return the observation for the given agent.
        """
        # Placeholder observation: agent's index and current step count for this agent
        agent_idx = self.possible_agents.index(agent)
        agent_steps = self.steps // len(self.possible_agents) # Approximate steps per agent
        return np.array([agent_idx, agent_steps], dtype=np.float32)

    def reset(self, seed=None, options=None):
        """
        Reset the environment to a starting state.
        """
        if seed is not None:
            # Seed the random number generator if needed (e.g., for procedural generation)
            random.seed(seed)
            np.random.seed(seed)
            # Note: PettingZoo recommends seeding action/observation spaces separately if needed

        self.agents = self.possible_agents[:] # Reset active agents
        self._agent_selector.reinit(self.agents) # Reinitialize selector
        self.agent_selection = self._agent_selector.reset() # Get first agent

        # Reset internal state
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.steps = 0

        # Reset agent object states if necessary
        for agent_obj in self.agent_objects.values():
             agent_obj.is_active = True # Example reset

        # Return initial observations and infos (PettingZoo standard)
        # observations = {agent: self.observe(agent) for agent in self.agents}
        # infos = {agent: {} for agent in self.agents}
        # According to AEC API, reset doesn't return obs/info directly.
        # The first observation is retrieved via last() after reset().

    def step(self, action):
        """
        Apply the action for the current agent_selection.
        Updates rewards, terminations, truncations, and selects the next agent.
        """
        agent_name = self.agent_selection

        if self.terminations[agent_name] or self.truncations[agent_name]:
            # If agent is done, handle potential cleanup and select next agent
            # In AEC, stepping a done agent usually consumes the action (often None)
            # and moves to the next agent without state change.
            self._was_done_step(action) # PZ utility
            # Select next agent if current agent is done
            if self._agent_selector.is_last():
                 # If it was the last agent in the cycle and it's done,
                 # we might need special handling or rely on the main loop to check all done.
                 pass # agent_iter in main.py handles stopping
            else:
                 # This ensures we move to the next agent even if the current one was done
                 # self.agent_selection = self._agent_selector.next() # _was_done_step might handle this
                 pass # Let the loop in main handle agent cycling via agent_iter

            return # Agent is done, no state update needed for it


        # --- Action Processing ---
        # Get the agent object
        current_agent_obj = self.agent_objects[agent_name]

        # Process the action (placeholder logic)
        # print(f"Step {self.steps}: Agent {agent_name} takes action {action}")

        # --- Update State ---
        # Simple reward based on action
        reward = 0
        if action == 1:
            reward = 1
        elif action == 0:
            reward = -1
        
        self.rewards[agent_name] = reward # Store reward for this step
        self._cumulative_rewards[agent_name] += reward # Update cumulative reward

        # --- Termination/Truncation Logic ---
        self.steps += 1
        # Truncate based on max_cycles (total steps / num_agents)
        agent_cycles = self.steps / len(self.possible_agents)
        if agent_cycles >= self.max_cycles:
            self.truncations[agent_name] = True
            # print(f"Agent {agent_name} truncated at step {self.steps} (cycle {agent_cycles})")

        # Termination condition (example: agent decides to terminate based on state)
        # self.terminations[agent_name] = some_condition

        # Update info dictionary for the agent
        self.infos[agent_name] = {"cumulative_reward": self._cumulative_rewards[agent_name]}

        # --- Communication Phase (Placeholder) ---
        # This is where agents might send/receive messages based on the action or state
        # 1. Agent decides to send messages (based on action or internal state)
        #    - Uses its communication model to select partners/content
        # 2. Environment facilitates message delivery
        #    - Stores messages in mailboxes?
        # 3. Next time an agent acts, it checks its mailbox
        #    - Uses its communication model to process messages
        # This needs careful integration with the AEC step flow. Often done before/after action.

        # --- Select Next Agent ---
        # Select the next agent to act.
        # If the current agent just became terminated/truncated, it should still get one last `last()` call
        # before the next agent is selected by the agent_iter loop.
        # The AEC standard is that `step` prepares the state for the *next* agent's `last()` call.
        if self._agent_selector.is_last():
            # End of a cycle (all agents have taken a step)
            # Potentially update global state, check for game end conditions affecting all agents
            # If the whole environment should terminate/truncate:
            # for ag in self.agents:
            #     self.truncations[ag] = True # Or terminations
            pass
        
        # Crucially, agent_selection is typically updated by the agent_iter loop in main.py
        # based on the selector state, rather than directly in step().
        # However, we need to ensure the internal selector is ready for the next agent.
        # PZ internal logic often handles this implicitly. Let's rely on agent_iter.
        # self.agent_selection = self._agent_selector.next() # Let agent_iter handle this

        # PZ utility: Accumulate rewards correctly for the next call to last()
        self._accumulate_rewards()

        # PZ utility: Handle render mode if active
        if self.render_mode == "human":
            self.render()


    def render(self):
        """
        Renders the environment.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if self.render_mode == "human":
            # Implement visualization here (e.g., using Pygame, Matplotlib)
            print("\n--- Rendering Frame ---")
            print(f"Step: {self.steps}")
            print(f"Current Agent: {self.agent_selection}")
            print("Agent States:")
            for name, agent_obj in self.agent_objects.items():
                state = "Done" if (self.terminations[name] or self.truncations[name]) else "Active"
                print(f"  - {name}: Reward={self._cumulative_rewards[name]:.2f}, State={state}")
            print("-----------------------\n")
        else:
            raise NotImplementedError(f"Render mode '{self.render_mode}' not supported.")


    def close(self):
        """
        Close the environment, release resources.
        """
        print("Closing environment.")
        # Add cleanup code here if needed (e.g., close pygame window)
        pass
