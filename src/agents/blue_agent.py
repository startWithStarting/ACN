from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from collections import defaultdict
import gymnasium.spaces as spaces

from .base_agent import BaseAgent, AgentType
from .registry import register_agent
from ..utils.regressor import VectorAutoRegressor
from ..utils.geometry import calculate_distance, is_within_detection_radius
from ..utils.logger import get_logger

# Import blue agent strategies
from .blue_strategies.static_strategy import static_blue_strategy
from .blue_strategies.pursuit_strategy import pursuit_blue_strategy

logger = get_logger("acn.agents.blue")

@register_agent("blue")
class BlueAgent(BaseAgent):
    """
    Represents a Blue agent in the simulation.
    """
    def __init__(self, name: str, communication_bandwidth: int, processing_capability: int, 
                 detection_radius: float = 20.0, strategy_type: str = "pursuit", prediction_timeout: int = 5,
                 observation_window_size: int = 5, prediction_interval: int = 1, grid_size: Tuple[float, float] = (100.0, 100.0),
                 debug_mode: bool = False):
        """
        Initializes a Blue agent.

        Args:
            name (str): A unique identifier for the agent.
            communication_bandwidth (int): Communication capacity.
            processing_capability (int): Computational power - also determines number of past time steps for prediction.
            detection_radius (float): Radius within which the agent can detect Red agents. Defaults to 20.0.
            strategy_type (str): Movement strategy to use. Options are:
                                  "static" - Remain stationary
                                  "pursuit" - Move toward predicted red agent positions
            prediction_timeout (int): Steps before prediction is considered stale
            observation_window_size (int): History window
            prediction_interval (int): How often to predict
            grid_size (Tuple[float, float]): Dimensions of the grid (width, height)
            debug_mode (bool): Whether to print debug logs
        """
        super().__init__(
            name=name,
            agent_type=AgentType.BLUE,
            communication_bandwidth=communication_bandwidth,
            processing_capability=processing_capability
        )
        self.grid_size = grid_size
        self.debug_mode = debug_mode
        # Dictionary to store observed Red agent paths
        # Key: Red agent name, Value: List of (position, timestamp) tuples
        self.observed_red_agents = defaultdict(list)
        self.detection_radius = detection_radius  # Detection radius in units

        # Dictionary to store prediction models for each red agent
        # Key: Red agent name, Value: LinearRegression model
        self.prediction_models = {}
        
        # Define the action space for Blue Agent (matching RedAgent for consistency)
        self.action_space = spaces.Dict({
            'direction': spaces.Box(low=np.array([-1.0, -1.0], dtype=np.float32),
                                    high=np.array([1.0, 1.0], dtype=np.float32),
                                    shape=(2,), dtype=np.float32),
            'speed': spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32)
        })

        # Dictionary to store predicted positions for each red agent
        # Key: Red agent name, Value: List of predicted future positions [(x1, y1), (x2, y2), ...]
        self.predicted_positions = defaultdict(list)

        # Dictionaries to store history of actual and predicted positions with timestamps
        # Key: Red agent name, Value: List of (position, timestamp) tuples
        self.actual_position_history = defaultdict(list)
        # Key: Red agent name, Value: List of (predicted_position, timestamp) tuples
        self.prediction_history = defaultdict(list)

        # Set the movement strategy type
        self.strategy_type = strategy_type

        # Optimization: Prediction interval
        self.prediction_interval = prediction_interval
        self.prediction_timeout = prediction_timeout
        self.observation_window_size = observation_window_size
        self.steps = 0

        # History cap to prevent unbounded memory growth
        self.max_history_length = 1000

        # Movement memory/momentum system
        self.memory_constant = 0.1  # How much to retain from previous path (0-1)
        self.current_target_position = None  # Current target the agent is moving toward

    def is_within_detection_radius(self, red_agent_pos: Tuple[float, float]) -> bool:
        """Check if a Red agent is within the detection radius."""
        if self.x is None or self.y is None:
            return False
        return is_within_detection_radius((self.x, self.y), red_agent_pos, self.detection_radius)

    def record_red_agent_movement(self, red_agent_name: str, position: Tuple[float, float], timestamp: float):
        """
        Record the movement of a Red agent if it's within detection radius.
        Only record if the position is not (0,0) and is a real detection.
        """
        if position is not None and not (position[0] == 0 and position[1] == 0):
            self.observed_red_agents[red_agent_name].append((position, timestamp))
            # Maintain sliding window
            if len(self.observed_red_agents[red_agent_name]) > self.observation_window_size:
                self.observed_red_agents[red_agent_name].pop(0)

    def get_observed_paths(self) -> Dict[str, List[Tuple[Tuple[float, float], float]]]:
        """
        Get all recorded paths of Red agents.

        Returns:
            Dict[str, List[Tuple[Tuple[float, float], float]]]: Dictionary of Red agent paths
        """
        return dict(self.observed_red_agents)
        
    def fit_prediction_model(self, red_agent_name: str) -> bool:
        """
        Fits a linear autoregressive model for predicting the future position of a red agent
        based on its observed path. The model considers the past n time steps where n is the
        agent's processing_capability.
        
        Args:
            red_agent_name (str): Name of the Red agent to fit model for
            
        Returns:
            bool: True if the model was successfully fit, False otherwise
        """
        # Check if we have enough observations for this red agent (need at least 2 points)
        if red_agent_name not in self.observed_red_agents or \
           len(self.observed_red_agents[red_agent_name]) < 2:
            return False
        positions = [pos for pos, _ in self.observed_red_agents[red_agent_name] if not (pos[0] == 0 and pos[1] == 0)]
        positions_array = np.array(positions)  # (num_observations, 2)
        
        # Determine number of lags (n)
        # We need to ensure we have enough samples to fit the model.
        # samples = len(positions) - n
        # We want samples to be at least 3 if possible for stability.
        min_samples = 3
        max_n = len(positions) - min_samples
        
        # n must be at least 1, and max_n might be negative if len(positions) is small
        n = max(1, min(self.processing_capability, max_n))
        
        # If we still don't have enough data for even n=1 (e.g. len=2), n will be 1
        # and we'll get 1 sample. That's the best we can do for very short history.
        if n < 1:
            return False 
            
        if len(positions_array) < n + 1:
            return False
            
        # Build windowed X and y
        X = []
        y = []
        for i in range(len(positions_array) - n):
            X.append(positions_array[i:i+n].flatten())  # shape (n*2,)
            y.append(positions_array[i+n])              # shape (2,)
        X = np.array(X)
        y = np.array(y)
        # Logging handled by logger.warning in the except block
        try:
            model = VectorAutoRegressor(n_lags=n)
            model.fit(X, y)
            self.prediction_models[red_agent_name] = model
            return True
        except Exception as e:
            logger.warning("Error fitting prediction model for {}: {}", red_agent_name, e)
            return False
            
    def predict_future_position(self, red_agent_name: str, steps_ahead: int = 1, current_time: float = None) -> Optional[Tuple[float, float]]:
        """
        Predicts the future position of a red agent based on its observed path.
        If the last observation is more than 3 time steps before the current time, do not predict.
        Args:
            red_agent_name (str): Name of the Red agent to predict position for
            steps_ahead (int): Number of steps into the future to predict. Defaults to 1.
            current_time (float, optional): The current simulation time. If None, disables the time check.
        Returns:
            Optional[Tuple[float, float]]: Predicted (x, y) position or None if prediction couldn't be made
        """
        if red_agent_name not in self.predicted_positions:
            self.predicted_positions[red_agent_name] = []
        if len(self.predicted_positions[red_agent_name]) >= steps_ahead:
            return self.predicted_positions[red_agent_name][steps_ahead-1]
        observations = [obs for obs in self.observed_red_agents.get(red_agent_name, []) if not (obs[0][0] == 0 and obs[0][1] == 0)]
        if len(observations) < 2:
            if observations:
                last_pos, _ = observations[-1]
                return last_pos
            return None
        # Check time gap: if current_time is provided and last observation is too old, do not predict
        if current_time is not None:
            last_obs_time = observations[-1][1]
            if current_time - last_obs_time > self.prediction_timeout:
                return None
        if red_agent_name not in self.prediction_models:
            if not self.fit_prediction_model(red_agent_name):
                last_pos, _ = observations[-1]
                return last_pos
        
        # Calculate n exactly as in fit_prediction_model
        min_samples = 3
        max_n = len(observations) - min_samples
        n = max(1, min(self.processing_capability, max_n))
        
        if n < 1:
            return None
        recent_positions = np.array([pos for pos, _ in observations])
        
        # FALLBACK: If we have very few points (e.g. only 2 points -> 1 sample),
        # LinearRegression fits a constant (mean) instead of a slope, causing "collapsed" predictions.
        # So for len < 3, we use simple velocity projection (Last - Prev).
        if len(recent_positions) < 3 and len(recent_positions) >= 2:
            # Calculate last velocity
            velocity = recent_positions[-1] - recent_positions[-2]
            
            # Project forward
            preds = []
            curr = recent_positions[-1].copy()
            for _ in range(steps_ahead):
                curr += velocity
                # Clamp
                curr[0] = max(0.0, min(curr[0], self.grid_size[0]))
                curr[1] = max(0.0, min(curr[1], self.grid_size[1]))
                # preds.append(curr.copy()) # We want the FINAL point? Or list?
                # The function returns a single point (or list?).
                # Wait, fit_prediction_model implies VectorAutoRegressor which predicts LIST.
                # BUT this function `predict_future_position` returns `last_pos` (single tuple) 
                # OR runs a loop `for _ in range(1, steps_ahead)`.
                # BUT the `VectorAutoRegressor` usually returns a sequence.
                # Let's check what `predict_future_position` usually returns.
                # It returns `predicted_position` which is `x, y`. Sample: `return last_pos`.
                # So it returns SINGLE coordinate.
                pass
            return tuple(curr)

        if recent_positions.shape[0] < n:
            return None
        # Use the last n positions for prediction
        features = recent_positions[-n:]  # shape (n, 2)
        model = self.prediction_models[red_agent_name]
        try:
            predicted_position = model.predict(features)
            for _ in range(1, steps_ahead):
                features = np.vstack([features[1:], predicted_position])
                predicted_position = model.predict(features)
            
            # SANITY CHECK: Clean up prediction
            if np.any(np.isnan(predicted_position)) or np.any(np.isinf(predicted_position)):
                last_pos, _ = observations[-1]
                return last_pos
                
            # Clamp to grid size
            x, y = predicted_position
            x = max(0.0, min(x, self.grid_size[0]))
            y = max(0.0, min(y, self.grid_size[1]))
            
            return (x, y)
        except Exception as e:
            # Logging handled by logger.warning in the except block
            last_pos, _ = observations[-1]
            return last_pos
    def choose_action(self, observation=None):
        """
        Chooses an action for the Blue agent.
        For now, Blue agents will stay still (they don't move in this version).
        Also attempts to fit prediction models for observed red agents and make predictions.

        Args:
            observation (Dict[str, Any], optional): The agent's observation.
                                                  Defaults to None for backward compatibility.

        Returns:
            Dict[str, Any]: A dictionary containing the 'direction' and 'speed' for consistency
                           with RedAgent's action space.
        """
        # Process observation if available
        if observation is not None:
            red_agents = observation.get('red_agents', {})
            timestamp = observation.get('timestamp', 0.0)
            
            # Record movements of Red agents within detection radius
            for red_name, red_data in red_agents.items():
                if 'position' in red_data:
                    position = red_data['position']
                    # Only record if the red agent is within detection radius and not (0,0)
                    if self.is_within_detection_radius(position) and not (position[0] == 0 and position[1] == 0):
                        self.record_red_agent_movement(red_name, position, timestamp)
                        self.actual_position_history[red_name].append((position, timestamp))
                        # Trim history if it exceeds the cap
                        if len(self.actual_position_history[red_name]) > self.max_history_length:
                            self.actual_position_history[red_name] = self.actual_position_history[red_name][-self.max_history_length:]
            
            # Try to fit/update prediction models for all observed red agents
        # Optimization: Only update predictions every prediction_interval steps
        self.steps += 1
        if self.steps % self.prediction_interval == 0:
            # Active Pruning: Remove stale agents that haven't been seen for a while
            # This prevents "ghost" attraction to where agents USED to be.
            stale_agents = []
            for red_name, observations in self.observed_red_agents.items():
                if not observations:
                    stale_agents.append(red_name)
                    continue
                last_obs_time = observations[-1][1]
                if timestamp - last_obs_time > self.prediction_timeout:
                    stale_agents.append(red_name)
            
            for red_name in stale_agents:
                if red_name in self.observed_red_agents:
                    del self.observed_red_agents[red_name]
                if red_name in self.predicted_positions:
                    del self.predicted_positions[red_name]
                if red_name in self.prediction_models:
                    del self.prediction_models[red_name]
                # maintain prediction_history for analysis
                # if red_name in self.prediction_history:
                #    del self.prediction_history[red_name]

            # Calculate new predicted target position
            new_predictions = {}
            
            for red_name in self.observed_red_agents.keys():
                # Only try to fit the model if we have at least 2 observations
                model_fitted = False
                if len(self.observed_red_agents[red_name]) >= 2:
                    # Fit or update the prediction model
                    model_fitted = self.fit_prediction_model(red_name)
                
                # If model was successfully fitted, make predictions for future positions
                if model_fitted or red_name in self.prediction_models:
                    # CRITICAL FIX: Clear old cached predictions before generating new ones
                    if red_name in self.predicted_positions:
                        del self.predicted_positions[red_name]
                        
                    # Make prediction for the next step (steps_ahead=1)
                    next_step_prediction = self.predict_future_position(red_name, 1, current_time=timestamp)
                    if next_step_prediction is not None and not (next_step_prediction[0] == 0 and next_step_prediction[1] == 0):
                        self.prediction_history[red_name].append((next_step_prediction, timestamp))
                        if len(self.prediction_history[red_name]) > self.max_history_length:
                            self.prediction_history[red_name] = self.prediction_history[red_name][-self.max_history_length:]
                    else:
                        if self.debug_mode:
                            logger.debug("{} prediction invalid or None for {}", self.name, red_name)
                        pass
                        
                    # Store predictions for the next 5 steps (for visualization)
                    future_positions = []
                    for steps_ahead in range(1, 6):
                        future_pos = self.predict_future_position(red_name, steps_ahead, current_time=timestamp)
                        if future_pos is not None:
                            future_positions.append(future_pos)
                    
                    # Store the new predictions
                    if future_positions:
                        new_predictions[red_name] = future_positions
            
            # Update predicted_positions with new predictions
            self.predicted_positions = new_predictions
            
            # Calculate new target position from predictions
            if self.predicted_positions:
                all_predictions = []
                for predictions in self.predicted_positions.values():
                    all_predictions.extend(predictions)
                new_target = np.mean(all_predictions, axis=0)
                
                # Blend with current target using memory constant
                # new_target = current_target * memory + new_observation * (1 - memory)
                if self.current_target_position is not None:
                    self.current_target_position = (
                        self.current_target_position * self.memory_constant +
                        new_target * (1 - self.memory_constant)
                    )
                else:
                    # First time, just use the new target
                    self.current_target_position = new_target

        # Collect detected predictions for red agents
        detected_predictions = []
        for red_name, predictions in self.predicted_positions.items():
            if predictions and self.is_within_detection_radius(self.observed_red_agents[red_name][-1][0]):
                detected_predictions.extend(predictions)
        
        # Select strategy based on strategy_type
        if self.strategy_type == "static":
            return static_blue_strategy()
        else:  # Default to pursuit strategy
            current_pos = np.array([self.x, self.y], dtype=np.float32)
            # Use momentum-based target position for smooth movement
            target_pos = self.current_target_position if self.current_target_position is not None else None
            return pursuit_blue_strategy(current_pos, target_pos)

    # You can override methods from BaseAgent if Blue agents behave differently
    # For example:
    # def receive_message(self, sender_name: str, message_content: dict):
    #     # Blue agent specific message processing
    #     super().receive_message(sender_name, message_content) # Optionally call parent
    #     # ... additional blue logic

    def __str__(self):
        position_str = f", Pos: ({self.x:.2f}, {self.y:.2f})" if self.x is not None and self.y is not None else ""
        return f"BlueAgent(Name: {self.name}, CommBW: {self.communication_bandwidth}, ProcCap: {self.processing_capability}{position_str})"

    def __repr__(self):
        position_repr = f", x={self.x}, y={self.y}" if self.x is not None and self.y is not None else ""
        return f"BlueAgent(name='{self.name}', communication_bandwidth={self.communication_bandwidth}, processing_capability={self.processing_capability}{position_repr})"
