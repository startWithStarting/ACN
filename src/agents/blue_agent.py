from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LinearRegression

from .base_agent import BaseAgent, AgentType

class BlueAgent(BaseAgent):
    """
    Represents a Blue agent in the simulation.
    """
    def __init__(self, name: str, communication_bandwidth: int, processing_capability: int, detection_radius: float = 20.0):
        """
        Initializes a Blue agent.

        Args:
            name (str): A unique identifier for the agent.
            communication_bandwidth (int): Communication capacity.
            processing_capability (int): Computational power - also determines number of past time steps for prediction.
            detection_radius (float): Radius within which the agent can detect Red agents. Defaults to 20.0.
        """
        super().__init__(
            name=name,
            agent_type=AgentType.BLUE,
            communication_bandwidth=communication_bandwidth,
            processing_capability=processing_capability
        )
        # Dictionary to store observed Red agent paths
        # Key: Red agent name, Value: List of (position, timestamp) tuples
        self.observed_red_agents = defaultdict(list)
        self.detection_radius = detection_radius  # Detection radius in units

        # Dictionary to store prediction models for each red agent
        # Key: Red agent name, Value: LinearRegression model
        self.prediction_models = {}
        
        # Dictionary to store predicted positions for each red agent
        # Key: Red agent name, Value: List of predicted future positions [(x1, y1), (x2, y2), ...]
        self.predicted_positions = defaultdict(list)

    def calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """
        Calculate Euclidean distance between two positions.

        Args:
            pos1 (Tuple[float, float]): First position (x, y)
            pos2 (Tuple[float, float]): Second position (x, y)

        Returns:
            float: Euclidean distance between the positions
        """
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def is_within_detection_radius(self, red_agent_pos: Tuple[float, float]) -> bool:
        """
        Check if a Red agent is within the detection radius.

        Args:
            red_agent_pos (Tuple[float, float]): Position of the Red agent

        Returns:
            bool: True if the Red agent is within detection radius, False otherwise
        """
        if self.x is None or self.y is None:
            return False
        return self.calculate_distance((self.x, self.y), red_agent_pos) <= self.detection_radius

    def record_red_agent_movement(self, red_agent_name: str, position: Tuple[float, float], timestamp: float):
        """
        Record the movement of a Red agent if it's within detection radius.

        Args:
            red_agent_name (str): Name of the Red agent
            position (Tuple[float, float]): Current position of the Red agent
            timestamp (float): Current simulation timestamp
        """
        if self.is_within_detection_radius(position):
            self.observed_red_agents[red_agent_name].append((position, timestamp))

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
            
        # Extract position data from observations
        positions = [pos for pos, _ in self.observed_red_agents[red_agent_name]]
        positions_array = np.array(positions)  # Shape: (num_observations, 2) for [x, y] coordinates
        
        # Determine how many past steps to use based on processing_capability,
        # but limited by available data minus one (for target)
        n = min(self.processing_capability, len(positions) - 1)  
        
        # Cannot predict with less than 2 points
        if n < 1:
            return False
            
        # Prepare training data
        X = []  # Features: sequences of n past positions
        y = []  # Targets: next position
        
        # For each position except the first n, use the previous n positions to predict it
        for i in range(n, len(positions)):
            # Past n positions flattened to a feature vector
            past_positions = positions_array[i-n:i].flatten()  # [x1, y1, x2, y2, ..., xn, yn]
            X.append(past_positions)
            y.append(positions_array[i])  # Target: current [x, y] position
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Train model to predict [x, y] coordinates directly
        model = LinearRegression()
        
        try:
            model.fit(X, y)  # Model predicts [x, y] as a single unit
            
            # Store model
            self.prediction_models[red_agent_name] = model
            return True
        except Exception as e:
            print(f"Error fitting prediction model for {red_agent_name}: {e}")
            return False
            
    def predict_future_position(self, red_agent_name: str, steps_ahead: int = 1) -> Optional[Tuple[float, float]]:
        """
        Predicts the future position of a red agent based on its observed path.
        
        Args:
            red_agent_name (str): Name of the Red agent to predict position for
            steps_ahead (int): Number of steps into the future to predict. Defaults to 1.
            
        Returns:
            Optional[Tuple[float, float]]: Predicted (x, y) position or None if prediction couldn't be made
        """
        # Check if we already have a prediction for this agent and steps_ahead
        if red_agent_name in self.predicted_positions and len(self.predicted_positions[red_agent_name]) >= steps_ahead:
            return self.predicted_positions[red_agent_name][steps_ahead-1]
            
        # Need at least 2 points to make a prediction
        observations = self.observed_red_agents.get(red_agent_name, [])
        if len(observations) < 2:
            return None
            
        # Check if we have a model for this red agent
        if red_agent_name not in self.prediction_models:
            # Try to fit a model first
            if not self.fit_prediction_model(red_agent_name):
                return None
                
        # Get number of past positions to use based on processing_capability
        n = min(self.processing_capability, len(observations) - 1)
        
        # Cannot predict with less than 2 points
        if n < 1:
            return None
            
        # Get the most recent n positions
        recent_positions = np.array([pos for pos, _ in observations[-n:]])
        
        # Create feature vector from recent positions
        features = recent_positions.flatten().reshape(1, -1)  # Reshape to (1, 2*n) for sklearn
        
        # Get the prediction model
        model = self.prediction_models[red_agent_name]
        
        # Make prediction
        predicted_position = model.predict(features)[0]  # Returns [x, y] coordinates
        
        # For multiple steps ahead prediction
        for _ in range(1, steps_ahead):
            # Update features by removing oldest position and adding predicted position
            features = features.flatten()
            features = np.concatenate([features[2:], predicted_position]).reshape(1, -1)
            
            # Make next prediction
            predicted_position = model.predict(features)[0]
        
        return tuple(predicted_position)

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
                    self.record_red_agent_movement(red_name, red_data['position'], timestamp)
            
            # Clear previous predictions
            self.predicted_positions.clear()
            
            # Try to fit/update prediction models for all observed red agents
            for red_name in self.observed_red_agents.keys():
                # Only try to fit the model if we have at least 2 observations
                if len(self.observed_red_agents[red_name]) >= 2:
                    # Fit or update the prediction model
                    model_fitted = self.fit_prediction_model(red_name)
                    
                    # If model was successfully fitted, make predictions for future positions
                    if model_fitted or red_name in self.prediction_models:
                        # Store predictions for the next 5 steps
                        future_positions = []
                        for steps_ahead in range(1, 6):
                            future_pos = self.predict_future_position(red_name, steps_ahead)
                            if future_pos is not None:
                                future_positions.append(future_pos)
                        
                        # Store the predictions if we have any
                        if future_positions:
                            self.predicted_positions[red_name] = future_positions

        # Blue agents stay still for now
        return {
            'direction': np.array([0.0, 0.0], dtype=np.float32),
            'speed': 0
        }

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
