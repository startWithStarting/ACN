# ACN (Agent Communication Networks)

## Overview
This project focuses on multi-agent systems with onboard sensing and actuation capabilities. We explore communication protocols and distributed control mechanisms for autonomous agents operating in a shared environment. Agents develop their own ego-centric "world models" and can share versions of these models with other agents. A key research focus is examining how ML-based communication protocols affect agent performance when each agent has limited computational capabilities.

## Current Features

### Agent Types
- **Blue Agents**: Defensive agents equipped with Vector Auto-Regressive (VAR) models to predict future positions of red agents. Their behavior can be configured using different strategies.
- **Red Agents**: Mobile agents with various movement strategies. Their behavior can be configured using different strategies.

### Agent Strategies

#### Blue Agent Strategies
1. **Static**: Agents remain stationary but still track and predict red agent movements.
2. **Pursuit**: Agents move toward the average predicted position of detected red agents.

#### Red Agent Strategies
1. **Center-based**: Agents move toward or away from the grid center, maintaining a minimum distance.
2. **Avoidant**: Agents detect blue agents and steer away from them.
3. **Aggressive**: Agents detect and pursue blue agents, prioritizing the closest ones.
4. **Team-based**: Agents detect other red teammates and move toward their average position.

## Configuration Files
The project includes multiple configuration files in the `config` directory:

- `experiment_config.yaml`: Default configuration with a mix of different agent types and strategies.
- `center_config.yaml`: All red agents use center-based movement.
- `avoidant_config.yaml`: All red agents use the avoidant strategy.
- `aggressive_config.yaml`: All red agents use the aggressive strategy.
- `team_config.yaml`: All red agents use the team-based strategy.

## Running the Simulation

To run the simulation with a specific configuration:

```bash
python main.py --config config/center_config.yaml
```

Replace the config path with any of the available configuration files.

## Project Structure

- `src/agents/`: Agent implementation classes
  - `blue_agent.py`: Blue agent implementation with prediction capabilities
  - `red_agent.py`: Red agent implementation with configurable strategies
  - `base_agent.py`: Base class for all agents
  - `strategies/`: Movement strategies for red agents
- `src/env/`: Environment implementation using PettingZoo framework
- `src/utils/`: Utility functions and helpers
- `config/`: Configuration files for experiments
- `results/`: Generated experiment results (plots, GIFs, etc.)

## Future Work
- Implementing ML-based communication protocols between agents
- Enhancing the agent world models with more sophisticated prediction techniques
- Developing cooperative and competitive scenarios for agent interaction
- Evaluating performance metrics across different communication protocols and agent types

##A note on the strategies
lets assume the grid is a 2D plane with the center at (50,50) and our red agents are peigens looking for food. In that case, the red agent strategies can be summarised as follows:

- center (peigens moving towards food spread in a ring around the center of the grid)
- avoidant (peigens moving towards the center of the grid while avoiding blue agents)
- aggressive (kamakazi peigens moving towards blue agents, perhaps to distract them from the other peigens)
- team (peigens moving towards the center of the mass of the team mates they can see (flocking without the repulsion component))


