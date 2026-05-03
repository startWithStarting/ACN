Configuration
=============

ACN uses YAML configuration files to define experiments without code changes.

Configuration Files
-------------------

Example configs are provided in the ``config/`` directory:

* ``experiment_config.yaml``: Mixed agent types and strategies
* ``center_config.yaml``: All red agents use center strategy
* ``avoidant_config.yaml``: All red agents use avoidant strategy
* ``aggressive_config.yaml``: All red agents use aggressive strategy
* ``team_config.yaml``: All red agents use team-based strategy

Configuration Structure
-----------------------

.. code-block:: yaml

   # Grid settings
   grid_width: 100
   grid_height: 100

   # Environment settings
   max_cycles: 1000
   render_mode: "human_matplotlib"

   # Blue agent configuration
   blue_agents:
     count: 5
     detection_radius: 30
     prediction_horizon: 10
     strategy: "pursuit"
     params:
       speed: 3.0

   # Red agent configuration
   red_agents:
     count: 10
     strategy: "flocking"
     params:
       cohesion_weight: 1.0
       alignment_weight: 1.0
       separation_weight: 1.5
       separation_radius: 10
       max_speed: 5.0
       max_force: 0.5

   # Physics settings
   physics:
     boundary_mode: "clamp"
     default_drag: 0.0
     default_max_speed: 5.0
     enable_collisions: true

   # Reward configuration
   reward:
     type: "attractor"
     params:
       reward_radius: 50.0
       tolerance: 1.0

   # Visualization
   save_episode_gifs: true
   episode_length: 500

Environment Variables
---------------------

ACN uses the following environment variables:

* ``ACN_CONFIG_PATH``: Path to default config file
* ``ACN_RESULTS_DIR``: Directory for results (default: ``results/``)
* ``ACN_LOG_LEVEL``: Logging level (DEBUG, INFO, WARNING, ERROR)

Loading Configuration
---------------------

.. code-block:: python

   from src.utils.config_loader import load_config

   # Load from file
   config = load_config("config/experiment_config.yaml")

   # Load from environment
   config = load_config()  # Uses ACN_CONFIG_PATH

   # Override via CLI
   # python main.py --config config/center_config.yaml