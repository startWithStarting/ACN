import yaml
import os

from src.utils.logger import get_logger

logger = get_logger("acn.config")


def load_config(config_path):
    """
    Loads a configuration file (YAML format).

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        dict: A dictionary containing the configuration, or None if loading fails.
    """
    if not os.path.exists(config_path):
        logger.error("Configuration file not found at {}", config_path)
        return None

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded successfully from {}", config_path)
        return config
    except yaml.YAMLError as e:
        logger.error("Error parsing YAML file {}: {}", config_path, e)
        return None
    except Exception as e:
        logger.error("Unexpected error while loading {}: {}", config_path, e)
        return None


# Example usage (optional, for testing this module directly)
if __name__ == '__main__':
    dummy_config_path = 'dummy_config.yaml'
    dummy_data = {
        'experiment_name': 'test_experiment',
        'agents': {
            'blue_agents': [{'count': 2, 'communication_bandwidth': 10}],
            'red_agents': [{'count': 1, 'processing_capability': 50}]
        }
    }
    with open(dummy_config_path, 'w') as f:
        yaml.dump(dummy_data, f)

    loaded = load_config(dummy_config_path)
    os.remove(dummy_config_path)
    load_config('non_existent_config.yaml')
