import numpy as np
from typing import Dict, Any, Optional, Tuple, List

def static_blue_strategy() -> Dict[str, Any]:
    """
    A simple strategy where blue agents remain stationary.
    
    Returns:
        Dict[str, Any]: A dictionary containing the 'direction' (normalized numpy array)
                         and 'speed' (integer).
    """
    return {
        'direction': np.array([0.0, 0.0], dtype=np.float32),
        'speed': 0
    }
