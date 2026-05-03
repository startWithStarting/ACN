import numpy as np
from typing import Dict, Any

from ...agents.registry import register_strategy


@register_strategy("static", side="blue")
def static_blue_strategy() -> Dict[str, Any]:
    """
    A simple strategy where blue agents remain stationary.

    Returns:
        Dict[str, Any]: A dictionary containing the 'direction' and 'speed'.
    """
    return {
        'direction': np.array([0.0, 0.0], dtype=np.float32),
        'speed': np.float32(0.0)
    }
