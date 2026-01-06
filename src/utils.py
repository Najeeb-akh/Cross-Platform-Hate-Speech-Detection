"""
Utility functions for I/O operations, run directory management, and reproducibility.
"""

import json
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import pickle

import numpy as np


def create_run_dir(base_dir: str = "runs") -> Path:
    """
    Create a timestamped run directory.
    
    Args:
        base_dir: Base directory for runs
    
    Returns:
        Path to created run directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created run directory: {run_dir}")
    return run_dir


def save_json(data: Dict, filepath: str):
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Path to JSON file
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved JSON to {filepath}")


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    
    print(f"Random seed set to {seed}")

