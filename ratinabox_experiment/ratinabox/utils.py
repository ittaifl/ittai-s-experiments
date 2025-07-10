"""
Utilities Module
================

Common helpers for file I/O, logging, configuration management, and Google Drive integration.
"""

import json
import yaml
import logging
import os
import sys
from typing import Dict, Any, Optional, Union
from pathlib import Path
import numpy as np


def setup_logging(
    level: str = "INFO",
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        format: Log message format
        log_file: Optional file to write logs to
        
    Returns:
        Configured logger
        
    Example:
        >>> logger = setup_logging("INFO", log_file="experiment.log")
        >>> logger.info("Starting experiment")
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    # Create logger
    logger = logging.getLogger("ratinabox_experiment")
    logger.setLevel(numeric_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_config(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from JSON or YAML file.
    
    Args:
        filepath: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Example:
        >>> config = load_config("config.json")
        >>> print(config["model"]["hidden_dim"])
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    
    filepath = Path(filepath)
    
    with open(filepath, 'r') as f:
        if filepath.suffix.lower() == '.json':
            config = json.load(f)
        elif filepath.suffix.lower() in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config format: {filepath.suffix}")
    
    return config


def save_config(config: Dict[str, Any], filepath: str) -> None:
    """
    Save configuration to JSON or YAML file.
    
    Args:
        config: Configuration dictionary
        filepath: Path to save configuration
        
    Example:
        >>> save_config({"model": {"hidden_dim": 100}}, "config.json")
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    filepath = Path(filepath)
    
    with open(filepath, 'w') as f:
        if filepath.suffix.lower() == '.json':
            json.dump(config, f, indent=2, default=str)
        elif filepath.suffix.lower() in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported config format: {filepath.suffix}")
    
    print(f"Configuration saved to {filepath}")


def create_default_config() -> Dict[str, Any]:
    """
    Create a default configuration dictionary.
    
    Returns:
        Default configuration
    """
    return {
        "environment": {
            "width": 1.0,
            "height": 1.0,
            "grid_size": 64,
            "num_wall_objects": 12,
            "object_types": [0, 1, 2],
            "seed": 42
        },
        "sensors": {
            "sensor_types": ["red", "green", "purple"],
            "distance_range": [0.01, 0.11],
            "spatial_resolution": 0.05,
            "angle_range": [0, 125],
            "fov_downscalar": 3,
            "num_beams": 6
        },
        "data": {
            "n_trials": 1000,
            "timesteps": 100,
            "test_timesteps": 50000,
            "batch_size": 128,
            "val_fraction": 0.2,
            "grid_size": 64,
            "seed": 42
        },
        "model": {
            "obs_dim": 24,
            "act_dim": 8,
            "hidden_dim": 100,
            "dropout_rate": 0.15,
            "noise_std": 0.03,
            "epsilon": 1e-2,
            "activation": "norm_relu",
            "init_method": "levinstein"
        },
        "training": {
            "num_epochs": 800,
            "learning_rate": 0.01,
            "batch_size": 128,
            "patience": 8,
            "max_lr_reductions": 5,
            "lr_reduction_factor": 0.5,
            "min_improvement": 5e-5,
            "l1_lambda": 0.0,
            "save_every": 50,
            "log_every": 10,
            "verbose": True
        },
        "plotting": {
            "figsize": [12, 8],
            "dpi": 100,
            "colormap": "jet",
            "interpolation": "gaussian",
            "units_to_plot": 100,
            "grid_size": [10, 10],
            "show_inline": True,
            "save_plots": True
        }
    }


def merge_configs(default_config: Dict[str, Any], user_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge user configuration with default configuration.
    
    Args:
        default_config: Default configuration dictionary
        user_config: User-provided configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    merged = default_config.copy()
    
    for key, value in user_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def is_google_colab() -> bool:
    """
    Check if running in Google Colab environment.
    
    Returns:
        True if running in Colab, False otherwise
    """
    try:
        import google.colab
        return True
    except ImportError:
        return False


def mount_google_drive(mount_point: str = "/content/drive") -> bool:
    """
    Mount Google Drive in Colab environment.
    
    Args:
        mount_point: Mount point for Google Drive
        
    Returns:
        True if successful, False otherwise
    """
    if not is_google_colab():
        print("Not running in Google Colab - skipping Drive mount")
        return False
    
    try:
        from google.colab import drive
        drive.mount(mount_point)
        print(f"Google Drive mounted at {mount_point}")
        return True
    except Exception as e:
        print(f"Failed to mount Google Drive: {e}")
        return False


def resolve_drive_path(path: str, drive_mount: str = "/content/drive") -> str:
    """
    Resolve a Google Drive path to absolute path.
    
    Args:
        path: Path that may be a Drive URI or relative path
        drive_mount: Mount point for Google Drive
        
    Returns:
        Resolved absolute path
        
    Example:
        >>> path = resolve_drive_path("MyDrive/experiments/data.npz")
        >>> print(path)  # "/content/drive/MyDrive/experiments/data.npz"
    """
    if path.startswith("drive://"):
        # Convert drive:// URI to mount path
        drive_path = path[8:]  # Remove "drive://" prefix
        return os.path.join(drive_mount, drive_path)
    elif path.startswith("MyDrive/"):
        # Direct MyDrive path
        return os.path.join(drive_mount, path)
    else:
        # Assume it's already a valid path
        return path


def ensure_dir(filepath: str) -> str:
    """
    Ensure directory exists for a file path.
    
    Args:
        filepath: File path
        
    Returns:
        The same file path
    """
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    return filepath


def get_device() -> str:
    """
    Get the best available computation device.
    
    Returns:
        Device string ("cuda" or "cpu")
    """
    import torch
    
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {gpu_name}")
    else:
        device = "cpu"
        print("Using CPU")
    
    return device


def set_random_seeds(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    import torch
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    print(f"Random seeds set to {seed}")


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
        
    Example:
        >>> print(format_duration(3661))  # "1h 1m 1s"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")
    
    return " ".join(parts)


def get_memory_usage() -> str:
    """
    Get current memory usage information.
    
    Returns:
        Memory usage string
    """
    import psutil
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    
    return f"Memory usage: {memory_mb:.1f} MB"


def print_system_info() -> None:
    """Print system information including Python version, packages, and hardware."""
    import platform
    import sys
    
    print("System Information:")
    print("=" * 50)
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.architecture()[0]}")
    print(f"Processor: {platform.processor()}")
    
    # Try to get GPU info
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
        else:
            print("GPU: Not available")
    except ImportError:
        print("PyTorch not installed")
    
    # Memory info
    try:
        print(get_memory_usage())
    except ImportError:
        print("psutil not available for memory info")
    
    print("=" * 50)


def validate_paths(paths: Dict[str, str]) -> Dict[str, str]:
    """
    Validate and resolve paths, including Google Drive paths.
    
    Args:
        paths: Dictionary of path names to paths
        
    Returns:
        Dictionary of validated paths
    """
    validated = {}
    
    for name, path in paths.items():
        if path is None:
            validated[name] = None
            continue
            
        # Resolve Drive paths if in Colab
        if is_google_colab():
            resolved_path = resolve_drive_path(path)
        else:
            resolved_path = path
        
        # Ensure directory exists for output paths
        if "output" in name.lower() or "checkpoint" in name.lower():
            ensure_dir(resolved_path)
        
        validated[name] = resolved_path
    
    return validated


class ConfigurationError(Exception):
    """Exception raised for configuration-related errors."""
    pass


class ExperimentTimer:
    """Context manager for timing experimental procedures."""
    
    def __init__(self, name: str, logger: Optional[logging.Logger] = None):
        """
        Initialize timer.
        
        Args:
            name: Name of the procedure being timed
            logger: Optional logger for output
        """
        self.name = name
        self.logger = logger
        self.start_time = None
    
    def __enter__(self):
        """Start timing."""
        import time
        self.start_time = time.time()
        message = f"Starting {self.name}..."
        
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and report duration."""
        import time
        duration = time.time() - self.start_time
        duration_str = format_duration(duration)
        
        if exc_type is None:
            message = f"Completed {self.name} in {duration_str}"
        else:
            message = f"Failed {self.name} after {duration_str}"
        
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
