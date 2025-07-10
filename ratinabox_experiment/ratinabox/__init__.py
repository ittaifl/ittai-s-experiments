"""
RatInABox Experiment Package
===========================

A modular package for structured latent representations through predictive learning.
Based on the paper: Recanatesi et al. - Predictive learning as a network mechanism 
for extracting low-dimensional latent space representations.

This package provides:
- Environment creation and parameterization
- Sensor suite definition and configuration  
- Data loading for train/test trajectories
- Neural agent models with load/save capabilities
- Training loops with checkpoint support
- Place-cell visualization and comparison tools
- CLI interfaces for local and Colab usage
"""

from .env import create_environment, EnvironmentConfig
from .sensors import create_sensor_suite, SensorConfig
from .data import load_training_data, load_test_data, DataLoader as RatDataLoader
from .model import NextStepRNN, load_model, save_model
from .train import train_model, TrainingConfig
from .plot import plot_place_cells, compare_place_cells, save_plots
from .utils import setup_logging, load_config, save_config

__version__ = "1.0.0"
__author__ = "Ittai's Experiments"

__all__ = [
    "create_environment",
    "EnvironmentConfig", 
    "create_sensor_suite",
    "SensorConfig",
    "load_training_data",
    "load_test_data", 
    "RatDataLoader",
    "NextStepRNN",
    "load_model",
    "save_model",
    "train_model",
    "TrainingConfig",
    "plot_place_cells",
    "compare_place_cells",
    "save_plots",
    "setup_logging",
    "load_config",
    "save_config",
]
