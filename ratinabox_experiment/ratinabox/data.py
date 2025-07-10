"""
Data Module
===========

Load train/test trajectories and place-cell targets for model training and evaluation.
Handles data generation, loading, preprocessing, and batching.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import os
import pickle
from tqdm import tqdm

from .env import create_environment, EnvironmentConfig, get_next_move, create_navigation_grid, get_head_directions
from .sensors import create_sensor_suite, SensorConfig, update_all_sensors, reset_all_sensors, get_sensor_observations
from ratinabox.Agent import Agent


@dataclass
class DataConfig:
    """Configuration for data generation and loading."""
    
    n_trials: int = 1000
    timesteps: int = 100
    test_timesteps: int = 50000
    batch_size: int = 128
    val_fraction: float = 0.2
    grid_size: int = 64
    seed: Optional[int] = None
    
    # Data paths (can be local paths or Google Drive URIs)
    train_data_path: Optional[str] = None
    test_data_path: Optional[str] = None
    cache_dir: Optional[str] = None


class RatDataLoader:
    """Custom data loader for RatInABox trajectory data."""
    
    def __init__(self, config: DataConfig):
        """
        Initialize the data loader.
        
        Args:
            config: Data configuration object
        """
        self.config = config
        if config.seed is not None:
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)
    
    def generate_training_data(
        self, 
        env_config: EnvironmentConfig,
        sensor_config: SensorConfig
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate training data by simulating agent trajectories.
        
        Args:
            env_config: Environment configuration
            sensor_config: Sensor configuration
            
        Returns:
            Tuple of (observations, actions) arrays
            
        Example:
            >>> loader = RatDataLoader(DataConfig())
            >>> obs, acts = loader.generate_training_data(env_config, sensor_config)
        """
        print(f"Generating training data: {self.config.n_trials} trials of {self.config.timesteps} timesteps...")
        
        # Create environment and agent
        env = create_environment(env_config)
        agent = Agent(env)
        sensors = create_sensor_suite(agent, sensor_config)
        
        # Setup navigation grid
        grid, grid_centers = create_navigation_grid(self.config.grid_size)
        head_directions = get_head_directions()
        
        # Initialize data arrays
        obs_dim = len(sensor_config.sensor_types) * 6 + 6  # RGB sensors + distances
        obs_array = np.zeros((self.config.n_trials, self.config.timesteps, obs_dim))
        act_array = np.zeros((self.config.n_trials, self.config.timesteps, 8))
        
        # Generate data for each trial
        for trial in tqdm(range(self.config.n_trials), desc="Generating trials"):
            # Reset for new trial
            agent.reset_history()
            reset_all_sensors(sensors)
            
            # Random starting position and direction
            coords = np.random.randint(high=self.config.grid_size-2, low=1, size=2)
            action_hd_idx = np.random.randint(high=7, low=0)
            
            agent.pos = np.array(grid_centers[coords[0], coords[1]], dtype=float)
            agent.head_direction = np.array(head_directions[action_hd_idx], dtype=float)
            agent.dt = 0.05
            
            actions_saved = []
            
            # Simulate trial trajectory
            for t in range(self.config.timesteps):
                # Get next move
                coords, action_hd_idx, action_encoding = get_next_move(
                    coords, action_hd_idx, self.config.grid_size
                )
                
                # Update agent
                next_pos = np.array(grid_centers[coords[0], coords[1]], dtype=float)
                agent.use_DRIFT = False
                agent.update(forced_next_position=next_pos)
                agent.head_direction = np.array(head_directions[action_hd_idx], dtype=float)
                agent.history["head_direction"][-1] = head_directions[action_hd_idx].tolist()
                
                actions_saved.append(action_encoding)
                
                # Update sensors
                update_all_sensors(sensors)
            
            # Store trial data
            obs_array[trial] = get_sensor_observations(sensors)
            act_array[trial] = np.array(actions_saved, dtype=float)
        
        return obs_array, act_array
    
    def generate_test_data(
        self,
        env_config: EnvironmentConfig,
        sensor_config: SensorConfig
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate test data with a long continuous trajectory.
        
        Args:
            env_config: Environment configuration
            sensor_config: Sensor configuration
            
        Returns:
            Tuple of (observations, actions, positions)
        """
        print(f"Generating test data: {self.config.test_timesteps} timesteps...")
        
        # Create environment and agent
        env = create_environment(env_config)
        agent = Agent(env)
        sensors = create_sensor_suite(agent, sensor_config)
        
        # Setup navigation
        grid, grid_centers = create_navigation_grid(self.config.grid_size)
        head_directions = get_head_directions()
        
        # Reset for test data generation
        agent.reset_history()
        reset_all_sensors(sensors)
        
        # Random starting conditions
        coords = np.random.randint(high=self.config.grid_size-2, low=1, size=2)
        action_hd_idx = np.random.randint(high=7, low=0)
        
        agent.pos = np.array(grid_centers[coords[0], coords[1]], dtype=float)
        agent.head_direction = np.array(head_directions[action_hd_idx], dtype=float)
        agent.dt = 0.05
        
        actions_saved = []
        
        # Generate long trajectory
        for t in tqdm(range(self.config.test_timesteps), desc="Generating test trajectory"):
            coords, action_hd_idx, action_encoding = get_next_move(
                coords, action_hd_idx, self.config.grid_size
            )
            
            next_pos = np.array(grid_centers[coords[0], coords[1]], dtype=float)
            agent.use_DRIFT = False
            agent.update(forced_next_position=next_pos)
            agent.head_direction = np.array(head_directions[action_hd_idx], dtype=float)
            agent.history["head_direction"][-1] = head_directions[action_hd_idx].tolist()
            
            actions_saved.append(action_encoding)
            update_all_sensors(sensors)
        
        # Extract data
        obs_array = get_sensor_observations(sensors)
        positions = agent.get_history_arrays()['pos']
        
        # Clean up NaN values
        nan_mask = np.isnan(obs_array)
        if nan_mask.any():
            print(f"Warning: Found {nan_mask.sum()} NaN values, replacing with 0")
            obs_array = np.nan_to_num(obs_array, nan=0.0)
        
        return obs_array, np.array(actions_saved), positions
    
    def save_data(self, data: Dict, filepath: str) -> None:
        """
        Save data to disk.
        
        Args:
            data: Dictionary containing data arrays
            filepath: Path to save the data
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if filepath.endswith('.npz'):
            np.savez_compressed(filepath, **data)
        elif filepath.endswith('.pkl'):
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        else:
            raise ValueError("Filepath must end with .npz or .pkl")
        
        print(f"Data saved to {filepath}")
    
    def load_data(self, filepath: str) -> Dict:
        """
        Load data from disk.
        
        Args:
            filepath: Path to load data from
            
        Returns:
            Dictionary containing loaded data
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        if filepath.endswith('.npz'):
            data = dict(np.load(filepath))
        elif filepath.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
        else:
            raise ValueError("Filepath must end with .npz or .pkl")
        
        print(f"Data loaded from {filepath}")
        return data


def load_training_data(
    config: DataConfig,
    env_config: EnvironmentConfig,
    sensor_config: SensorConfig,
    force_regenerate: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    Load or generate training data and return data loaders.
    
    Args:
        config: Data configuration
        env_config: Environment configuration  
        sensor_config: Sensor configuration
        force_regenerate: Whether to force data regeneration
        
    Returns:
        Tuple of (train_loader, val_loader)
        
    Example:
        >>> train_loader, val_loader = load_training_data(
        ...     data_config, env_config, sensor_config
        ... )
    """
    loader = RatDataLoader(config)
    
    # Check if cached data exists
    if config.train_data_path and os.path.exists(config.train_data_path) and not force_regenerate:
        print("Loading cached training data...")
        data = loader.load_data(config.train_data_path)
        obs_array = data['observations']
        act_array = data['actions']
    else:
        # Generate new data
        obs_array, act_array = loader.generate_training_data(env_config, sensor_config)
        
        # Save if path provided
        if config.train_data_path:
            data = {'observations': obs_array, 'actions': act_array}
            loader.save_data(data, config.train_data_path)
    
    # Clean NaN values
    obs_array = np.nan_to_num(obs_array, nan=0.0)
    
    # Convert to tensors for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    obs_seq = torch.tensor(obs_array[:, :-1], dtype=torch.float32)      # (trials, T-1, obs_dim)
    act_seq = torch.tensor(act_array[:, 1:], dtype=torch.float32)       # (trials, T-1, act_dim)
    next_obs_seq = torch.tensor(obs_array[:, 1:], dtype=torch.float32)  # (trials, T-1, obs_dim)
    
    # Create dataset
    full_dataset = TensorDataset(obs_seq, act_seq, next_obs_seq)
    
    # Split into train/validation
    val_size = int(len(full_dataset) * config.val_fraction)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=0, 
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=0, 
        pin_memory=False
    )
    
    return train_loader, val_loader


def load_test_data(
    config: DataConfig,
    env_config: EnvironmentConfig,
    sensor_config: SensorConfig,
    force_regenerate: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Load or generate test data.
    
    Args:
        config: Data configuration
        env_config: Environment configuration
        sensor_config: Sensor configuration
        force_regenerate: Whether to force data regeneration
        
    Returns:
        Tuple of (obs_tensor, act_tensor, positions_array)
    """
    loader = RatDataLoader(config)
    
    # Check if cached data exists
    if config.test_data_path and os.path.exists(config.test_data_path) and not force_regenerate:
        print("Loading cached test data...")
        data = loader.load_data(config.test_data_path)
        obs_array = data['observations']
        act_array = data['actions']
        positions = data['positions']
    else:
        # Generate new data
        obs_array, act_array, positions = loader.generate_test_data(env_config, sensor_config)
        
        # Save if path provided
        if config.test_data_path:
            data = {
                'observations': obs_array,
                'actions': act_array, 
                'positions': positions
            }
            loader.save_data(data, config.test_data_path)
    
    # Convert to tensors
    obs_tensor = torch.tensor(obs_array, dtype=torch.float32).unsqueeze(1)
    act_tensor = torch.tensor(act_array, dtype=torch.float32).unsqueeze(1)
    
    return obs_tensor, act_tensor, positions


def create_autoencoding_data(
    config: DataConfig,
    env_config: EnvironmentConfig,
    sensor_config: SensorConfig
) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders for autoencoding training (reconstruct current input).
    
    Args:
        config: Data configuration
        env_config: Environment configuration
        sensor_config: Sensor configuration
        
    Returns:
        Tuple of (train_loader, val_loader) for autoencoding
    """
    # Load training data
    train_loader, val_loader = load_training_data(config, env_config, sensor_config)
    
    # Convert to autoencoding format (input = output, no actions)
    autoencoding_train_data = []
    autoencoding_val_data = []
    
    for obs_seq, act_seq, next_obs_seq in train_loader:
        # For autoencoding: input observation -> same observation
        # Zero out actions since we're not using them
        zero_actions = torch.zeros_like(act_seq)
        autoencoding_train_data.append((obs_seq, zero_actions, obs_seq))
    
    for obs_seq, act_seq, next_obs_seq in val_loader:
        zero_actions = torch.zeros_like(act_seq)
        autoencoding_val_data.append((obs_seq, zero_actions, obs_seq))
    
    # Create new data loaders
    train_dataset_autoencoding = TensorDataset(*zip(*autoencoding_train_data))
    val_dataset_autoencoding = TensorDataset(*zip(*autoencoding_val_data))
    
    train_loader_autoencoding = DataLoader(
        train_dataset_autoencoding,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    val_loader_autoencoding = DataLoader(
        val_dataset_autoencoding,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    return train_loader_autoencoding, val_loader_autoencoding
