"""
Environment Module
==================

Create and parameterize the RatInABox environment with customizable dimensions,
wall objects, and navigation grids.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from ratinabox.Environment import Environment


@dataclass
class EnvironmentConfig:
    """Configuration for the RatInABox environment."""
    
    width: float = 1.0
    height: float = 1.0
    grid_size: int = 64
    num_wall_objects: int = 12
    object_types: List[int] = None
    seed: Optional[int] = None
    
    def __post_init__(self):
        if self.object_types is None:
            self.object_types = [0, 1, 2]  # Red, Green, Purple objects


def create_environment(config: EnvironmentConfig) -> Environment:
    """
    Create a RatInABox environment with specified configuration.
    
    Args:
        config: Environment configuration object
        
    Returns:
        Configured Environment instance
        
    Example:
        >>> config = EnvironmentConfig(width=2.0, height=2.0, grid_size=128)
        >>> env = create_environment(config)
    """
    if config.seed is not None:
        np.random.seed(config.seed)
    
    # Create the base environment
    env = Environment()
    
    # Add wall objects for each type
    for obj_type in config.object_types:
        locations = _generate_wall_object_locations(
            config.num_wall_objects, 
            config.width, 
            config.height
        )
        for location in locations:
            env.add_object(location, type=obj_type)
    
    return env


def _generate_wall_object_locations(
    num_objects: int, 
    width: float, 
    height: float
) -> np.ndarray:
    """
    Generate random locations for wall objects.
    
    Args:
        num_objects: Number of objects to place
        width: Environment width
        height: Environment height
        
    Returns:
        Array of object locations with shape (num_objects, 2)
    """
    # Generate locations along walls (either x=0/1 or y=0/1)
    locations = np.concatenate([
        np.round(np.random.uniform(low=0.0, high=width, size=(num_objects, 1))),
        np.random.uniform(low=0.0, high=height, size=(num_objects, 1))
    ], axis=1)
    
    # Randomly swap columns to distribute along different walls
    swap_mask = np.random.rand(len(locations), 1) < 0.5
    locations = np.where(swap_mask, locations, locations[:, ::-1])
    
    return locations


def create_navigation_grid(grid_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a navigation grid for agent movement.
    
    Args:
        grid_size: Size of the grid (grid_size x grid_size)
        
    Returns:
        Tuple of (grid_coordinates, grid_centers)
        
    Example:
        >>> coords, centers = create_navigation_grid(64)
        >>> print(f"Grid shape: {centers.shape}")  # (64, 64, 2)
    """
    grid = np.array([(x, y) for x in range(grid_size) for y in range(grid_size)])
    grid_centers = np.array([
        [(x / grid_size, y / grid_size) for y in range(grid_size)]
        for x in range(grid_size)
    ])
    
    return grid, grid_centers


def get_head_directions() -> np.ndarray:
    """
    Get the 8 possible head directions for agent movement.
    
    Returns:
        Array of unit vectors representing 8 cardinal/diagonal directions
    """
    sqrt2_2 = np.sqrt(2) / 2
    return np.array([
        (1, 0),           # E
        (sqrt2_2, sqrt2_2),   # NE  
        (0, 1),           # N
        (-sqrt2_2, sqrt2_2),  # NW
        (-1, 0),          # W
        (-sqrt2_2, -sqrt2_2), # SW
        (0, -1),          # S
        (sqrt2_2, -sqrt2_2)   # SE
    ])


def get_action_options() -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """
    Get action options for agent movement.
    
    Returns:
        Tuple of (action_indices, action_names, coordinate_updates)
    """
    action_idxs = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    action_options = np.array(['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE'])
    coord_updates = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
    
    return action_idxs, action_options, coord_updates


def get_next_move(
    coords: np.ndarray, 
    hd_idx: int, 
    grid_size: int = 64
) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Compute the agent's next move based on current position and head direction.
    
    The agent moves along a grid structure, randomly selecting from 8 adjacent tiles
    with probabilities biased toward forward movement. Wall collisions are avoided.
    
    Args:
        coords: Current coordinates (x, y)
        hd_idx: Current head direction index (0-7)
        grid_size: Size of the navigation grid
        
    Returns:
        Tuple of (next_coordinates, next_hd_idx, action_encoding)
        
    Example:
        >>> coords = np.array([32, 32])
        >>> next_coords, next_hd, action = get_next_move(coords, 0, 64)
    """
    wall_buffer = 1
    
    # Check border conditions (E, N, W, S)
    border_flags = [
        coords[0] >= grid_size - wall_buffer,  # East wall
        coords[1] >= grid_size - wall_buffer,  # North wall  
        coords[0] <= wall_buffer,              # West wall
        coords[1] <= wall_buffer               # South wall
    ]
    
    # Default action probabilities (biased toward forward movement)
    action_probs = np.roll(np.array([0.5, 0.15, 0.05, 0.04, 0.02, 0.04, 0.05, 0.15]), hd_idx)
    
    action_idxs, action_options, coord_updates = get_action_options()
    
    # Remove impossible actions near walls
    if any(border_flags):
        if border_flags[0]:  # East wall
            action_probs[np.isin(action_options, ['E', 'NE', 'SE'])] = 0
        if border_flags[1]:  # North wall
            action_probs[np.isin(action_options, ['N', 'NE', 'NW'])] = 0
        if border_flags[2]:  # West wall
            action_probs[np.isin(action_options, ['W', 'NW', 'SW'])] = 0
        if border_flags[3]:  # South wall
            action_probs[np.isin(action_options, ['S', 'SE', 'SW'])] = 0
        
        # Renormalize probabilities
        action_probs = action_probs / action_probs.sum()
    
    # Choose next action
    next_action_idx = np.random.choice(action_idxs, p=action_probs)
    next_coords = coords + np.array(coord_updates[next_action_idx])
    
    # Create one-hot action encoding
    action_encoding = np.zeros(8)
    action_encoding[next_action_idx] = 1
    
    return next_coords, next_action_idx, action_encoding
