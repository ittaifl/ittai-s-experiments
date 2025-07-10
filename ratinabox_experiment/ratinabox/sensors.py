"""
Sensors Module
==============

Define and configure different sensor suites for the rat agent,
including Field of View Object Vector Cells (FOV-OVCs) for wall detection.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings
from matplotlib.collections import EllipseCollection
import matplotlib
import matplotlib.pyplot as plt

from ratinabox.Neurons import FieldOfViewOVCs
from ratinabox import utils
import ratinabox

warnings.filterwarnings('ignore')


@dataclass 
class SensorConfig:
    """Configuration for sensor suites."""
    
    sensor_types: List[str] = None
    distance_range: Tuple[float, float] = (0.01, 0.11)
    spatial_resolution: float = 0.05
    angle_range: Tuple[float, float] = (0, 125)
    fov_downscalar: int = 3
    num_beams: int = 6
    max_ray_length: float = 100.0
    
    def __post_init__(self):
        if self.sensor_types is None:
            self.sensor_types = ["red", "green", "purple"]


class WallFixedFOVc(FieldOfViewOVCs):
    """
    Enhanced Field of View Object Vector Cells that are fixed to walls.
    
    This class extends the standard FOV-OVCs to create sensors that detect
    wall objects at fixed distances, simulating 6 beams of light that detect
    RGB colors and distances to walls.
    """
    
    def __init__(self, agent, params=None):
        """
        Initialize wall-fixed FOV cells.
        
        Args:
            agent: The RatInABox agent
            params: Dictionary of parameters for the FOV cells
        """
        if params is None:
            params = {}
        
        super().__init__(agent, params)
        self.history["tuning_distances"] = []
        self.history["sigma_distances"] = []
        self.history["sigma_angles"] = []
        self.history["dists"] = []

    def display_vector_cells(self, fig=None, ax=None, t=None, **kwargs):
        """
        Visualize the current firing rate of these cells relative to the Agent.
        
        Each cell is plotted as an ellipse where the alpha value reflects the
        current firing rate. The ellipses approximate the receptive fields.
        
        Args:
            fig: Matplotlib figure (optional)
            ax: Matplotlib axis (optional)  
            t: Time to plot at (optional)
            **kwargs: Additional plotting arguments
            
        Returns:
            fig, ax: Updated matplotlib objects
        """
        if t is None:
            t = self.Agent.history["t"][-1]
        t_id = np.argmin(np.abs(np.array(self.Agent.history["t"]) - t))

        if fig is None and ax is None:
            fig, ax = self.Agent.plot_trajectory(t_start=t - 10, t_end=t, **kwargs)

        pos = self.Agent.history["pos"][t_id]
        head_direction = self.Agent.history["head_direction"][t_id]
        head_direction_angle = 0.0

        if self.reference_frame == "egocentric":
            head_direction_angle = (180 / np.pi) * ratinabox.utils.get_angle(head_direction)
            x_axis_wrt_agent = head_direction / np.linalg.norm(head_direction)
            y_axis_wrt_agent = utils.rotate(x_axis_wrt_agent, np.pi / 2)
        else:
            x_axis_wrt_agent = np.array([1, 0])
            y_axis_wrt_agent = np.array([0, 1])

        fr = np.array(self.history["firingrate"][t_id])
        tuning_distances = np.array(self.history["tuning_distances"][t_id])
        sigma_angles = np.array(self.history["sigma_angles"][t_id])
        sigma_distances = np.array(self.history["sigma_distances"][t_id])
        tuning_angles = self.tuning_angles

        x = tuning_distances * np.cos(tuning_angles)
        y = tuning_distances * np.sin(tuning_angles)

        pos_of_cells = pos + np.outer(x, x_axis_wrt_agent) + np.outer(y, y_axis_wrt_agent)

        ww = sigma_angles * tuning_distances
        hh = sigma_distances
        aa = 1.0 * head_direction_angle + tuning_angles * 180 / np.pi

        ec = EllipseCollection(
            ww, hh, aa, units='x',
            offsets=pos_of_cells,
            offset_transform=ax.transData,
            linewidth=0.5,
            edgecolor="dimgrey",
            zorder=2.1,
        )
        
        if self.cell_colors is None:
            facecolor = self.color if self.color is not None else "C1"
            facecolor = np.array(matplotlib.colors.to_rgba(facecolor))
            facecolor_array = np.tile(np.array(facecolor), (self.n, 1))
        else:
            facecolor_array = self.cell_colors.copy()
            
        facecolor_array[:, -1] = 0.7 * np.maximum(
            0, np.minimum(1, fr / (0.5 * self.max_fr))
        )
        ec.set_facecolors(facecolor_array)
        ax.add_collection(ec)

        return fig, ax

    def ray_distances_to_walls(self, agent_pos, head_direction, thetas, ray_length, walls):
        """
        Compute distances from agent to walls along multiple ray directions.
        
        This simulates the agent shooting out beams of light at different angles
        and measuring the distance to wall intersections.
        
        Args:
            agent_pos: Current agent position
            head_direction: Current head direction vector
            thetas: Array of angles relative to head direction
            ray_length: Maximum ray length
            walls: Array of wall segments
            
        Returns:
            Array of distances to walls for each ray
        """
        n = len(thetas)
        heading = np.arctan2(head_direction[1], head_direction[0])

        # Build ray endpoints
        ends = np.stack([
            agent_pos + ray_length * np.array([np.cos(heading + θ), np.sin(heading + θ)])
            for θ in thetas
        ], axis=0)

        # Ray start points (agent position)
        starts = np.tile(agent_pos[np.newaxis, :], (n, 1))

        # Build ray segments
        segs = np.zeros((n, 2, 2))
        segs[:, 0, :] = starts  # Start points
        segs[:, 1, :] = ends    # End points

        # Calculate intersections with walls
        intercepts = utils.vector_intercepts(segs, walls, return_collisions=False)
        la = intercepts[..., 0]
        lb = intercepts[..., 1]

        # Valid segment-segment intersections
        valid = (la > 0) & (la < 1) & (lb > 0) & (lb < 1)

        # Find nearest intersection for each ray
        la_valid = np.where(valid, la, np.inf)
        min_la = la_valid.min(axis=1)

        # Convert to distances
        distances = np.minimum(min_la * ray_length, ray_length)
        return distances

    def update_cell_locations(self):
        """Update cell locations to remain fixed on walls."""
        thetas = self.Agent.Neurons[0].tuning_angles
        n = len(thetas)

        # Get distances to walls for each sensor beam
        self.dists = self.ray_distances_to_walls(
            self.Agent.pos, 
            self.Agent.head_direction, 
            thetas, 
            100.0, 
            np.array(self.Agent.Environment.walls)
        )

        # Update tuning parameters based on wall distances
        s = self.dists * 16.5
        self.tuning_distances = np.ones(n) * 0.06 * s
        self.sigma_distances = np.ones(n) * 0.05
        self.sigma_angles = np.ones(n) * 0.83333333 / s

    def save_to_history(self):
        """Save current state to history including distance measurements."""
        super().save_to_history()
        self.history["tuning_distances"].append(self.tuning_distances.copy())
        self.history["sigma_distances"].append(self.sigma_distances.copy())
        self.history["sigma_angles"].append(self.sigma_angles.copy())
        # Normalize distances to [0,1] range (max distance in unit square is sqrt(2))
        self.history["dists"].append(self.dists.copy() / np.sqrt(2))

    def reset_history(self):
        """Reset history for new simulation runs."""
        super().reset_history()
        self.history["dists"] = []


def create_sensor_suite(agent, config: SensorConfig) -> Dict[str, WallFixedFOVc]:
    """
    Create a suite of wall-fixed FOV sensors for different object types.
    
    Args:
        agent: The RatInABox agent
        config: Sensor configuration
        
    Returns:
        Dictionary mapping sensor names to WallFixedFOVc instances
        
    Example:
        >>> config = SensorConfig(sensor_types=["red", "green"])
        >>> sensors = create_sensor_suite(agent, config)
        >>> red_sensor = sensors["red"]
    """
    sensors = {}
    
    # Object type mapping
    type_mapping = {"red": 0, "green": 1, "purple": 2}
    
    for i, sensor_type in enumerate(config.sensor_types):
        if sensor_type not in type_mapping:
            raise ValueError(f"Unknown sensor type: {sensor_type}")
            
        obj_type = type_mapping[sensor_type]
        
        params = {
            "object_tuning_type": obj_type,
            "distance_range": config.distance_range,
            "spatial_resolution": config.spatial_resolution,
            "cell_arrangement": "uniform_manifold",
            "angle_range": config.angle_range
        }
        
        sensor = WallFixedFOVc(agent, params=params)
        
        # Apply FOV downscaling
        sensor.tuning_angles = sensor.tuning_angles / config.fov_downscalar
        
        sensors[sensor_type] = sensor
    
    return sensors


def update_all_sensors(sensors: Dict[str, WallFixedFOVc]) -> None:
    """
    Update all sensors in the suite.
    
    Args:
        sensors: Dictionary of sensor instances
    """
    for sensor in sensors.values():
        sensor.update_cell_locations()
        sensor.update()


def reset_all_sensors(sensors: Dict[str, WallFixedFOVc]) -> None:
    """
    Reset history for all sensors in the suite.
    
    Args:
        sensors: Dictionary of sensor instances  
    """
    for sensor in sensors.values():
        sensor.reset_history()


def get_sensor_observations(sensors: Dict[str, WallFixedFOVc]) -> np.ndarray:
    """
    Extract current observations from all sensors.
    
    Args:
        sensors: Dictionary of sensor instances
        
    Returns:
        Concatenated observations from all sensors
        
    Example:
        >>> observations = get_sensor_observations(sensors)
        >>> print(f"Observation shape: {observations.shape}")  # (24,) for 3 sensors
    """
    observations = []
    
    # Concatenate firing rates from all sensor types
    for sensor_type in ["red", "green", "purple"]:
        if sensor_type in sensors:
            firing_rates = sensors[sensor_type].get_history_arrays()['firingrate']
            observations.append(firing_rates)
    
    # Add distance measurements from the first sensor
    if sensors:
        first_sensor = list(sensors.values())[0]
        distances = first_sensor.get_history_arrays()['dists']
        observations.append(distances)
    
    return np.concatenate(observations, axis=1)


def get_sensor_observation_dim(config: SensorConfig) -> int:
    """
    Calculate the total observation dimension for a sensor configuration.
    
    Args:
        config: Sensor configuration
        
    Returns:
        Total observation dimension
    """
    # Each sensor contributes 6 values (for 6 beams)
    # Plus 6 distance values from one sensor
    return len(config.sensor_types) * config.num_beams + config.num_beams
