"""
Plotting Module
===============

Generate before/after place-cell maps, difference plots, and training visualizations.
Supports both inline display (for Colab) and PNG export.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import os
from dataclasses import dataclass

from .model import NextStepRNN


@dataclass
class PlotConfig:
    """Configuration for plotting functions."""
    
    figsize: Tuple[int, int] = (12, 8)
    dpi: int = 100
    save_format: str = "png"
    colormap: str = "jet"
    interpolation: str = "gaussian"
    units_to_plot: int = 100
    grid_size: Tuple[int, int] = (10, 10)
    
    # Output settings
    output_dir: str = "./plots"
    show_inline: bool = True
    save_plots: bool = True


def plot_place_cells(
    activity_maps: np.ndarray,
    config: PlotConfig,
    title: str = "Place Cell Activity",
    filename: Optional[str] = None
) -> plt.Figure:
    """
    Plot place cell activity maps for multiple neurons.
    
    Args:
        activity_maps: Activity maps with shape (n_neurons, height, width)
        config: Plotting configuration
        title: Figure title
        filename: Optional filename to save plot
        
    Returns:
        Matplotlib figure object
        
    Example:
        >>> activity_maps = model.extract_place_cells(obs_seq, act_seq, positions)
        >>> fig = plot_place_cells(activity_maps, plot_config)
    """
    n_neurons = min(activity_maps.shape[0], config.units_to_plot)
    grid_h, grid_w = config.grid_size
    
    if n_neurons > grid_h * grid_w:
        print(f"Warning: Showing only {grid_h * grid_w} of {n_neurons} neurons")
        n_neurons = grid_h * grid_w
    
    # Create figure
    fig = plt.figure(figsize=config.figsize, dpi=config.dpi)
    fig.suptitle(title, fontsize=16)
    
    # Plot each neuron's activity map
    for i in range(n_neurons):
        ax = fig.add_subplot(grid_h, grid_w, i + 1)
        
        # Plot activity map
        im = ax.imshow(
            activity_maps[i], 
            origin='lower', 
            cmap=config.colormap,
            interpolation=config.interpolation
        )
        
        # Remove ticks and labels for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Optional: add neuron number as title
        # ax.set_title(f'Unit {i}', fontsize=8)
    
    plt.tight_layout()
    
    # Save plot if requested
    if config.save_plots and filename:
        save_path = os.path.join(config.output_dir, f"{filename}.{config.save_format}")
        os.makedirs(config.output_dir, exist_ok=True)
        fig.savefig(save_path, dpi=config.dpi, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
    
    # Show inline if requested
    if config.show_inline:
        plt.show()
    
    return fig


def compare_place_cells(
    activity_maps_1: np.ndarray,
    activity_maps_2: np.ndarray,
    config: PlotConfig,
    labels: Tuple[str, str] = ("Model 1", "Model 2"),
    filename: Optional[str] = None
) -> plt.Figure:
    """
    Compare place cell activity maps between two models or conditions.
    
    Args:
        activity_maps_1: First set of activity maps
        activity_maps_2: Second set of activity maps
        config: Plotting configuration
        labels: Labels for the two conditions
        filename: Optional filename to save plot
        
    Returns:
        Matplotlib figure object
        
    Example:
        >>> fig = compare_place_cells(
        ...     predictive_maps, autoencoding_maps, 
        ...     config, ("Predictive", "Autoencoding")
        ... )
    """
    n_neurons = min(
        activity_maps_1.shape[0], 
        activity_maps_2.shape[0], 
        config.units_to_plot
    )
    
    # Calculate difference maps
    difference_maps = activity_maps_1[:n_neurons] - activity_maps_2[:n_neurons]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(config.figsize[0] * 1.5, config.figsize[1]), dpi=config.dpi)
    gs = gridspec.GridSpec(config.grid_size[0], config.grid_size[1] * 3, figure=fig)
    
    fig.suptitle(f"Place Cell Comparison: {labels[0]} vs {labels[1]}", fontsize=16)
    
    for i in range(min(n_neurons, config.grid_size[0] * config.grid_size[1])):
        row = i // config.grid_size[1]
        col = i % config.grid_size[1]
        
        # First condition
        ax1 = fig.add_subplot(gs[row, col])
        ax1.imshow(
            activity_maps_1[i], 
            origin='lower', 
            cmap=config.colormap,
            interpolation=config.interpolation
        )
        ax1.set_xticks([])
        ax1.set_yticks([])
        if i < config.grid_size[1]:
            ax1.set_title(labels[0], fontsize=10)
        
        # Second condition
        ax2 = fig.add_subplot(gs[row, col + config.grid_size[1]])
        ax2.imshow(
            activity_maps_2[i], 
            origin='lower', 
            cmap=config.colormap,
            interpolation=config.interpolation
        )
        ax2.set_xticks([])
        ax2.set_yticks([])
        if i < config.grid_size[1]:
            ax2.set_title(labels[1], fontsize=10)
        
        # Difference
        ax3 = fig.add_subplot(gs[row, col + 2 * config.grid_size[1]])
        ax3.imshow(
            difference_maps[i], 
            origin='lower', 
            cmap='RdBu_r',  # Diverging colormap for differences
            interpolation=config.interpolation
        )
        ax3.set_xticks([])
        ax3.set_yticks([])
        if i < config.grid_size[1]:
            ax3.set_title("Difference", fontsize=10)
    
    plt.tight_layout()
    
    # Save plot if requested
    if config.save_plots and filename:
        save_path = os.path.join(config.output_dir, f"{filename}.{config.save_format}")
        os.makedirs(config.output_dir, exist_ok=True)
        fig.savefig(save_path, dpi=config.dpi, bbox_inches='tight')
        print(f"Comparison plot saved: {save_path}")
    
    # Show inline if requested
    if config.show_inline:
        plt.show()
    
    return fig


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    config: PlotConfig,
    title: str = "Training Progress",
    filename: Optional[str] = None
) -> plt.Figure:
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        config: Plotting configuration
        title: Figure title
        filename: Optional filename to save plot
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=config.dpi)
    
    epochs = range(1, len(train_losses) + 1)
    
    ax.plot(epochs, train_losses, label='Training Loss', linewidth=2)
    ax.plot(epochs, val_losses, label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save plot if requested
    if config.save_plots and filename:
        save_path = os.path.join(config.output_dir, f"{filename}.{config.save_format}")
        os.makedirs(config.output_dir, exist_ok=True)
        fig.savefig(save_path, dpi=config.dpi, bbox_inches='tight')
        print(f"Training curve saved: {save_path}")
    
    # Show inline if requested
    if config.show_inline:
        plt.show()
    
    return fig


def plot_latent_space_evolution(
    activity_maps_by_epoch: Dict[int, np.ndarray],
    config: PlotConfig,
    neuron_indices: Optional[List[int]] = None,
    filename: Optional[str] = None
) -> plt.Figure:
    """
    Plot the evolution of latent space across training epochs.
    
    Args:
        activity_maps_by_epoch: Dictionary mapping epoch -> activity maps
        config: Plotting configuration
        neuron_indices: Specific neurons to plot (if None, plot first few)
        filename: Optional filename to save plot
        
    Returns:
        Matplotlib figure object
    """
    epochs = sorted(activity_maps_by_epoch.keys())
    n_epochs = len(epochs)
    
    if neuron_indices is None:
        neuron_indices = list(range(min(10, activity_maps_by_epoch[epochs[0]].shape[0])))
    
    n_neurons = len(neuron_indices)
    
    # Create figure
    fig, axes = plt.subplots(
        n_neurons, n_epochs, 
        figsize=(2 * n_epochs, 2 * n_neurons), 
        dpi=config.dpi
    )
    
    if n_neurons == 1:
        axes = axes.reshape(1, -1)
    if n_epochs == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle("Latent Space Evolution Across Training", fontsize=14)
    
    for i, neuron_idx in enumerate(neuron_indices):
        for j, epoch in enumerate(epochs):
            ax = axes[i, j]
            
            activity_map = activity_maps_by_epoch[epoch][neuron_idx]
            ax.imshow(
                activity_map, 
                origin='lower', 
                cmap=config.colormap,
                interpolation=config.interpolation
            )
            
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Labels
            if i == 0:
                ax.set_title(f"Epoch {epoch}", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"Unit {neuron_idx}", fontsize=10)
    
    plt.tight_layout()
    
    # Save plot if requested
    if config.save_plots and filename:
        save_path = os.path.join(config.output_dir, f"{filename}.{config.save_format}")
        os.makedirs(config.output_dir, exist_ok=True)
        fig.savefig(save_path, dpi=config.dpi, bbox_inches='tight')
        print(f"Evolution plot saved: {save_path}")
    
    # Show inline if requested
    if config.show_inline:
        plt.show()
    
    return fig


def plot_trajectory_overlay(
    positions: np.ndarray,
    activity_maps: np.ndarray,
    config: PlotConfig,
    neuron_indices: Optional[List[int]] = None,
    filename: Optional[str] = None
) -> plt.Figure:
    """
    Plot agent trajectory overlaid with place cell firing.
    
    Args:
        positions: Agent positions (T, 2)
        activity_maps: Place cell activity maps
        config: Plotting configuration
        neuron_indices: Specific neurons to plot
        filename: Optional filename to save plot
        
    Returns:
        Matplotlib figure object
    """
    if neuron_indices is None:
        neuron_indices = list(range(min(6, activity_maps.shape[0])))
    
    n_neurons = len(neuron_indices)
    cols = min(3, n_neurons)
    rows = (n_neurons + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), dpi=config.dpi)
    
    if n_neurons == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    fig.suptitle("Trajectory with Place Cell Activity", fontsize=14)
    
    for i, neuron_idx in enumerate(neuron_indices):
        ax = axes[i] if i < len(axes) else None
        if ax is None:
            break
        
        # Plot place cell activity as background
        ax.imshow(
            activity_maps[neuron_idx], 
            origin='lower', 
            cmap=config.colormap,
            alpha=0.7,
            extent=[0, 1, 0, 1],
            interpolation=config.interpolation
        )
        
        # Overlay trajectory
        ax.plot(positions[:, 0], positions[:, 1], 'k-', alpha=0.5, linewidth=0.5)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"Unit {neuron_idx}")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
    
    # Hide unused subplots
    for i in range(n_neurons, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot if requested
    if config.save_plots and filename:
        save_path = os.path.join(config.output_dir, f"{filename}.{config.save_format}")
        os.makedirs(config.output_dir, exist_ok=True)
        fig.savefig(save_path, dpi=config.dpi, bbox_inches='tight')
        print(f"Trajectory overlay saved: {save_path}")
    
    # Show inline if requested
    if config.show_inline:
        plt.show()
    
    return fig


def save_plots(
    plots: Dict[str, plt.Figure],
    output_dir: str,
    format: str = "png"
) -> None:
    """
    Save multiple plots to disk.
    
    Args:
        plots: Dictionary mapping plot names to figures
        output_dir: Directory to save plots
        format: File format (png, pdf, svg, etc.)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for name, fig in plots.items():
        filepath = os.path.join(output_dir, f"{name}.{format}")
        fig.savefig(filepath, dpi=100, bbox_inches='tight')
        print(f"Saved: {filepath}")


def create_plot_config(
    output_dir: str = "./plots",
    show_inline: bool = True,
    save_plots: bool = True,
    **kwargs
) -> PlotConfig:
    """
    Create a plotting configuration with common defaults.
    
    Args:
        output_dir: Directory to save plots
        show_inline: Whether to show plots inline (for Colab)
        save_plots: Whether to save plots to disk
        **kwargs: Additional configuration parameters
        
    Returns:
        PlotConfig object
    """
    return PlotConfig(
        output_dir=output_dir,
        show_inline=show_inline,
        save_plots=save_plots,
        **kwargs
    )


def setup_colab_plotting():
    """
    Setup plotting for Google Colab environment.
    
    Call this function at the beginning of Colab notebooks.
    """
    try:
        from google.colab import files
        import matplotlib.pyplot as plt
        
        # Enable inline plotting
        plt.rcParams['figure.facecolor'] = 'white'
        print("Colab plotting setup complete")
        return True
    except ImportError:
        print("Not running in Colab, using standard matplotlib")
        return False
