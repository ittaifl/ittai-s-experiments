#!/usr/bin/env python3
"""
Local Runner Script
===================

Thin wrapper to call ratinabox training via CLI locally.

Usage Examples:
    # Basic training with default parameters
    python run_local.py --output-dir ./outputs

    # Train with custom environment size and sensor types
    python run_local.py --env-width 2.0 --env-height 2.0 --sensor-types red,green --output-dir ./outputs

    # Resume training from checkpoint
    python run_local.py --pretrained-weights ./checkpoints/model.pth --output-dir ./outputs

    # Load custom configuration
    python run_local.py --config config.json --output-dir ./outputs

    # Generate comparison plots
    python run_local.py --mode compare --model1-dir ./outputs1 --model2-dir ./outputs2 --output-dir ./comparison
"""

import argparse
import sys
import os
from pathlib import Path

# Add ratinabox package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ratinabox import (
    create_environment, EnvironmentConfig,
    create_sensor_suite, SensorConfig, 
    load_training_data, load_test_data, DataConfig,
    create_model, train_model, load_model, save_model, ModelConfig, TrainingConfig,
    plot_place_cells, compare_place_cells, plot_training_curves, PlotConfig,
    setup_logging, load_config, save_config, create_default_config, merge_configs,
    set_random_seeds, get_device, ExperimentTimer
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="RatInABox Experiment - Local Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Mode selection
    parser.add_argument(
        "--mode", 
        choices=["train", "compare", "plot"],
        default="train",
        help="Execution mode: train model, compare models, or generate plots"
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON/YAML configuration file"
    )
    
    # Environment parameters
    env_group = parser.add_argument_group("Environment")
    env_group.add_argument("--env-width", type=float, default=1.0, help="Environment width")
    env_group.add_argument("--env-height", type=float, default=1.0, help="Environment height")
    env_group.add_argument("--grid-size", type=int, default=64, help="Navigation grid size")
    env_group.add_argument("--num-wall-objects", type=int, default=12, help="Number of wall objects per type")
    
    # Sensor parameters
    sensor_group = parser.add_argument_group("Sensors")
    sensor_group.add_argument(
        "--sensor-types",
        type=str,
        default="red,green,purple",
        help="Comma-separated list of sensor types"
    )
    sensor_group.add_argument(
        "--sensor-params",
        type=str,
        help="Path to JSON file with detailed sensor parameters"
    )
    
    # Data parameters
    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--train-data", type=str, help="Path to training data file")
    data_group.add_argument("--test-data", type=str, help="Path to test data file")
    data_group.add_argument("--n-trials", type=int, default=1000, help="Number of training trials")
    data_group.add_argument("--timesteps", type=int, default=100, help="Timesteps per trial")
    data_group.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    data_group.add_argument("--force-regenerate", action="store_true", help="Force data regeneration")
    
    # Model parameters
    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--hidden-dim", type=int, default=100, help="Hidden layer dimension")
    model_group.add_argument("--activation", choices=["norm_relu", "hard_sigmoid", "tanh"], 
                           default="norm_relu", help="Activation function")
    model_group.add_argument("--pretrained-weights", type=str, help="Path to pretrained model weights")
    
    # Training parameters
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--num-epochs", type=int, default=800, help="Number of training epochs")
    train_group.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate")
    train_group.add_argument("--training-mode", choices=["predictive", "autoencoding"], 
                           default="predictive", help="Training mode")
    
    # Comparison parameters (for compare mode)
    compare_group = parser.add_argument_group("Comparison")
    compare_group.add_argument("--model1-dir", type=str, help="Directory containing first model")
    compare_group.add_argument("--model2-dir", type=str, help="Directory containing second model")
    compare_group.add_argument("--model1-weights", type=str, help="Path to first model weights")
    compare_group.add_argument("--model2-weights", type=str, help="Path to second model weights")
    compare_group.add_argument("--comparison-labels", type=str, default="Model 1,Model 2",
                             help="Comma-separated labels for comparison")
    
    # Output parameters
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for models, plots, and logs"
    )
    output_group.add_argument("--save-plots", action="store_true", default=True, help="Save plots to disk")
    output_group.add_argument("--show-plots", action="store_true", default=True, help="Show plots inline")
    
    # General parameters
    general_group = parser.add_argument_group("General")
    general_group.add_argument("--seed", type=int, default=42, help="Random seed")
    general_group.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Computation device")
    general_group.add_argument("--verbose", action="store_true", default=True, help="Verbose output")
    general_group.add_argument("--log-file", type=str, help="Path to log file")
    
    return parser.parse_args()


def create_configs_from_args(args):
    """Create configuration objects from command line arguments."""
    # Load base configuration
    if args.config:
        base_config = load_config(args.config)
    else:
        base_config = create_default_config()
    
    # Override with command line arguments
    env_config = EnvironmentConfig(
        width=args.env_width,
        height=args.env_height,
        grid_size=args.grid_size,
        num_wall_objects=args.num_wall_objects,
        seed=args.seed
    )
    
    sensor_config = SensorConfig(
        sensor_types=args.sensor_types.split(','),
        **base_config.get("sensors", {})
    )
    
    data_config = DataConfig(
        n_trials=args.n_trials,
        timesteps=args.timesteps,
        batch_size=args.batch_size,
        train_data_path=args.train_data,
        test_data_path=args.test_data,
        seed=args.seed,
        **base_config.get("data", {})
    )
    
    obs_dim = len(sensor_config.sensor_types) * 6 + 6  # RGB sensors + distances
    model_config = ModelConfig(
        obs_dim=obs_dim,
        act_dim=8,
        hidden_dim=args.hidden_dim,
        activation=args.activation,
        **base_config.get("model", {})
    )
    
    training_config = TrainingConfig(
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        checkpoint_dir=os.path.join(args.output_dir, "checkpoints"),
        resume_from=args.pretrained_weights,
        device=args.device,
        verbose=args.verbose,
        **base_config.get("training", {})
    )
    
    plot_config = PlotConfig(
        output_dir=os.path.join(args.output_dir, "plots"),
        show_inline=args.show_plots,
        save_plots=args.save_plots,
        **base_config.get("plotting", {})
    )
    
    return env_config, sensor_config, data_config, model_config, training_config, plot_config


def train_mode(args):
    """Execute training mode."""
    logger = setup_logging("INFO", log_file=args.log_file)
    logger.info("Starting training mode")
    
    # Setup
    set_random_seeds(args.seed)
    device = get_device() if args.device == "auto" else args.device
    
    # Create configurations
    env_config, sensor_config, data_config, model_config, training_config, plot_config = create_configs_from_args(args)
    
    # Save configuration
    config_path = os.path.join(args.output_dir, "config.json")
    full_config = {
        "environment": env_config.__dict__,
        "sensors": sensor_config.__dict__,
        "data": data_config.__dict__,
        "model": model_config.__dict__,
        "training": training_config.__dict__,
        "plotting": plot_config.__dict__
    }
    save_config(full_config, config_path)
    
    # Load or generate data
    with ExperimentTimer("Data loading", logger):
        if args.training_mode == "predictive":
            train_loader, val_loader = load_training_data(
                data_config, env_config, sensor_config, args.force_regenerate
            )
        else:  # autoencoding
            from ratinabox.data import create_autoencoding_data
            train_loader, val_loader = create_autoencoding_data(
                data_config, env_config, sensor_config
            )
    
    # Create or load model
    if args.pretrained_weights and os.path.exists(args.pretrained_weights):
        logger.info(f"Loading pretrained model from {args.pretrained_weights}")
        model = load_model(args.pretrained_weights, model_config, device)
    else:
        logger.info("Creating new model")
        model = create_model(model_config.obs_dim, model_config.act_dim, model_config.hidden_dim)
    
    # Train model
    with ExperimentTimer("Model training", logger):
        trained_model, train_losses = train_model(
            model, train_loader, val_loader, training_config, model_config
        )
    
    # Generate test data and extract place cells
    with ExperimentTimer("Place cell extraction", logger):
        obs_test, act_test, positions = load_test_data(
            data_config, env_config, sensor_config, args.force_regenerate
        )
        
        activity_maps = trained_model.extract_place_cells(
            obs_test[:-2, :, :], act_test[1:-1, :, :], positions
        )
    
    # Generate plots
    with ExperimentTimer("Plot generation", logger):
        # Training curves
        val_losses = [0] * len(train_losses)  # Placeholder - would need to extract from trainer
        plot_training_curves(train_losses, val_losses, plot_config, 
                           f"Training Progress - {args.training_mode.title()}", 
                           f"training_curves_{args.training_mode}")
        
        # Place cell maps
        plot_place_cells(activity_maps, plot_config,
                        f"Place Cells - {args.training_mode.title()}",
                        f"place_cells_{args.training_mode}")
    
    logger.info(f"Training completed. Results saved to {args.output_dir}")


def compare_mode(args):
    """Execute comparison mode."""
    logger = setup_logging("INFO", log_file=args.log_file)
    logger.info("Starting comparison mode")
    
    if not (args.model1_weights and args.model2_weights):
        logger.error("Comparison mode requires --model1-weights and --model2-weights")
        sys.exit(1)
    
    # Create configurations (using defaults for comparison)
    env_config, sensor_config, data_config, model_config, _, plot_config = create_configs_from_args(args)
    
    # Load models
    model1 = load_model(args.model1_weights, model_config)
    model2 = load_model(args.model2_weights, model_config)
    
    # Generate test data
    obs_test, act_test, positions = load_test_data(
        data_config, env_config, sensor_config
    )
    
    # Extract place cells
    activity_maps_1 = model1.extract_place_cells(
        obs_test[:-2, :, :], act_test[1:-1, :, :], positions
    )
    activity_maps_2 = model2.extract_place_cells(
        obs_test[:-2, :, :], act_test[1:-1, :, :], positions
    )
    
    # Generate comparison plots
    labels = tuple(args.comparison_labels.split(','))
    compare_place_cells(
        activity_maps_1, activity_maps_2, plot_config,
        labels, "place_cell_comparison"
    )
    
    logger.info(f"Comparison completed. Results saved to {args.output_dir}")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Execute based on mode
    if args.mode == "train":
        train_mode(args)
    elif args.mode == "compare":
        compare_mode(args)
    elif args.mode == "plot":
        # TODO: Implement standalone plotting mode
        print("Plot mode not yet implemented")
        sys.exit(1)
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
