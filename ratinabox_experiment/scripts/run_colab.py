#!/usr/bin/env python3
"""
Google Colab Runner Script
==========================

Runner script with Google Drive mount support and CLI entrypoint for Colab.

Usage Examples:
    # Basic training in Colab (auto-mounts Drive)
    !python run_colab.py --output-dir "MyDrive/experiments/run1"

    # Train with data from Drive
    !python run_colab.py --train-data "MyDrive/data/train.npz" --test-data "MyDrive/data/test.npz" --output-dir "MyDrive/experiments/run1"

    # Resume from Drive checkpoint
    !python run_colab.py --pretrained-weights "MyDrive/experiments/run1/checkpoints/model.pth" --output-dir "MyDrive/experiments/run1_continued"

    # Compare models from Drive
    !python run_colab.py --mode compare --model1-weights "MyDrive/experiments/predictive/model.pth" --model2-weights "MyDrive/experiments/autoencoding/model.pth" --output-dir "MyDrive/experiments/comparison"
"""

import argparse
import sys
import os
from pathlib import Path

# Google Colab specific imports and setup
def setup_colab_environment():
    """Setup Google Colab environment."""
    print("Setting up Google Colab environment...")
    
    # Mount Google Drive
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✓ Google Drive mounted successfully")
    except ImportError:
        print("⚠ Not running in Google Colab - skipping Drive mount")
        return False
    except Exception as e:
        print(f"✗ Failed to mount Google Drive: {e}")
        return False
    
    # Install required packages if not already installed
    try:
        import ratinabox
    except ImportError:
        print("Installing ratinabox...")
        os.system("pip install ratinabox")
    
    # Setup matplotlib for Colab
    import matplotlib.pyplot as plt
    plt.rcParams['figure.facecolor'] = 'white'
    
    return True


# Add current directory to path for imports
sys.path.insert(0, '/content')
if os.path.exists('/content/ratinabox_experiment'):
    sys.path.insert(0, '/content/ratinabox_experiment')

# Try to import the ratinabox package
try:
    from ratinabox import (
        create_environment, EnvironmentConfig,
        create_sensor_suite, SensorConfig, 
        load_training_data, load_test_data, DataConfig,
        create_model, train_model, load_model, save_model, ModelConfig, TrainingConfig,
        plot_place_cells, compare_place_cells, plot_training_curves, PlotConfig,
        setup_logging, load_config, save_config, create_default_config, merge_configs,
        set_random_seeds, get_device, ExperimentTimer, resolve_drive_path, mount_google_drive
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure the ratinabox package is available in the current directory")
    sys.exit(1)


def parse_args():
    """Parse command line arguments with Colab-specific defaults."""
    parser = argparse.ArgumentParser(
        description="RatInABox Experiment - Google Colab Runner",
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
        help="Path to JSON/YAML configuration file (can be Drive path like 'MyDrive/config.json')"
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
        help="Path to JSON file with detailed sensor parameters (can be Drive path)"
    )
    
    # Data parameters
    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--train-data", type=str, help="Path to training data file (can be Drive path)")
    data_group.add_argument("--test-data", type=str, help="Path to test data file (can be Drive path)")
    data_group.add_argument("--n-trials", type=int, default=1000, help="Number of training trials")
    data_group.add_argument("--timesteps", type=int, default=100, help="Timesteps per trial")
    data_group.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    data_group.add_argument("--force-regenerate", action="store_true", help="Force data regeneration")
    
    # Model parameters
    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--hidden-dim", type=int, default=100, help="Hidden layer dimension")
    model_group.add_argument("--activation", choices=["norm_relu", "hard_sigmoid", "tanh"], 
                           default="norm_relu", help="Activation function")
    model_group.add_argument("--pretrained-weights", type=str, help="Path to pretrained model weights (can be Drive path)")
    
    # Training parameters
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--num-epochs", type=int, default=800, help="Number of training epochs")
    train_group.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate")
    train_group.add_argument("--training-mode", choices=["predictive", "autoencoding"], 
                           default="predictive", help="Training mode")
    
    # Comparison parameters (for compare mode)
    compare_group = parser.add_argument_group("Comparison")
    compare_group.add_argument("--model1-dir", type=str, help="Directory containing first model (can be Drive path)")
    compare_group.add_argument("--model2-dir", type=str, help="Directory containing second model (can be Drive path)")
    compare_group.add_argument("--model1-weights", type=str, help="Path to first model weights (can be Drive path)")
    compare_group.add_argument("--model2-weights", type=str, help="Path to second model weights (can be Drive path)")
    compare_group.add_argument("--comparison-labels", type=str, default="Model 1,Model 2",
                             help="Comma-separated labels for comparison")
    
    # Output parameters (Colab-friendly defaults)
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output-dir",
        type=str,
        default="/content/drive/MyDrive/ratinabox_experiments/default_run",
        help="Output directory for models, plots, and logs (use 'MyDrive/path' for Drive storage)"
    )
    output_group.add_argument("--save-plots", action="store_true", default=True, help="Save plots to disk")
    output_group.add_argument("--show-plots", action="store_true", default=True, help="Show plots inline")
    
    # General parameters
    general_group = parser.add_argument_group("General")
    general_group.add_argument("--seed", type=int, default=42, help="Random seed")
    general_group.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Computation device")
    general_group.add_argument("--verbose", action="store_true", default=True, help="Verbose output")
    general_group.add_argument("--log-file", type=str, help="Path to log file (can be Drive path)")
    general_group.add_argument("--skip-drive-mount", action="store_true", help="Skip automatic Drive mounting")
    
    return parser.parse_args()


def resolve_all_paths(args):
    """Resolve all paths to handle Google Drive paths."""
    path_fields = [
        'config', 'train_data', 'test_data', 'sensor_params', 
        'pretrained_weights', 'model1_dir', 'model2_dir', 
        'model1_weights', 'model2_weights', 'output_dir', 'log_file'
    ]
    
    for field in path_fields:
        path = getattr(args, field, None)
        if path:
            resolved_path = resolve_drive_path(path)
            setattr(args, field, resolved_path)
    
    return args


def create_configs_from_args(args):
    """Create configuration objects from command line arguments."""
    # Load base configuration
    if args.config and os.path.exists(args.config):
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
    """Execute training mode in Colab."""
    print("🚀 Starting training mode in Google Colab")
    
    # Setup logging
    logger = setup_logging("INFO", log_file=args.log_file)
    logger.info("Starting training mode in Colab")
    
    # Setup
    set_random_seeds(args.seed)
    device = get_device() if args.device == "auto" else args.device
    
    # Create configurations
    env_config, sensor_config, data_config, model_config, training_config, plot_config = create_configs_from_args(args)
    
    # Save configuration to Drive
    os.makedirs(args.output_dir, exist_ok=True)
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
    print(f"📋 Configuration saved to {config_path}")
    
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
        print(f"📂 Loading pretrained model from {args.pretrained_weights}")
        model = load_model(args.pretrained_weights, model_config, device)
    else:
        print("🏗 Creating new model")
        model = create_model(model_config.obs_dim, model_config.act_dim, model_config.hidden_dim)
    
    print(f"🧠 Model summary: {model_config.hidden_dim} hidden units, {args.training_mode} mode")
    
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
        
        print("🗺 Extracting place cell activity maps...")
        activity_maps = trained_model.extract_place_cells(
            obs_test[:-2, :, :], act_test[1:-1, :, :], positions
        )
    
    # Generate plots
    with ExperimentTimer("Plot generation", logger):
        print("📊 Generating visualizations...")
        
        # Training curves (simplified for demo)
        val_losses = [0] * len(train_losses)  # Would need actual validation losses
        plot_training_curves(train_losses, val_losses, plot_config, 
                           f"Training Progress - {args.training_mode.title()}", 
                           f"training_curves_{args.training_mode}")
        
        # Place cell maps
        plot_place_cells(activity_maps, plot_config,
                        f"Place Cells - {args.training_mode.title()}",
                        f"place_cells_{args.training_mode}")
    
    print(f"✅ Training completed! Results saved to {args.output_dir}")
    print("📁 Output files:")
    print(f"   - Model: {os.path.join(training_config.checkpoint_dir, 'model_final.pth')}")
    print(f"   - Plots: {plot_config.output_dir}")
    print(f"   - Config: {config_path}")


def compare_mode(args):
    """Execute comparison mode in Colab."""
    print("🔍 Starting comparison mode in Google Colab")
    
    logger = setup_logging("INFO", log_file=args.log_file)
    logger.info("Starting comparison mode in Colab")
    
    if not (args.model1_weights and args.model2_weights):
        print("❌ Comparison mode requires --model1-weights and --model2-weights")
        sys.exit(1)
    
    # Create configurations (using defaults for comparison)
    env_config, sensor_config, data_config, model_config, _, plot_config = create_configs_from_args(args)
    
    # Load models
    print(f"📂 Loading model 1 from {args.model1_weights}")
    model1 = load_model(args.model1_weights, model_config)
    print(f"📂 Loading model 2 from {args.model2_weights}")
    model2 = load_model(args.model2_weights, model_config)
    
    # Generate test data
    print("🔄 Generating test data...")
    obs_test, act_test, positions = load_test_data(
        data_config, env_config, sensor_config
    )
    
    # Extract place cells
    print("🗺 Extracting place cells from both models...")
    activity_maps_1 = model1.extract_place_cells(
        obs_test[:-2, :, :], act_test[1:-1, :, :], positions
    )
    activity_maps_2 = model2.extract_place_cells(
        obs_test[:-2, :, :], act_test[1:-1, :, :], positions
    )
    
    # Generate comparison plots
    print("📊 Generating comparison plots...")
    labels = tuple(args.comparison_labels.split(','))
    compare_place_cells(
        activity_maps_1, activity_maps_2, plot_config,
        labels, "place_cell_comparison"
    )
    
    print(f"✅ Comparison completed! Results saved to {args.output_dir}")


def main():
    """Main entry point for Colab runner."""
    print("🌟 RatInABox Experiment - Google Colab Runner")
    print("=" * 50)
    
    # Parse arguments
    args = parse_args()
    
    # Setup Colab environment
    if not args.skip_drive_mount:
        colab_ready = setup_colab_environment()
        if not colab_ready and not args.skip_drive_mount:
            print("⚠ Colab setup failed, but continuing...")
    
    # Resolve Drive paths
    args = resolve_all_paths(args)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"📁 Output directory: {args.output_dir}")
    
    # Execute based on mode
    try:
        if args.mode == "train":
            train_mode(args)
        elif args.mode == "compare":
            compare_mode(args)
        elif args.mode == "plot":
            print("📊 Plot mode not yet implemented")
            sys.exit(1)
        else:
            print(f"❌ Unknown mode: {args.mode}")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
