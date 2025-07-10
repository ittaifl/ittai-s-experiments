#!/usr/bin/env python3
"""
Example Usage Script
====================

Demonstrates basic usage of the RatInABox experiment package.
"""

import sys
import os
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

from ratinabox import (
    EnvironmentConfig, SensorConfig, DataConfig, ModelConfig, TrainingConfig, PlotConfig,
    create_environment, create_sensor_suite, RatDataLoader,
    create_model, train_model, load_model, save_model,
    plot_place_cells, compare_place_cells, plot_training_curves,
    setup_logging, set_random_seeds, get_device, ExperimentTimer
)


def basic_training_example():
    """Example: Basic model training and place cell extraction."""
    print("🚀 Running basic training example...")
    
    # Setup
    logger = setup_logging("INFO")
    set_random_seeds(42)
    device = get_device()
    
    # Create configurations
    env_config = EnvironmentConfig(width=1.0, height=1.0, grid_size=32, seed=42)
    sensor_config = SensorConfig(sensor_types=["red", "green"])
    data_config = DataConfig(n_trials=100, timesteps=50, batch_size=32)
    model_config = ModelConfig(obs_dim=18, act_dim=8, hidden_dim=50)  # 2 sensors * 6 + 6 distances = 18
    training_config = TrainingConfig(num_epochs=50, learning_rate=0.01, verbose=True)
    plot_config = PlotConfig(output_dir="./example_plots", units_to_plot=25)
    
    # Load data
    with ExperimentTimer("Data generation", logger):
        loader = RatDataLoader(data_config)
        train_loader, val_loader = loader.load_training_data(data_config, env_config, sensor_config)
        obs_test, act_test, positions = loader.load_test_data(data_config, env_config, sensor_config)
    
    # Create and train model
    with ExperimentTimer("Model training", logger):
        model = create_model(model_config.obs_dim, model_config.act_dim, model_config.hidden_dim)
        trained_model, losses = train_model(model, train_loader, val_loader, training_config, model_config)
    
    # Extract place cells
    with ExperimentTimer("Place cell extraction", logger):
        activity_maps = trained_model.extract_place_cells(
            obs_test[:-2, :, :], act_test[1:-1, :, :], positions, nbins=16
        )
    
    # Generate plots
    with ExperimentTimer("Plotting", logger):
        plot_training_curves(losses, losses, plot_config, "Training Example", "training_example")
        plot_place_cells(activity_maps, plot_config, "Example Place Cells", "place_cells_example")
    
    print("✅ Basic training example completed!")
    return trained_model, activity_maps


def comparison_example():
    """Example: Compare predictive vs autoencoding models."""
    print("🔍 Running comparison example...")
    
    logger = setup_logging("INFO")
    set_random_seeds(42)
    
    # Small-scale configurations for quick demo
    env_config = EnvironmentConfig(width=1.0, height=1.0, grid_size=32, seed=42)
    sensor_config = SensorConfig(sensor_types=["red"])
    data_config = DataConfig(n_trials=50, timesteps=30, batch_size=16)
    model_config = ModelConfig(obs_dim=12, act_dim=8, hidden_dim=25)  # 1 sensor * 6 + 6 distances = 12
    training_config = TrainingConfig(num_epochs=20, learning_rate=0.01, verbose=False)
    plot_config = PlotConfig(output_dir="./comparison_plots", units_to_plot=16, grid_size=[4, 4])
    
    # Generate data once
    loader = RatDataLoader(data_config)
    
    # Train predictive model
    print("📈 Training predictive model...")
    train_loader, val_loader = loader.load_training_data(data_config, env_config, sensor_config)
    predictive_model = create_model(model_config.obs_dim, model_config.act_dim, model_config.hidden_dim)
    predictive_model, _ = train_model(predictive_model, train_loader, val_loader, training_config, model_config)
    
    # Train autoencoding model (reconstruct current input)
    print("🔄 Training autoencoding model...")
    from ratinabox.data import create_autoencoding_data
    auto_train_loader, auto_val_loader = create_autoencoding_data(data_config, env_config, sensor_config)
    autoencoding_model = create_model(model_config.obs_dim, model_config.act_dim, model_config.hidden_dim)
    autoencoding_model, _ = train_model(autoencoding_model, auto_train_loader, auto_val_loader, training_config, model_config)
    
    # Extract place cells from both models
    obs_test, act_test, positions = loader.load_test_data(data_config, env_config, sensor_config)
    
    predictive_maps = predictive_model.extract_place_cells(
        obs_test[:-2, :, :], act_test[1:-1, :, :], positions, nbins=16
    )
    autoencoding_maps = autoencoding_model.extract_place_cells(
        obs_test[:-2, :, :], act_test[1:-1, :, :], positions, nbins=16
    )
    
    # Compare models
    print("📊 Generating comparison plots...")
    compare_place_cells(
        predictive_maps, autoencoding_maps, plot_config,
        ("Predictive", "Autoencoding"), "model_comparison"
    )
    
    print("✅ Comparison example completed!")
    return predictive_maps, autoencoding_maps


def configuration_example():
    """Example: Using configuration files."""
    print("⚙️ Configuration example...")
    
    from ratinabox.utils import create_default_config, save_config, load_config, merge_configs
    
    # Create and save default configuration
    default_config = create_default_config()
    save_config(default_config, "example_config.json")
    print("📋 Saved default configuration to example_config.json")
    
    # Load and modify configuration
    config = load_config("example_config.json")
    
    # Override some settings
    user_overrides = {
        "model": {"hidden_dim": 150},
        "training": {"num_epochs": 1000, "learning_rate": 0.005},
        "environment": {"grid_size": 128}
    }
    
    merged_config = merge_configs(config, user_overrides)
    save_config(merged_config, "example_config_modified.json")
    print("📝 Saved modified configuration to example_config_modified.json")
    
    print("✅ Configuration example completed!")


def main():
    """Run all examples."""
    print("🌟 RatInABox Experiment Package Examples")
    print("=" * 50)
    
    try:
        # Create output directories
        os.makedirs("./example_plots", exist_ok=True)
        os.makedirs("./comparison_plots", exist_ok=True)
        
        # Run examples
        basic_training_example()
        print()
        comparison_example()
        print()
        configuration_example()
        
        print("\n🎉 All examples completed successfully!")
        print("📁 Check the following directories for outputs:")
        print("   - ./example_plots/ - Basic training plots")
        print("   - ./comparison_plots/ - Model comparison plots")
        print("   - example_config*.json - Configuration files")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        raise


if __name__ == "__main__":
    main()
