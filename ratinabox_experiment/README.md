# RatInABox Experiment Package

A modular Python package for studying structured latent representations through predictive learning, based on [Recanatesi et al. (2021)](https://www.nature.com/articles/s41467-021-21696-1).

## 🚀 Features

- **Modular Architecture**: Clean separation of environment, sensors, data, models, training, and plotting
- **CLI Interface**: Full command-line support for both local and Google Colab environments
- **Google Drive Integration**: Seamless integration with Google Drive for data and model storage
- **Place Cell Analysis**: Automatic extraction and visualization of place-cell-like activity
- **Model Comparison**: Side-by-side comparison of predictive vs autoencoding models
- **Checkpoint Support**: Resume training from saved checkpoints
- **Flexible Configuration**: JSON/YAML configuration files with CLI overrides

## 📁 Project Structure

```
ratinabox_experiment/
├── ratinabox/                    # Main package
│   ├── __init__.py              # Package initialization
│   ├── env.py                   # Environment creation & parameterization
│   ├── sensors.py               # Sensor suite definition & configuration
│   ├── data.py                  # Data loading & trajectory generation
│   ├── model.py                 # Neural agent models & utilities
│   ├── train.py                 # Training loops & checkpointing
│   ├── plot.py                  # Visualization & comparison tools
│   └── utils.py                 # Common utilities & Drive integration
├── scripts/
│   ├── run_local.py             # Local execution script
│   └── run_colab.py             # Google Colab execution script
└── README.md                    # This file
```

## 🔧 Installation

### Local Installation

```bash
# Clone or download the repository
cd ratinabox_experiment

# Install dependencies
pip install torch torchvision matplotlib numpy scipy scikit-learn tqdm pyyaml psutil
pip install ratinabox

# Optional: Install Jupyter for notebook support
pip install jupyter
```

### Google Colab Installation

```python
# In a Colab cell:
!git clone <your-repo-url> /content/ratinabox_experiment
%cd /content/ratinabox_experiment
!pip install ratinabox pyyaml psutil

# Mount Google Drive (done automatically by run_colab.py)
from google.colab import drive
drive.mount('/content/drive')
```

## 🎯 Quick Start

### Local Usage

```bash
# Basic training with default parameters
python scripts/run_local.py --output-dir ./outputs

# Train with larger environment
python scripts/run_local.py \
    --env-width 2.0 \
    --env-height 2.0 \
    --hidden-dim 200 \
    --num-epochs 1000 \
    --output-dir ./outputs/large_env

# Resume from checkpoint
python scripts/run_local.py \
    --pretrained-weights ./outputs/checkpoints/model.pth \
    --output-dir ./outputs/resumed

# Compare predictive vs autoencoding
python scripts/run_local.py \
    --mode compare \
    --model1-weights ./predictive_model.pth \
    --model2-weights ./autoencoding_model.pth \
    --comparison-labels "Predictive,Autoencoding" \
    --output-dir ./comparison
```

### Google Colab Usage

```python
# In a Colab cell:
!python scripts/run_colab.py --output-dir "MyDrive/experiments/run1"

# Train with custom parameters
!python scripts/run_colab.py \
    --env-width 2.0 \
    --env-height 2.0 \
    --hidden-dim 200 \
    --num-epochs 1000 \
    --output-dir "MyDrive/experiments/large_env"

# Resume training from Drive
!python scripts/run_colab.py \
    --pretrained-weights "MyDrive/experiments/run1/checkpoints/model_final.pth" \
    --output-dir "MyDrive/experiments/run1_continued"

# Compare models stored on Drive
!python scripts/run_colab.py \
    --mode compare \
    --model1-weights "MyDrive/experiments/predictive/model_final.pth" \
    --model2-weights "MyDrive/experiments/autoencoding/model_final.pth" \
    --output-dir "MyDrive/experiments/comparison"
```

## 📋 Configuration

### Configuration Files

Create a `config.json` file to specify parameters:

```json
{
  "environment": {
    "width": 1.0,
    "height": 1.0,
    "grid_size": 64,
    "num_wall_objects": 12,
    "object_types": [0, 1, 2],
    "seed": 42
  },
  "model": {
    "hidden_dim": 100,
    "activation": "norm_relu",
    "init_method": "levinstein"
  },
  "training": {
    "num_epochs": 800,
    "learning_rate": 0.01,
    "batch_size": 128
  }
}
```

Use with: `python scripts/run_local.py --config config.json --output-dir ./outputs`

### CLI Parameters

#### Environment Parameters
- `--env-width`, `--env-height`: Environment dimensions (default: 1.0, 1.0)
- `--grid-size`: Navigation grid resolution (default: 64)
- `--num-wall-objects`: Objects per type on walls (default: 12)

#### Model Parameters
- `--hidden-dim`: Hidden layer size (default: 100)
- `--activation`: Activation function (norm_relu, hard_sigmoid, tanh)
- `--pretrained-weights`: Path to pretrained model

#### Training Parameters
- `--num-epochs`: Training epochs (default: 800)
- `--learning-rate`: Learning rate (default: 0.01)
- `--training-mode`: predictive or autoencoding (default: predictive)
- `--batch-size`: Batch size (default: 128)

#### Data Parameters
- `--n-trials`: Training trials (default: 1000)
- `--timesteps`: Steps per trial (default: 100)
- `--train-data`, `--test-data`: Paths to cached data files

## 🧠 Model Architecture

The package implements a Recurrent Neural Network (RNN) for next-step prediction:

```
h_t = g(W_in * obs_t + W_act * act_t + W_rec * h_{t-1} + beta)
pred_t = sigmoid(W_out * h_t)
```

Where:
- `obs_t`: Current sensory observations (24D: RGB + distance sensors)
- `act_t`: Current action (8D: one-hot encoding of movement direction)
- `h_t`: Hidden state (100D by default)
- `g`: Activation function (NormReLU, HardSigmoid, or Tanh)

### Training Modes

1. **Predictive Learning** (default): Learn to predict next observation
   - Develops spatially-tuned place cells in hidden layer
   - Input: current observation + action → Output: next observation

2. **Autoencoding** (control): Learn to reconstruct current observation
   - Does NOT develop place cells
   - Input: current observation → Output: same observation

## 📊 Visualization

The package generates several types of plots:

### Training Curves
- Training and validation loss over epochs
- Learning rate schedule visualization

### Place Cell Maps
- 2D activity maps for each hidden unit
- Spatial tuning visualization
- Color-coded firing rate intensity

### Model Comparison
- Side-by-side place cell maps
- Difference plots highlighting changes
- Quantitative metrics

### Trajectory Overlays
- Agent path overlaid on place cell activity
- Demonstrates spatial coding emergence

## 🔬 Example Experiments

### Experiment 1: Basic Place Cell Emergence

```bash
# Train predictive model
python scripts/run_local.py \
    --training-mode predictive \
    --hidden-dim 100 \
    --num-epochs 800 \
    --output-dir ./exp1_predictive

# Train autoencoding control
python scripts/run_local.py \
    --training-mode autoencoding \
    --hidden-dim 100 \
    --num-epochs 800 \
    --output-dir ./exp1_autoencoding

# Compare results
python scripts/run_local.py \
    --mode compare \
    --model1-weights ./exp1_predictive/checkpoints/model_final.pth \
    --model2-weights ./exp1_autoencoding/checkpoints/model_final.pth \
    --output-dir ./exp1_comparison
```

### Experiment 2: Environment Size Effects

```bash
# Small environment
python scripts/run_local.py \
    --env-width 1.0 --env-height 1.0 \
    --output-dir ./exp2_small

# Large environment  
python scripts/run_local.py \
    --env-width 3.0 --env-height 3.0 \
    --grid-size 128 \
    --output-dir ./exp2_large

# Compare environments
python scripts/run_local.py \
    --mode compare \
    --model1-weights ./exp2_small/checkpoints/model_final.pth \
    --model2-weights ./exp2_large/checkpoints/model_final.pth \
    --comparison-labels "Small Env,Large Env" \
    --output-dir ./exp2_comparison
```

### Experiment 3: Architecture Comparison

```bash
# Small network
python scripts/run_local.py \
    --hidden-dim 50 \
    --output-dir ./exp3_small_net

# Large network
python scripts/run_local.py \
    --hidden-dim 200 \
    --output-dir ./exp3_large_net
```

## 🔧 Advanced Usage

### Custom Data Generation

```python
from ratinabox import DataConfig, EnvironmentConfig, SensorConfig, RatDataLoader

# Create configurations
data_config = DataConfig(n_trials=2000, timesteps=200)
env_config = EnvironmentConfig(width=2.0, height=2.0)
sensor_config = SensorConfig(sensor_types=["red", "green"])

# Generate data
loader = RatDataLoader(data_config)
obs_array, act_array = loader.generate_training_data(env_config, sensor_config)

# Save for later use
loader.save_data({"observations": obs_array, "actions": act_array}, "custom_data.npz")
```

### Custom Model Training

```python
from ratinabox import create_model, ModelConfig, TrainingConfig, train_model

# Create model
model_config = ModelConfig(hidden_dim=150, activation="hard_sigmoid")
model = create_model(24, 8, 150, activation="hard_sigmoid")

# Custom training
training_config = TrainingConfig(
    num_epochs=1000,
    learning_rate=0.005,
    patience=10
)

trained_model, losses = train_model(model, train_loader, val_loader, training_config, model_config)
```

### Place Cell Analysis

```python
# Extract place cell activity
activity_maps = model.extract_place_cells(obs_seq, act_seq, positions, nbins=64)

# Find spatially tuned cells
spatial_info = calculate_spatial_information(activity_maps, positions)
place_cells = np.where(spatial_info > threshold)[0]

# Visualize specific cells
plot_place_cells(activity_maps[place_cells], plot_config, "Significant Place Cells")
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors in Colab**
   ```python
   # Add the package to Python path
   import sys
   sys.path.append('/content/ratinabox_experiment')
   ```

2. **Google Drive Mount Issues**
   ```python
   # Manual mount
   from google.colab import drive
   drive.mount('/content/drive', force_remount=True)
   ```

3. **Memory Issues with Large Datasets**
   ```bash
   # Reduce data size or batch size
   python scripts/run_local.py --n-trials 500 --batch-size 64
   ```

4. **CUDA Out of Memory**
   ```bash
   # Force CPU usage
   python scripts/run_local.py --device cpu
   ```

### Performance Tips

- Use GPU acceleration when available (`--device auto`)
- Cache generated data to avoid regeneration (`--train-data`, `--test-data`)
- Use smaller batch sizes for large models (`--batch-size 64`)
- Save checkpoints frequently (`--save-every 25`)

## 📚 References

- **Primary Paper**: Recanatesi, S., Pereira-Obilinovic, U., Mancusi, M., Mehta, M., Cranmer, K., & Fiete, I. (2021). Predictive learning as a network mechanism for extracting low-dimensional latent space representations. *Nature Communications*, 12(1), 1-13.

- **Supplementary**: Levinstein, Y., Swaminathan, A., & Fiete, I. (2024). Sequential predictive learning is a unifying theory for hippocampal representation and replay. *bioRxiv*.

- **RatInABox**: Geometric navigation toolkit - [GitHub Repository](https://github.com/RatInABox-Lab/RatInABox)

## 📄 License

This project is for educational and research purposes. Please cite the original papers when using this code.

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📧 Support

For questions or issues:

1. Check the troubleshooting section above
2. Review the original papers for theoretical background
3. Create an issue with detailed error information
