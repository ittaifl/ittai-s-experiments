"""
Model Module
============

Neural agent models with load/save capabilities.
Implements the NextStepRNN for predictive learning and place cell emergence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import os


@dataclass
class ModelConfig:
    """Configuration for neural network models."""
    
    obs_dim: int = 24
    act_dim: int = 8
    hidden_dim: int = 100
    dropout_rate: float = 0.15
    noise_std: float = 0.03
    epsilon: float = 1e-2
    activation: str = "norm_relu"  # "norm_relu", "hard_sigmoid", "tanh"
    init_method: str = "levinstein"  # "levinstein", "recanatesi"


class NormReLU(nn.Module):
    """
    Normalized ReLU activation as described in Levinstein et al.
    
    Applies layer normalization followed by ReLU with learnable bias and noise.
    """
    
    def __init__(self, hidden_size: int, epsilon: float = 1e-2, noise_std: float = 0.03):
        """
        Initialize NormReLU activation.
        
        Args:
            hidden_size: Size of hidden layer
            epsilon: Small constant for numerical stability
            noise_std: Standard deviation of Gaussian noise
        """
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.epsilon = epsilon
        self.noise_std = noise_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply normalized ReLU activation."""
        x_norm = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.epsilon)
        noise = torch.randn_like(x) * self.noise_std
        return F.relu(x_norm + self.bias + noise)


class HardSigmoid(nn.Module):
    """
    Hard sigmoid activation as described in Recanatesi et al.
    
    Applies piecewise linear approximation to sigmoid: max(0, min(1, 0.2*x + 0.5))
    """
    
    def __init__(self):
        """Initialize HardSigmoid activation."""
        super(HardSigmoid, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply hard sigmoid activation."""
        return torch.clamp(0.2 * x + 0.5, min=0.0, max=1.0)


class NextStepRNN(nn.Module):
    """
    Recurrent Neural Network for next-step prediction learning.
    
    This model learns to predict the next sensory observation given the current
    observation and action. Through this predictive learning, spatially-tuned
    place cells emerge in the hidden layer.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the NextStepRNN.
        
        Args:
            config: Model configuration object
            
        Example:
            >>> config = ModelConfig(obs_dim=24, act_dim=8, hidden_dim=100)
            >>> model = NextStepRNN(config)
        """
        super().__init__()
        
        # Store configuration
        self.config = config
        self.obs_dim = config.obs_dim
        self.act_dim = config.act_dim
        self.hidden_dim = config.hidden_dim
        
        # Define layers
        self.W_in = nn.Linear(config.obs_dim, config.hidden_dim, bias=False)
        self.W_act = nn.Linear(config.act_dim, config.hidden_dim, bias=False)
        self.W_rec = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.W_out = nn.Linear(config.hidden_dim, config.obs_dim)
        self.beta = nn.Parameter(torch.zeros(1))
        
        # Activation functions
        self.norm_relu = NormReLU(config.hidden_dim, config.epsilon, config.noise_std)
        self.hard_sigmoid = HardSigmoid()
        self.dropout = nn.Dropout(p=config.dropout_rate)
        
        # Select activation function
        if config.activation == "norm_relu":
            self.g = self.norm_relu
        elif config.activation == "hard_sigmoid":
            self.g = self.hard_sigmoid
        elif config.activation == "tanh":
            self.g = torch.tanh
        else:
            raise ValueError(f"Unknown activation: {config.activation}")
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize network weights according to specified method."""
        if self.config.init_method == "levinstein":
            self._init_weights_levinstein()
        elif self.config.init_method == "recanatesi":
            self._init_weights_recanatesi()
        else:
            raise ValueError(f"Unknown init method: {self.config.init_method}")
    
    def _init_weights_levinstein(self):
        """Initialize weights as described in Levinstein et al."""
        tau = 2.0
        k_in = 1.0 / np.sqrt(self.obs_dim + self.act_dim)
        k_out = 1.0 / np.sqrt(self.hidden_dim)
        
        # Initialize input and action weights
        init.uniform_(self.W_in.weight, -k_in, k_in)
        init.uniform_(self.W_act.weight, -k_in, k_in)
        init.uniform_(self.W_out.weight, -k_out, k_out)
        
        # Identity-initialized + uniform recurrent weights
        W_rec_data = torch.empty(self.hidden_dim, self.hidden_dim)
        init.uniform_(W_rec_data, -k_out, k_out)
        identity_boost = torch.eye(self.hidden_dim) * (1 - 1 / tau)
        W_rec_data += identity_boost
        self.W_rec.weight.data = W_rec_data
    
    def _init_weights_recanatesi(self):
        """Initialize weights as described in Recanatesi et al."""
        # Initialize recurrent weights to identity matrix
        nn.init.eye_(self.W_rec.weight)
        
        # Initialize other weights to normal distribution
        nn.init.normal_(self.W_in.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.W_act.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.W_out.weight, mean=0.0, std=0.02)
    
    def forward(
        self, 
        obs_seq: torch.Tensor, 
        act_seq: torch.Tensor, 
        return_hidden: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the network.
        
        Args:
            obs_seq: Observation sequence (T, B, obs_dim)
            act_seq: Action sequence (T, B, act_dim)
            return_hidden: Whether to return hidden states
            
        Returns:
            Tuple of (outputs, hidden_states). Hidden states only returned if
            return_hidden=True.
            
        Example:
            >>> outputs = model(obs_seq, act_seq)
            >>> outputs, hiddens = model(obs_seq, act_seq, return_hidden=True)
        """
        T, B, _ = obs_seq.size()
        device = obs_seq.device
        
        # Initialize hidden state and output
        h = torch.zeros(B, self.hidden_dim, device=device)
        outputs, hiddens = [], []
        
        # Process each timestep
        for t in range(T):
            # Compute hidden state update
            o_in = self.W_in(obs_seq[t, :, :])      # Observation input
            a_in = self.W_act(act_seq[t, :, :])     # Action input  
            h_in = self.W_rec(h)                    # Recurrent input
            bias = self.beta                        # Learnable bias
            
            # Update hidden state: h = g(W_in*obs + W_act*act + W_rec*h + bias)
            h = self.g(o_in + a_in + h_in + bias)
            
            if return_hidden:
                hiddens.append(h.detach().cpu())
            
            # Compute output (next observation prediction)
            y = torch.sigmoid(self.W_out(h))
            outputs.append(y)
        
        # Stack outputs
        outputs = torch.stack(outputs)  # (T, B, obs_dim)
        
        if return_hidden:
            hiddens = torch.stack(hiddens)  # (T, B, hidden_dim)
            return outputs, hiddens
        
        return outputs, None
    
    def extract_place_cells(
        self, 
        obs_seq: torch.Tensor, 
        act_seq: torch.Tensor,
        positions: np.ndarray,
        nbins: int = 32
    ) -> np.ndarray:
        """
        Extract place cell activity maps from the hidden layer.
        
        Args:
            obs_seq: Observation sequence
            act_seq: Action sequence
            positions: Agent positions (T, 2)
            nbins: Number of spatial bins for activity maps
            
        Returns:
            Activity maps with shape (hidden_dim, nbins, nbins)
        """
        self.eval()
        
        with torch.no_grad():
            _, hiddens = self.forward(obs_seq, act_seq, return_hidden=True)
        
        if hiddens is None:
            raise ValueError("Hidden states not returned")
        
        # Convert to numpy and reshape
        hidden_activity = hiddens.numpy().reshape(hiddens.shape[0], hiddens.shape[2])
        T = hidden_activity.shape[0]
        
        # Initialize activity maps
        activity_map = np.zeros((self.hidden_dim, nbins, nbins))
        counts = np.zeros((nbins, nbins))
        
        # Accumulate activity for each spatial bin
        for t in range(T):
            x_bin = int(positions[t, 0] * nbins)
            y_bin = int(positions[t, 1] * nbins)
            
            if 0 <= x_bin < nbins and 0 <= y_bin < nbins:
                activity_map[:, y_bin, x_bin] += hidden_activity[t]
                counts[y_bin, x_bin] += 1
        
        # Normalize to get average firing rates
        for i in range(self.hidden_dim):
            activity_map[i] /= (counts + 1e-5)
        
        return activity_map


def load_model(filepath: str, config: ModelConfig, device: str = "cpu") -> NextStepRNN:
    """
    Load a trained model from disk.
    
    Args:
        filepath: Path to model checkpoint
        config: Model configuration
        device: Device to load model on
        
    Returns:
        Loaded NextStepRNN model
        
    Example:
        >>> model = load_model("model.pth", config, "cuda")
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    model = NextStepRNN(config)
    
    # Load state dict
    state_dict = torch.load(filepath, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    
    print(f"Model loaded from {filepath}")
    return model


def save_model(model: NextStepRNN, filepath: str) -> None:
    """
    Save a trained model to disk.
    
    Args:
        model: Trained NextStepRNN model
        filepath: Path to save model
        
    Example:
        >>> save_model(model, "model_final.pth")
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def create_model(
    obs_dim: int,
    act_dim: int, 
    hidden_dim: int = 100,
    **kwargs
) -> NextStepRNN:
    """
    Create a new NextStepRNN model with specified dimensions.
    
    Args:
        obs_dim: Observation dimension
        act_dim: Action dimension
        hidden_dim: Hidden layer dimension
        **kwargs: Additional configuration parameters
        
    Returns:
        Initialized NextStepRNN model
        
    Example:
        >>> model = create_model(obs_dim=24, act_dim=8, hidden_dim=100)
    """
    config = ModelConfig(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=hidden_dim,
        **kwargs
    )
    
    return NextStepRNN(config)


def count_parameters(model: NextStepRNN) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: NextStepRNN model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: NextStepRNN) -> str:
    """
    Get a summary string for the model architecture.
    
    Args:
        model: NextStepRNN model
        
    Returns:
        Summary string
    """
    n_params = count_parameters(model)
    
    summary = f"""
NextStepRNN Model Summary:
========================
Observation dim: {model.obs_dim}
Action dim: {model.act_dim}
Hidden dim: {model.hidden_dim}
Activation: {model.config.activation}
Initialization: {model.config.init_method}
Dropout rate: {model.config.dropout_rate}
Total parameters: {n_params:,}

Layer dimensions:
- W_in: {list(model.W_in.weight.shape)}
- W_act: {list(model.W_act.weight.shape)}
- W_rec: {list(model.W_rec.weight.shape)}
- W_out: {list(model.W_out.weight.shape)}
"""
    
    return summary
