"""
Training Module
===============

Training loops with checkpoint support, learning rate scheduling, and validation.
Supports both predictive learning and autoencoding modes.
"""

import torch
import torch.nn as nn
from torch.optim import RMSprop
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
import os
import time
from tqdm import tqdm

from .model import NextStepRNN, save_model, load_model, ModelConfig


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    num_epochs: int = 800
    learning_rate: float = 0.01
    batch_size: int = 128
    patience: int = 8
    max_lr_reductions: int = 5
    lr_reduction_factor: float = 0.5
    min_improvement: float = 5e-5
    l1_lambda: float = 0.0
    
    # Checkpoint settings
    save_every: int = 50
    checkpoint_dir: str = "./checkpoints"
    resume_from: Optional[str] = None
    
    # Device settings
    device: str = "auto"  # "auto", "cpu", "cuda"
    
    # Logging
    log_every: int = 10
    verbose: bool = True


class ModelTrainer:
    """Trainer class for NextStepRNN models."""
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.device = self._setup_device()
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'epochs': []
        }
    
    def _setup_device(self) -> str:
        """Setup computation device."""
        if self.config.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.config.device
        
        if self.config.verbose:
            print(f"Using device: {device}")
        
        return device
    
    def train_model(
        self,
        model: NextStepRNN,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model_config: ModelConfig
    ) -> List[float]:
        """
        Train a NextStepRNN model.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            model_config: Model configuration for saving checkpoints
            
        Returns:
            List of training losses per epoch
            
        Example:
            >>> trainer = ModelTrainer(training_config)
            >>> losses = trainer.train_model(model, train_loader, val_loader, model_config)
        """
        # Move model to device
        model = model.to(self.device)
        
        # Setup optimizer and loss function
        optimizer = RMSprop(
            model.parameters(), 
            lr=self.config.learning_rate, 
            alpha=0.95, 
            eps=1e-8
        )
        loss_fn = nn.MSELoss()
        
        # Training state
        best_val_loss = float('inf')
        epochs_no_improve = 0
        lr_reductions = 0
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if self.config.resume_from:
            start_epoch = self._load_checkpoint(model, optimizer)
        
        if self.config.verbose:
            print(f"Starting training from epoch {start_epoch}")
            print(f"Training for {self.config.num_epochs} epochs")
        
        # Training loop
        for epoch in range(start_epoch, self.config.num_epochs):
            # Training phase
            train_loss = self._train_epoch(model, train_loader, optimizer, loss_fn)
            
            # Validation phase
            val_loss = self._validate_epoch(model, val_loader, loss_fn)
            
            # Update history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            self.training_history['epochs'].append(epoch)
            
            # Logging
            if epoch % self.config.log_every == 0 and self.config.verbose:
                print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, "
                      f"Val Loss = {val_loss:.6f}, LR = {optimizer.param_groups[0]['lr']:.6f}")
            
            # Learning rate scheduling
            if val_loss < best_val_loss - self.config.min_improvement:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
                if epochs_no_improve >= self.config.patience:
                    if lr_reductions < self.config.max_lr_reductions:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= self.config.lr_reduction_factor
                        
                        if self.config.verbose:
                            print(f"Reducing learning rate to {optimizer.param_groups[0]['lr']:.6f}")
                        
                        epochs_no_improve = 0
                        lr_reductions += 1
                    else:
                        if self.config.verbose:
                            print("Early stopping - maximum LR reductions reached")
                        break
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self._save_checkpoint(model, optimizer, epoch, model_config)
        
        # Save final model
        final_path = os.path.join(self.config.checkpoint_dir, "model_final.pth")
        save_model(model, final_path)
        
        if self.config.verbose:
            print(f"Training completed. Final model saved to {final_path}")
        
        return self.training_history['train_loss']
    
    def _train_epoch(
        self, 
        model: NextStepRNN, 
        train_loader: DataLoader, 
        optimizer: RMSprop, 
        loss_fn: nn.Module
    ) -> float:
        """Train for one epoch."""
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for obs_seq, act_seq, next_obs_seq in train_loader:
            # Move data to device and reshape
            obs_seq = obs_seq.permute(1, 0, 2).to(self.device)      # (T, B, D)
            act_seq = act_seq.permute(1, 0, 2).to(self.device)
            next_obs_seq = next_obs_seq.permute(1, 0, 2).to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            pred, _ = model(obs_seq, act_seq)
            
            # Compute loss
            mse_loss = loss_fn(pred, next_obs_seq)
            
            # Add L1 regularization if specified
            l1_loss = 0.0
            if self.config.l1_lambda > 0:
                for param in model.parameters():
                    l1_loss += torch.norm(param, 1)
                
                total_loss = mse_loss + self.config.l1_lambda * l1_loss
            else:
                total_loss = mse_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            num_batches += 1
        
        return epoch_loss / num_batches
    
    def _validate_epoch(
        self, 
        model: NextStepRNN, 
        val_loader: DataLoader, 
        loss_fn: nn.Module
    ) -> float:
        """Validate for one epoch."""
        model.eval()
        val_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for obs_seq, act_seq, next_obs_seq in val_loader:
                # Move data to device and reshape
                obs_seq = obs_seq.permute(1, 0, 2).to(self.device)
                act_seq = act_seq.permute(1, 0, 2).to(self.device)
                next_obs_seq = next_obs_seq.permute(1, 0, 2).to(self.device)
                
                # Forward pass
                pred, _ = model(obs_seq, act_seq)
                
                # Compute loss
                mse_loss = loss_fn(pred, next_obs_seq)
                
                # Add L1 regularization if specified
                l1_loss = 0.0
                if self.config.l1_lambda > 0:
                    for param in model.parameters():
                        l1_loss += torch.norm(param, 1)
                    
                    total_loss = mse_loss + self.config.l1_lambda * l1_loss
                else:
                    total_loss = mse_loss
                
                val_loss += total_loss.item()
                num_batches += 1
        
        return val_loss / num_batches
    
    def _save_checkpoint(
        self, 
        model: NextStepRNN, 
        optimizer: RMSprop, 
        epoch: int,
        model_config: ModelConfig
    ) -> None:
        """Save training checkpoint."""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_history': self.training_history,
            'model_config': model_config,
            'training_config': self.config
        }
        
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir, 
            f"checkpoint_epoch_{epoch+1}.pth"
        )
        torch.save(checkpoint, checkpoint_path)
        
        if self.config.verbose:
            print(f"Checkpoint saved: {checkpoint_path}")
    
    def _load_checkpoint(self, model: NextStepRNN, optimizer: RMSprop) -> int:
        """Load training checkpoint."""
        if not os.path.exists(self.config.resume_from):
            raise FileNotFoundError(f"Checkpoint not found: {self.config.resume_from}")
        
        checkpoint = torch.load(self.config.resume_from, map_location=self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
        
        epoch = checkpoint['epoch']
        
        if self.config.verbose:
            print(f"Resumed from checkpoint: {self.config.resume_from}")
            print(f"Resuming from epoch {epoch+1}")
        
        return epoch + 1


def train_model(
    model: NextStepRNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    model_config: ModelConfig
) -> Tuple[NextStepRNN, List[float]]:
    """
    Train a NextStepRNN model with the specified configuration.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        model_config: Model configuration
        
    Returns:
        Tuple of (trained_model, training_losses)
        
    Example:
        >>> trained_model, losses = train_model(
        ...     model, train_loader, val_loader, train_config, model_config
        ... )
    """
    trainer = ModelTrainer(config)
    losses = trainer.train_model(model, train_loader, val_loader, model_config)
    return model, losses


def create_training_config(
    num_epochs: int = 800,
    learning_rate: float = 0.01,
    checkpoint_dir: str = "./checkpoints",
    **kwargs
) -> TrainingConfig:
    """
    Create a training configuration with common defaults.
    
    Args:
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        checkpoint_dir: Directory to save checkpoints
        **kwargs: Additional configuration parameters
        
    Returns:
        TrainingConfig object
    """
    return TrainingConfig(
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        checkpoint_dir=checkpoint_dir,
        **kwargs
    )


def train_predictive_model(
    model: NextStepRNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    model_config: ModelConfig
) -> Tuple[NextStepRNN, List[float]]:
    """
    Train a model for predictive learning (predict next observation).
    
    This is the main training mode that leads to place cell emergence.
    """
    if config.verbose:
        print("Training in PREDICTIVE mode (next-step prediction)")
    
    return train_model(model, train_loader, val_loader, config, model_config)


def train_autoencoding_model(
    model: NextStepRNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    model_config: ModelConfig
) -> Tuple[NextStepRNN, List[float]]:
    """
    Train a model for autoencoding (reconstruct current observation).
    
    This is the control condition that should NOT develop place cells.
    """
    if config.verbose:
        print("Training in AUTOENCODING mode (reconstruction)")
    
    return train_model(model, train_loader, val_loader, config, model_config)


def get_training_summary(trainer: ModelTrainer) -> str:
    """
    Get a summary of training progress.
    
    Args:
        trainer: ModelTrainer instance
        
    Returns:
        Summary string
    """
    history = trainer.training_history
    
    if not history['train_loss']:
        return "No training history available."
    
    final_train_loss = history['train_loss'][-1]
    final_val_loss = history['val_loss'][-1]
    min_val_loss = min(history['val_loss'])
    num_epochs = len(history['train_loss'])
    
    summary = f"""
Training Summary:
================
Epochs completed: {num_epochs}
Final train loss: {final_train_loss:.6f}
Final val loss: {final_val_loss:.6f}
Best val loss: {min_val_loss:.6f}
Final learning rate: {history['learning_rates'][-1]:.6f}
"""
    
    return summary
