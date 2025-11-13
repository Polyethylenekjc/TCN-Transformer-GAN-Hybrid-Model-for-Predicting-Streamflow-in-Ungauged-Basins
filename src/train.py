"""Training script for streamflow prediction model."""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import os
from pathlib import Path
from typing import Dict
import logging
from datetime import datetime
import numpy as np

from src.model import StreamflowPredictionModel, MultiTaskStreamflowModel
from src.dataset import StreamflowDataset, DataLoader
from src.loss import CombinedLoss, calculate_metrics
from src.utils.config_loader import ConfigLoader


class Trainer:
    """Model trainer with mixed precision and gradient accumulation support."""
    
    def __init__(self, config: Dict, model_path: str = None):
        """
        Initialize trainer.
        
        Args:
            config: Configuration dictionary
            model_path: Path to load pre-trained model (optional)
        """
        self.config = config
        self.device = torch.device(config.get('train', {}).get('device', 'cpu'))
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Model setup
        self.model = StreamflowPredictionModel(config)
        
        if model_path and os.path.exists(model_path):
            self.logger.info(f"Loading model from {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.to(self.device)
        
        # Loss function
        self.loss_fn = CombinedLoss(config)
        self.loss_fn.to(self.device)
        
        # Optimizer
        lr = config.get('train', {}).get('lr', 1e-4)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr
        )
        
        # Mixed precision
        self.use_amp = config.get('system', {}).get('mixed_precision', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Gradient accumulation
        self.grad_accum_steps = config.get('system', {}).get('grad_accum_steps', 1)
        
        # Training tracking
        self.epoch = 0
        self.best_loss = float('inf')
        self.log_interval = config.get('system', {}).get('log_interval', 10)
        
        self.logger.info(f"Model parameters: {self.model.get_num_parameters()}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Mixed precision: {self.use_amp}")
        self.logger.info(f"Grad accumulation steps: {self.grad_accum_steps}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger('Trainer')
        logger.setLevel(logging.INFO)
        
        # File handler
        log_dir = Path(self.config.get('data', {}).get('output_dir', './output'))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        handler = logging.FileHandler(
            log_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Console handler
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)
        
        return logger
    
    def _log_gpu_memory(self) -> None:
        """Log GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 ** 2
            reserved = torch.cuda.memory_reserved() / 1024 ** 2
            self.logger.info(
                f"GPU Memory - Allocated: {allocated:.1f}MB, Reserved: {reserved:.1f}MB"
            )
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Data loader
            
        Returns:
            Dictionary with loss metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_img_loss = 0.0
        total_stn_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            images = batch['images'].to(self.device)
            output_image = batch['output_image'].to(self.device)
            
            # Move station data to device
            station_positions = [
                pos.to(self.device) if isinstance(pos, torch.Tensor) else pos
                for pos in batch['station_positions']
            ]
            station_runoffs = [
                runoff.to(self.device) if isinstance(runoff, torch.Tensor) else runoff
                for runoff in batch['stations']
            ]
            
            # Forward pass
            with autocast(enabled=self.use_amp):
                predictions = self.model(images)
                
                loss_dict = self.loss_fn(
                    predictions,
                    output_image,
                    station_positions,
                    station_runoffs
                )
                
                loss = loss_dict['total'] / self.grad_accum_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Accumulate metrics
            total_loss += loss_dict['total'].item()
            total_img_loss += loss_dict['image'].item()
            total_stn_loss += loss_dict['station'].item()
            num_batches += 1
            
            # Optimizer step
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Log GPU memory
                if (batch_idx + 1) % (self.log_interval * self.grad_accum_steps) == 0:
                    self._log_gpu_memory()
            
            # Periodic logging
            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = total_loss / num_batches
                avg_img = total_img_loss / num_batches
                avg_stn = total_stn_loss / num_batches
                
                self.logger.info(
                    f"Epoch {self.epoch} Batch {batch_idx + 1}: "
                    f"Loss={avg_loss:.6f}, Img={avg_img:.6f}, Stn={avg_stn:.6f}"
                )
        
        return {
            'loss': total_loss / num_batches,
            'image_loss': total_img_loss / num_batches,
            'station_loss': total_stn_loss / num_batches
        }
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Validate model.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Dictionary with metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch['images'].to(self.device)
                output_image = batch['output_image'].to(self.device)
                
                # Forward pass
                predictions = self.model(images)
                
                # Calculate loss
                loss_dict = self.loss_fn(
                    predictions,
                    output_image,
                    batch['station_positions'],
                    batch['stations']
                )
                
                total_loss += loss_dict['total'].item()
                num_batches += 1
                
                # Collect predictions for metrics
                all_predictions.append(predictions.cpu())
                all_targets.append(output_image.cpu())
        
        # Calculate metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = calculate_metrics(all_predictions, all_targets)
        
        metrics['loss'] = total_loss / num_batches
        
        return metrics
    
    def train(
        self,
        train_dataset: StreamflowDataset,
        num_epochs: int = 100,
        val_dataset: StreamflowDataset = None
    ) -> None:
        """
        Full training loop.
        
        Args:
            train_dataset: Training dataset
            num_epochs: Number of epochs
            val_dataset: Validation dataset (optional)
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('train', {}).get('batch_size', 4),
            shuffle=True
        )
        
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.get('train', {}).get('batch_size', 4),
                shuffle=False
            )
        else:
            val_loader = None
        
        output_dir = Path(self.config.get('data', {}).get('output_dir', './output'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            self.logger.info(
                f"Epoch {epoch} - Train Loss: {train_metrics['loss']:.6f}"
            )
            
            # Validate
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                self.logger.info(
                    f"Epoch {epoch} - Val Loss: {val_metrics['loss']:.6f}, "
                    f"RMSE: {val_metrics['RMSE']:.4f}, "
                    f"NSE: {val_metrics['NSE']:.4f}"
                )
                
                # Save best model
                if val_metrics['loss'] < self.best_loss:
                    self.best_loss = val_metrics['loss']
                    model_path = output_dir / 'best_model.pt'
                    torch.save(self.model.state_dict(), model_path)
                    self.logger.info(f"Model saved to {model_path}")
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                ckpt_path = output_dir / f'checkpoint_epoch_{epoch}.pt'
                torch.save(self.model.state_dict(), ckpt_path)
                self.logger.info(f"Checkpoint saved to {ckpt_path}")
            
            self._log_gpu_memory()
        
        self.logger.info("Training complete!")


def main(config_path: str = './data/config.yaml'):
    """
    Main training function.
    
    Args:
        config_path: Path to configuration file
    """
    # Load configuration
    config = ConfigLoader.load_config(config_path)
    
    # Create dataset
    image_dir = config.get('data', {}).get('image_dir', './data/images')
    station_dir = config.get('data', {}).get('station_dir', './data/stations')
    
    dataset = StreamflowDataset(
        image_dir=image_dir,
        station_dir=station_dir,
        config=config,
        normalize=True
    )
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Train
    num_epochs = config.get('train', {}).get('num_epochs', 100)
    trainer.train(dataset, num_epochs=num_epochs)


if __name__ == '__main__':
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else './data/config.yaml'
    main(config_path)
