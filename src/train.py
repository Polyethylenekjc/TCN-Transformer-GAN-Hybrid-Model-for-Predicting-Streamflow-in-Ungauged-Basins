"""Training script for streamflow prediction model."""

import torch
import torch.nn as nn
from torch import amp
from torch.cuda.amp import GradScaler
import os
from pathlib import Path
from typing import Dict
import logging
from datetime import datetime
import numpy as np
try:
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn
    from rich.logging import RichHandler
    RICH_AVAILABLE = True
except Exception:
    RICH_AVAILABLE = False

from torch.utils.data import DataLoader as TorchDataLoader

from src.model import StreamflowPredictionModel, MultiTaskStreamflowModel
from src.model.gan import PatchDiscriminator, GANLoss
from src.dataset import StreamflowDataset
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
        
        lr = config.get('train', {}).get('lr', 1e-4)
        
        # GAN setup
        self.use_gan = config.get('model', {}).get('use_gan', False)
        if self.use_gan:
            self.logger.info("Initializing GAN components...")
            # Discriminator
            self.discriminator = PatchDiscriminator(
                input_channels=1,  # Output is 1 channel (flow)
                hidden_channels=64
            ).to(self.device)
            
            # GAN Loss
            self.gan_weight = config.get('model', {}).get('gan_weight', 0.1)
            self.gan_loss_fn = GANLoss(
                lambda_gan=self.gan_weight,
                lambda_pixel=1.0  # Pixel loss is handled by CombinedLoss
            ).to(self.device)
            
            # Discriminator Optimizer
            # Use lower learning rate for discriminator to prevent it from overpowering generator
            self.optimizer_D = torch.optim.Adam(
                self.discriminator.parameters(),
                lr=lr * 0.2,  # 20% of generator LR
                betas=(0.5, 0.999)  # Standard GAN betas
            )
            self.logger.info(f"GAN enabled with weight {self.gan_weight}")
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr
        )
        
        # Mixed precision
        self.use_amp = config.get('system', {}).get('mixed_precision', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Gradient accumulation
        self.grad_accum_steps = config.get('system', {}).get('grad_accum_steps', 1)
        
        # Discriminator update interval (Train D every N steps)
        self.d_update_interval = 3  # Update D only once every 3 batches
        
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
        # Allow verbose debug in config
        level = logging.DEBUG if self.config.get('system', {}).get('debug', False) else logging.INFO
        logger.setLevel(level)
        
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
        
        # Console handler: prefer rich if available
        if RICH_AVAILABLE and self.config.get('system', {}).get('use_rich', True):
            rich_handler = RichHandler()
            rich_handler.setLevel(level)
            logger.addHandler(rich_handler)
        else:
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
    
    def train_epoch(self, dataloader) -> Dict[str, float]:
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
        total_batches = len(dataloader)
        
        # Use rich progress bar if available and enabled
        use_rich = RICH_AVAILABLE and self.config.get('system', {}).get('use_rich', True)
        # When using external Progress (from train), train_epoch will receive progress and task id

        for batch_idx, batch in enumerate(dataloader):
            images = batch['images'].to(self.device, non_blocking=True)
            output_image = batch['output_image'].to(self.device, non_blocking=True)
            
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
            # Use torch.amp.autocast with explicit device_type to avoid FutureWarning
            with amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                if torch.isnan(images).any() or torch.isinf(images).any():
                    self.logger.warning("NaN or Inf found in input images!")
                
                predictions = self.model(images)

                if torch.isnan(predictions).any() or torch.isinf(predictions).any():
                    self.logger.warning("NaN or Inf found in model predictions!")
                
                loss_dict = self.loss_fn(
                    predictions,
                    output_image,
                    station_positions,
                    station_runoffs
                )
                
                loss_G = loss_dict['total']
                
                if self.use_gan:
                    # Generator Adversarial Loss
                    # We want discriminator to classify generated images as real
                    pred_fake = self.discriminator(predictions)
                    loss_G_gan = self.gan_loss_fn.gan_loss(pred_fake, torch.ones_like(pred_fake))
                    
                    # Combine losses
                    loss_G = loss_G + self.gan_weight * loss_G_gan
                    loss_dict['gan_g'] = loss_G_gan
                    loss_dict['total'] = loss_G  # Update total for logging

                if torch.isnan(loss_dict['total']).any() or torch.isinf(loss_dict['total']).any():
                    self.logger.warning("NaN or Inf found in loss!")

            # Backward Generator
            if self.use_amp:
                self.scaler.scale(loss_G / self.grad_accum_steps).backward()
            else:
                (loss_G / self.grad_accum_steps).backward()
            
            # Discriminator Step
            if self.use_gan and (batch_idx % self.d_update_interval == 0):
                with amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    # Real loss
                    pred_real = self.discriminator(output_image)
                    # Fake loss (detach to avoid backprop to generator)
                    pred_fake_d = self.discriminator(predictions.detach())
                    
                    loss_D = self.gan_loss_fn.discriminator_loss(pred_real, pred_fake_d)
                    loss_dict['gan_d'] = loss_D
                
                # Backward Discriminator
                if self.use_amp:
                    self.scaler.scale(loss_D / self.grad_accum_steps).backward()
                else:
                    (loss_D / self.grad_accum_steps).backward()
            
            # Accumulate metrics
            total_loss += loss_dict['total'].item()
            total_img_loss += loss_dict['image'].item()
            total_stn_loss += loss_dict['station'].item()
            num_batches += 1
            
            # Optimizer step
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    # Only step D optimizer if we computed gradients for it
                    if self.use_gan and (batch_idx % self.d_update_interval == 0):
                        self.scaler.step(self.optimizer_D)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                    if self.use_gan and (batch_idx % self.d_update_interval == 0):
                        self.optimizer_D.step()
                
                self.optimizer.zero_grad()
                if self.use_gan:
                    self.optimizer_D.zero_grad()
                
                # Log GPU memory
                if (batch_idx + 1) % (self.log_interval * self.grad_accum_steps) == 0:
                    self._log_gpu_memory()
            
            # Periodic logging
            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = total_loss / num_batches
                avg_img = total_img_loss / num_batches
                avg_stn = total_stn_loss / num_batches
                
                log_msg = (
                    f"Epoch {self.epoch} Batch {batch_idx + 1}: "
                    f"Loss={avg_loss:.6f}, Img={avg_img:.6f}, Stn={avg_stn:.6f}"
                )
                if self.use_gan and 'gan_d' in loss_dict:
                    log_msg += f", D_Loss={loss_dict['gan_d'].item():.6f}"
                
                self.logger.info(log_msg)
            # If an external progress/task were provided in self._active_progress, update it
            active = getattr(self, '_active_progress', None)
            if active is not None:
                prog, batch_task = active
                try:
                    batch_loss = loss_dict['total'].item()
                except Exception:
                    batch_loss = 0.0
                try:
                    prog.update(batch_task, advance=1, loss=batch_loss)
                except Exception:
                    pass
        
        return {
            'loss': total_loss / num_batches,
            'image_loss': total_img_loss / num_batches,
            'station_loss': total_stn_loss / num_batches
        }
    
    def validate(self, dataloader) -> Dict[str, float]:
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
                images = batch['images'].to(self.device, non_blocking=True)
                output_image = batch['output_image'].to(self.device, non_blocking=True)
                
                # Forward pass
                predictions = self.model(images)
                
                # Post-processing for validation metrics (optional but recommended for consistency)
                # We don't apply this for loss calculation, only for metrics if needed
                # But here we keep raw predictions for loss calculation
                
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
        # Use native PyTorch DataLoader with multi-worker + pinned memory
        batch_size = self.config.get('train', {}).get('batch_size', 4)
        num_workers = self.config.get('train', {}).get('num_workers', 4)
        pin_memory = self.device.type == 'cuda'

        def _collate_fn(batch):
            # batch is a list of samples from StreamflowDataset
            images = torch.stack([b['images'] for b in batch], dim=0)
            outputs = torch.stack([b['output_image'] for b in batch], dim=0)
            stations = [b['stations'] for b in batch]
            station_positions = [b['station_positions'] for b in batch]
            return {
                'images': images,
                'output_image': outputs,
                'stations': stations,
                'station_positions': station_positions
            }

        train_loader = TorchDataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=_collate_fn,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
        )

        if val_dataset is not None:
            val_loader = TorchDataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=_collate_fn,
                persistent_workers=(num_workers > 0),
                prefetch_factor=2 if num_workers > 0 else None,
            )
        else:
            val_loader = None
        
        output_dir = Path(self.config.get('data', {}).get('output_dir', './output'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Optionally use a single rich Progress with two tasks: epochs and batches
        use_rich = RICH_AVAILABLE and self.config.get('system', {}).get('use_rich', True)
        if use_rich:
            progress = Progress(
                SpinnerColumn(),
                "[progress.description]{task.description}",
                BarColumn(),
                TextColumn("Loss: {task.fields[loss]:.4f}"),
                TimeRemainingColumn()
            )
            progress.start()
            epoch_task = progress.add_task("Training", total=num_epochs, loss=0.0)
            # batch_task will be updated per-epoch

        for epoch in range(num_epochs):
            self.epoch = epoch

            # If using rich, set up active progress and batch task
            if use_rich:
                total_batches = len(train_loader)
                batch_task = progress.add_task(f"Epoch {epoch}", total=total_batches, loss=0.0)
                # expose to train_epoch for updates
                self._active_progress = (progress, batch_task)
            else:
                self._active_progress = None

            # Train
            train_metrics = self.train_epoch(train_loader)
            self.logger.info(
                f"Epoch {epoch} - Train Loss: {train_metrics['loss']:.6f}"
            )

            if use_rich:
                # advance epoch progress and remove batch task
                progress.update(epoch_task, advance=1)
                try:
                    progress.remove_task(batch_task)
                except Exception:
                    pass
                # clear active progress
                self._active_progress = None
            
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
            
            # Save GAN checkpoints every epoch
            if self.use_gan:
                gan_dir = output_dir / 'gan'
                gan_dir.mkdir(parents=True, exist_ok=True)
                gan_ckpt_path = gan_dir / f'gan_epoch_{epoch}.pt'
                torch.save(self.model.state_dict(), gan_ckpt_path)
                self.logger.info(f"GAN Checkpoint saved to {gan_ckpt_path}")
            
            self._log_gpu_memory()
        
        self.logger.info("Training complete!")


def run_training(config_path: str, resume_path: str = None, epochs: int = None):
    """
    Run training programmatically.
    
    Args:
        config_path: Path to config file
        resume_path: Path to checkpoint to resume from
        epochs: Override number of epochs
    """
    # Load configuration
    config = ConfigLoader.load_config(config_path)
    
    # Override epochs if provided
    if epochs is not None:
        if 'train' not in config:
            config['train'] = {}
        config['train']['num_epochs'] = epochs
    
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
    trainer = Trainer(config, model_path=resume_path)
    
    # Train
    num_epochs = config.get('train', {}).get('num_epochs', 100)
    trainer.train(dataset, num_epochs=num_epochs)


def main():
    """
    Main training function.
    """
    import argparse
    parser = argparse.ArgumentParser(description='Train Streamflow Prediction Model')
    parser.add_argument('--config', type=str, default='./data/config.yaml', help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=None, help='Override number of epochs from config')
    args = parser.parse_args()

    run_training(args.config, args.resume, args.epochs)


if __name__ == '__main__':
    main()
