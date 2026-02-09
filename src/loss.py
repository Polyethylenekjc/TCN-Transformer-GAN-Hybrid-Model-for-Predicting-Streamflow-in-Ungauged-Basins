"""Loss functions for streamflow prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class ImageLoss(nn.Module):
    """Pixel-wise image loss for flow prediction with optional weighting for non-zero regions."""
    
    def __init__(self, metric: str = 'MSE', upscale_factor: int = 2, 
                 nonzero_weight: float = 10.0, zero_threshold: float = 1e-6):
        """
        Initialize image loss.
        
        Args:
            metric: Loss metric ('MSE' or 'MAE')
            upscale_factor: Upscaling factor for target images
            nonzero_weight: Weight multiplier for non-zero (river) pixels
            zero_threshold: Threshold below which a pixel is considered zero
        """
        super().__init__()
        self.metric = metric.upper()
        self.upscale_factor = upscale_factor
        self.nonzero_weight = nonzero_weight
        self.zero_threshold = zero_threshold
        
        if self.metric not in ['MSE', 'MAE']:
            raise ValueError(f"Unknown metric: {metric}")
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate weighted image loss.
        
        Args:
            predictions: Predicted flow map (B, 1, H, W)
            targets: Target flow map (B, 1, H_orig, W_orig)
            
        Returns:
            Loss value
        """
        # Upsample targets to match predictions if needed
        if targets.shape != predictions.shape:
            targets = F.interpolate(
                targets,
                size=(predictions.shape[2], predictions.shape[3]),
                mode='bilinear',
                align_corners=False
            )
        
        # Create weight map: higher weight for non-zero (river) pixels
        nonzero_mask = (torch.abs(targets) > self.zero_threshold).float()
        weights = torch.ones_like(targets)
        weights = weights + (self.nonzero_weight - 1.0) * nonzero_mask
        
        # Calculate element-wise loss
        if self.metric == 'MSE':
            pixel_loss = (predictions - targets) ** 2
        else:  # MAE
            pixel_loss = torch.abs(predictions - targets)
        
        # Apply weights and compute mean
        weighted_loss = (pixel_loss * weights).sum() / weights.sum()
        
        return weighted_loss


class StationLoss(nn.Module):
    """Loss for station-level runoff predictions."""
    
    def __init__(
        self,
        kernel_size: int = 3,
        metric: str = 'MSE'
    ):
        """
        Initialize station loss.
        
        Args:
            kernel_size: Neighborhood kernel size (3, 5, 7, etc.)
            metric: Loss metric ('MSE' or 'MAE')
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.metric = metric.upper()
        
        if self.metric == 'MSE':
            self.loss_fn = nn.MSELoss()
        elif self.metric == 'MAE':
            self.loss_fn = nn.L1Loss()
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def forward(
        self,
        predictions: torch.Tensor,
        station_positions: List[torch.Tensor],
        station_runoffs: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Calculate station loss.
        
        Args:
            predictions: Predicted flow map (B, 1, H, W)
            station_positions: List of (num_stations, 2) position tensors
            station_runoffs: List of (num_stations,) runoff value tensors
            
        Returns:
            Loss value
        """
        total_loss = 0.0
        num_stations = 0
        
        B = predictions.shape[0]
        kernel_half = self.kernel_size // 2
        
        for b in range(B):
            positions = station_positions[b]
            runoffs = station_runoffs[b]
            
            if len(runoffs) == 0:
                continue
            
            for pos, runoff in zip(positions, runoffs):
                px, py = pos[0].item(), pos[1].item()
                
                # Extract neighborhood
                y_start = max(0, py - kernel_half)
                y_end = min(predictions.shape[2], py + kernel_half + 1)
                x_start = max(0, px - kernel_half)
                x_end = min(predictions.shape[3], px + kernel_half + 1)
                
                neighborhood = predictions[
                    b, 0, y_start:y_end, x_start:x_end
                ]
                
                if neighborhood.numel() == 0:
                    continue
                
                # Average over neighborhood
                avg_prediction = neighborhood.mean()
                
                # Calculate loss for this station
                loss = self.loss_fn(
                    avg_prediction.unsqueeze(0),
                    runoff.unsqueeze(0).to(predictions.device)
                )
                
                total_loss += loss
                num_stations += 1
        
        if num_stations == 0:
            return torch.tensor(0.0, device=predictions.device)
        
        return total_loss / num_stations


class CombinedLoss(nn.Module):
    """Combined loss with image and station components."""
    
    def __init__(self, config: Dict):
        """
        Initialize combined loss.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        
        loss_cfg = config.get('loss', {})
        weights = loss_cfg.get('weights', {})
        upscale_factor = config.get('data', {}).get('upscale_factor', 2)
        
        self.image_weight = weights.get('image', 1.0)
        self.station_weight = weights.get('station', 0.5)
        
        # Loss functions
        self.image_loss = ImageLoss(
            metric=loss_cfg.get('metric', 'MSE'),
            upscale_factor=upscale_factor,
            nonzero_weight=float(loss_cfg.get('nonzero_weight', 10.0)),
            zero_threshold=float(loss_cfg.get('zero_threshold', 1e-6))
        )
        
        station_cfg = config.get('stations', {})
        self.station_loss = StationLoss(
            kernel_size=station_cfg.get('kernel_size', 3),
            metric=station_cfg.get('metric', 'MSE')
        )
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        station_positions: Optional[List[torch.Tensor]] = None,
        station_runoffs: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate combined loss.
        
        Args:
            predictions: Predicted flow map (B, 1, H, W)
            targets: Target flow map (B, 1, H, W)
            station_positions: List of station positions
            station_runoffs: List of station runoff values
            
        Returns:
            Dictionary with loss breakdown
        """
        # Image loss
        img_loss = self.image_loss(predictions, targets)
        
        # Station loss
        if station_positions is not None and station_runoffs is not None:
            stn_loss = self.station_loss(
                predictions, station_positions, station_runoffs
            )
        else:
            stn_loss = torch.tensor(0.0, device=predictions.device)
        
        # Combined loss
        total_loss = (
            self.image_weight * img_loss +
            self.station_weight * stn_loss
        )
        
        return {
            'total': total_loss,
            'image': img_loss,
            'station': stn_loss
        }


def calculate_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> Dict[str, float]:
    """
    Calculate evaluation metrics.
    
    Args:
        predictions: Predicted values
        targets: Target values
        
    Returns:
        Dictionary with metrics (RMSE, MAE, R², NSE)
    """
    # Ensure same shape
    if predictions.shape != targets.shape:
        # Upsample targets to match predictions
        if targets.dim() == 4:  # (B, C, H, W)
            targets = F.interpolate(
                targets,
                size=(predictions.shape[2], predictions.shape[3]),
                mode='bilinear',
                align_corners=False
            )
        elif targets.dim() == 3:  # (C, H, W) -> add batch dim
            targets = targets.unsqueeze(0)
            targets = F.interpolate(
                targets,
                size=(predictions.shape[-2], predictions.shape[-1]),
                mode='bilinear',
                align_corners=False
            )
            targets = targets.squeeze(0)
    
    pred = predictions.detach().cpu().numpy().flatten()
    target = targets.detach().cpu().numpy().flatten()

    # Ensure finite values
    if not (pred.size and target.size):
        return {
            'RMSE': 0.0,
            'MAE': 0.0,
            'R2': 0.0,
            'NSE': 0.0,
            'MAPE': 0.0,
            'KGE': 0.0,
        }

    # Replace NaN/Inf to avoid propagation
    import numpy as np

    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)

    # RMSE
    rmse = ((pred - target) ** 2).mean() ** 0.5

    # MAE
    mae = np.abs(pred - target).mean()

    # R² and NSE share the same numerator/denominator
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    if ss_tot > 0:
        r2 = 1 - (ss_res / ss_tot)
        nse = 1 - (ss_res / ss_tot)
    else:
        r2 = 0.0
        nse = 0.0

    # MAPE (Mean Absolute Percentage Error, in %)
    # Ignore very small targets to avoid division by zero explosion
    eps = 1e-6
    mask = np.abs(target) > eps
    if mask.any():
        mape = (np.abs((pred[mask] - target[mask]) / target[mask])).mean() * 100.0
    else:
        mape = 0.0

    # KGE (Kling-Gupta Efficiency)
    # KGE = 1 - sqrt( (r-1)^2 + (alpha-1)^2 + (beta-1)^2 )
    # r: correlation, alpha: std(pred)/std(obs), beta: mean(pred)/mean(obs)
    try:
        mu_p = float(pred.mean())
        mu_o = float(target.mean())
        std_p = float(pred.std())
        std_o = float(target.std())

        if std_p > 0 and std_o > 0 and mu_o != 0.0:
            r = float(np.corrcoef(pred, target)[0, 1])
            alpha = std_p / std_o
            beta = mu_p / mu_o
            kge = 1.0 - float(np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))
        else:
            kge = 0.0
    except Exception:
        kge = 0.0

    return {
        'RMSE': float(rmse),
        'MAE': float(mae),
        'R2': float(r2),
        'NSE': float(nse),
        'MAPE': float(mape),
        'KGE': float(kge),
    }
