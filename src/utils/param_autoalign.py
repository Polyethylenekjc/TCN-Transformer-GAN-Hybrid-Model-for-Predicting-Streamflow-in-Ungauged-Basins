"""Automatic parameter alignment module for model dimension matching."""

import torch
import torch.nn as nn
from typing import Tuple, List


class ParamAutoAlign:
    """Auto-align dimensions across model components."""
    
    @staticmethod
    def calculate_cnn_output_shape(
        input_shape: Tuple[int, int, int],
        input_channels: int,
        kernel_sizes: List[int] = None,
        strides: List[int] = None,
        paddings: List[int] = None
    ) -> Tuple[int, int, int]:
        """
        Calculate CNN output shape (C, H, W).
        
        Args:
            input_shape: (H, W) of input
            input_channels: Number of input channels
            kernel_sizes: List of kernel sizes for each conv layer
            strides: List of strides for each conv layer
            paddings: List of paddings for each conv layer
            
        Returns:
            (output_channels, output_height, output_width)
        """
        if kernel_sizes is None:
            kernel_sizes = [3, 3, 3]
        if strides is None:
            strides = [1, 1, 1]
        if paddings is None:
            paddings = [1, 1, 1]
        
        H, W = input_shape
        C = input_channels
        
        # Each conv layer doubles channels
        for i in range(len(kernel_sizes)):
            C = C * 2
            H = (H + 2 * paddings[i] - kernel_sizes[i]) // strides[i] + 1
            W = (W + 2 * paddings[i] - kernel_sizes[i]) // strides[i] + 1
        
        return (C, H, W)
    
    @staticmethod
    def calculate_upsampling_output_shape(
        input_shape: Tuple[int, int, int],
        upscale_factor: int
    ) -> Tuple[int, int, int]:
        """
        Calculate output shape after upsampling.
        
        Args:
            input_shape: (C, H, W) of input
            upscale_factor: Upscaling factor (e.g., 2 for 2x)
            
        Returns:
            (output_channels, output_height, output_width)
        """
        C, H, W = input_shape
        return (1, H * upscale_factor, W * upscale_factor)
    
    @staticmethod
    def reshape_for_temporal(
        features: torch.Tensor,
        sequence_length: int,
        hidden_dim: int
    ) -> torch.Tensor:
        """
        Reshape CNN features for temporal module input.
        
        Args:
            features: (B*T, C, H, W) tensor from CNN
            sequence_length: T (number of time steps)
            hidden_dim: Target hidden dimension
            
        Returns:
            Reshaped tensor for temporal module
        """
        B_T, C, H, W = features.shape
        B = B_T // sequence_length
        
        # Reshape to (B, T, C*H*W)
        features_flat = features.view(B, sequence_length, C * H * W)
        
        # Project to hidden_dim if needed
        if C * H * W != hidden_dim:
            linear = nn.Linear(C * H * W, hidden_dim)
            features_flat = linear(features_flat)
        
        return features_flat
    
    @staticmethod
    def reshape_from_temporal(
        temporal_output: torch.Tensor,
        spatial_shape: Tuple[int, int, int]
    ) -> torch.Tensor:
        """
        Reshape temporal output back to spatial format.
        
        Args:
            temporal_output: (B, hidden_dim) or (B, 1, hidden_dim)
            spatial_shape: (C, H, W) target shape
            
        Returns:
            Reshaped tensor (B, C, H, W)
        """
        if temporal_output.dim() == 3:
            temporal_output = temporal_output.squeeze(1)
        
        B = temporal_output.shape[0]
        C, H, W = spatial_shape
        
        # Project from hidden_dim to C*H*W
        if temporal_output.shape[-1] != C * H * W:
            linear = nn.Linear(temporal_output.shape[-1], C * H * W)
            temporal_output = linear(temporal_output)
        
        return temporal_output.view(B, C, H, W)
    
    @staticmethod
    def adaptive_pool_output(
        features: torch.Tensor,
        target_h: int,
        target_w: int
    ) -> torch.Tensor:
        """
        Use adaptive pooling to ensure specific spatial dimensions.
        
        Args:
            features: Input tensor (B, C, H, W)
            target_h: Target height
            target_w: Target width
            
        Returns:
            Pooled tensor (B, C, target_h, target_w)
        """
        pool = nn.AdaptiveAvgPool2d((target_h, target_w))
        return pool(features)
    
    @staticmethod
    def validate_dimensions(
        input_shape: Tuple[int, ...],
        expected_shape: Tuple[int, ...],
        name: str = "Tensor"
    ) -> bool:
        """
        Validate tensor dimensions match expected shape.
        
        Args:
            input_shape: Actual input shape
            expected_shape: Expected shape
            name: Name for error message
            
        Returns:
            True if valid, raises error otherwise
        """
        if input_shape != expected_shape:
            raise ValueError(
                f"{name} dimension mismatch: expected {expected_shape}, "
                f"got {input_shape}"
            )
        return True
