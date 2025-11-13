"""CNN Encoder for spatial feature extraction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    """Multi-layer CNN encoder for spatial feature extraction."""
    
    def __init__(
        self,
        input_channels: int,
        hidden_dim: int,
        num_layers: int = 2,
        kernel_size: int = 3,
        padding: int = 1,
        use_batchnorm: bool = True
    ):
        """
        Initialize CNN encoder.
        
        Args:
            input_channels: Number of input channels
            hidden_dim: Hidden feature dimension
            num_layers: Number of convolutional layers
            kernel_size: Convolution kernel size
            padding: Convolution padding
            use_batchnorm: Whether to use batch normalization
        """
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # target spatial size after encoder (H', W')
        self.target_spatial = 32
        
        layers = []
        in_channels = input_channels
        
        for i in range(num_layers):
            out_channels = hidden_dim * (2 ** i)
            
            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=not use_batchnorm
                )
            )
            
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))
            
            layers.append(nn.ReLU(inplace=True))
            
            # Optional: add max pooling to reduce spatial dimensions
            if i < num_layers - 1:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            
            in_channels = out_channels
        
        self.encoder = nn.Sequential(*layers)
        self.out_channels = in_channels

        # Adaptive pooling to enforce fixed spatial output size (e.g., 32x32)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((self.target_spatial, self.target_spatial))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder.
        
        Args:
            x: Input tensor (B, T, C, H, W) or (B*T, C, H, W)
            
        Returns:
            Encoded features (B*T, out_channels, H', W')
        """
        # Handle different input shapes
        if x.dim() == 5:  # (B, T, C, H, W)
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
        
        x = self.encoder(x)
        # Ensure output spatial dimensions are fixed to target_spatial
        if x.dim() == 4:
            _, _, h, w = x.shape
            if h != self.target_spatial or w != self.target_spatial:
                x = self.adaptive_pool(x)

        return x
    
    def get_output_channels(self) -> int:
        """Get output number of channels."""
        return self.out_channels
