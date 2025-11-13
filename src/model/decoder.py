"""Decoder for upsampling to high-resolution streamflow prediction."""

import torch
import torch.nn as nn


class UpsamplingDecoder(nn.Module):
    """Decoder for upsampling features to target resolution."""
    
    def __init__(
        self,
        input_channels: int,
        input_height: int,
        input_width: int,
        output_height: int,
        output_width: int,
        upscale_factor: int = 2,
        hidden_channels: int = 64
    ):
        """
        Initialize upsampling decoder.
        
        Args:
            input_channels: Number of input feature channels
            input_height: Input feature map height
            input_width: Input feature map width
            output_height: Target output height
            output_width: Target output width
            upscale_factor: Upscaling factor (2 for 2x)
            hidden_channels: Hidden layer channels
        """
        super().__init__()
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.upscale_factor = upscale_factor
        
        # Decoder blocks: alternating conv + upsample
        self.decoder = nn.Sequential()
        
        current_channels = input_channels
        current_h = input_height
        current_w = input_width
        
        # Calculate number of upsampling blocks needed
        scale_h = output_height / input_height
        scale_w = output_width / input_width
        
        if abs(scale_h - scale_w) > 0.1:
            # Heights and widths don't scale uniformly
            # Will use adaptive pooling at the end
            num_upsample_blocks = 1
        else:
            # Determine how many 2x upsampling blocks needed
            num_upsample_blocks = int(round(math.log2(upscale_factor)))
        
        import math
        
        for i in range(num_upsample_blocks):
            # Convolution block
            self.decoder.add_module(
                f'conv{i}',
                nn.Conv2d(
                    current_channels,
                    hidden_channels,
                    kernel_size=3,
                    padding=1
                )
            )
            self.decoder.add_module(f'relu{i}', nn.ReLU(inplace=True))
            
            # Upsample using PixelShuffle for efficiency
            # First expand channels for PixelShuffle
            self.decoder.add_module(
                f'expand{i}',
                nn.Conv2d(
                    hidden_channels,
                    hidden_channels * 4,
                    kernel_size=1
                )
            )
            self.decoder.add_module(
                f'pixelshuffle{i}',
                nn.PixelShuffle(upscale_factor=2)
            )
            
            current_channels = hidden_channels
            current_h *= 2
            current_w *= 2
        
        # Final output layer: produce single channel
        self.decoder.add_module(
            'final_conv',
            nn.Conv2d(current_channels, 32, kernel_size=3, padding=1)
        )
        self.decoder.add_module('final_relu', nn.ReLU(inplace=True))
        self.decoder.add_module(
            'output_conv',
            nn.Conv2d(32, 1, kernel_size=1)
        )
        
        # Adaptive pooling for final spatial adjustment
        self.adaptive_pool = nn.AdaptiveAvgPool2d((output_height, output_width))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            x: Input tensor (B, input_channels, H, W)
            
        Returns:
            Upsampled output (B, 1, output_height, output_width)
        """
        x = self.decoder(x)
        
        # Ensure exact output size
        x = self.adaptive_pool(x)
        
        return x


class SimpleUpsamplingDecoder(nn.Module):
    """Simpler decoder using bilinear interpolation."""
    
    def __init__(
        self,
        input_channels: int,
        output_height: int,
        output_width: int,
        hidden_channels: int = 64
    ):
        """
        Initialize simple upsampling decoder.
        
        Args:
            input_channels: Number of input channels
            output_height: Target output height
            output_width: Target output width
            hidden_channels: Hidden layer channels
        """
        super().__init__()
        self.output_height = output_height
        self.output_width = output_width
        
        self.conv1 = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.output_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            x: Input tensor (B, input_channels, H, W)
            
        Returns:
            Upsampled output (B, 1, output_height, output_width)
        """
        # Upsample using bilinear interpolation
        x = torch.nn.functional.interpolate(
            x,
            size=(self.output_height, self.output_width),
            mode='bilinear',
            align_corners=False
        )
        
        # Refine with convolutions
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.output_conv(x)
        
        return x
