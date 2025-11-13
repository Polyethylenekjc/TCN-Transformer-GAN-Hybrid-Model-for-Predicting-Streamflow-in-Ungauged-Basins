"""Complete streamflow prediction model with CNN + LSTM/Transformer + Decoder."""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from .cnn_encoder import CNNEncoder
from .lstm_module import LSTMModule
from .transformer_module import TransformerModule
from .decoder import SimpleUpsamplingDecoder


class StreamflowPredictionModel(nn.Module):
    """End-to-end streamflow prediction model."""
    
    def __init__(
        self,
        config: Dict
    ):
        """
        Initialize streamflow prediction model from config.
        
        Args:
            config: Configuration dictionary from YAML
        """
        super().__init__()
        self.config = config
        
        # Extract configuration
        model_cfg = config.get('model', {})
        data_cfg = config.get('data', {})
        
        self.input_channels = model_cfg.get('input_channels', 10)
        self.hidden_dim = model_cfg.get('hidden_dim', 128)
        self.num_layers = model_cfg.get('num_layers', 2)
        self.backbone = model_cfg.get('backbone', 'CNN')
        self.temporal_module_type = model_cfg.get('temporal_module', 'LSTM')
        self.upscale_factor = data_cfg.get('upscale_factor', 2)
        self.window_size = data_cfg.get('window_size', 5)
        
        # Input/output dimensions
        # Input image size from config (height, width)
        img_size = data_cfg.get('image_size', [128, 128])
        if isinstance(img_size, int):
            self.input_height = int(img_size)
            self.input_width = int(img_size)
        else:
            self.input_height = int(img_size[0])
            self.input_width = int(img_size[1])
        self.output_height = self.input_height * self.upscale_factor
        self.output_width = self.input_width * self.upscale_factor
        
        # CNN Encoder
        self.cnn_encoder = CNNEncoder(
            input_channels=self.input_channels,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            kernel_size=3,
            padding=1,
            use_batchnorm=True
        )
        
        cnn_out_channels = self.cnn_encoder.get_output_channels()
        
        # Encoder output spatial dimensions are enforced by CNNEncoder's adaptive pool
        encoded_h = getattr(self.cnn_encoder, 'target_spatial', 32)
        encoded_w = getattr(self.cnn_encoder, 'target_spatial', 32)
        
        # Flatten features for temporal module
        cnn_feature_size = cnn_out_channels * encoded_h * encoded_w
        
        # Temporal Module (LSTM or Transformer)
        if self.temporal_module_type.upper() == 'TRANSFORMER':
            self.temporal_module = TransformerModule(
                input_dim=cnn_feature_size,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                num_heads=4,
                dropout=0.2
            )
        else:  # Default to LSTM
            self.temporal_module = LSTMModule(
                input_dim=cnn_feature_size,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=0.2
            )
        
        temporal_out_dim = self.temporal_module.get_output_dim()
        
        # Project temporal output back to spatial features
        self.temporal_to_spatial = nn.Linear(
            temporal_out_dim,
            cnn_out_channels * encoded_h * encoded_w
        )
        
        self.encoded_h = encoded_h
        self.encoded_w = encoded_w
        self.cnn_out_channels = cnn_out_channels
        
        # Decoder for upsampling
        self.decoder = SimpleUpsamplingDecoder(
            input_channels=cnn_out_channels,
            output_height=self.output_height,
            output_width=self.output_width,
            hidden_channels=64
        )
    
    def forward(
        self,
        images: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            images: Input image sequence (B, T, C, H, W)
            
        Returns:
            Predicted streamflow map (B, 1, H_out, W_out)
        """
        B, T, C, H, W = images.shape
        
        # Reshape for CNN: (B*T, C, H, W)
        images_reshaped = images.view(B * T, C, H, W)
        
        # CNN encoding
        encoded = self.cnn_encoder(images_reshaped)  # (B*T, out_ch, h, w)
        _, out_ch, h_enc, w_enc = encoded.shape
        
        # Flatten and reshape for temporal module
        encoded_flat = encoded.view(B * T, -1)  # (B*T, out_ch*h*w)
        encoded_temporal = encoded_flat.view(B, T, -1)  # (B, T, out_ch*h*w)
        
        # Temporal modeling
        temporal_out = self.temporal_module(encoded_temporal)  # (B, hidden_dim)
        
        # Project back to spatial features
        spatial_features = self.temporal_to_spatial(temporal_out)  # (B, out_ch*h*w)
        spatial_features = spatial_features.view(
            B, out_ch, h_enc, w_enc
        )  # (B, out_ch, h, w)
        
        # Decode to high resolution
        output = self.decoder(spatial_features)  # (B, 1, H_out, W_out)
        
        return output
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultiTaskStreamflowModel(nn.Module):
    """Streamflow model with image and station outputs."""
    
    def __init__(self, config: Dict):
        """
        Initialize multi-task model.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self.config = config
        
        # Main streamflow prediction model
        self.streamflow_model = StreamflowPredictionModel(config)
        
        # Station output head: predict single value per station
        hidden_dim = config.get('model', {}).get('hidden_dim', 128)
        self.station_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(
        self,
        images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with multi-task outputs.
        
        Args:
            images: Input image sequence (B, T, C, H, W)
            
        Returns:
            Tuple of (streamflow_map, station_runoff)
                streamflow_map: (B, 1, H_out, W_out)
                station_runoff: (B, 1)
        """
        # Get streamflow prediction
        streamflow_map = self.streamflow_model(images)
        
        # Get temporal features for station prediction
        B, T, C, H, W = images.shape
        images_reshaped = images.view(B * T, C, H, W)
        encoded = self.streamflow_model.cnn_encoder(images_reshaped)
        encoded_flat = encoded.view(B * T, -1)
        encoded_temporal = encoded_flat.view(B, T, -1)
        
        # Get temporal output
        temporal_out = self.streamflow_model.temporal_module(
            encoded_temporal
        )  # (B, hidden_dim)
        
        # Predict station runoff
        station_runoff = self.station_head(temporal_out)  # (B, 1)
        
        return streamflow_map, station_runoff
