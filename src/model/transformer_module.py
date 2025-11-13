"""Transformer temporal module for sequence modeling."""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:, :x.size(1)]


class TransformerModule(nn.Module):
    """Transformer-based temporal modeling module."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
        feedforward_dim: int = None
    ):
        """
        Initialize Transformer module.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Transformer hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            feedforward_dim: Feed-forward network dimension
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        if feedforward_dim is None:
            feedforward_dim = hidden_dim * 4
        
        # Input projection
        if input_dim != hidden_dim:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.input_proj = None
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=100)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.output_dim = hidden_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Transformer.
        
        Args:
            x: Input tensor (B, T, input_dim)
            
        Returns:
            Aggregated temporal feature (B, hidden_dim)
        """
        # Project input if necessary
        if self.input_proj is not None:
            x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer forward
        transformer_out = self.transformer_encoder(x)
        
        # Aggregate over time dimension (use mean pooling)
        out = transformer_out.mean(dim=1)  # (B, hidden_dim)
        
        return out
    
    def get_output_dim(self) -> int:
        """Get output dimension."""
        return self.output_dim
