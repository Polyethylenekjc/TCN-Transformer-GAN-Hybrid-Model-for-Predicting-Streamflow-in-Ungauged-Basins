"""LSTM temporal module for sequence modeling."""

import torch
import torch.nn as nn


class LSTMModule(nn.Module):
    """LSTM-based temporal modeling module."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        """
        Initialize LSTM module.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Input projection if needed
        if input_dim != hidden_dim:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.input_proj = None
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        self.output_dim = hidden_dim * (2 if bidirectional else 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM.
        
        Args:
            x: Input tensor (B, T, input_dim)
            
        Returns:
            Last hidden state (B, output_dim)
        """
        # Project input if necessary
        if self.input_proj is not None:
            x = self.input_proj(x)
        
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Return last hidden state
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            h_n = h_n.view(self.num_layers, 2, x.shape[0], self.hidden_dim)
            h_n = h_n[-1].transpose(0, 1).contiguous()  # (B, 2, hidden_dim)
            h_n = h_n.view(x.shape[0], -1)  # (B, 2*hidden_dim)
        else:
            h_n = h_n[-1]  # (B, hidden_dim)
        
        return h_n
    
    def get_output_dim(self) -> int:
        """Get output dimension."""
        return self.output_dim
