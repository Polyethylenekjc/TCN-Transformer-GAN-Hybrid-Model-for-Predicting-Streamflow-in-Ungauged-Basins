"""Baseline comparison models: CNN-LSTM, CNN-BP (MLP), CNN-Attention-GRU.

Each baseline prepends a lightweight CNN encoder to compress the high-res
input (e.g. 256×256×10) into a compact feature vector before feeding into
LSTM / MLP / Attention-GRU.  The CNN front-end dramatically reduces the
number of FLOPs in the temporal/feedforward stages.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict


# ---------------------------------------------------------------------------
# Shared lightweight CNN reducer
# 3 × (Conv-BN-ReLU with stride-2) + AdaptiveAvgPool → (B*T, out_ch, 8, 8)
# ---------------------------------------------------------------------------

class LightweightCNNReducer(nn.Module):
    """3-layer strided CNN that compresses (B*T, C_in, H, W) → (B*T, out_ch, 8, 8).

    256×256×10  →  conv s2  →  128×128×32
                →  conv s2  →  64×64×32
                →  conv s2  →  32×32×out_ch
                →  adaptive pool  →  8×8×out_ch

    Default out_ch=16 ⇒ flat_dim = 16×8×8 = 1024  (vs old 10×16×16 = 2560)
    """

    def __init__(self, in_channels: int = 10, out_channels: int = 16,
                 target_size: int = 8):
        super().__init__()
        self.target_size = target_size
        self.out_channels = out_channels
        self.encoder = nn.Sequential(
            # layer 1: C_in → 32, stride 2
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # layer 2: 32 → 32, stride 2
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # layer 3: 32 → out_channels, stride 2
            nn.Conv2d(32, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((target_size, target_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.pool(x)
        return x


# ---------------------------------------------------------------------------
# 1. Pure LSTM Model
# ---------------------------------------------------------------------------

class PureLSTMModel(nn.Module):
    """CNN-LSTM model.

    Pipeline:
        CNN(stride-2×3) → Flatten → Linear → LSTM → Linear → Reshape → Upsample
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        model_cfg = config.get('model', {})
        data_cfg = config.get('data', {})

        self.input_channels = model_cfg.get('input_channels', 10)
        self.hidden_dim = model_cfg.get('hidden_dim', 64)
        self.num_layers = model_cfg.get('num_layers', 2)
        self.window_size = data_cfg.get('window_size', 5)

        img_size = data_cfg.get('image_size', [256, 256])
        if isinstance(img_size, int):
            self.output_h = img_size
            self.output_w = img_size
        else:
            self.output_h = int(img_size[0])
            self.output_w = int(img_size[1])
        upscale = data_cfg.get('upscale_factor', 1)
        self.output_h *= upscale
        self.output_w *= upscale

        # Lightweight CNN reducer  (→ 16ch × 8×8 = 1024)
        self.reduce_size = 8
        self.cnn_out_ch = 16
        self.reducer = LightweightCNNReducer(self.input_channels, self.cnn_out_ch,
                                             self.reduce_size)
        flat_dim = self.cnn_out_ch * self.reduce_size * self.reduce_size  # 1024

        # Input projection
        self.input_proj = nn.Linear(flat_dim, self.hidden_dim)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=0.2 if self.num_layers > 1 else 0,
            batch_first=True,
        )

        # Output head
        self.decode_size = 8
        self.head_proj = nn.Linear(self.hidden_dim, 64 * self.decode_size * self.decode_size)
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = images.shape
        x = images.reshape(B * T, C, H, W)
        x = self.reducer(x)                          # (B*T, 16, 8, 8)
        x = x.reshape(B, T, -1)                      # (B, T, 1024)
        x = self.input_proj(x)                        # (B, T, hidden)
        _, (h_n, _) = self.lstm(x)                    # h_n: (layers, B, hidden)
        h = h_n[-1]                                   # (B, hidden)
        feat = self.head_proj(h)                      # (B, 64*8*8)
        feat = feat.view(B, 64, self.decode_size, self.decode_size)
        out = self.decoder(feat)                      # (B, 1, 8, 8)
        out = F.interpolate(out, size=(self.output_h, self.output_w),
                            mode='bilinear', align_corners=False)
        return out

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 2. Pure BP (MLP) Model
# ---------------------------------------------------------------------------

class PureBPModel(nn.Module):
    """CNN-BP / MLP model.

    Pipeline:
        CNN(stride-2×3) → Flatten (all T steps concat) → MLP → Reshape → Upsample
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        model_cfg = config.get('model', {})
        data_cfg = config.get('data', {})

        self.input_channels = model_cfg.get('input_channels', 10)
        self.hidden_dim = model_cfg.get('hidden_dim', 64)
        self.window_size = data_cfg.get('window_size', 5)

        img_size = data_cfg.get('image_size', [256, 256])
        if isinstance(img_size, int):
            self.output_h = img_size
            self.output_w = img_size
        else:
            self.output_h = int(img_size[0])
            self.output_w = int(img_size[1])
        upscale = data_cfg.get('upscale_factor', 1)
        self.output_h *= upscale
        self.output_w *= upscale

        # Lightweight CNN reducer  (→ 16ch × 8×8 = 1024 per step)
        self.reduce_size = 8
        self.cnn_out_ch = 16
        self.reducer = LightweightCNNReducer(self.input_channels, self.cnn_out_ch,
                                             self.reduce_size)

        per_step = self.cnn_out_ch * self.reduce_size * self.reduce_size  # 1024
        flat_dim = self.window_size * per_step  # 5*1024=5120  (was 5*10*16*16=12800)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(flat_dim, self.hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2),
            nn.ReLU(inplace=True),
        )

        self.decode_size = 8
        self.head_proj = nn.Linear(self.hidden_dim * 2, 64 * self.decode_size * self.decode_size)
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = images.shape
        x = images.reshape(B * T, C, H, W)
        x = self.reducer(x)                           # (B*T, 16, 8, 8)
        x = x.reshape(B, -1)                          # (B, T*1024)
        x = self.mlp(x)                               # (B, hidden*2)
        feat = self.head_proj(x)                      # (B, 64*8*8)
        feat = feat.view(B, 64, self.decode_size, self.decode_size)
        out = self.decoder(feat)                      # (B, 1, 8, 8)
        out = F.interpolate(out, size=(self.output_h, self.output_w),
                            mode='bilinear', align_corners=False)
        return out

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 3. Attention-GRU Model
# ---------------------------------------------------------------------------

class TemporalAttention(nn.Module):
    """Scaled dot-product attention over GRU hidden states."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = math.sqrt(hidden_dim)

    def forward(self, gru_outputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gru_outputs: (B, T, hidden_dim)
        Returns:
            context: (B, hidden_dim) — attention-weighted summary
        """
        Q = self.query(gru_outputs)   # (B, T, H)
        K = self.key(gru_outputs)     # (B, T, H)
        V = self.value(gru_outputs)   # (B, T, H)

        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale   # (B, T, T)
        weights = torch.softmax(scores, dim=-1)                   # (B, T, T)
        context = torch.bmm(weights, V)                           # (B, T, H)

        # Aggregate over time (take mean of attended outputs)
        return context.mean(dim=1)  # (B, H)


class AttentionGRUModel(nn.Module):
    """CNN-Attention-GRU model.

    Pipeline:
        CNN(stride-2×3) → Flatten → Linear → GRU → Attention → Linear → Reshape → Upsample
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        model_cfg = config.get('model', {})
        data_cfg = config.get('data', {})

        self.input_channels = model_cfg.get('input_channels', 10)
        self.hidden_dim = model_cfg.get('hidden_dim', 64)
        self.num_layers = model_cfg.get('num_layers', 2)
        self.window_size = data_cfg.get('window_size', 5)

        img_size = data_cfg.get('image_size', [256, 256])
        if isinstance(img_size, int):
            self.output_h = img_size
            self.output_w = img_size
        else:
            self.output_h = int(img_size[0])
            self.output_w = int(img_size[1])
        upscale = data_cfg.get('upscale_factor', 1)
        self.output_h *= upscale
        self.output_w *= upscale

        # Lightweight CNN reducer  (→ 16ch × 8×8 = 1024)
        self.reduce_size = 8
        self.cnn_out_ch = 16
        self.reducer = LightweightCNNReducer(self.input_channels, self.cnn_out_ch,
                                             self.reduce_size)
        flat_dim = self.cnn_out_ch * self.reduce_size * self.reduce_size  # 1024

        # Input projection
        self.input_proj = nn.Linear(flat_dim, self.hidden_dim)

        # GRU
        self.gru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=0.2 if self.num_layers > 1 else 0,
            batch_first=True,
        )

        # Temporal attention
        self.attention = TemporalAttention(self.hidden_dim)

        # Decode
        self.decode_size = 8
        self.head_proj = nn.Linear(self.hidden_dim, 64 * self.decode_size * self.decode_size)
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = images.shape
        x = images.reshape(B * T, C, H, W)
        x = self.reducer(x)                           # (B*T, 16, 8, 8)
        x = x.reshape(B, T, -1)                       # (B, T, 1024)
        x = self.input_proj(x)                         # (B, T, hidden)
        gru_out, _ = self.gru(x)                       # (B, T, hidden)
        context = self.attention(gru_out)              # (B, hidden)
        feat = self.head_proj(context)                 # (B, 64*8*8)
        feat = feat.view(B, 64, self.decode_size, self.decode_size)
        out = self.decoder(feat)                       # (B, 1, 8, 8)
        out = F.interpolate(out, size=(self.output_h, self.output_w),
                            mode='bilinear', align_corners=False)
        return out

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

BASELINE_MODELS = {
    'LSTM': PureLSTMModel,
    'BP': PureBPModel,
    'Attention-GRU': AttentionGRUModel,
}


def build_baseline_model(name: str, config: Dict) -> nn.Module:
    """Build a baseline model by name.

    Args:
        name: One of 'LSTM', 'BP', 'Attention-GRU'
        config: Full YAML config dict

    Returns:
        nn.Module instance
    """
    if name not in BASELINE_MODELS:
        raise ValueError(f"Unknown baseline model: {name}. Choose from {list(BASELINE_MODELS)}")
    return BASELINE_MODELS[name](config)
