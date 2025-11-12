import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def forward(self, x, mask=None):
        batch_size, seq_length, d_model = x.size()
        
        Q = self.W_q(x).view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, d_model)
        output = self.W_o(attn_output)
        return output


class PositionWiseFeedForward(nn.Module):
    """
    位置前馈网络
    """
    def __init__(self, d_model, hidden_dim):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    """
    位置编码
    """
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        # 如果序列长度超过了预定义的最大长度，则进行截断
        if x.size(1) > self.pe.size(1):
            return x
        return x + self.pe[:, :x.size(1)]


class TransformerEncoderLayer(nn.Module):
    """
    Transformer编码器层
    """
    def __init__(self, d_model, num_heads, hidden_dim):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, hidden_dim)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        attn_output = self.attention(x, mask)
        x = self.norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer编码器
    将时序后的特征做 patch/flatten 后输入 Transformer
    """
    def __init__(self, d_model=64, num_heads=4, num_layers=2, hidden_dim=128, max_seq_length=256):
        """
        Args:
            d_model: 特征维度
            num_heads: 注意力头数
            num_layers: 编码器层数
            hidden_dim: 前馈网络隐藏层维度
            max_seq_length: 最大序列长度
        """
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, hidden_dim) 
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        Args:
            x: 输入特征，形状为 (B, d_model, H, W)
            
        Returns:
            编码后的特征，形状为 (B, d_model, H, W)
        """
        B, d_model, H, W = x.shape
        # 将空间位置作为tokens
        x = x.view(B, d_model, H*W).permute(0, 2, 1)  # (B, H*W, d_model)
        
        # 添加位置编码
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # 通过Transformer层
        for layer in self.encoder_layers:
            x = layer(x)
            
        # 恢复空间结构
        x = x.permute(0, 2, 1).view(B, d_model, H, W)  # (B, d_model, H, W)
        return x
