import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1D(nn.Module):
    """
    因果一维卷积，确保不会看到未来的信息
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1D, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0, dilation=dilation)
        
    def forward(self, x):
        # x shape: (B, C, T)
        x = F.pad(x, (self.padding, 0))  # 左侧填充
        return self.conv(x)


class TCNBlock(nn.Module):
    """
    TCN基本块，包含残差连接
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(TCNBlock, self).__init__()
        self.conv1 = CausalConv1D(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = CausalConv1D(out_channels, out_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 如果输入输出通道不匹配，则使用1x1卷积进行匹配
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (B, C, T)
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            residual = self.downsample(residual)
            
        out += residual
        out = self.relu(out)
        return out


class TemporalTCN(nn.Module):
    """
    时序编码器 (TCN)
    对每个空间位置的时序做 TCN（1D causal/dilated conv）
    """
    def __init__(self, d_model, num_channels=[64, 64], kernel_size=3, dilations=[1, 2]):
        """
        Args:
            d_model: 输入特征维度
            num_channels: 每层的通道数
            kernel_size: 卷积核大小
            dilations: 膨胀系数
        """
        super(TemporalTCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            in_channels = d_model if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            dilation = dilations[i] if i < len(dilations) else 2 ** i
            layers.append(TCNBlock(in_channels, out_channels, kernel_size, dilation))
            
        self.network = nn.Sequential(*layers)
        self.output_dim = num_channels[-1] if num_channels else d_model

    def forward(self, x):
        """
        Args:
            x: 时空特征，形状为 (B, T, d_model, H, W)
            
        Returns:
            时序编码后的特征，形状为 (B, d_model, H, W)
        """
        B, T, d_model, H, W = x.shape
        # 将空间维度合并，对每个空间位置独立处理时序
        x = x.permute(0, 3, 4, 2, 1)  # (B, H, W, d_model, T)
        x = x.contiguous().view(B * H * W, d_model, T)  # (B*H*W, d_model, T)
        
        # 应用TCN
        out = self.network(x)  # (B*H*W, out_channels, T)
        
        # 取最后一个时间步的输出
        out = out[:, :, -1]  # (B*H*W, out_channels)
        
        # 恢复空间结构
        out = out.view(B, H, W, self.output_dim)  # (B, H, W, out_channels)
        out = out.permute(0, 3, 1, 2)  # (B, out_channels, H, W)
        
        return out