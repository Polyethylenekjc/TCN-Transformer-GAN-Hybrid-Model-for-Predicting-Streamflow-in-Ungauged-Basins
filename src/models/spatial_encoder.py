import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """
    残差块，用于提高空间编码器的表达能力
    """
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResBlock, self).__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SpatialEncoder(nn.Module):
    """
    空间特征提取器 (Shared CNN)
    对每天的多通道图像做同一套轻量 CNN 编码，得到每帧的低维空间特征图
    """
    def __init__(self, input_channels, d_model=64, downsample_steps=2):
        """
        Args:
            input_channels: 输入图像的通道数
            d_model: 输出特征图的通道数
            downsample_steps: 下采样步数
        """
        super(SpatialEncoder, self).__init__()
        self.d_model = d_model
        
        # 初始卷积层
        layers = [
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        ]
        
        # 添加下采样残差块
        in_channels = 32
        for i in range(downsample_steps):
            layers.append(ResBlock(in_channels, 64 if i == downsample_steps-1 else in_channels, downsample=True))
            in_channels = 64 if i == downsample_steps-1 else in_channels
            
        # 添加保持分辨率的残差块
        for i in range(2):
            layers.append(ResBlock(in_channels, d_model if i == 1 else in_channels))
            
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: 输入图像，形状为 (B, T, C, H, W)
            
        Returns:
            特征图，形状为 (B, T, d_model, H', W')
        """
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)  # (B*T, C, H, W)
        features = self.encoder(x)  # (B*T, d_model, H', W')
        _, d, h, w = features.shape
        features = features.view(B, T, d, h, w)  # (B, T, d_model, H', W')
        return features
