import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """
    正弦位置编码，用于时间步编码
    """
    def __init__(self, dim):
        super(SinusoidalPositionEmbeddings, self).__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    """
    UNet基础块
    """
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        
        # 时间嵌入投影
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        
    def forward(self, x, t):
        x = self.relu(self.bn1(self.conv1(x)))
        # 添加时间信息
        t = self.time_mlp(t)
        x = x + t[:, :, None, None]
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class DownBlock(nn.Module):
    """
    下采样块
    """
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super(DownBlock, self).__init__()
        self.block = Block(in_ch, out_ch, time_emb_dim)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, t):
        x = self.block(x, t)
        pooled = self.pool(x)
        return x, pooled


class UpBlock(nn.Module):
    """
    上采样块
    """
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super(UpBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.block = Block(in_ch, out_ch, time_emb_dim)

    def forward(self, x, skip, t):
        x = self.up(x)
        # 解决尺寸不匹配问题
        if x.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.block(x, t)
        return x


class ConditionalUNet(nn.Module):
    """
    条件扩散模型 (Conditional DDPM/UNet)
    """
    def __init__(self, in_channels=1, out_channels=1, time_emb_dim=16, d_model=64):
        """
        Args:
            in_channels: 输入通道数（通常是噪声图像）
            out_channels: 输出通道数
            time_emb_dim: 时间嵌入维度
            d_model: 条件编码维度（来自Transformer）
        """
        super(ConditionalUNet, self).__init__()
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
        )
        
        # 条件编码适配器（用于将Transformer输出适配到不同分辨率）
        self.cond_adapters = nn.ModuleList([
            nn.Conv2d(d_model, 16, 1),   # 适配到第一层
            nn.Conv2d(d_model, 32, 1),   # 适配到第二层
        ])
        
        # 下采样路径 (进一步减少层数和通道数)
        self.down1 = DownBlock(in_channels, 16, time_emb_dim)
        self.down2 = DownBlock(16, 32, time_emb_dim)
        
        # 中间块
        self.bottleneck = Block(32, 32, time_emb_dim)
        
        # 上采样路径
        self.up1 = UpBlock(32 + 32, 32, time_emb_dim)
        self.up2 = UpBlock(32 + 16, 16, time_emb_dim)
        
        # 输出层
        self.output = nn.Conv2d(16, out_channels, 1)

    def forward(self, x, timestep, cond):
        """
        Args:
            x: 噪声图像，形状为 (B, 1, H_out, W_out)
            timestep: 时间步，形状为 (B,)
            cond: 条件编码（来自Transformer），形状为 (B, d_model, H, W)
            
        Returns:
            噪声预测，形状为 (B, 1, H_out, W_out)
        """
        # 记录原始输入尺寸
        orig_size = x.shape[2:]
        
        # 时间嵌入
        t = self.time_mlp(timestep)
        
        # 适配条件编码到不同分辨率
        cond1 = self.cond_adapters[0](cond)  # 适配到第一层分辨率
        cond2 = self.cond_adapters[1](cond)  # 适配到第二层分辨率
        
        # 下采样过程
        skips = []
        x, skip = self.down1(x, t)
        # 添加条件信息到skip连接
        skip = skip + F.interpolate(cond1, size=skip.shape[2:], mode='bilinear', align_corners=True)
        skips.append(skip)
        
        x, skip = self.down2(x, t)
        # 添加条件信息到skip连接
        skip = skip + F.interpolate(cond2, size=skip.shape[2:], mode='bilinear', align_corners=True)
        skips.append(skip)
        
        # 中间块
        x = self.bottleneck(x, t)
        
        # 上采样过程
        x = self.up1(x, skips[-1], t)
        x = self.up2(x, skips[-2], t)
        
        # 输出
        output = self.output(x)
        
        # 确保输出尺寸与原始输入尺寸一致
        if output.shape[2:] != orig_size:
            output = F.interpolate(output, size=orig_size, mode='bilinear', align_corners=True)
            
        return output
