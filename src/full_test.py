#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
完整的测试流程，包括数据生成、模型训练、评估和可视化
"""

import torch
import torch.nn as nn
import sys
import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__)))

from models.full_model import ForecastingModel, initialize_model
from utils.losses import CombinedLoss
from utils.metrics import calculate_all_metrics
from utils.logger import create_logger


class MockDataset(torch.utils.data.Dataset):
    """
    模拟数据集，生成类似Moving MNIST的数据
    """
    def __init__(self, size=100, sequence_length=3, input_size=(32, 32), output_size=(128, 128)):
        self.size = size
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.output_size = output_size
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # 生成随机移动的简单形状（圆形）
        sequence = torch.zeros(self.sequence_length, 3, *self.input_size)
        
        # 随机选择起始位置和移动方向
        center_x = random.randint(5, self.input_size[0] - 6)
        center_y = random.randint(5, self.input_size[1] - 6)
        move_x = random.uniform(-1, 1)
        move_y = random.uniform(-1, 1)
        
        # 生成序列
        for t in range(self.sequence_length):
            frame = torch.zeros(3, *self.input_size)
            
            # 更新位置
            curr_x = int(center_x + move_x * t)
            curr_y = int(center_y + move_y * t)
            
            # 确保位置在图像范围内
            curr_x = max(3, min(self.input_size[0] - 4, curr_x))
            curr_y = max(3, min(self.input_size[1] - 4, curr_y))
            
            # 绘制一个简单的圆形
            for i in range(curr_x-2, curr_x+3):
                for j in range(curr_y-2, curr_y+3):
                    if 0 <= i < self.input_size[0] and 0 <= j < self.input_size[1]:
                        distance = ((i - curr_x) ** 2 + (j - curr_y) ** 2) ** 0.5
                        if distance <= 2:
                            frame[:, i, j] = 1.0
            
            sequence[t] = frame
        
        # 生成对应的目标图像（更大尺寸）
        target = torch.zeros(1, *self.output_size)
        scale_factor = self.output_size[0] // self.input_size[0]
        
        # 在更大尺寸的图像上绘制同样的圆形
        center_x_large = (center_x + move_x * (self.sequence_length - 1)) * scale_factor
        center_y_large = (center_y + move_y * (self.sequence_length - 1)) * scale_factor
        
        for i in range(int(center_x_large)-4, int(center_x_large)+5):
            for j in range(int(center_y_large)-4, int(center_y_large)+5):
                if 0 <= i < self.output_size[0] and 0 <= j < self.output_size[1]:
                    distance = ((i - center_x_large) ** 2 + (j - center_y_large) ** 2) ** 0.5
                    if distance <= 4:
                        target[0, i, j] = 1.0
        
        return sequence, target


def create_data_loader(dataset, batch_size=4, shuffle=True):
    """
    创建数据加载器
    
    Args:
        dataset: 数据集
        batch_size: 批次大小
        shuffle: 是否打乱数据
        
    Returns:
        DataLoader对象
    """
    from torch.utils.data import DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def visualize_samples(x_seq, y_gt, generated=None, batch_idx=0, save_path="./samples"):
    """
    可视化样本
    
    Args:
        x_seq: 输入序列 (B, T, C, H, W)
        y_gt: 真实标签 (B, 1, H_out, W_out)
        generated: 生成的图像 (B, 1, H_out, W_out)
        batch_idx: 批次索引
        save_path: 保存路径
    """
    os.makedirs(save_path, exist_ok=True)
    
    # 可视化输入序列的几个帧
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle(f'Batch {batch_idx} - Input Sequence and Ground Truth')
    
    # 显示输入序列的前5帧
    for i in range(min(5, x_seq.shape[1])):
        frame = x_seq[0, i].permute(1, 2, 0).cpu().numpy()
        axes[0, i].imshow(frame)
        axes[0, i].set_title(f'Frame {i+1}')
        axes[0, i].axis('off')
    
    # 显示真实标签
    gt_image = y_gt[0, 0].cpu().numpy()
    axes[1, 0].imshow(gt_image, cmap='gray')
    axes[1, 0].set_title('Ground Truth')
    axes[1, 0].axis('off')
    
    # 如果有生成结果，显示生成图像
    if generated is not None:
        gen_image = generated[0, 0].cpu().numpy()
        axes[1, 1].imshow(gen_image, cmap='gray')
        axes[1, 1].set_title('Generated')
        axes[1, 1].axis('off')
        
        # 显示差异
        diff = np.abs(gt_image - gen_image)
        axes[1, 2].imshow(diff, cmap='hot')
        axes[1, 2].set_title('Difference')
        axes[1, 2].axis('off')
    
    # 隐藏多余的子图
    for i in range(3, 5):
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"sample_batch_{batch_idx}.png"))
    plt.close()


def train_model(model, train_loader, val_loader, config, logger):
    """
    训练模型
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        config: 配置
        logger: 日志记录器
    """
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    model = model.to(device)
    
    # 损失函数
    criterion = CombinedLoss(
        lambda_diff=config.get('lambda_diff', 1.0),
        lambda_pix=config.get('lambda_pix', 0.1),
        lambda_perc=config.get('lambda_perc', 0.01)
    )
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get('learning_rate', 1e-4))
    
    # 训练循环
    num_epochs = config.get('epochs', 3)
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, (x_seq, y_gt) in enumerate(train_loader):
            x_seq = x_seq.to(device)
            y_gt = y_gt.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            predicted_noise, true_noise = model(x_seq, y_gt, mode='train')
            
            # 计算损失
            loss = criterion(predicted_noise, true_noise)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            # 每隔一定批次记录一次
            if batch_idx % 5 == 0:
                logger.info(f"  Batch {batch_idx}, Loss: {loss.item():.4f}")
                
                # 可视化前几个批次的结果
                if batch_idx < 3:
                    model.eval()
                    with torch.no_grad():
                        generated = model(x_seq, mode='sample')
                        visualize_samples(x_seq, y_gt, generated, batch_idx)
                    model.train()
        
        avg_train_loss = train_loss / train_batches
        logger.info(f"Epoch {epoch+1} - Average Training Loss: {avg_train_loss:.4f}")
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_batches = 0
        total_metrics = {'mse': 0, 'rmse': 0, 'psnr': 0, 'ssim': 0}
        
        with torch.no_grad():
            for batch_idx, (x_seq, y_gt) in enumerate(val_loader):
                x_seq = x_seq.to(device)
                y_gt = y_gt.to(device)
                
                # 前向传播
                predicted_noise, true_noise = model(x_seq, y_gt, mode='train')
                
                # 计算损失
                loss = criterion(predicted_noise, true_noise)
                val_loss += loss.item()
                val_batches += 1
                
                # 计算评估指标
                generated = model(x_seq, mode='sample')
                batch_metrics = calculate_all_metrics(generated, y_gt)
                
                for key in total_metrics:
                    total_metrics[key] += batch_metrics.get(key, 0)
                
                # 可视化验证集结果
                if epoch == num_epochs - 1 and batch_idx < 2:  # 最后一轮可视化
                    visualize_samples(x_seq, y_gt, generated, f"val_{batch_idx}")
        
        avg_val_loss = val_loss / val_batches
        for key in total_metrics:
            total_metrics[key] /= val_batches
            
        logger.info(f"Epoch {epoch+1} - Validation Loss: {avg_val_loss:.4f}")
        logger.info("Validation Metrics:")
        for key, value in total_metrics.items():
            logger.info(f"  {key.upper()}: {value:.4f}")


def main():
    """
    主函数
    """
    # 创建日志记录器
    logger = create_logger('full_test', max_log_files=10)
    logger.info("Starting full test pipeline...")
    
    # 设置随机种子以确保结果可复现
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # 创建配置
    config = {
        'input_channels': 3,
        'input_height': 32,
        'input_width': 32,
        'output_height': 128,
        'output_width': 128,
        'd_model': 64,
        'tcn_dilations': [1, 2],
        'transformer_num_heads': 4,
        'transformer_num_layers': 2,
        'diffusion_time_steps': 100,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'learning_rate': 1e-4,
        'epochs': 2,
        'batch_size': 4
    }
    
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # 创建数据集
    logger.info("Creating datasets...")
    input_size = (config['input_height'], config['input_width'])
    output_size = (config['output_height'], config['output_width'])

    train_dataset = MockDataset(
        size=20,
        sequence_length=3,
        input_size=input_size,
        output_size=output_size
    )
    val_dataset = MockDataset(
        size=8,
        sequence_length=3,
        input_size=input_size,
        output_size=output_size
    )
    
    train_loader = create_data_loader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = create_data_loader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # 创建模型
    logger.info("Creating model...")
    model = ForecastingModel(
        input_channels=config['input_channels'],
        input_height=32,  # 默认值
        input_width=32,   # 默认值
        output_height=128,  # 默认值
        output_width=128,   # 默认值
        d_model=config['d_model'],
        tcn_dilations=config['tcn_dilations'],
        transformer_num_heads=config['transformer_num_heads'],
        transformer_num_layers=config['transformer_num_layers'],
        diffusion_time_steps=config['diffusion_time_steps']
    )
    
    model = initialize_model(model)
    logger.log_model_info(model)
    
    # 训练模型
    logger.info("Starting training...")
    train_model(model, train_loader, val_loader, config, logger)
    
    # 保存模型
    model_path = "./checkpoints/test_model.pth"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    logger.info("Full test pipeline completed successfully!")


if __name__ == "__main__":
    main()