#!/usr/bin/env python
"""Demo script showing system capabilities."""

import sys
import torch
import yaml
from pathlib import Path

sys.path.insert(0, '.')

from src.utils.config_loader import ConfigLoader
from src.dataset import StreamflowDataset
from src.model import StreamflowPredictionModel
from src.loss import CombinedLoss, calculate_metrics


def demo():
    """Run a demonstration of the system."""
    
    print("\n" + "="*60)
    print("时空径流预测系统 - 演示")
    print("="*60 + "\n")
    
    # Load config
    print("1️⃣  加载配置...")
    config = ConfigLoader.load_config('./data/config.yaml')
    print(f"   ✓ 时间模块: {config['model']['temporal_module']}")
    print(f"   ✓ 输入通道: {config['model']['input_channels']}")
    print(f"   ✓ 隐层维度: {config['model']['hidden_dim']}")
    
    # Load dataset
    print("\n2️⃣  加载数据集...")
    dataset = StreamflowDataset(
        image_dir='./data/images',
        station_dir='./data/stations',
        config=config,
        normalize=True
    )
    print(f"   ✓ 样本数量: {len(dataset)}")
    print(f"   ✓ 时间步长: {config['data']['window_size']}")
    
    # Initialize model
    print("\n3️⃣  初始化模型...")
    model = StreamflowPredictionModel(config)
    total_params = model.get_num_parameters()
    print(f"   ✓ 模型参数: {total_params:,}")
    print(f"   ✓ 内存占用: ~{total_params * 4 / 1e9:.2f}GB (float32)")
    
    # Get a sample
    print("\n4️⃣  获取样本...")
    sample = dataset[0]
    images = sample['images'].unsqueeze(0)
    target = sample['output_image'].unsqueeze(0)
    
    print(f"   ✓ 输入形状: {images.shape}")
    print(f"     (batch=1, time=5, channels=10, height=128, width=128)")
    print(f"   ✓ 目标形状: {target.shape}")
    print(f"     (batch=1, channels=1, height=128, width=128)")
    
    # Forward pass
    print("\n5️⃣  前向传播...")
    model.eval()
    with torch.no_grad():
        output = model(images)
    
    print(f"   ✓ 输出形状: {output.shape}")
    print(f"     (batch=1, channels=1, height=256, width=256)")
    print(f"   ✓ 上采样倍数: 2x (128→256)")
    
    # Calculate loss
    print("\n6️⃣  计算损失...")
    loss_fn = CombinedLoss(config)
    loss_dict = loss_fn(output, target)
    
    print(f"   ✓ 总损失: {loss_dict['total'].item():.6f}")
    print(f"   ✓ 图像损失: {loss_dict['image'].item():.6f}")
    print(f"   ✓ 站点损失: {loss_dict['station'].item():.6f}")
    
    # Calculate metrics
    print("\n7️⃣  计算指标...")
    metrics = calculate_metrics(output, target)
    print(f"   ✓ RMSE: {metrics['RMSE']:.4f}")
    print(f"   ✓ MAE: {metrics['MAE']:.4f}")
    print(f"   ✓ R²: {metrics['R2']:.4f}")
    print(f"   ✓ NSE: {metrics['NSE']:.4f}")
    
    # Show how to use different temporal modules
    print("\n8️⃣  可切换的时间模块...")
    print("   当前: LSTM")
    print("   ")
    print("   切换到Transformer:")
    print("   1. 编辑 data/config.yaml")
    print("   2. 修改: temporal_module: Transformer")
    print("   3. 重新运行训练")
    print("   ")
    print("   系统会自动初始化相应的模块!")
    
    # Training info
    print("\n9️⃣  训练命令...")
    print("   python main.py train --config data/config.yaml")
    print("   ")
    print("   防OOM优化:")
    print("   - 混合精度: enabled")
    print("   - 梯度累积: 配置化")
    print("   - 自动GPU监控: 每10步打印显存使用")
    
    # Evaluation info
    print("\n🔟 评估命令...")
    print("   python main.py evaluate --config data/config.yaml --model output/best_model.pt")
    print("   ")
    print("   输出:")
    print("   - output/predictions/ : 预测的径流图")
    print("   - output/stations_eval.csv : 站点预测对比")
    
    print("\n" + "="*60)
    print("✅ 演示完成!")
    print("="*60 + "\n")


if __name__ == '__main__':
    demo()
