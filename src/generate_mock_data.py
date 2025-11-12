#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
生成模拟训练数据的脚本
"""

import torch
import numpy as np
import pandas as pd
import os
from pathlib import Path


def generate_mock_stations_csv(filepath, num_stations=10):
    """
    生成模拟站点CSV文件
    
    Args:
        filepath: 文件保存路径
        num_stations: 站点数量
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 生成随机站点数据
    station_ids = [f"station_{i+1}" for i in range(num_stations)]
    # 生成纬度范围在39-41度之间
    latitudes = np.random.uniform(39.0, 41.0, num_stations)
    # 生成经度范围在-75到-73度之间
    longitudes = np.random.uniform(-75.0, -73.0, num_stations)
    # 生成随机径流值
    values = np.random.uniform(0.1, 2.0, num_stations)
    
    # 创建DataFrame，使用代码中期望的列名
    df = pd.DataFrame({
        'station_id': station_ids,
        'lat': latitudes,
        'lon': longitudes,
        'runoff': values
    })
    
    # 保存到CSV文件
    df.to_csv(filepath, index=False)
    print(f"Generated mock stations CSV with {num_stations} stations at {filepath}")


def generate_mock_training_data(data_dir, num_samples=100, sequence_length=3, 
                               input_channels=5, input_height=128, input_width=128,
                               output_height=256, output_width=256):
    """
    生成模拟训练数据
    
    Args:
        data_dir: 数据保存目录
        num_samples: 样本数量
        sequence_length: 时间序列长度
        input_channels: 输入通道数
        input_height: 输入高度
        input_width: 输入宽度
        output_height: 输出高度
        output_width: 输出宽度
    """
    os.makedirs(data_dir, exist_ok=True)
    
    for i in range(num_samples):
        # 生成输入序列数据 (T, C, H, W)
        input_seq = np.random.randn(sequence_length, input_channels, input_height, input_width).astype(np.float32)
        
        # 生成目标图像数据 (1, H_out, W_out)
        target_image = np.random.randn(1, output_height, output_width).astype(np.float32)
        
        # 保存数据
        sample_dir = os.path.join(data_dir, f"sample_{i:04d}")
        os.makedirs(sample_dir, exist_ok=True)
        
        np.save(os.path.join(sample_dir, "input_sequence.npy"), input_seq)
        np.save(os.path.join(sample_dir, "target_image.npy"), target_image)
        
        if i % 10 == 0:
            print(f"Generated {i+1}/{num_samples} samples")
    
    print(f"Generated {num_samples} mock training samples in {data_dir}")


def generate_mock_validation_data(data_dir, num_samples=20, sequence_length=3, 
                                 input_channels=5, input_height=128, input_width=128,
                                 output_height=256, output_width=256):
    """
    生成模拟验证数据
    
    Args:
        data_dir: 数据保存目录
        num_samples: 样本数量
        sequence_length: 时间序列长度
        input_channels: 输入通道数
        input_height: 输入高度
        input_width: 输入宽度
        output_height: 输出高度
        output_width: 输出宽度
    """
    os.makedirs(data_dir, exist_ok=True)
    
    for i in range(num_samples):
        # 生成输入序列数据 (T, C, H, W)
        input_seq = np.random.randn(sequence_length, input_channels, input_height, input_width).astype(np.float32)
        
        # 生成目标图像数据 (1, H_out, W_out)
        target_image = np.random.randn(1, output_height, output_width).astype(np.float32)
        
        # 保存数据
        sample_dir = os.path.join(data_dir, f"sample_{i:04d}")
        os.makedirs(sample_dir, exist_ok=True)
        
        np.save(os.path.join(sample_dir, "input_sequence.npy"), input_seq)
        np.save(os.path.join(sample_dir, "target_image.npy"), target_image)
        
        if i % 10 == 0:
            print(f"Generated {i+1}/{num_samples} samples")
    
    print(f"Generated {num_samples} mock validation samples in {data_dir}")


def main():
    """
    主函数
    """
    # 创建数据目录
    data_root = "./data"
    os.makedirs(data_root, exist_ok=True)
    
    # 生成站点CSV文件
    stations_csv_path = os.path.join(data_root, "stations.csv")
    generate_mock_stations_csv(stations_csv_path, num_stations=10)
    
    # 生成训练数据
    train_data_dir = os.path.join(data_root, "train")
    generate_mock_training_data(
        train_data_dir, 
        num_samples=100, 
        sequence_length=3,
        input_channels=5, 
        input_height=128, 
        input_width=128,
        output_height=256, 
        output_width=256
    )
    
    # 生成验证数据
    val_data_dir = os.path.join(data_root, "val")
    generate_mock_validation_data(
        val_data_dir, 
        num_samples=20, 
        sequence_length=3,
        input_channels=5, 
        input_height=128, 
        input_width=128,
        output_height=256, 
        output_width=256
    )
    
    print("Mock data generation completed!")


if __name__ == "__main__":
    main()