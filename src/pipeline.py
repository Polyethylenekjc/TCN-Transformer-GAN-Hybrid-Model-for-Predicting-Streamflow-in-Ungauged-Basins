#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
端到端推理管线，可以直接从图像目录和站点CSV目录开始处理
支持z-score归一化、模型预测和结果保存
"""

import torch
import numpy as np
import pandas as pd
import os
import yaml
import argparse
from pathlib import Path

from models.full_model import ForecastingModel, initialize_model
from data.preprocessing import DataPreprocessor
from utils.logger import create_logger


class InferencePipeline:
    """
    端到端推理管线
    """
    def __init__(self, config_path, checkpoint_path, input_dir, stations_csv, output_dir):
        """
        初始化推理管线
        
        Args:
            config_path: 配置文件路径
            checkpoint_path: 模型检查点路径
            input_dir: 输入图像目录
            stations_csv: 站点CSV文件路径
            output_dir: 输出结果目录
        """
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.input_dir = input_dir
        self.stations_csv = stations_csv
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 创建日志记录器
        self.logger = create_logger('inference_pipeline')
        
        # 加载配置
        self.config = self._load_config()
        
        # 初始化模型
        self.model = self._init_model()
        
        # 初始化预处理器
        self.preprocessor = DataPreprocessor(normalize_method='z_score')
        
        # 加载统计数据
        self._load_statistics()

    def _load_config(self):
        """
        加载配置文件
        """
        self.logger.info(f"[bold blue]Loading config from {self.config_path}...[/bold blue]")
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # 将嵌套的配置展平为单层字典
        flat_config = {}
        for section, values in config.items():
            if isinstance(values, dict):
                flat_config.update(values)
            else:
                flat_config[section] = values
                
        return flat_config

    def _init_model(self):
        """
        初始化模型
        """
        self.logger.info("[bold blue]Initializing model...[/bold blue]")
        model = ForecastingModel(
            input_channels=self.config['input_channels'],
            input_height=self.config['input_height'],
            input_width=self.config['input_width'],
            output_height=self.config['output_height'],
            output_width=self.config['output_width'],
            d_model=self.config['d_model'],
            tcn_dilations=self.config['tcn_dilations'],
            transformer_num_heads=self.config['transformer_num_heads'],
            transformer_num_layers=self.config['transformer_num_layers'],
            diffusion_time_steps=self.config['diffusion_time_steps']
        )
        
        model = initialize_model(model)
        
        # 加载检查点
        if self.checkpoint_path and os.path.exists(self.checkpoint_path) and not os.path.isdir(self.checkpoint_path):
            self.logger.info(f"[bold blue]Loading checkpoint from {self.checkpoint_path}...[/bold blue]")
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                self.logger.info("[green]Checkpoint loaded successfully[/green]")
            except Exception as e:
                self.logger.warning(f"[bold yellow]Failed to load checkpoint: {str(e)}, using initialized model[/bold yellow]")
        else:
            if self.checkpoint_path:
                self.logger.warning(f"[bold yellow]Checkpoint path {self.checkpoint_path} is invalid or not found, using initialized model[/bold yellow]")
            else:
                self.logger.info("[cyan]No checkpoint provided, using initialized model[/cyan]")
            
        model = model.to(self.config['device'])
        model.eval()
        self.logger.log_model_info(model)
        
        return model

    def _load_statistics(self):
        """
        加载预计算的统计数据（均值和标准差）
        在实际使用中，这些统计数据应该在训练阶段计算并保存
        """
        # 这里我们简单地创建一些虚拟统计数据
        # 在实际应用中，应该从训练集计算这些统计数据
        # 对于输入数据，通道数为input_channels
        self.preprocessor.stats['mean'] = torch.zeros(self.config['input_channels'])
        self.preprocessor.stats['std'] = torch.ones(self.config['input_channels'])

    def _load_input_data(self):
        """
        从目录加载输入数据
        
        Returns:
            输入序列张量 (B, T, C, H, W)
        """
        self.logger.info(f"[bold blue]Loading input data from {self.input_dir}...[/bold blue]")
        
        # 获取所有.npy文件
        npy_files = sorted(list(Path(self.input_dir).glob("*.npy")))
        if not npy_files:
            raise ValueError(f"No .npy files found in {self.input_dir}")
            
        # 加载第一个文件作为示例
        # 在实际应用中，你可能需要处理多个样本
        data_file = npy_files[0]
        data = np.load(data_file)
        
        # 确保数据形状正确
        # 假设数据形状为 (T, C, H, W) 或 (C, H, W)
        if len(data.shape) == 3:
            # 添加时间维度
            data = np.expand_dims(data, axis=0)
            
        # 转换为张量
        data_tensor = torch.from_numpy(data).float()
        
        # 调整维度顺序为 (1, T, C, H, W)
        if len(data_tensor.shape) == 4:
            data_tensor = data_tensor.unsqueeze(0)
            
        self.logger.log_data_shape(data_tensor, "Input data")
        return data_tensor

    def _load_stations_data(self):
        """
        加载站点数据
        
        Returns:
            站点数据DataFrame
        """
        self.logger.info(f"[bold blue]Loading stations data from {self.stations_csv}...[/bold blue]")
        stations_df = pd.read_csv(self.stations_csv)
        self.logger.info(f"[cyan]Loaded {len(stations_df)} stations[/cyan]")
        return stations_df

    def run(self):
        """
        运行推理管线
        """
        self.logger.info("[bold blue]Starting inference pipeline...[/bold blue]")
        
        try:
            # 加载输入数据
            x_seq = self._load_input_data()
            
            # 数据归一化
            self.logger.info("[bold blue]Normalizing input data...[/bold blue]")
            x_seq_normalized = self.preprocessor.normalize(x_seq, channel_axis=2)  # (B, T, C, H, W)
            
            # 移动到设备
            x_seq_normalized = x_seq_normalized.to(self.config['device'])
            
            # 模型预测
            self.logger.info("[bold blue]Running model prediction...[/bold blue]")
            with torch.no_grad():
                generated = self.model.sample(x_seq_normalized)
                
            # 反归一化结果
            self.logger.info("[bold blue]Denormalizing results...[/bold blue]")
            # 创建输出数据的预处理器和统计数据
            output_preprocessor = DataPreprocessor(normalize_method='z_score')
            output_preprocessor.stats['mean'] = torch.zeros(1)  # 只有一个通道
            output_preprocessor.stats['std'] = torch.ones(1)    # 只有一个通道
            generated_denormalized = output_preprocessor.denormalize(generated, channel_axis=1)  # (B, 1, H, W)
            
            # 保存结果
            self._save_results(generated_denormalized)
            
            self.logger.info("[bold green]Inference pipeline completed successfully![/bold green]")
            
        except Exception as e:
            self.logger.error(f"[bold red]Error during inference: {str(e)}[/bold red]")
            raise

    def _save_results(self, results):
        """
        保存结果
        
        Args:
            results: 生成的结果张量 (B, 1, H, W)
        """
        self.logger.info(f"[bold blue]Saving results to {self.output_dir}...[/bold blue]")
        
        # 转换为numpy数组
        results_np = results.cpu().numpy()
        
        # 保存为.npy文件
        output_path = os.path.join(self.output_dir, "predictions.npy")
        np.save(output_path, results_np)
        
        # 保存为.csv文件（可选）
        # 展平结果并保存
        flattened = results_np.flatten()
        csv_path = os.path.join(self.output_dir, "predictions.csv")
        pd.DataFrame(flattened).to_csv(csv_path, index=False, header=["value"])
        
        self.logger.info(f"[green]Results saved to {output_path} and {csv_path}[/green]")


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='Run inference pipeline')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default=None, help='模型检查点路径')
    parser.add_argument('--input-dir', type=str, required=True, help='输入图像目录')
    parser.add_argument('--stations-csv', type=str, required=True, help='站点CSV文件路径')
    parser.add_argument('--output-dir', type=str, required=True, help='输出结果目录')
    
    args = parser.parse_args()
    
    # 创建并运行推理管线
    pipeline = InferencePipeline(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        input_dir=args.input_dir,
        stations_csv=args.stations_csv,
        output_dir=args.output_dir
    )
    
    pipeline.run()


if __name__ == "__main__":
    main()