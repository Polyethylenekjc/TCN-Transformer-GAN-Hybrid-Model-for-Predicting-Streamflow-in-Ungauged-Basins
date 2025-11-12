import logging
import os
import sys
from torchsummary import summary
import torch
from datetime import datetime
from rich.logging import RichHandler
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import glob


class ModelLogger:
    """
    模型训练日志记录器
    """
    def __init__(self, name, log_dir='./logs', level=logging.INFO, max_log_files=10):
        """
        初始化日志记录器
        
        Args:
            name: 日志记录器名称
            log_dir: 日志文件目录
            level: 日志级别
            max_log_files: 最大日志文件数量
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # 清除现有处理程序以避免重复
        self.logger.handlers.clear()
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 清理多余的日志文件，只保留最近的max_log_files个
        self._cleanup_old_logs(log_dir, max_log_files)
        
        # 文件处理器 - 使用指定的时间格式
        file_handler = logging.FileHandler(os.path.join(log_dir, f'{name}_{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}.log'))
        file_handler.setLevel(level)
        
        # 控制台处理器 - 使用Rich提供彩色输出
        console_handler = RichHandler(show_time=True, show_path=False, markup=True)
        console_handler.setLevel(level)
        
        # 格式化器
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        # 添加处理器到日志记录器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # 创建控制台实例用于彩色输出
        self.console = Console()
    
    def _cleanup_old_logs(self, log_dir, max_log_files):
        """
        清理旧的日志文件，只保留最近的max_log_files个
        
        Args:
            log_dir: 日志目录
            max_log_files: 最大日志文件数量
        """
        # 获取所有日志文件
        log_files = glob.glob(os.path.join(log_dir, f'*.*.log'))
        
        # 按修改时间排序
        log_files.sort(key=os.path.getmtime)
        
        # 删除多余的日志文件
        if len(log_files) > max_log_files:
            files_to_delete = log_files[:len(log_files) - max_log_files]
            for file_path in files_to_delete:
                try:
                    os.remove(file_path)
                    print(f"Deleted old log file: {file_path}")
                except OSError as e:
                    print(f"Error deleting log file {file_path}: {e}")
    
    def info(self, message):
        """记录INFO级别日志"""
        self.logger.info(message)
    
    def warning(self, message):
        """记录WARNING级别日志"""
        self.logger.warning(message)
    
    def error(self, message):
        """记录ERROR级别日志"""
        self.logger.error(message)
    
    def log_model_info(self, model):
        """
        记录模型信息
        
        Args:
            model: PyTorch模型
        """
        self.info("[bold blue]=[/bold blue]" * 50)
        self.info(f"[bold green]Model:[/bold green] {model.__class__.__name__}")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.info(f"[bold cyan]Total parameters:[/bold cyan] {total_params:,}")
        self.info(f"[bold cyan]Trainable parameters:[/bold cyan] {trainable_params:,}")
        self.info("[bold blue]=[/bold blue]" * 50)
    
    def log_data_shape(self, data, name="Data"):
        """
        记录数据形状
        
        Args:
            data: 输入数据 (tensor or tuple of tensors)
            name: 数据名称
        """
        if isinstance(data, (list, tuple)):
            for i, d in enumerate(data):
                if torch.is_tensor(d):
                    self.info(f"[yellow]{name}[{i}][/yellow] shape: {d.shape}")
                else:
                    self.info(f"[yellow]{name}[{i}[/yellow]: {type(d)}")
        elif torch.is_tensor(data):
            self.info(f"[yellow]{name}[/yellow] shape: {data.shape}")
        else:
            self.info(f"[yellow]{name}[/yellow]: {type(d)}")
    
    def log_config(self, config):
        """
        记录配置信息
        
        Args:
            config: 配置字典
        """
        self.info("[bold magenta]Configuration:[/bold magenta]")
        for key, value in config.items():
            self.info(f"  [green]{key}[/green]: {value}")


def create_logger(name, log_dir='./logs', level=logging.INFO, max_log_files=10):
    """
    创建日志记录器的便捷函数
    
    Args:
        name: 日志记录器名称
        log_dir: 日志文件目录
        level: 日志级别
        max_log_files: 最大日志文件数量
        
    Returns:
        ModelLogger实例
    """
    return ModelLogger(name, log_dir, level, max_log_files)