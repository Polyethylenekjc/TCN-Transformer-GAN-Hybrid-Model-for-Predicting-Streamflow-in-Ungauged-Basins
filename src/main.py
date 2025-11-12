import torch
import argparse
import sys
import os
import yaml

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__)))

from models.full_model import ForecastingModel, initialize_model
from train.trainer import create_trainer
from utils.logger import create_logger


def load_config(config_path):
    """
    加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    # 将嵌套的配置展平为单层字典以保持向后兼容性
    flat_config = {}
    for section, values in config.items():
        if isinstance(values, dict):
            flat_config.update(values)
        else:
            flat_config[section] = values
    return flat_config


def main():
    """
    主程序入口
    """
    parser = argparse.ArgumentParser(description='Forecasting Model Training')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'sample', 'pipeline'],
                        help='运行模式')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='检查点文件路径')
    parser.add_argument('--input-dir', type=str, default=None,
                        help='输入图像目录 (仅在pipeline模式下使用)')
    parser.add_argument('--stations-csv', type=str, default=None,
                        help='站点CSV文件路径 (仅在pipeline模式下使用)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出结果目录 (仅在pipeline模式下使用)')
    
    args = parser.parse_args()
    
    # 创建日志记录器
    logger = create_logger('main', max_log_files=10)
    
    # 加载配置文件
    logger.info(f"[bold blue]Loading config from {args.config}...[/bold blue]")
    config = load_config(args.config)
    
    # 确保设备设置正确
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        logger.warning("[bold yellow]CUDA not available, falling back to CPU[/bold yellow]")
        config['device'] = 'cpu'
    
    # 根据模式执行不同操作
    if args.mode == 'train':
        logger.info("[bold blue]Starting training mode...[/bold blue]")
        # 创建训练器并开始训练
        trainer = create_trainer(config)
        
        # 如果提供了检查点路径，则加载检查点
        if args.checkpoint and os.path.exists(args.checkpoint):
            trainer.load_checkpoint(args.checkpoint)
            
        # 开始训练
        trainer.train(config['epochs'])
        
    elif args.mode == 'sample':
        logger.info("[bold blue]Starting sampling mode...[/bold blue]")
        # 创建模型
        model = ForecastingModel(
            input_channels=config['input_channels'],
            input_height=config['input_height'],
            input_width=config['input_width'],
            output_height=config['output_height'],
            output_width=config['output_width'],
            d_model=config['d_model'],
            tcn_dilations=config['tcn_dilations'],
            transformer_num_heads=config['transformer_num_heads'],
            transformer_num_layers=config['transformer_num_layers'],
            diffusion_time_steps=config['diffusion_time_steps']
        )
        
        model = initialize_model(model)
        logger.log_model_info(model)
        
        # 如果提供了检查点路径，则加载检查点
        if args.checkpoint and os.path.exists(args.checkpoint):
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            logger.info("[green]Checkpoint loaded successfully[/green]")
        
        # 示例采样过程
        model.eval()
        model = model.to(config['device'])  # 确保模型在正确的设备上
        logger.info(f"[cyan]Model moved to device: {config['device']}[/cyan]")
        with torch.no_grad():
            # 创建示例输入 - 根据配置调整尺寸
            logger.info("[bold blue]Creating sample input tensor...[/bold blue]")
            x_seq = torch.randn(1, config['sequence_length'], config['input_channels'], 
                               config['input_height'], config['input_width'])  # (B, T, C, H, W)
            x_seq = x_seq.to(config['device'])  # 确保输入在正确的设备上
            logger.log_data_shape(x_seq, "Input sequence")
            
            # 生成预测结果
            logger.info("[bold blue]Generating predictions...[/bold blue]")
            generated = model.sample(x_seq)
            logger.log_data_shape(generated, "Generated output")
            
            logger.info("[bold green]Sampling completed successfully![/bold green]")
            
    elif args.mode == 'pipeline':
        logger.info("[bold blue]Starting pipeline mode...[/bold blue]")
        # 检查必要参数
        if not args.input_dir or not args.stations_csv or not args.output_dir:
            logger.error("[bold red]Pipeline mode requires --input-dir, --stations-csv, and --output-dir arguments[/bold red]")
            return
            
        # 导入pipeline模块
        try:
            from pipeline import InferencePipeline
            
            # 创建并运行推理管线
            pipeline = InferencePipeline(
                config_path=args.config,
                checkpoint_path=args.checkpoint,
                input_dir=args.input_dir,
                stations_csv=args.stations_csv,
                output_dir=args.output_dir
            )
            
            pipeline.run()
            
        except ImportError as e:
            logger.error(f"[bold red]Failed to import pipeline module: {str(e)}[/bold red]")
            return
        except Exception as e:
            logger.error(f"[bold red]Error in pipeline mode: {str(e)}[/bold red]")
            raise
            
    else:
        logger.error(f"[bold red]Unknown mode: {args.mode}[/bold red]")


if __name__ == "__main__":
    main()