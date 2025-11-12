import torch
import torch.optim as optim
import os

from models.full_model import ForecastingModel, initialize_model
from utils.losses import CombinedLoss
from utils.metrics import calculate_all_metrics
from utils.logger import create_logger


class Trainer:
    """
    模型训练器
    """
    def __init__(self, model, train_loader, val_loader, config):
        """
        Args:
            model: 待训练的模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            config: 训练配置
        """
        self.model = initialize_model(model)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # 创建日志记录器
        self.logger = create_logger('trainer', config.get('log_dir', './logs'))
        
        # 设置设备
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)
        
        # 记录设备信息
        self.logger.info(f"Using device: {self.device}")
        
        # 损失函数
        self.criterion = CombinedLoss(
            lambda_diff=config.get('lambda_diff', 1.0),
            lambda_pix=config.get('lambda_pix', 0.1),
            lambda_perc=config.get('lambda_perc', 0.01),
            pix_loss_type=config.get('pix_loss_type', 'l1')
        )
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.get('lr_decay_step', 10),
            gamma=config.get('lr_decay_rate', 0.9)
        )
        
        # 训练状态
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        
        # 检查点路径
        self.checkpoint_dir = config.get('checkpoint_dir', './checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 记录模型信息
        self.logger.log_model_info(self.model)
        
        # 记录配置信息
        self.logger.log_config(config)

    def train_epoch(self, epoch):
        """
        训练一个epoch
        
        Returns:
            平均训练损失
        """
        self.model.train()
        total_loss = 0.0
        
        # 添加进度条
        with self.logger.create_progress_bar() as progress:
            task = progress.add_task(f"[cyan]Training Epoch {epoch+1}[/cyan]", total=len(self.train_loader))
            
            for batch_idx, (x_seq, y_gt) in enumerate(self.train_loader):
                x_seq = x_seq.to(self.device)
                y_gt = y_gt.to(self.device)
                
                # 记录输入数据形状（仅在前几个批次记录）
                if batch_idx == 0:
                    self.logger.log_data_shape(x_seq, "Input sequence (x_seq)")
                    self.logger.log_data_shape(y_gt, "Ground truth (y_gt)")
                
                # 清零梯度
                self.optimizer.zero_grad()
                
                # 前向传播
                predicted_noise, true_noise = self.model(x_seq, y_gt, mode='train')
                
                # 记录输出数据形状（仅在前几个批次记录）
                if batch_idx == 0:
                    self.logger.log_data_shape(predicted_noise, "Predicted noise")
                    self.logger.log_data_shape(true_noise, "True noise")
                
                # 计算损失
                loss = self.criterion(predicted_noise, true_noise)
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # 更新参数
                self.optimizer.step()
                
                # 累计损失
                total_loss += loss.item()
                
                # 更新进度条
                progress.update(task, advance=1, description=f"[cyan]Training Epoch {epoch+1}[/cyan] - Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(self.train_loader)
        self.logger.info(f"[green]Epoch {epoch+1} - Training completed.[/green] Average loss: [bold]{avg_loss:.4f}[/bold]")
        return avg_loss

    def validate(self, epoch):
        """
        验证模型
        
        Returns:
            平均验证损失和指标
        """
        self.model.eval()
        total_loss = 0.0
        total_metrics = {'mse': 0, 'rmse': 0, 'psnr': 0, 'ssim': 0}
        
        with torch.no_grad():
            with self.logger.create_progress_bar() as progress:
                task = progress.add_task(f"[magenta]Validation Epoch {epoch+1}[/magenta]", total=len(self.val_loader))
                
                for batch_idx, (x_seq, y_gt) in enumerate(self.val_loader):
                    x_seq = x_seq.to(self.device)
                    y_gt = y_gt.to(self.device)
                    
                    # 前向传播
                    predicted_noise, true_noise = self.model(x_seq, y_gt, mode='train')
                    
                    # 计算损失
                    loss = self.criterion(predicted_noise, true_noise)
                    total_loss += loss.item()
                    
                    # 采样生成图像以计算指标
                    if batch_idx % 10 == 0:  # 每10个批次计算一次指标以节省时间
                        generated = self.model(x_seq, mode='sample')
                        batch_metrics = calculate_all_metrics(generated, y_gt)
                        
                        for key in total_metrics:
                            total_metrics[key] += batch_metrics.get(key, 0)
                    
                    # 更新进度条
                    progress.update(task, advance=1)
        
        # 计算平均值
        avg_loss = total_loss / len(self.val_loader)
        for key in total_metrics:
            total_metrics[key] /= (len(self.val_loader) // 10 + 1)
            
        self.logger.info(f"[green]Epoch {epoch+1} - Validation completed.[/green] Average loss: [bold]{avg_loss:.4f}[/bold]")
        return avg_loss, total_metrics

    def train(self, epochs):
        """
        完整训练过程
        
        Args:
            epochs: 训练轮数
        """
        self.logger.info(f"[bold blue]Starting training for {epochs} epochs[/bold blue]")
        
        with self.logger.create_progress_bar() as progress:
            epoch_task = progress.add_task("[yellow]Overall Training Progress[/yellow]", total=epochs-self.start_epoch)
            
            for epoch in range(self.start_epoch, epochs):
                self.logger.info(f"\n[bold underline]Epoch {epoch+1}/{epochs}[/bold underline]")
                
                # 训练
                train_loss = self.train_epoch(epoch)
                
                # 验证
                val_loss, val_metrics = self.validate(epoch)
                
                # 更新学习率
                self.scheduler.step()
                
                # 打印结果
                self.logger.info(f"[bold blue]Epoch {epoch+1} Results:[/bold blue]")
                self.logger.info(f"  Train Loss: [bold]{train_loss:.4f}[/bold]")
                self.logger.info(f"  Val Loss: [bold]{val_loss:.4f}[/bold]")
                self.logger.info("[bold]  Val Metrics:[/bold]")
                for key, value in val_metrics.items():
                    self.logger.info(f"    {key.upper()}: [bold]{value:.4f}[/bold]")
                
                # 更新总体进度条
                progress.update(epoch_task, advance=1)
                
                # 保存检查点
                self.save_checkpoint(epoch, val_loss)
                
                # 保存最佳模型
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(epoch, val_loss, is_best=True)
                    self.logger.info(f"[bold green]New best model saved with validation loss: {val_loss:.4f}[/bold green]")

    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """
        保存检查点
        
        Args:
            epoch: 当前epoch
            val_loss: 验证损失
            is_best: 是否是最佳模型
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        # 保存定期检查点
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"[green]New best model saved with validation loss: {val_loss:.4f}[/green]")

    def load_checkpoint(self, checkpoint_path):
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点路径
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['val_loss']
        self.logger.info(f"[cyan]Checkpoint loaded. Resume training from epoch {self.start_epoch}[/cyan]")


def create_trainer(config):
    """
    创建训练器实例
    
    Args:
        config: 训练配置
        
    Returns:
        Trainer实例
    """
    # 创建模型
    model = ForecastingModel(
        input_channels=config.get('input_channels', 3),
        d_model=config.get('d_model', 128),
        tcn_dilations=config.get('tcn_dilations', [1, 2, 4, 8]),
        transformer_num_heads=config.get('transformer_num_heads', 8),
        transformer_num_layers=config.get('transformer_num_layers', 4),
        diffusion_time_steps=config.get('diffusion_time_steps', 1000)
    )
    
    # 这里应该传入实际的data_loader，此处仅为示例
    # train_loader = ...
    # val_loader = ...
    
    # trainer = Trainer(model, train_loader, val_loader, config)
    # return trainer
    
    return model  # 简化返回模型本身