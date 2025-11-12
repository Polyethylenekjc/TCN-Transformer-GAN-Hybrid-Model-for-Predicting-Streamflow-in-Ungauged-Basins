import torch
import torch.optim as optim
import torch.nn as nn
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from models.full_model import ForecastingModel, initialize_model
from utils.losses import CombinedLoss, TotalLoss
from utils.metrics import calculate_all_metrics
from utils.logger import create_logger


class RealDataset(torch.utils.data.Dataset):
    """
    真实数据集类
    """
    def __init__(self, data_dir, stations_csv):
        """
        Args:
            data_dir: 数据目录，包含多个子目录，每个子目录包含input_sequence.npy和target_image.npy
            stations_csv: 站点CSV文件路径
        """
        self.data_dir = data_dir
        self.stations_csv = stations_csv
        
        # 获取所有样本目录
        self.sample_dirs = [d for d in os.listdir(data_dir) 
                           if os.path.isdir(os.path.join(data_dir, d))]
        
        # 加载站点数据
        self.stations_df = pd.read_csv(stations_csv)
        self.num_stations = len(self.stations_df)
        
    def __len__(self):
        return len(self.sample_dirs)
        
    def __getitem__(self, idx):
        # 获取样本目录
        sample_dir = os.path.join(self.data_dir, self.sample_dirs[idx])
        
        # 加载输入序列 [T, C, H, W]
        input_seq = np.load(os.path.join(sample_dir, "input_sequence.npy"))
        input_seq = torch.from_numpy(input_seq).float()
        
        # 加载目标图像 [1, H_out, W_out]
        target_image = np.load(os.path.join(sample_dir, "target_image.npy"))
        target_image = torch.from_numpy(target_image).float()
        
        # 生成站点径流值 [N_stations] 或 [N_stations, T_target]
        # 这里我们简化为 [N_stations]，表示单个时间步的观测值
        station_runoffs = torch.randn(self.num_stations)
        
        return input_seq, target_image, station_runoffs


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
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


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
        
        # 混合精度训练设置
        self.use_mixed_precision = config.get('use_mixed_precision', False)
        if self.use_mixed_precision:
            # 使用新API解决FutureWarning
            if hasattr(torch.amp, 'GradScaler'):
                self.scaler = torch.amp.GradScaler('cuda')
            else:
                # 回退到旧API以防新API不可用
                self.scaler = torch.cuda.amp.GradScaler()
            self.logger.info("Mixed precision training enabled")
        
        # 梯度累积设置
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
        if self.gradient_accumulation_steps > 1:
            self.logger.info(f"Gradient accumulation enabled with {self.gradient_accumulation_steps} steps")
        
        # 损失函数
        if config.get('use_station_loss', False):
            self.criterion = TotalLoss(
                stations_csv=config.get('csv_file', 'data/stations.csv'),
                resolution=config.get('resolution', 1.0),
                lambda_recon=config.get('lambda_recon', 1.0),
                lambda_station=config.get('lambda_station', 0.1),
                recon_loss_type=config.get('recon_loss_type', 'l1'),
                station_loss_type=config.get('station_loss_type', 'l1'),
                lat_range=(config.get('lat_min', 0.0), config.get('lat_max', 256.0)),
                lon_range=(config.get('lon_min', 0.0), config.get('lon_max', 256.0))
            )
        else:
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

    def train(self, epochs):
        """
        完整训练过程
        
        Args:
            epochs: 训练轮数
        """
        self.logger.info(f"Starting training for {epochs} epochs")
        
        # 总体训练进度条
        overall_progress = tqdm(range(self.start_epoch, epochs), desc="Training", 
                               leave=True, position=0)
        
        for epoch in overall_progress:
            
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_metrics = self.validate(epoch)
            
            # 更新学习率
            self.scheduler.step()
            
            # 打印结果
            self.logger.info(f"Epoch {epoch+1} - Validation completed. Average loss: {val_loss:.4f}")
            self.logger.info(f"Epoch {epoch+1} Results:")
            self.logger.info(f"  Train Loss: {train_loss:.4f}")
            self.logger.info(f"  Val Loss: {val_loss:.4f}")
            self.logger.info("  Val Metrics:")
            for key, value in val_metrics.items():
                self.logger.info(f"    {key.upper()}: {value:.4f}")
            
            # 更新总体进度条描述
            overall_progress.set_postfix({
                "Train Loss": f"{train_loss:.4f}",
                "Val Loss": f"{val_loss:.4f}"
            })
            
            # 保存检查点
            self.save_checkpoint(epoch, val_loss)
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, is_best=True)

    def train_epoch(self, epoch):
        """
        训练一个epoch
        
        Returns:
            平均训练损失
        """
        self.model.train()
        total_loss = 0.0
        train_batches = 0
        
        # 创建训练进度条
        progress = tqdm(enumerate(self.train_loader), total=len(self.train_loader), 
                       desc=f"Epoch {epoch+1}/Train", leave=False, position=1)
        
        for batch_idx, (x_seq, y_gt, station_runoffs) in progress:
            x_seq = x_seq.to(self.device)
            y_gt = y_gt.to(self.device)
            station_runoffs = station_runoffs.to(self.device)

            
            # 清零梯度
            if batch_idx % self.gradient_accumulation_steps == 0:
                self.optimizer.zero_grad()
            
            if self.config.get('use_station_loss', False):
                # 使用站点损失的训练方式
                # 前向传播和损失计算（使用混合精度）
                if self.use_mixed_precision:
                    with (torch.amp.autocast('cuda') if hasattr(torch.amp, 'autocast') 
                          else torch.cuda.amp.autocast()):
                        # 直接在加噪前计算站点损失，避免内存消耗
                        result = self.model(x_seq, y_gt, mode='train', target_runoff_values=station_runoffs)
                        
                        if len(result) == 3:
                            predicted_noise, true_noise, station_loss = result
                        else:
                            predicted_noise, true_noise = result
                            station_loss = torch.tensor(0.0, device=self.device)
                        
                        # 计算扩散损失
                        diffusion_loss = self.criterion.diffusion_loss(predicted_noise, true_noise) if hasattr(self.criterion, 'diffusion_loss') else nn.MSELoss()(predicted_noise, true_noise)
                        
                        # 计算总损失
                        lambda_station = self.config.get('lambda_station', 0.1)
                        total_loss_val = diffusion_loss + lambda_station * station_loss
                        
                        # 仅记录重建损失用于日志输出
                        recon_loss = diffusion_loss
                        
                        # 梯度累积时需要对损失进行缩放
                        if self.gradient_accumulation_steps > 1:
                            total_loss_val = total_loss_val / self.gradient_accumulation_steps
                else:
                    # 直接在加噪前计算站点损失，避免内存消耗
                    result = self.model(x_seq, y_gt, mode='train', target_runoff_values=station_runoffs)
                    
                    if len(result) == 3:
                        predicted_noise, true_noise, station_loss = result
                    else:
                        predicted_noise, true_noise = result
                        station_loss = torch.tensor(0.0, device=self.device)
                    
                    # 计算扩散损失
                    diffusion_loss = self.criterion.diffusion_loss(predicted_noise, true_noise) if hasattr(self.criterion, 'diffusion_loss') else nn.MSELoss()(predicted_noise, true_noise)
                    
                    # 计算总损失
                    lambda_station = self.config.get('lambda_station', 0.1)
                    total_loss_val = diffusion_loss + lambda_station * station_loss
                    
                    # 仅记录重建损失用于日志输出
                    recon_loss = diffusion_loss
                    
                    # 梯度累积时需要对损失进行缩放
                    if self.gradient_accumulation_steps > 1:
                        total_loss_val = total_loss_val / self.gradient_accumulation_steps
                
                # 反向传播
                if self.use_mixed_precision:
                    self.scaler.scale(total_loss_val).backward()
                else:
                    total_loss_val.backward()
                
                # 梯度裁剪
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    # 更新参数
                    if self.use_mixed_precision:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                
                total_loss += total_loss_val.item() * self.gradient_accumulation_steps  # 恢复原始损失值
                train_batches += 1
                
                # 更新进度条
                progress.set_postfix({"Loss": f"{total_loss_val.item() * self.gradient_accumulation_steps:.4f}"})
            else:
                # 原始训练方式
                # 前向传播和损失计算（使用混合精度）
                if self.use_mixed_precision:
                    with (torch.amp.autocast('cuda') if hasattr(torch.amp, 'autocast') 
                          else torch.cuda.amp.autocast()):
                        predicted_noise, true_noise = self.model(x_seq, y_gt, mode='train')
                        
                        loss = self.criterion(predicted_noise, true_noise)
                        # 梯度累积时需要对损失进行缩放
                        if self.gradient_accumulation_steps > 1:
                            loss = loss / self.gradient_accumulation_steps
                else:
                    predicted_noise, true_noise = self.model(x_seq, y_gt, mode='train')
                    loss = self.criterion(predicted_noise, true_noise)
                    # 梯度累积时需要对损失进行缩放
                    if self.gradient_accumulation_steps > 1:
                        loss = loss / self.gradient_accumulation_steps
                
                # 反向传播
                if self.use_mixed_precision:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # 梯度裁剪
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    # 更新参数
                    if self.use_mixed_precision:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                
                # 累计损失
                total_loss += loss.item() * self.gradient_accumulation_steps  # 恢复原始损失值
                train_batches += 1
                
                # 更新进度条
                progress.set_postfix({"Loss": f"{loss.item() * self.gradient_accumulation_steps:.4f}"})
                
        avg_loss = total_loss / train_batches
        return avg_loss

    def validate(self, epoch):
        """
        验证模型
        
        Returns:
            平均验证损失和指标
        """
        self.model.eval()
        total_loss = 0.0
        val_batches = 0
        total_metrics = {'mse': 0, 'rmse': 0, 'psnr': 0, 'ssim': 0}
        
        with torch.no_grad():
            # 创建验证进度条
            progress = tqdm(enumerate(self.val_loader), total=len(self.val_loader),
                           desc=f"Epoch {epoch+1}/Val", leave=False, position=1)
            
            for batch_idx, (x_seq, y_gt, station_runoffs) in progress:
                x_seq = x_seq.to(self.device)
                y_gt = y_gt.to(self.device)
                station_runoffs = station_runoffs.to(self.device)
                
                if self.config.get('use_station_loss', False):
                    # 使用站点损失的验证方式
                    # 混合精度推理
                    if self.use_mixed_precision:
                        with (torch.amp.autocast('cuda') if hasattr(torch.amp, 'autocast') 
                              else torch.cuda.amp.autocast()):
                            # 生成图像并计算站点损失
                            result = self.model(x_seq, target_runoff_values=station_runoffs, mode='sample')
                            if isinstance(result, tuple):
                                generated, station_loss = result
                            else:
                                generated = result
                                station_loss = torch.tensor(0.0, device=self.device)
                            
                            # 计算重建损失
                            recon_loss = nn.L1Loss()(generated, y_gt)
                            
                            # 计算总损失
                            lambda_station = self.config.get('lambda_station', 0.1)
                            lambda_recon = self.config.get('lambda_recon', 1.0)
                            total_loss_val = lambda_recon * recon_loss + lambda_station * station_loss
                    else:
                        # 生成图像并计算站点损失
                        result = self.model(x_seq, target_runoff_values=station_runoffs, mode='sample')
                        if isinstance(result, tuple):
                            generated, station_loss = result
                        else:
                            generated = result
                            station_loss = torch.tensor(0.0)
                        
                        # 计算重建损失
                        recon_loss = nn.L1Loss()(generated, y_gt)
                        
                        # 计算总损失
                        lambda_station = self.config.get('lambda_station', 0.1)
                        lambda_recon = self.config.get('lambda_recon', 1.0)
                        total_loss_val = lambda_recon * recon_loss + lambda_station * station_loss
                    
                    total_loss += total_loss_val.item()
                    val_batches += 1
                else:
                    # 原始验证方式
                    # 混合精度推理
                    if self.use_mixed_precision:
                        with (torch.amp.autocast('cuda') if hasattr(torch.amp, 'autocast') 
                              else torch.cuda.amp.autocast()):
                            predicted_noise, true_noise = self.model(x_seq, y_gt, mode='train')
                            loss = self.criterion(predicted_noise, true_noise)
                    else:
                        predicted_noise, true_noise = self.model(x_seq, y_gt, mode='train')
                        loss = self.criterion(predicted_noise, true_noise)
                    total_loss += loss.item()
                    val_batches += 1
                
                # 采样生成图像以计算指标
                if batch_idx % 10 == 0:  # 每10个批次计算一次指标以节省时间
                    # 混合精度采样
                    if self.use_mixed_precision:
                        with (torch.amp.autocast('cuda') if hasattr(torch.amp, 'autocast') 
                              else torch.cuda.amp.autocast()):
                            if not self.config.get('use_station_loss', False):
                                generated = self.model(x_seq, mode='sample')
                    else:
                        if not self.config.get('use_station_loss', False):
                            generated = self.model(x_seq, mode='sample')
                            
                    batch_metrics = calculate_all_metrics(generated, y_gt)
                    
                    for key in total_metrics:
                        total_metrics[key] += batch_metrics.get(key, 0)
                
                # 更新进度条
                progress.set_postfix({"Loss": f"{total_loss / (batch_idx + 1):.4f}"})
        
        # 计算平均值
        avg_loss = total_loss / val_batches
        for key in total_metrics:
            total_metrics[key] /= (val_batches // 10 + 1)
            
        return avg_loss, total_metrics

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
        self.logger.info(f"Checkpoint loaded. Resume training from epoch {self.start_epoch}")


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
        input_height=config.get('input_height', 128),
        input_width=config.get('input_width', 128),
        output_height=config.get('output_height', 256),
        output_width=config.get('output_width', 256),
        d_model=config.get('d_model', 128),
        tcn_dilations=config.get('tcn_dilations', [1, 2, 4, 8]),
        transformer_num_heads=config.get('transformer_num_heads', 8),
        transformer_num_layers=config.get('transformer_num_layers', 4),
        diffusion_time_steps=config.get('diffusion_time_steps', 100),
        stations_csv=config.get('csv_file', None),
        lambda_station=config.get('lambda_station', 0.1),
        lat_range=(config.get('lat_min', 0.0), config.get('lat_max', 256.0)),
        lon_range=(config.get('lon_min', 0.0), config.get('lon_max', 256.0)),
        resolution=config.get('resolution', 1.0)
    )
    
    # 创建数据集和数据加载器
    stations_csv_path = config.get('csv_file', 'data/stations.csv')
    
    train_dataset = RealDataset(
        data_dir=config.get('train_data_dir', 'data/train'),
        stations_csv=stations_csv_path
    )
    val_dataset = RealDataset(
        data_dir=config.get('val_data_dir', 'data/val'),
        stations_csv=stations_csv_path
    )
    
    train_loader = create_data_loader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = create_data_loader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    trainer = Trainer(model, train_loader, val_loader, config)
    return trainer