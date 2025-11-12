import torch
import torchvision.transforms as transforms
import numpy as np
import random


class DataPreprocessor:
    """
    数据预处理类
    """
    def __init__(self, normalize_method='z_score'):
        """
        Args:
            normalize_method: 归一化方法 ('z_score' 或 'min_max')
        """
        self.normalize_method = normalize_method
        self.stats = {}  # 存储各通道的统计信息

    def fit_stats(self, data_loader):
        """
        从数据集中计算统计数据
        
        Args:
            data_loader: 数据加载器
        """
        if self.normalize_method == 'z_score':
            self._compute_zscore_stats(data_loader)
        elif self.normalize_method == 'min_max':
            self._compute_minmax_stats(data_loader)

    def _compute_zscore_stats(self, data_loader):
        """
        计算z-score归一化所需的均值和标准差
        """
        # 初始化统计量
        channels = None
        mean_sum = None
        squared_diff_sum = None
        total_samples = 0
        
        # 遍历数据集计算均值和标准差
        for batch_idx, (x_seq, y_gt) in enumerate(data_loader):
            # x_seq形状: (B, T, C, H, W)
            batch_size = x_seq.size(0)
            total_samples += batch_size
            
            if channels is None:
                channels = x_seq.size(2)  # C维度
                mean_sum = torch.zeros(channels)
                squared_diff_sum = torch.zeros(channels)
            
            # 计算每个通道的均值
            for c in range(channels):
                channel_data = x_seq[:, :, c, :, :]  # (B, T, H, W)
                mean_sum[c] += channel_data.mean().item() * batch_size
                
        # 计算总体均值
        mean = mean_sum / total_samples
        
        # 第二次遍历计算标准差
        for batch_idx, (x_seq, y_gt) in enumerate(data_loader):
            batch_size = x_seq.size(0)
            for c in range(channels):
                channel_data = x_seq[:, :, c, :, :]  # (B, T, H, W)
                squared_diff_sum[c] += ((channel_data - mean[c]) ** 2).sum().item()
                
        # 计算标准差
        std = torch.sqrt(squared_diff_sum / total_samples)
        
        self.stats['mean'] = mean
        self.stats['std'] = std

    def _compute_minmax_stats(self, data_loader):
        """
        计算min-max归一化所需的最小值和最大值
        """
        min_vals = None
        max_vals = None
        
        # 遍历数据集计算最小值和最大值
        for batch_idx, (x_seq, y_gt) in enumerate(data_loader):
            channels = x_seq.size(2)  # C维度
            
            if min_vals is None:
                min_vals = torch.full((channels,), float('inf'))
                max_vals = torch.full((channels,), float('-inf'))
            
            # 计算每个通道的最小值和最大值
            for c in range(channels):
                channel_data = x_seq[:, :, c, :, :]  # (B, T, H, W)
                batch_min = channel_data.min()
                batch_max = channel_data.max()
                
                if batch_min < min_vals[c]:
                    min_vals[c] = batch_min
                if batch_max > max_vals[c]:
                    max_vals[c] = batch_max
                    
        self.stats['min'] = min_vals
        self.stats['max'] = max_vals

    def normalize(self, data, channel_axis=1):
        """
        归一化数据
        
        Args:
            data: 输入数据
            channel_axis: 通道轴索引
            
        Returns:
            归一化后的数据
        """
        if self.normalize_method == 'z_score':
            return self._zscore_normalize(data, channel_axis)
        elif self.normalize_method == 'min_max':
            return self._minmax_normalize(data, channel_axis)
        else:
            return data

    def _zscore_normalize(self, data, channel_axis):
        """
        Z-score归一化
        """
        if 'mean' not in self.stats or 'std' not in self.stats:
            raise ValueError("Statistics not computed. Call fit_stats() first.")
            
        normalized_data = data.clone()
        mean = self.stats['mean']
        std = self.stats['std']
        
        # 对每个通道进行归一化
        for c in range(len(mean)):
            if channel_axis == 1:
                normalized_data[:, c, ...] = (data[:, c, ...] - mean[c]) / (std[c] + 1e-8)
            else:
                # 根据实际情况调整索引
                normalized_data[..., c, :, :] = (data[..., c, :, :] - mean[c]) / (std[c] + 1e-8)
                
        return normalized_data

    def _minmax_normalize(self, data, channel_axis):
        """
        Min-Max归一化
        """
        if 'min' not in self.stats or 'max' not in self.stats:
            raise ValueError("Statistics not computed. Call fit_stats() first.")
            
        normalized_data = data.clone()
        min_vals = self.stats['min']
        max_vals = self.stats['max']
        
        # 对每个通道进行归一化
        for c in range(len(min_vals)):
            if channel_axis == 1:
                normalized_data[:, c, ...] = (data[:, c, ...] - min_vals[c]) / (max_vals[c] - min_vals[c] + 1e-8)
            else:
                # 根据实际情况调整索引
                normalized_data[..., c, :, :] = (data[..., c, :, :] - min_vals[c]) / (max_vals[c] - min_vals[c] + 1e-8)
                
        return normalized_data

    def denormalize(self, data, channel_axis=1):
        """
        反归一化数据
        
        Args:
            data: 归一化后的数据
            channel_axis: 通道轴索引
            
        Returns:
            原始范围的数据
        """
        if self.normalize_method == 'z_score':
            return self._zscore_denormalize(data, channel_axis)
        elif self.normalize_method == 'min_max':
            return self._minmax_denormalize(data, channel_axis)
        else:
            return data
            
    def _zscore_denormalize(self, data, channel_axis):
        """
        Z-score反归一化
        """
        if 'mean' not in self.stats or 'std' not in self.stats:
            raise ValueError("Statistics not computed. Call fit_stats() first.")
            
        denormalized_data = data.clone()
        mean = self.stats['mean']
        std = self.stats['std']
        
        # 对每个通道进行反归一化
        for c in range(len(mean)):
            if channel_axis == 1:
                denormalized_data[:, c, ...] = data[:, c, ...] * std[c] + mean[c]
            else:
                # 根据实际情况调整索引
                denormalized_data[..., c, :, :] = data[..., c, :, :] * std[c] + mean[c]
                
        return denormalized_data
        
    def _minmax_denormalize(self, data, channel_axis):
        """
        Min-Max反归一化
        """
        if 'min' not in self.stats or 'max' not in self.stats:
            raise ValueError("Statistics not computed. Call fit_stats() first.")
            
        denormalized_data = data.clone()
        min_vals = self.stats['min']
        max_vals = self.stats['max']
        
        # 对每个通道进行反归一化
        for c in range(len(min_vals)):
            if channel_axis == 1:
                denormalized_data[:, c, ...] = data[:, c, ...] * (max_vals[c] - min_vals[c] + 1e-8) + min_vals[c]
            else:
                # 根据实际情况调整索引
                denormalized_data[..., c, :, :] = data[..., c, :, :] * (max_vals[c] - min_vals[c] + 1e-8) + min_vals[c]
                
        return denormalized_data


class DataAugmentation:
    """
    数据增强类
    """
    def __init__(self, augmentations=None):
        """
        Args:
            augmentations: 增强操作列表
        """
        if augmentations is None:
            self.augmentations = ['flip', 'rotate']
        else:
            self.augmentations = augmentations

    def apply_augmentation(self, x_seq, y_gt=None):
        """
        应用数据增强
        
        Args:
            x_seq: 输入序列 (B, T, C, H, W)
            y_gt: 真实标签 (B, 1, H_out, W_out)
            
        Returns:
            增强后的数据
        """
        # 随机选择是否进行增强
        if random.random() < 0.5:
            return x_seq, y_gt
            
        # 应用各种增强操作
        for aug in self.augmentations:
            if aug == 'flip':
                x_seq, y_gt = self._random_flip(x_seq, y_gt)
            elif aug == 'rotate':
                x_seq, y_gt = self._random_rotate(x_seq, y_gt)
            elif aug == 'crop':
                x_seq, y_gt = self._random_crop(x_seq, y_gt)
            elif aug == 'noise':
                x_seq, y_gt = self._add_noise(x_seq, y_gt)
            elif aug == 'brightness':
                x_seq, y_gt = self._adjust_brightness(x_seq, y_gt)
                
        return x_seq, y_gt

    def _random_flip(self, x_seq, y_gt):
        """
        随机翻转
        """
        if random.random() < 0.5:
            # 水平翻转
            x_seq = torch.flip(x_seq, [-1])
            if y_gt is not None:
                y_gt = torch.flip(y_gt, [-1])
                
        if random.random() < 0.5:
            # 垂直翻转
            x_seq = torch.flip(x_seq, [-2])
            if y_gt is not None:
                y_gt = torch.flip(y_gt, [-2])
                
        return x_seq, y_gt

    def _random_rotate(self, x_seq, y_gt):
        """
        随机旋转
        """
        if random.random() < 0.5:
            k = random.randint(1, 3)  # 90, 180, 或 270度旋转
            x_seq = torch.rot90(x_seq, k, [-2, -1])
            if y_gt is not None:
                y_gt = torch.rot90(y_gt, k, [-2, -1])
                
        return x_seq, y_gt

    def _random_crop(self, x_seq, y_gt):
        """
        随机裁剪
        """
        # 获取输入张量的尺寸
        b, t, c, h, w = x_seq.shape
        
        # 定义裁剪比例 (保留80%-100%的原始尺寸)
        scale = random.uniform(0.8, 1.0)
        crop_h, crop_w = int(h * scale), int(w * scale)
        
        # 随机选择裁剪位置
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)
        
        # 执行裁剪
        x_seq = x_seq[:, :, :, top:top+crop_h, left:left+crop_w]
        
        # 如果有标签数据，也进行相应的裁剪
        if y_gt is not None:
            _, _, _, h_out, w_out = y_gt.shape
            scale_h, scale_w = h_out // h, w_out // w  # 假设输出是输入的整数倍
            crop_h_out, crop_w_out = crop_h * scale_h, crop_w * scale_w
            top_out, left_out = top * scale_h, left * scale_w
            y_gt = y_gt[:, :, :, top_out:top_out+crop_h_out, left_out:left_out+crop_w_out]
            
        # 调整尺寸回原始大小
        x_seq = torch.nn.functional.interpolate(x_seq.view(-1, c, crop_h, crop_w), 
                                               size=(h, w), mode='bilinear', align_corners=False)
        x_seq = x_seq.view(b, t, c, h, w)
        
        if y_gt is not None:
            y_gt = torch.nn.functional.interpolate(y_gt.view(-1, 1, crop_h_out, crop_w_out), 
                                                  size=(h_out, w_out), mode='bilinear', align_corners=False)
            y_gt = y_gt.view(b, 1, 1, h_out, w_out)
                
        return x_seq, y_gt

    def _add_noise(self, x_seq, y_gt):
        """
        添加噪声
        """
        # 添加高斯噪声
        noise = torch.randn_like(x_seq) * 0.01  # 小噪声
        x_seq = x_seq + noise
        return x_seq, y_gt

    def _adjust_brightness(self, x_seq, y_gt):
        """
        调整亮度
        """
        # 随机调整亮度 (0.8到1.2倍)
        factor = random.uniform(0.8, 1.2)
        x_seq = x_seq * factor
        return x_seq, y_gt


def create_data_loader(dataset, batch_size=8, shuffle=True, num_workers=4):
    """
    创建数据加载器
    
    Args:
        dataset: 数据集
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 加载数据的工作进程数
        
    Returns:
        DataLoader对象
    """
    from torch.utils.data import DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


# 测试样例
def test_preprocessing():
    """
    测试数据预处理功能
    """
    print("Testing Data Preprocessor...")
    
    # 创建示例数据
    x_seq = torch.randn(2, 10, 3, 32, 32)  # (B, T, C, H, W)
    y_gt = torch.randn(2, 1, 1, 128, 128)  # (B, 1, H_out, W_out)
    
    # 测试数据预处理器
    preprocessor = DataPreprocessor(normalize_method='z_score')
    
    # 由于我们没有实际的数据加载器，这里只是简单测试归一化功能
    # 在实际使用中，需要先调用fit_stats方法
    
    print("Testing Data Augmentation...")
    
    # 测试数据增强
    augmenter = DataAugmentation(augmentations=['flip', 'rotate', 'crop', 'noise', 'brightness'])
    augmented_x, augmented_y = augmenter.apply_augmentation(x_seq, y_gt)
    
    print(f"Original x_seq shape: {x_seq.shape}")
    print(f"Augmented x_seq shape: {augmented_x.shape}")
    print(f"Original y_gt shape: {y_gt.shape}")
    print(f"Augmented y_gt shape: {augmented_y.shape}")
    
    print("Test completed!")


if __name__ == "__main__":
    test_preprocessing()