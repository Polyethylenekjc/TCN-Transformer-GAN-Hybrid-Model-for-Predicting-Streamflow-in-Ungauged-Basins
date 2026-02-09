"""
对比实验脚本：训练并评估 Pure LSTM / BP (MLP) / Attention-GRU 三个基线模型，
与主模型 (CNN-Transformer) 进行对比。

评估两个层面：
  - 站点层面：NSE, MSE, MAE, MAPE, KGE（按四个站点分别保存）
  - 流域层面：像素级 MSE, SSIM, PSNR

功能特性：
  - 丰富的 debug 日志输出（时间、进度、GPU 内存、loss 分量）
  - 中途保存功能：每完成一个模型即保存结果到独立 CSV，支持断点续传
  - 站点级评价指标按四个站点分别保存

基线模型只使用单通道（径流）输入预测单通道（径流）输出。
"""

import os
import sys
import copy
import json
import time
import argparse
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from src.dataset import StreamflowDataset
from src.model import StreamflowPredictionModel
from src.model.baseline_models import build_baseline_model, BASELINE_MODELS
from src.loss import CombinedLoss
from src.evaluate import calculate_ssim, calculate_psnr
from src.utils.config_loader import ConfigLoader


# ════════════════════════════════════════════════════════════
#  站点坐标 → 站点名称映射
#    坎迪:   (181, 66)
#    东吁:   (185, 106)
#    帕本:   (189, 136)   ← 第一次出现
#    曼德勒: (189, 136)   ← 第二次出现（同坐标）
# ════════════════════════════════════════════════════════════
STATION_POSITION_MAP = {
    (181, 66):  ['坎迪'],
    (185, 106): ['东吁'],
    (189, 136): ['帕本', '曼德勒'],
}


def get_station_name(px: int, py: int, occurrence: int = 0) -> str:
    """根据像素坐标 (px, py) 和该坐标在同一日期内的出现次序映射站点名称。"""
    names = STATION_POSITION_MAP.get((px, py))
    if names is None:
        return f'Unknown_({px},{py})'
    if occurrence < len(names):
        return names[occurrence]
    return f'{names[0]}_dup{occurrence}'


# ════════════════════════════════════════════════════════════
#  Logging
# ════════════════════════════════════════════════════════════
def setup_logging(output_dir: Path, debug: bool = False):
    """配置日志：同时输出到控制台和文件。"""
    log_level = logging.DEBUG if debug else logging.INFO
    log_file = output_dir / 'comparison.log'

    root = logging.getLogger()
    root.setLevel(log_level)
    root.handlers.clear()

    fmt = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    return logging.getLogger('CompareExperiment')


# ════════════════════════════════════════════════════════════
#  Checkpoint / 断点续传
# ════════════════════════════════════════════════════════════
CHECKPOINT_FILE = 'comparison_checkpoint.json'


def load_checkpoint(output_dir: Path) -> Dict:
    """加载中途保存的检查点，返回已完成模型列表。"""
    ckpt_path = output_dir / CHECKPOINT_FILE
    if ckpt_path.exists():
        try:
            with open(ckpt_path, 'r', encoding='utf-8') as f:
                ckpt = json.load(f)
            return ckpt
        except Exception as e:
            logging.getLogger('CompareExperiment').warning(
                f"检查点文件损坏，将重新开始: {e}")
    return {'completed_models': []}


def save_checkpoint(output_dir: Path, completed_models: List[str]):
    """保存检查点：记录已完成的模型名称。"""
    ckpt = {
        'completed_models': completed_models,
        'last_update': datetime.now().isoformat(),
    }
    ckpt_path = output_dir / CHECKPOINT_FILE
    with open(ckpt_path, 'w', encoding='utf-8') as f:
        json.dump(ckpt, f, ensure_ascii=False, indent=2)


def save_model_results(
    output_dir: Path,
    model_name: str,
    basin_metrics: Dict,
    station_df: pd.DataFrame,
    per_station_metrics: pd.DataFrame,
):
    """每完成一个模型，将其结果保存到独立目录，防止中断丢失。"""
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([basin_metrics]).to_csv(model_dir / 'basin_metrics.csv', index=False)

    if not station_df.empty:
        station_df.to_csv(model_dir / 'station_eval.csv', index=False)

    if not per_station_metrics.empty:
        per_station_metrics.to_csv(model_dir / 'station_metrics_detailed.csv', index=False)


def load_model_results(output_dir: Path, model_name: str):
    """从磁盘恢复已完成模型的结果。"""
    model_dir = output_dir / model_name

    basin_csv = model_dir / 'basin_metrics.csv'
    basin_metrics = pd.read_csv(basin_csv).iloc[0].to_dict() if basin_csv.exists() else None

    stn_csv = model_dir / 'station_eval.csv'
    station_df = pd.read_csv(stn_csv) if stn_csv.exists() else pd.DataFrame()

    ps_csv = model_dir / 'station_metrics_detailed.csv'
    per_station_metrics = pd.read_csv(ps_csv) if ps_csv.exists() else pd.DataFrame()

    return basin_metrics, station_df, per_station_metrics


# ════════════════════════════════════════════════════════════
#  Collate Function
# ════════════════════════════════════════════════════════════
def _collate_fn(batch):
    images = torch.stack([b['images'] for b in batch], dim=0)
    outputs = torch.stack([b['output_image'] for b in batch], dim=0)
    stations = [b['stations'] for b in batch]
    station_positions = [b['station_positions'] for b in batch]
    dates = [b['date'] for b in batch]
    return {
        'images': images,
        'output_image': outputs,
        'stations': stations,
        'station_positions': station_positions,
        'dates': dates,
    }


# ════════════════════════════════════════════════════════════
#  训练循环
# ════════════════════════════════════════════════════════════
def train_model(
    model: nn.Module,
    dataset: StreamflowDataset,
    config: Dict,
    num_epochs: int,
    model_name: str,
    output_dir: Path,
    single_channel: bool = False,
) -> nn.Module:
    """训练单个模型并返回训练后的模型。

    Args:
        single_channel: 是否只使用第 0 通道（径流）作为输入
    """
    logger = logging.getLogger('CompareExperiment')

    device = torch.device(config.get('train', {}).get('device', 'cpu'))
    model = model.to(device)

    batch_size = config.get('train', {}).get('batch_size', 4)
    num_workers = config.get('train', {}).get('num_workers', 4)
    lr = config.get('train', {}).get('lr', 1e-4)
    use_amp = config.get('system', {}).get('mixed_precision', True)
    log_interval = config.get('system', {}).get('log_interval', 10)

    loss_fn = CombinedLoss(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler() if use_amp else None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        collate_fn=_collate_fn,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float('inf')

    # ── DEBUG: 模型信息 ──
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[{model_name}] --- 模型信息 ---")
    logger.info(f"[{model_name}]   可训练参数: {trainable_params:,} / 总参数: {total_params:,}")
    logger.info(f"[{model_name}]   输入通道: {'1 (仅径流)' if single_channel else config.get('model', {}).get('input_channels', 10)}")
    logger.info(f"[{model_name}]   设备={device} | Batch={batch_size} | LR={lr} | AMP={use_amp}")
    logger.info(f"[{model_name}]   Epochs={num_epochs} | Batches/Epoch={len(loader)}")
    logger.debug(f"[{model_name}]   模型结构:\n{model}")

    logger.info(f"[{model_name}] 开始训练 ...")
    train_start = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        running_image_loss = 0.0
        running_station_loss = 0.0
        n_batch = 0
        batch_losses = []

        for batch_idx, batch in enumerate(loader):
            images = batch['images'].to(device, non_blocking=True)
            output_image = batch['output_image'].to(device, non_blocking=True)

            # ── 单通道：只取径流通道 (channel 0) ──
            if single_channel:
                images = images[:, :, :1, :, :]     # (B, T, 1, H, W)

            station_positions = [
                pos.to(device) if isinstance(pos, torch.Tensor) else pos
                for pos in batch['station_positions']
            ]
            station_runoffs = [
                r.to(device) if isinstance(r, torch.Tensor) else r
                for r in batch['stations']
            ]

            optimizer.zero_grad()
            with amp.autocast(device_type=device.type, enabled=use_amp):
                predictions = model(images)
                loss_dict = loss_fn(predictions, output_image, station_positions, station_runoffs)
                loss = loss_dict['total']

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            batch_loss = loss.item()
            running_loss += batch_loss
            running_image_loss += float(loss_dict.get('image', 0))
            running_station_loss += float(loss_dict.get('station', 0))
            batch_losses.append(batch_loss)
            n_batch += 1

            # ── 每 log_interval 个 batch 输出 ──
            if (batch_idx + 1) % log_interval == 0:
                logger.debug(
                    f"  [{model_name}] E{epoch+1} B{batch_idx+1}/{len(loader)} "
                    f"loss={batch_loss:.6f} "
                    f"(img={loss_dict.get('image', 0):.6f}, "
                    f"stn={loss_dict.get('station', 0):.6f})"
                )

        avg_loss = running_loss / max(n_batch, 1)
        avg_img = running_image_loss / max(n_batch, 1)
        avg_stn = running_station_loss / max(n_batch, 1)
        epoch_time = time.time() - epoch_start

        # ── Epoch 日志 ──
        should_log = (
            (epoch + 1) % max(1, num_epochs // 10) == 0
            or epoch == 0
            or epoch == num_epochs - 1
        )
        if should_log:
            loss_std = float(np.std(batch_losses)) if batch_losses else 0.0
            elapsed_total = time.time() - train_start
            eta_total = elapsed_total / (epoch + 1) * (num_epochs - epoch - 1)
            logger.info(
                f"[{model_name}] Epoch {epoch+1:3d}/{num_epochs} | "
                f"Loss={avg_loss:.6f} (std={loss_std:.6f}) | "
                f"img={avg_img:.6f} stn={avg_stn:.6f} | "
                f"Best={best_loss:.6f} | "
                f"Time={epoch_time:.1f}s | ETA={eta_total:.0f}s"
            )
            if device.type == 'cuda':
                ma = torch.cuda.memory_allocated(device) / 1024**2
                mr = torch.cuda.memory_reserved(device) / 1024**2
                logger.debug(f"  [{model_name}] GPU: {ma:.0f}MB alloc / {mr:.0f}MB reserved")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_dir / 'best_model.pt')

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), model_dir / f'checkpoint_epoch_{epoch}.pt')
            logger.debug(f"  [{model_name}] 训练检查点: epoch {epoch+1}")

    total_time = time.time() - train_start

    best_path = model_dir / 'best_model.pt'
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))

    logger.info(
        f"[{model_name}] 训练完成 | Best Loss={best_loss:.6f} | "
        f"总耗时={total_time:.1f}s ({total_time/60:.1f}min)"
    )
    return model


# ════════════════════════════════════════════════════════════
#  评估单个模型
# ════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataset: StreamflowDataset,
    config: Dict,
    model_name: str,
    single_channel: bool = False,
) -> Tuple[Dict, pd.DataFrame]:
    """评估一个模型并返回流域级指标和站点级逐条记录。

    Returns:
        basin_metrics: dict {'Model', 'Pixel_MSE', 'SSIM', 'PSNR'}
        station_df: DataFrame 含 model, station_id, date, position_x, position_y,
                     actual_runoff, predicted_runoff
    """
    logger = logging.getLogger('CompareExperiment')

    device = torch.device(config.get('train', {}).get('device', 'cpu'))
    model = model.to(device)
    model.eval()

    kernel_size = config.get('stations', {}).get('kernel_size', 3)
    half_k = kernel_size // 2

    logger.info(f"  [{model_name}] 评估开始: {len(dataset)} 样本 | "
                f"kernel_size={kernel_size} | single_channel={single_channel}")

    pixel_mse_sum = 0.0
    total_pixels = 0
    ssim_list = []
    psnr_list = []
    station_records = []

    eval_start = time.time()

    for idx in range(len(dataset)):
        sample = dataset[idx]
        images = sample['images'].unsqueeze(0).to(device)

        if single_channel:
            images = images[:, :, :1, :, :]

        target = sample['output_image'].unsqueeze(0)
        pred = model(images).cpu()

        if pred.shape != target.shape:
            target = F.interpolate(
                target,
                size=(pred.shape[2], pred.shape[3]),
                mode='bilinear', align_corners=False,
            )

        pred_2d = pred[0, 0].numpy()
        tgt_2d = target[0, 0].numpy()

        # ── 流域级指标 ──
        mse_map = (pred_2d - tgt_2d) ** 2
        pixel_mse_sum += mse_map.sum()
        total_pixels += mse_map.size

        ssim_val = calculate_ssim(pred_2d, tgt_2d)
        ssim_list.append(ssim_val)

        psnr_val = calculate_psnr(pred_2d, tgt_2d)
        if not np.isinf(psnr_val):
            psnr_list.append(psnr_val)

        # ── 站点级指标（含站点名称映射）──
        positions = sample['station_positions']
        runoffs = sample['stations']
        if len(runoffs) > 0:
            pos_counter = {}
            for pos, actual in zip(positions, runoffs):
                px, py = pos[0].item(), pos[1].item()
                key = (px, py)
                occurrence = pos_counter.get(key, 0)
                station_name = get_station_name(px, py, occurrence)
                pos_counter[key] = occurrence + 1

                y0 = max(0, py - half_k)
                y1 = min(pred_2d.shape[0], py + half_k + 1)
                x0 = max(0, px - half_k)
                x1 = min(pred_2d.shape[1], px + half_k + 1)
                pred_val = float(pred_2d[y0:y1, x0:x1].mean())

                station_records.append({
                    'model': model_name,
                    'station_id': station_name,
                    'date': sample['date'],
                    'position_x': px,
                    'position_y': py,
                    'actual_runoff': float(actual),
                    'predicted_runoff': pred_val,
                })

        if (idx + 1) % 50 == 0 or idx == len(dataset) - 1:
            elapsed = time.time() - eval_start
            speed = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (len(dataset) - idx - 1) / speed if speed > 0 else 0
            logger.info(
                f"  [{model_name}] 评估 {idx+1}/{len(dataset)} | "
                f"{speed:.1f} 样本/s | ETA {eta:.0f}s"
            )

    eval_time = time.time() - eval_start

    basin_mse = pixel_mse_sum / max(total_pixels, 1)
    basin_ssim = float(np.mean(ssim_list)) if ssim_list else 0.0
    basin_psnr = float(np.mean(psnr_list)) if psnr_list else 0.0

    basin_metrics = {
        'Model': model_name,
        'Pixel_MSE': float(basin_mse),
        'SSIM': basin_ssim,
        'PSNR': basin_psnr,
    }

    station_df = pd.DataFrame(station_records)

    logger.info(
        f"  [{model_name}] 评估完成 ({eval_time:.1f}s) | "
        f"MSE={basin_mse:.6f} SSIM={basin_ssim:.4f} PSNR={basin_psnr:.2f}"
    )
    if not station_df.empty:
        logger.debug(
            f"  [{model_name}] 站点记录 {len(station_df)} 条 | "
            f"站点: {station_df['station_id'].unique().tolist()}"
        )

    return basin_metrics, station_df


# ════════════════════════════════════════════════════════════
#  使用已保存的预测文件评估主模型
# ════════════════════════════════════════════════════════════
def evaluate_from_predictions(
    predictions_dir: str,
    dataset: StreamflowDataset,
    config: Dict,
    model_name: str,
) -> Tuple[Dict, pd.DataFrame]:
    """使用 pred_*.npy 预测文件评估主模型（无需加载模型权重）。"""
    logger = logging.getLogger('CompareExperiment')

    pred_dir = Path(predictions_dir)
    kernel_size = config.get('stations', {}).get('kernel_size', 3)
    half_k = kernel_size // 2

    pixel_mse_sum = 0.0
    total_pixels = 0
    ssim_list = []
    psnr_list = []
    station_records = []

    matched = 0
    skipped = 0

    eval_start = time.time()
    total_samples = len(dataset)
    logger.info(f"  [{model_name}] 从预测文件评估: {total_samples} 样本 | 目录={pred_dir}")

    for idx in range(total_samples):
        sample = dataset[idx]
        date_str = sample['date']

        pred_path = pred_dir / f'pred_{date_str}.npy'
        target_path = pred_dir / f'target_{date_str}.npy'

        if not pred_path.exists():
            skipped += 1
            if skipped <= 5:
                logger.debug(f"  [{model_name}] 跳过: {pred_path.name} 不存在")
            continue

        pred_2d = np.load(pred_path).astype(np.float32)

        if target_path.exists():
            tgt_2d = np.load(target_path).astype(np.float32)
        else:
            tgt_2d = sample['output_image'][0].numpy()

        if pred_2d.shape != tgt_2d.shape:
            tgt_tensor = torch.from_numpy(tgt_2d).unsqueeze(0).unsqueeze(0)
            tgt_tensor = F.interpolate(
                tgt_tensor, size=pred_2d.shape,
                mode='bilinear', align_corners=False,
            )
            tgt_2d = tgt_tensor[0, 0].numpy()

        # ── 流域级指标 ──
        mse_map = (pred_2d - tgt_2d) ** 2
        pixel_mse_sum += mse_map.sum()
        total_pixels += mse_map.size

        ssim_val = calculate_ssim(pred_2d, tgt_2d)
        ssim_list.append(ssim_val)

        psnr_val = calculate_psnr(pred_2d, tgt_2d)
        if not np.isinf(psnr_val):
            psnr_list.append(psnr_val)

        # ── 站点级指标（含站点名称映射）──
        positions = sample['station_positions']
        runoffs = sample['stations']
        if len(runoffs) > 0:
            pos_counter = {}
            for pos, actual in zip(positions, runoffs):
                px, py = pos[0].item(), pos[1].item()
                key = (px, py)
                occurrence = pos_counter.get(key, 0)
                station_name = get_station_name(px, py, occurrence)
                pos_counter[key] = occurrence + 1

                y0 = max(0, py - half_k)
                y1 = min(pred_2d.shape[0], py + half_k + 1)
                x0 = max(0, px - half_k)
                x1 = min(pred_2d.shape[1], px + half_k + 1)
                pred_val = float(pred_2d[y0:y1, x0:x1].mean())

                station_records.append({
                    'model': model_name,
                    'station_id': station_name,
                    'date': date_str,
                    'position_x': px,
                    'position_y': py,
                    'actual_runoff': float(actual),
                    'predicted_runoff': pred_val,
                })

        matched += 1
        if matched % 200 == 0 or matched == 1:
            elapsed = time.time() - eval_start
            speed = matched / elapsed if elapsed > 0 else 0
            logger.info(
                f"  [{model_name}] 已加载 {matched} 预测 (跳过 {skipped}) | "
                f"{speed:.1f} 样本/s"
            )

    eval_time = time.time() - eval_start
    logger.info(
        f"  [{model_name}] 预测文件匹配: {matched}/{total_samples} "
        f"(跳过 {skipped}) | 耗时 {eval_time:.1f}s"
    )

    basin_mse = pixel_mse_sum / max(total_pixels, 1)
    basin_ssim = float(np.mean(ssim_list)) if ssim_list else 0.0
    basin_psnr = float(np.mean(psnr_list)) if psnr_list else 0.0

    basin_metrics = {
        'Model': model_name,
        'Pixel_MSE': float(basin_mse),
        'SSIM': basin_ssim,
        'PSNR': basin_psnr,
    }

    station_df = pd.DataFrame(station_records)
    return basin_metrics, station_df


# ════════════════════════════════════════════════════════════
#  按站点分别计算评价指标
# ════════════════════════════════════════════════════════════
def compute_per_station_metrics(station_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """按站点分别计算 MSE, MAPE, NSE, KGE。

    输出格式与 station_metrics_detailed.csv 一致：
        station_id, position_x, position_y, MSE, MAPE, NSE, KGE
    """
    if station_df.empty:
        return pd.DataFrame()

    results = []
    for station_id in station_df['station_id'].unique():
        sdf = station_df[station_df['station_id'] == station_id]
        actual = sdf['actual_runoff'].values.astype(np.float64)
        predicted = sdf['predicted_runoff'].values.astype(np.float64)
        n = len(actual)

        # MSE
        mse = float(np.mean((actual - predicted) ** 2))

        # MAE
        mae = float(np.mean(np.abs(actual - predicted)))

        # MAPE
        mask = np.abs(actual) > 1e-8
        if mask.sum() > 0:
            mape = float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])))
        else:
            mape = 0.0

        # NSE
        ss_res = float(np.sum((actual - predicted) ** 2))
        ss_tot = float(np.sum((actual - actual.mean()) ** 2))
        nse = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0

        # KGE
        if n > 1 and np.std(actual) > 1e-10 and np.std(predicted) > 1e-10:
            r = float(np.corrcoef(actual, predicted)[0, 1])
            alpha = float(np.std(predicted) / np.std(actual))
            mean_actual = float(np.mean(actual))
            beta = float(np.mean(predicted) / mean_actual) if abs(mean_actual) > 1e-10 else 1.0
            kge = 1.0 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
            kge = float(kge)
        else:
            kge = 0.0

        px = int(sdf['position_x'].iloc[0])
        py = int(sdf['position_y'].iloc[0])

        results.append({
            'model': model_name,
            'station_id': station_id,
            'position_x': px,
            'position_y': py,
            'MSE': mse,
            'MAE': mae,
            'MAPE': mape,
            'NSE': nse,
            'KGE': kge,
        })

    return pd.DataFrame(results)


def compute_station_summary(station_df: pd.DataFrame, model_name: str) -> Dict:
    """计算全站点汇总指标 (NSE, MSE, MAE)。"""
    if station_df.empty:
        return {'Model': model_name, 'NSE': 0.0, 'MSE': 0.0, 'MAE': 0.0}

    actual = station_df['actual_runoff'].values.astype(np.float64)
    predicted = station_df['predicted_runoff'].values.astype(np.float64)

    mse = float(np.mean((actual - predicted) ** 2))
    mae = float(np.mean(np.abs(actual - predicted)))
    ss_res = float(np.sum((actual - predicted) ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    nse = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0

    return {'Model': model_name, 'NSE': nse, 'MSE': mse, 'MAE': mae}


# ════════════════════════════════════════════════════════════
#  辅助：打印系统信息 / 站点信息
# ════════════════════════════════════════════════════════════
def log_system_info(logger):
    """打印系统/GPU 信息。"""
    import platform
    logger.info(f"系统: {platform.system()} {platform.release()}")
    logger.info(f"Python: {sys.version}")
    logger.info(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        logger.info(f"CUDA: {torch.version.cuda} | GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        logger.info("CUDA: 不可用, 使用 CPU")


def _log_per_station(logger, model_name: str, ps_metrics: pd.DataFrame):
    """输出各站点指标到日志。"""
    if ps_metrics.empty:
        return
    for _, row in ps_metrics.iterrows():
        logger.info(
            f"  [{model_name}]   {row['station_id']:6s} "
            f"({int(row['position_x'])},{int(row['position_y'])}) | "
            f"MSE={row['MSE']:.6f} MAPE={row['MAPE']:.4f} "
            f"NSE={row['NSE']:.4f} KGE={row['KGE']:.4f}"
        )


# ════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description='对比实验: LSTM / BP / Attention-GRU vs 主模型 (CNN-Transformer)')
    parser.add_argument('--config', type=str, default='./data/config.yaml')
    parser.add_argument('--main-model', type=str, default=None,
                        help='主模型 checkpoint 路径 (CNN-Transformer)')
    parser.add_argument('--predictions-dir', type=str, default='./output/predictions',
                        help='主模型预测文件目录 (pred_*.npy)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='覆盖训练轮数')
    parser.add_argument('--output-dir', type=str, default='./output/comparison')
    parser.add_argument('--resume', action='store_true', default=True,
                        help='从检查点恢复（默认启用）')
    parser.add_argument('--no-resume', action='store_true',
                        help='忽略检查点，从头开始')
    parser.add_argument('--debug', action='store_true',
                        help='启用 DEBUG 级别日志')
    args = parser.parse_args()

    config = ConfigLoader.load_config(args.config)
    if args.epochs is not None:
        config.setdefault('train', {})['num_epochs'] = args.epochs
    num_epochs = config.get('train', {}).get('num_epochs', 100)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 设置日志 ──
    debug = args.debug or config.get('system', {}).get('debug', False)
    logger = setup_logging(output_dir, debug=debug)

    logger.info("=" * 70)
    logger.info("          对比实验启动")
    logger.info("=" * 70)
    log_system_info(logger)
    logger.info(f"配置文件: {args.config}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"训练轮数: {num_epochs}")
    logger.info(f"站点映射: {STATION_POSITION_MAP}")

    # ── 检查点/断点续传 ──
    if args.no_resume:
        completed_models = []
        logger.info("已禁用断点续传，从头开始")
    else:
        ckpt = load_checkpoint(output_dir)
        completed_models = ckpt.get('completed_models', [])
        if completed_models:
            logger.info(f"从检查点恢复 | 已完成: {completed_models}")
        else:
            logger.info("未找到检查点，从头开始")

    # ── 数据集 ──
    image_dir = config.get('data', {}).get('image_dir', './data/images')
    station_dir = config.get('data', {}).get('station_dir', './data/stations')
    logger.info(f"加载数据集: {image_dir}")

    dataset = StreamflowDataset(
        image_dir=image_dir,
        station_dir=station_dir,
        config=config,
        normalize=True,
    )
    logger.info(f"数据集大小: {len(dataset)} 样本")
    logger.info(f"图像尺寸: {config.get('data', {}).get('image_size', 'N/A')}")
    logger.info(f"时间窗口: {config.get('data', {}).get('window_size', 5)}")

    # ── DEBUG: 输出首个样本形状 ──
    if len(dataset) > 0:
        s0 = dataset[0]
        logger.debug(f"首个样本形状: images={s0['images'].shape}, "
                     f"output={s0['output_image'].shape}, "
                     f"stations={len(s0['stations'])}, "
                     f"positions={len(s0['station_positions'])}")

    device = torch.device(config.get('train', {}).get('device', 'cpu'))

    # ── 结果容器 ──
    basin_results = []
    station_summary_results = []
    all_per_station_metrics = []
    all_station_records = []

    # ── 所有模型列表 ──
    all_models = ['CNN-Transformer', 'LSTM', 'BP', 'Attention-GRU']

    total_start = time.time()

    for model_idx, model_name in enumerate(all_models):
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"  [{model_idx+1}/{len(all_models)}] 处理模型: {model_name}")
        logger.info("=" * 70)

        # ── 检查是否已完成（断点续传）──
        if model_name in completed_models:
            logger.info(f"  [{model_name}] 已完成，从磁盘加载结果 ...")
            try:
                basin_m, stn_df, ps_metrics = load_model_results(output_dir, model_name)
                if basin_m is not None:
                    basin_results.append(basin_m)
                    stn_summary = compute_station_summary(stn_df, model_name)
                    station_summary_results.append(stn_summary)
                    all_per_station_metrics.append(ps_metrics)
                    all_station_records.append(stn_df)
                    logger.info(f"  [{model_name}] 已恢复 | "
                                f"Basin MSE={basin_m['Pixel_MSE']:.6f}")
                    _log_per_station(logger, model_name, ps_metrics)
                    continue
                else:
                    logger.warning(f"  [{model_name}] 结果文件缺失，重新处理")
                    completed_models.remove(model_name)
            except Exception as e:
                logger.warning(f"  [{model_name}] 加载失败: {e}，重新处理")
                completed_models.remove(model_name)

        model_start = time.time()

        try:
            # ════════════════════════════════════════════
            #  主模型 (CNN-Transformer)
            # ════════════════════════════════════════════
            if model_name == 'CNN-Transformer':
                predictions_dir = Path(args.predictions_dir)
                has_saved_preds = (
                    predictions_dir.is_dir()
                    and any(predictions_dir.glob('pred_*.npy'))
                )

                if has_saved_preds:
                    pred_count = len(list(predictions_dir.glob('pred_*.npy')))
                    logger.info(f"  [{model_name}] 发现 {pred_count} 个预测文件 -> 快速评估")
                    basin_m, stn_df = evaluate_from_predictions(
                        str(predictions_dir), dataset, config, model_name,
                    )
                else:
                    main_model = StreamflowPredictionModel(config)
                    if args.main_model and os.path.exists(args.main_model):
                        logger.info(f"  [{model_name}] 加载预训练权重: {args.main_model}")
                        main_model.load_state_dict(
                            torch.load(args.main_model, map_location=device)
                        )
                    else:
                        logger.info(f"  [{model_name}] 从头训练 ...")
                        main_model = train_model(
                            main_model, dataset, config, num_epochs,
                            model_name, output_dir, single_channel=False,
                        )

                    basin_m, stn_df = evaluate_model(
                        main_model, dataset, config, model_name,
                        single_channel=False,
                    )
                    del main_model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # ════════════════════════════════════════════
            #  基线模型 (LSTM, BP, Attention-GRU)
            #  只用单通道（径流）输入
            # ════════════════════════════════════════════
            else:
                bl_config = copy.deepcopy(config)
                bl_config['model']['input_channels'] = 1
                logger.info(f"  [{model_name}] 基线模型使用单通道（径流）输入")

                bl_model = build_baseline_model(model_name, bl_config)
                bl_model = train_model(
                    bl_model, dataset, bl_config, num_epochs,
                    model_name, output_dir, single_channel=True,
                )

                basin_m, stn_df = evaluate_model(
                    bl_model, dataset, config, model_name,
                    single_channel=True,
                )
                del bl_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # ── 计算指标 ──
            stn_summary = compute_station_summary(stn_df, model_name)
            ps_metrics = compute_per_station_metrics(stn_df, model_name)

            basin_results.append(basin_m)
            station_summary_results.append(stn_summary)
            all_per_station_metrics.append(ps_metrics)
            all_station_records.append(stn_df)

            model_time = time.time() - model_start

            # ── 输出结果 ──
            logger.info(f"  [{model_name}] --- 结果汇总 ---")
            logger.info(f"  [{model_name}]   流域: MSE={basin_m['Pixel_MSE']:.6f}, "
                        f"SSIM={basin_m['SSIM']:.4f}, PSNR={basin_m['PSNR']:.2f}")
            logger.info(f"  [{model_name}]   站点(汇总): NSE={stn_summary['NSE']:.4f}, "
                        f"MSE={stn_summary['MSE']:.6f}, MAE={stn_summary['MAE']:.6f}")
            _log_per_station(logger, model_name, ps_metrics)
            logger.info(f"  [{model_name}] 耗时: {model_time:.1f}s ({model_time/60:.1f}min)")

            # ── 即时保存到磁盘（防止中断丢失）──
            save_model_results(output_dir, model_name, basin_m, stn_df, ps_metrics)
            completed_models.append(model_name)
            save_checkpoint(output_dir, completed_models)
            logger.info(f"  [{model_name}] 结果已保存 | 检查点已更新")

        except Exception as e:
            logger.error(f"  [{model_name}] 处理失败: {e}")
            logger.error(traceback.format_exc())
            save_checkpoint(output_dir, completed_models)
            continue

    # ════════════════════════════════════════════════════════════
    #  汇总并保存最终结果
    # ════════════════════════════════════════════════════════════
    total_time = time.time() - total_start
    logger.info("")
    logger.info("=" * 70)
    logger.info("  保存最终汇总结果")
    logger.info("=" * 70)

    if basin_results:
        basin_df = pd.DataFrame(basin_results)
        basin_csv = output_dir / 'basin_comparison.csv'
        basin_df.to_csv(basin_csv, index=False)
        logger.info(f"  流域级对比 -> {basin_csv}")

    if station_summary_results:
        stn_sum_df = pd.DataFrame(station_summary_results)
        station_csv = output_dir / 'station_comparison.csv'
        stn_sum_df.to_csv(station_csv, index=False)
        logger.info(f"  站点级对比(汇总) -> {station_csv}")

    if all_per_station_metrics:
        valid_ps = [df for df in all_per_station_metrics if not df.empty]
        if valid_ps:
            ps_all = pd.concat(valid_ps, ignore_index=True)
            ps_csv = output_dir / 'station_metrics_per_station.csv'
            ps_all.to_csv(ps_csv, index=False)
            logger.info(f"  各站点指标(所有模型) -> {ps_csv}")

    if all_station_records:
        valid_records = [df for df in all_station_records if not df.empty]
        if valid_records:
            detail_df = pd.concat(valid_records, ignore_index=True)
            detail_csv = output_dir / 'station_predictions_detail.csv'
            detail_df.to_csv(detail_csv, index=False)
            logger.info(f"  站点预测明细 -> {detail_csv}")

    # ── 打印最终汇总表 ──
    print("\n" + "=" * 70)
    print("                    对比实验结果汇总")
    print("=" * 70)

    if basin_results:
        print("\n  流域层面 (Basin-level)")
        basin_df = pd.DataFrame(basin_results)
        print(basin_df.to_string(index=False, float_format='%.6f'))

    if station_summary_results:
        print("\n  站点层面 - 整体 (Station-level Overall)")
        stn_sum_df = pd.DataFrame(station_summary_results)
        print(stn_sum_df.to_string(index=False, float_format='%.6f'))

    if all_per_station_metrics:
        valid_ps = [df for df in all_per_station_metrics if not df.empty]
        if valid_ps:
            print("\n  站点层面 - 分站点 (Station-level Per-Station)")
            ps_all = pd.concat(valid_ps, ignore_index=True)
            print(ps_all.to_string(index=False, float_format='%.6f'))

    print(f"\n  总耗时: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"  结果已保存至 {output_dir}/")
    print("=" * 70)


if __name__ == '__main__':
    main()
