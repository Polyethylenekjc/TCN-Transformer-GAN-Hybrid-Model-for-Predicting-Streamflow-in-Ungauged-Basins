"""Visualization script for model predictions and performance analysis."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import argparse
import yaml
import torch
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from matplotlib import rcParams

from src.evaluate import calculate_ssim, calculate_psnr

def load_config(config_path: str) -> Dict:
    """Load configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _set_chinese_font():
    """Configure matplotlib to use a Chinese font (Song style) if available."""
    # Prefer SimSun (宋体); fall back to common CJK fonts
    rcParams["font.sans-serif"] = ["SimSun"]
    rcParams["axes.unicode_minus"] = False
    # Increase base font size for better readability
    rcParams["font.size"] = 11


def _panel_label(idx: int) -> str:
    """Return subplot panel label like '(a)', '(b)', ... dynamically."""
    base = ord('a') + idx
    # Wrap after 'z' just in case, though we won't reach it here
    base = ord('a') + (base - ord('a')) % 26
    return f"({chr(base)})"

def visualize_prediction(
    pred_path: str,
    target_path: str,
    output_path: str
):
    """
    Visualize prediction vs target.
    
    Args:
        pred_path: Path to prediction .npy file
        target_path: Path to target .npy file
        output_path: Path to save visualization
    """
    pred = np.load(pred_path)
    target = np.load(target_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Target
    im0 = axes[0].imshow(target, cmap='Blues')
    axes[0].set_title('Ground Truth', fontweight='bold')
    plt.colorbar(im0, ax=axes[0])
    
    # Raw Prediction
    im1 = axes[1].imshow(pred, cmap='Blues')
    axes[1].set_title('Raw Prediction', fontweight='bold')
    plt.colorbar(im1, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def visualize_station_metrics(
    csv_path: str,
    output_dir: str
):
    """
    Visualize station metrics.
    
    Args:
        csv_path: Path to stations_eval.csv
        output_dir: Directory to save plots
    """
    df = pd.read_csv(csv_path)
    output_path = Path(output_dir)
    
    # Scatter plot: Predicted vs Actual
    plt.figure(figsize=(8, 8))
    plt.scatter(df['actual_runoff'], df['predicted_runoff'], alpha=0.5)
    
    # Perfect line
    min_val = min(df['actual_runoff'].min(), df['predicted_runoff'].min())
    max_val = max(df['actual_runoff'].max(), df['predicted_runoff'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Actual Runoff')
    plt.ylabel('Predicted Runoff')
    plt.title('Station Runoff: Predicted vs Actual')
    plt.grid(True)
    plt.savefig(output_path / 'station_scatter.png')
    plt.close()
    
    # Time series plot (if date is available and parseable)
    try:
        # Try to parse date, assuming YYYYMMDD format from filename
        # The CSV has 'date' column which might be just the filename stem
        # Let's assume it's sortable
        df_sorted = df.sort_values('date')
        
        # Group by station location (approximate by x,y)
        # Create a unique station ID based on position
        df_sorted['station_id'] = df_sorted.apply(
            lambda row: f"{row['position_x']:.1f}_{row['position_y']:.1f}", axis=1
        )
        
        unique_stations = df_sorted['station_id'].unique()
        
        for station in unique_stations:
            station_data = df_sorted[df_sorted['station_id'] == station]
            
            plt.figure(figsize=(12, 6))
            plt.plot(range(len(station_data)), station_data['actual_runoff'], label='Actual', marker='o')
            plt.plot(range(len(station_data)), station_data['predicted_runoff'], label='Predicted', marker='x')
            
            plt.xlabel('Time Step')
            plt.ylabel('Runoff')
            plt.title(f'Station {station} Time Series')
            plt.legend()
            plt.grid(True)
            plt.savefig(output_path / f'station_{station}_timeseries.png')
            plt.close()
            
    except Exception as e:
        print(f"Could not plot time series: {e}")


def _attach_station_names(df: pd.DataFrame) -> pd.DataFrame:
    """Attach human-readable station names based on (x, y) and row order.

    Mapping rules (position_x, position_y):
      - 坎迪: 181, 66
      - 东吁: 185, 106
      - 帕本: first (189, 136) for each date
      - 曼德勒: second (189, 136) for each date
    """

    df = df.copy()
    df["station_name"] = None

    # Simple one-to-one mappings by coordinates
    mask_kandi = (df["position_x"] == 181) & (df["position_y"] == 66)
    df.loc[mask_kandi, "station_name"] = "坎迪"

    mask_dongyou = (df["position_x"] == 185) & (df["position_y"] == 106)
    df.loc[mask_dongyou, "station_name"] = "东吁"

    # Two distinct logical stations share the same (x, y) = (189, 136)
    mask_189_136 = (df["position_x"] == 189) & (df["position_y"] == 136)
    dup_df = df[mask_189_136]
    if not dup_df.empty:
        for date_value, idxs in dup_df.groupby("date").groups.items():
            sorted_indices = sorted(idxs)
            if len(sorted_indices) >= 1:
                # First occurrence -> 帕本
                df.loc[sorted_indices[0], "station_name"] = "帕本"
            if len(sorted_indices) >= 2:
                # Second occurrence -> 曼德勒
                df.loc[sorted_indices[1], "station_name"] = "曼德勒"

    return df


def _create_station_id(df: pd.DataFrame) -> pd.Series:
    """Create a stable station id.

    Prefer a human-readable name if ``station_name`` is available,
    otherwise fall back to position-based id.
    """
    def _make_id(row: pd.Series) -> str:
        name = row.get("station_name", None)
        if isinstance(name, str) and name:
            return name
        return f"{row['position_x']:.1f}_{row['position_y']:.1f}"

    return df.apply(_make_id, axis=1)


def _compute_station_statistics(
    df: pd.DataFrame
) -> pd.DataFrame:
    """Compute MSE, MAPE, NSE, KGE for each station.

    Returns a dataframe indexed by station_id with one row per station.
    """
    df = _attach_station_names(df)
    df["station_id"] = _create_station_id(df)

    def nse(obs: np.ndarray, sim: np.ndarray) -> float:
        obs_mean = np.mean(obs)
        denom = np.sum((obs - obs_mean) ** 2)
        if denom <= 0:
            return np.nan
        return 1.0 - float(np.sum((sim - obs) ** 2) / denom)

    def kge(obs: np.ndarray, sim: np.ndarray) -> float:
        if len(obs) < 2:
            return np.nan
        obs_mean = np.mean(obs)
        sim_mean = np.mean(sim)
        obs_std = np.std(obs)
        sim_std = np.std(sim)
        if obs_std <= 0 or sim_std <= 0 or obs_mean == 0:
            return np.nan
        r = float(np.corrcoef(obs, sim)[0, 1])
        alpha = sim_std / obs_std
        beta = sim_mean / obs_mean
        return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))

    records: List[Dict] = []
    for station_id, g in df.groupby("station_id"):
        obs = g["actual_runoff"].values.astype(float)
        sim = g["predicted_runoff"].values.astype(float)

        mse = float(np.mean((sim - obs) ** 2)) if len(obs) > 0 else np.nan

        # Avoid division by zero in MAPE
        with np.errstate(divide="ignore", invalid="ignore"):
            mape_arr = np.abs((sim - obs) / obs)
            mape_arr[~np.isfinite(mape_arr)] = np.nan
        mape = float(np.nanmean(mape_arr)) if np.any(np.isfinite(mape_arr)) else np.nan

        records.append(
            {
                "station_id": station_id,
                "position_x": g["position_x"].iloc[0],
                "position_y": g["position_y"].iloc[0],
                "MSE": mse,
                "MAPE": mape,
                "NSE": nse(obs, sim),
                "KGE": kge(obs, sim),
            }
        )

    return pd.DataFrame.from_records(records)


def plot_station_model_fit_grid(
    detail_csv_path: str,
    metrics_csv_path: str,
    output_path: str,
):
    """绘制 4×4 拟合散点图：

    - 行：站点（按 CSV 中出现顺序）
    - 列：模型（按 CSV 中出现顺序）
    - 子图内容：该站点-模型下的观测值 vs 预测值散点图
      * 仅显示拟合点、y=x 参考线与 MSE 标注
      * 不使用子图标题，仅在左下角标注 (a)、(b)...
    """

    _set_chinese_font()

    detail_df = pd.read_csv(detail_csv_path)
    metrics_df = pd.read_csv(metrics_csv_path)

    # 保证包含所需列
    required_detail_cols = {"model", "station_id", "actual_runoff", "predicted_runoff"}
    if not required_detail_cols.issubset(detail_df.columns):
        raise ValueError(f"detail CSV 缺少必要列: {required_detail_cols - set(detail_df.columns)}")

    required_metric_cols = {"model", "station_id", "MSE"}
    if not required_metric_cols.issubset(metrics_df.columns):
        raise ValueError(f"metrics CSV 缺少必要列: {required_metric_cols - set(metrics_df.columns)}")

    # 模型和站点顺序：按 metrics CSV 首次出现的顺序
    models = list(dict.fromkeys(metrics_df["model"].tolist()))
    stations = list(dict.fromkeys(metrics_df["station_id"].tolist()))

    n_rows = len(stations)
    n_cols = len(models)
    if n_rows == 0 or n_cols == 0:
        raise ValueError("在 metrics CSV 中未找到有效的模型或站点信息")

    # 不同模型不同颜色（全图统一配色）
    default_colors = [
        "#1f77b4",  # 蓝
        "#ff7f0e",  # 橙
        "#2ca02c",  # 绿
        "#d62728",  # 红
        "#9467bd",  # 紫
        "#8c564b",  # 棕
    ]
    model_colors: Dict[str, str] = {}
    for idx, m in enumerate(models):
        model_colors[m] = default_colors[idx % len(default_colors)]

    # 全局坐标范围，保持 y=x 参考线一致
    global_min = float(
        min(detail_df["actual_runoff"].min(), detail_df["predicted_runoff"].min())
    )
    global_max = float(
        max(detail_df["actual_runoff"].max(), detail_df["predicted_runoff"].max())
    )
    if not np.isfinite(global_min) or not np.isfinite(global_max):
        raise ValueError("detail CSV 中 actual_runoff / predicted_runoff 数据异常")

    # 适当放宽一点边界
    padding = 0.02 * (global_max - global_min)
    line_min = global_min - padding
    line_max = global_max + padding

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), sharex=True, sharey=True)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    panel_idx = 0

    for i, station in enumerate(stations):
        for j, model in enumerate(models):
            ax = axes[i, j]

            mask = (detail_df["station_id"] == station) & (detail_df["model"] == model)
            sub_df = detail_df[mask]

            color = model_colors.get(model, "#1f77b4")

            if not sub_df.empty:
                x = sub_df["actual_runoff"].values.astype(float)
                y = sub_df["predicted_runoff"].values.astype(float)
                ax.scatter(x, y, s=8, alpha=0.6, color=color, edgecolors="none")

            # y = x 参考线
            ax.plot([line_min, line_max], [line_min, line_max], linestyle="--", color="gray", linewidth=1.0)

            ax.set_xlim(line_min, line_max)
            ax.set_ylim(line_min, line_max)
            ax.set_aspect("equal", adjustable="box")

            # 只在最左列和最后一行设置坐标轴标签，避免过于拥挤
            if j == 0:
                ax.set_ylabel("预测值", fontsize=11)
            else:
                ax.set_ylabel("")
            if i == n_rows - 1:
                ax.set_xlabel("观测值", fontsize=11)
            else:
                ax.set_xlabel("")

            # MSE 标注（子图内左上角）
            mse_row = metrics_df[(metrics_df["station_id"] == station) & (metrics_df["model"] == model)]
            if not mse_row.empty:
                mse_val = float(mse_row["MSE"].iloc[0])
                ax.text(
                    0.03,
                    0.95,
                    f"MSE = {mse_val:.3f}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=11,
                )

            # 子图左下角 panel 标注 (a), (b), ...
            ax.text(
                0.03,
                0.03,
                _panel_label(panel_idx),
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=11,
            )
            panel_idx += 1

    # 在图外增加行/列标签（不算子图标题）
    for j, model in enumerate(models):
        fig.text(
            (j + 0.5) / n_cols,
            0.98,
            model,
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
        )

    for i, station in enumerate(stations):
        fig.text(
            0.02,
            (n_rows - i - 0.5) / n_rows,
            station,
            ha="left",
            va="center",
            fontsize=12,
            fontweight="bold",
            rotation=90,
        )

    # 全局图例：模型-颜色 对应
    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=color, label=model, markersize=6)
        for model, color in model_colors.items()
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(len(models), 4),
        frameon=False,
        fontsize=11,
    )

    fig.tight_layout(rect=[0.06, 0.06, 0.98, 0.94])

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved station-model fit grid figure to {out_path}")


def _load_image_pairs(
    pred_dir: Path,
    start_date: datetime,
    end_date: datetime,
) -> Tuple[List[datetime], List[np.ndarray], List[np.ndarray]]:
    """Load all (pred, target) image pairs within the given date range."""

    pairs: List[Tuple[datetime, Path, Path]] = []
    for pred_file in pred_dir.glob("pred_*.npy"):
        date_str = pred_file.stem.replace("pred_", "")
        try:
            dt = datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            continue
        if not (start_date <= dt <= end_date):
            continue

        target_file = pred_dir / f"target_{date_str}.npy"
        if not target_file.exists():
            continue
        pairs.append((dt, pred_file, target_file))

    pairs.sort(key=lambda x: x[0])

    dates: List[datetime] = []
    preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    for dt, p_path, t_path in pairs:
        dates.append(dt)
        preds.append(np.load(p_path))
        targets.append(np.load(t_path))

    return dates, preds, targets


def _crop_region(arr: np.ndarray) -> np.ndarray:
    """Crop image to the specified sub-region for analysis.

        New rule:
            Rows: 86-168 (inclusive)
            Cols: 160-202 (inclusive)
    """
    row_slice = slice(86, 169)
    col_slice = slice(160, 203)
    return arr[row_slice, col_slice]


def _compute_basin_metrics(
    preds: List[np.ndarray],
    targets: List[np.ndarray],
) -> Dict[str, float]:
    """Compute basin-wide pixel metrics (MSE, SSIM, PSNR) on the cropped region."""

    if not preds:
        return {"MSE": np.nan, "SSIM": np.nan, "PSNR": np.nan}

    sse = 0.0
    n_pixels = 0
    ssim_list: List[float] = []
    psnr_list: List[float] = []

    for pred, target in zip(preds, targets):
        pred_c = _crop_region(pred)
        target_c = _crop_region(target)

        diff = pred_c - target_c
        sse += float(np.sum(diff ** 2))
        n_pixels += diff.size

        ssim_list.append(calculate_ssim(pred_c, target_c))
        psnr_list.append(calculate_psnr(pred_c, target_c))

    mse = sse / n_pixels if n_pixels > 0 else np.nan
    ssim_mean = float(np.mean(ssim_list)) if ssim_list else np.nan
    psnr_mean = float(np.mean(psnr_list)) if psnr_list else np.nan

    return {"MSE": mse, "SSIM": ssim_mean, "PSNR": psnr_mean}


def _compute_basin_mae_series(
    preds: List[np.ndarray],
    targets: List[np.ndarray],
) -> np.ndarray:
    """Compute basin-wide MAE time series (cropped region, one value per time step)."""
    maes: List[float] = []
    for pred, target in zip(preds, targets):
        pred_c = _crop_region(pred)
        target_c = _crop_region(target)
        maes.append(float(np.mean(np.abs(pred_c - target_c))))
    return np.asarray(maes, dtype=float)


def _compute_basin_ssim_series(
    preds: List[np.ndarray],
    targets: List[np.ndarray],
) -> np.ndarray:
    """Compute basin-wide SSIM time series on the cropped region."""

    ssim_vals: List[float] = []
    for pred, target in zip(preds, targets):
        pred_c = _crop_region(pred)
        target_c = _crop_region(target)
        ssim_vals.append(float(calculate_ssim(pred_c, target_c)))
    return np.asarray(ssim_vals, dtype=float)


def visualize_full_performance(
    output_dir: Path,
    start_date: datetime,
    end_date: datetime,
):
    """Create a comprehensive performance figure and save detailed metrics.

    Layout (5 rows x 4 columns):
      - Row 1: 1 plot spanning all 4 columns – basin MAE vs time (8:2 red line).
      - Rows 2-3: 4 stations total (2 per row), each spanning 2 columns – true vs pred.
      - Rows 4-5: 4 random days (2 per row), each spanning 2 columns – pred map + error map.
    """

    _set_chinese_font()

    pred_dir = output_dir / "predictions"
    if not pred_dir.exists():
        print(f"No predictions directory found at {pred_dir}, skip full performance visualization.")
        return

    # Load image pairs
    dates, preds, targets = _load_image_pairs(pred_dir, start_date, end_date)
    if not dates:
        print("No prediction/target image pairs found in the specified date range.")
        return

    # Basin metrics (cropped region)
    basin_metrics = _compute_basin_metrics(preds, targets)
    basin_metrics_df = pd.DataFrame([basin_metrics])
    basin_metrics_df.to_csv(output_dir / "basin_metrics_detailed.csv", index=False)

    # Station statistics
    station_csv = output_dir / "stations_eval.csv"
    if not station_csv.exists():
        print(f"No station CSV found at {station_csv}, skip station-based plots.")
        return

    station_df = pd.read_csv(station_csv)
    # Attach station_name column according to custom mapping
    station_df = _attach_station_names(station_df)
    station_df["date_str"] = station_df["date"].astype(str)

    # Parse dates, keep those within range and present in image pairs
    date_map = {dt.strftime("%Y%m%d"): dt for dt in dates}
    station_df = station_df[station_df["date_str"].isin(date_map.keys())].copy()
    station_df["datetime"] = station_df["date_str"].map(date_map)
    station_df.sort_values("datetime", inplace=True)

    if station_df.empty:
        print("Station dataframe is empty after aligning with image dates, skip station plots.")
        return

    station_stats_df = _compute_station_statistics(station_df)
    station_stats_df.to_csv(output_dir / "station_metrics_detailed.csv", index=False)

    # Prepare figure layout: 5 rows x 4 columns (first row slightly taller)
    fig = plt.figure(figsize=(16, 18))
    gs = fig.add_gridspec(
        5,
        4,
        hspace=0.35,
        wspace=0.30,
        height_ratios=[1.6, 1.0, 1.0, 1.0, 1.0],
    )

    # Panel index for '(a)', '(b)', ... labels
    panel_idx = 0

    # Global time axis and 8:2 split
    all_dates = dates
    n_all = len(all_dates)
    split_idx = int(n_all * 0.8)
    split_idx = max(1, min(split_idx, n_all - 1))
    split_date = all_dates[split_idx]

    # ── Row 1: basin-wide MAE & SSIM vs time (span all 4 columns) ──
    basin_mae = _compute_basin_mae_series(preds, targets)
    basin_ssim = _compute_basin_ssim_series(preds, targets)

    ax_mae = fig.add_subplot(gs[0, :])

    # Split MAE into train / test with different colors
    train_dates = all_dates[:split_idx + 1]
    test_dates  = all_dates[split_idx:]
    train_mae   = basin_mae[:split_idx + 1]
    test_mae    = basin_mae[split_idx:]
    train_ssim  = basin_ssim[:split_idx + 1]
    test_ssim   = basin_ssim[split_idx:]

    mae_train_line, = ax_mae.plot(train_dates, train_mae, color="#2ca02c", linewidth=1.2, label="MAE (训练集)")
    mae_test_line,  = ax_mae.plot(test_dates,  test_mae,  color="#1f77b4", linewidth=1.2, label="MAE (测试集)")
    ax_mae.axvline(split_date, color="red", linestyle="--", linewidth=1.2)
    # Light gray shading on the training side
    ax_mae.axvspan(all_dates[0], split_date, color="lightgray", alpha=0.25, zorder=0)
    ax_mae.text(split_date, ax_mae.get_ylim()[1] * 0.95, "训练集 ",
                ha="right", va="top", color="green", fontsize=10, fontweight="bold")
    ax_mae.text(split_date, ax_mae.get_ylim()[1] * 0.95, " 测试集",
                ha="left", va="top", color="#1f77b4", fontsize=10, fontweight="bold")
    ax_mae.set_xlabel("时间")
    ax_mae.set_ylabel("MAE")

    # Secondary y-axis for SSIM
    ax_ssim = ax_mae.twinx()
    ssim_train_line, = ax_ssim.plot(train_dates, train_ssim, color="#9467bd", linewidth=1.0, label="SSIM (训练集)")
    ssim_test_line,  = ax_ssim.plot(test_dates,  test_ssim,  color="#d62728", linewidth=1.0, label="SSIM (测试集)")
    ax_ssim.set_ylabel("SSIM")
    ax_ssim.set_ylim(top=1.25)

    ax_mae.set_title(f"{_panel_label(panel_idx)} 全流域 MAE / SSIM 随时间变化", fontweight="bold")
    panel_idx += 1
    # Combined legend from both axes
    lines  = [mae_train_line, mae_test_line, ssim_train_line, ssim_test_line]
    labels = ["MAE (训练集)", "MAE (测试集)", "SSIM (训练集)", "SSIM (测试集)"]
    ax_mae.legend(lines, labels, loc="upper left", fontsize=8)

    # ── Rows 2-3: 4 stations (2 per row, each spanning 2 columns) ──
    station_df["station_id"] = _create_station_id(station_df)
    unique_stations = station_df["station_id"].unique().tolist()
    if len(unique_stations) == 0:
        print("No stations found for time-series plots.")
        return

    selected_stations = unique_stations[:4]
    for idx, station_id in enumerate(selected_stations):
        row = 1 + idx // 2          # row 1 or 2
        col_start = (idx % 2) * 2   # 0 or 2
        ax = fig.add_subplot(gs[row, col_start : col_start + 2])

        g = station_df[station_df["station_id"] == station_id].sort_values("datetime")
        g_train = g[g["datetime"] <= split_date]
        g_test  = g[g["datetime"] >= split_date]

        # Train segment
        ax.plot(g_train["datetime"], g_train["actual_runoff"],  linewidth=1.0, color="#2ca02c", label="真实值 (训练)" if idx == 0 else None)
        ax.plot(g_train["datetime"], g_train["predicted_runoff"], linewidth=1.0, color="#9467bd", label="预测值 (训练)" if idx == 0 else None)
        # Test segment
        ax.plot(g_test["datetime"], g_test["actual_runoff"],  linewidth=1.0, color="#1f77b4", label="真实值 (测试)" if idx == 0 else None)
        ax.plot(g_test["datetime"], g_test["predicted_runoff"], linewidth=1.0, color="#d62728", label="预测值 (测试)" if idx == 0 else None)

        ax.axvline(split_date, color="red", linestyle="--", linewidth=1.2)
        ax.axvspan(g["datetime"].min(), split_date, color="lightgray", alpha=0.25, zorder=0)
        ax.set_xlabel("时间")
        ax.set_ylabel("径流")
        ax.set_title(f"{_panel_label(panel_idx)} 站点{station_id}", fontweight="bold")
        panel_idx += 1
        if idx == 0:
            ax.legend(loc="upper left", fontsize=7)

    # ── Rows 4-5: 4 random days (2 per row, each spanning 2 columns) ──
    #    2 days from train segment, 2 from test segment
    import random
    rng = random.Random(42)
    train_indices = list(range(split_idx))
    test_indices = list(range(split_idx, n_all))

    chosen_train = rng.sample(train_indices, min(2, len(train_indices)))
    chosen_test  = rng.sample(test_indices,  min(2, len(test_indices)))
    chosen_indices = sorted(chosen_train) + sorted(chosen_test)
    chosen_indices = chosen_indices[:4]

    for j, ci in enumerate(chosen_indices):
        dt = all_dates[ci]
        pred_crop  = _crop_region(preds[ci]).T
        target_crop = _crop_region(targets[ci]).T
        error = np.abs(pred_crop - target_crop)

        row = 3 + j // 2           # row 3 or 4
        col_base = (j % 2) * 2     # 0 or 2

        # Left column: prediction
        ax_pred = fig.add_subplot(gs[row, col_base])
        im_pred = ax_pred.imshow(pred_crop, cmap="viridis")
        ax_pred.set_title(f"{_panel_label(panel_idx)} {dt.strftime('%Y-%m-%d')} 预测", fontweight="bold")
        panel_idx += 1
        plt.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)

        # Right column: |prediction − truth|
        ax_err = fig.add_subplot(gs[row, col_base + 1])
        im_err = ax_err.imshow(error, cmap="magma")
        ax_err.set_title(f"{_panel_label(panel_idx)} {dt.strftime('%Y-%m-%d')} 误差", fontweight="bold")
        panel_idx += 1
        plt.colorbar(im_err, ax=ax_err, fraction=0.046, pad=0.04)

    vis_dir = output_dir / "visualization"
    vis_dir.mkdir(parents=True, exist_ok=True)
    fig_path = vis_dir / "performance_overview.png"
    # Further tighten around edges while保持子图不重叠
    fig.tight_layout(rect=[0, 0, 1, 1])
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)
    print(f"Saved full performance visualization to {fig_path}")

def main():
    parser = argparse.ArgumentParser(description='Visualize model predictions')
    parser.add_argument('--config', type=str, default='./data/config.yaml', help='Path to config file')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (overrides config)')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("./output")
    
    pred_dir = output_dir / 'predictions'
    vis_dir = output_dir / 'visualization'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Visualizing results from {output_dir}...")
    
    # 1. Visualize Images
    # Find all prediction files
    pred_files = sorted(list(pred_dir.glob('pred_*.npy')))
    
    # Limit to first 20 for quick check, or random sample
    # For now, let's do first 10
    for pred_file in pred_files[:10]:
        date_str = pred_file.stem.replace('pred_', '')
        target_file = pred_dir / f'target_{date_str}.npy'
        
        if target_file.exists():
            output_path = vis_dir / f'vis_{date_str}.png'
            visualize_prediction(
                str(pred_file),
                str(target_file),
                str(output_path)
            )
            print(f"Saved visualization to {output_path}")
    
    # 2. Visualize Station Metrics
    csv_path = output_dir / 'stations_eval.csv'
    if csv_path.exists():
        visualize_station_metrics(str(csv_path), str(vis_dir))
        print(f"Saved station metrics plots to {vis_dir}")
    else:
        print("No station metrics CSV found.")

    # 3. Full performance visualization and detailed metrics on cropped region
    start_date = datetime(2015, 1, 1)
    end_date = datetime(2024, 12, 31)
    visualize_full_performance(output_dir, start_date, end_date)

if __name__ == '__main__':
    main()
