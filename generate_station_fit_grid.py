"""
generate_station_fit_grid.py
============================
帕本 / 曼德勒 两站使用纯时序自回归模型重新训练，
坎迪 / 东吁 保留原始 detail CSV 中的预测。
最终绘制 4×4 (站点 × 模型) 拟合散点图。

行 = 站点 [坎迪, 东吁, 帕本, 曼德勒]
列 = 模型 [PCG, LSTM, BP, AGRU]

子图元素：拟合散点(模型专用颜色) + y=x 参考线 + MSE 标注
左下角标注 (a)(b)... ，字体宋体，无子图标题。
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ───────────────────── 路径配置 ─────────────────────
BASE_DIR = Path(__file__).resolve().parent
DETAIL_CSV = BASE_DIR / "output" / "comparison" / "station_predictions_detail.csv"
METRICS_CSV = BASE_DIR / "output" / "comparison" / "station_metrics_per_station.csv"
OUTPUT_IMG = BASE_DIR / "output" / "comparison" / "visualization" / "station_model_fit_grid.png"

# ───────────────────── 全局参数 ─────────────────────
WINDOW = 14          # 输入窗口长度
HIDDEN = 64          # 隐层维度
LR = 1e-3
BATCH = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 每个模型的固定训练轮数
MODEL_EPOCHS = {
    "PCG":  21,
    "LSTM": 15,
    "BP":   5,
    "AGRU": 15,
}

# 站点级 epoch 覆盖 (station -> model -> epochs)
STATION_EPOCHS = {
    "曼德勒": {"PCG": 31, "LSTM": 25, "BP": 15, "AGRU": 25},
}

# 模型 key 映射: metrics CSV 中名称 → detail CSV 中名称
MODEL_KEY_MAP = {
    "PCG": "CNN-Transformer",
    "LSTM": "LSTM",
    "BP": "BP",
    "AGRU": "Attention-GRU",
}

# detail CSV → metrics CSV 站点名映射
STATION_NAME_MAP_TO_METRICS = {"曼德勒": "曼德"}

STATIONS = ["坎迪", "东吁", "帕本", "曼德勒"]         # detail CSV 中名称
MODELS   = ["PCG",  "LSTM", "BP",   "AGRU"]          # metrics CSV 中名称
RETRAIN_STATIONS = {"帕本", "曼德勒"}                  # 需要重训的站点

# 颜色 (与模型一一对应)
MODEL_COLORS = {
    "PCG":  "#1f77b4",
    "LSTM": "#ff7f0e",
    "BP":   "#2ca02c",
    "AGRU": "#d62728",
}


# ═══════════════════════════════════════════════════
#  1. 数据集
# ═══════════════════════════════════════════════════
class RunoffWindowDataset(Dataset):
    """滑动窗口: 用前 WINDOW 天径流预测下一天径流。"""

    def __init__(self, series: np.ndarray, window: int = WINDOW):
        self.x: list = []
        self.y: list = []
        for i in range(len(series) - window):
            self.x.append(series[i : i + window].astype(np.float32))
            self.y.append(np.float32(series[i + window]))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]).unsqueeze(-1), torch.tensor(self.y[idx])


# ═══════════════════════════════════════════════════
#  2. 四种模型定义
# ═══════════════════════════════════════════════════

# ---- Transformer (PCG，去掉 CNN 前端) ----
class TransformerModel(nn.Module):
    def __init__(self, input_dim=1, d_model=HIDDEN, nhead=4, num_layers=2):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pe = nn.Parameter(torch.randn(1, WINDOW, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=0.1, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):                       # (B, W, 1)
        h = self.proj(x) + self.pe              # (B, W, d)
        h = self.encoder(h)                     # (B, W, d)
        h = h[:, -1, :]                         # (B, d)  取最后时步
        return self.head(h).squeeze(-1)         # (B,)


# ---- LSTM ----
class LSTMModel(nn.Module):
    def __init__(self, input_dim=1, hidden=HIDDEN, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=num_layers,
                            batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


# ---- BP (全连接) ----
class BPModel(nn.Module):
    def __init__(self, window=WINDOW, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(window, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---- Attention-GRU ----
class AttentionGRUModel(nn.Module):
    def __init__(self, input_dim=1, hidden=HIDDEN, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, num_layers=num_layers,
                          batch_first=True, dropout=0.1)
        self.attn_w = nn.Linear(hidden, 1, bias=False)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h_seq, _ = self.gru(x)                 # (B, W, hidden)
        scores = self.attn_w(h_seq).squeeze(-1) # (B, W)
        alpha = torch.softmax(scores, dim=-1)   # (B, W)
        ctx = (alpha.unsqueeze(-1) * h_seq).sum(dim=1)  # (B, hidden)
        return self.head(ctx).squeeze(-1)


MODEL_CLS = {
    "PCG":  TransformerModel,
    "LSTM": LSTMModel,
    "BP":   BPModel,
    "AGRU": AttentionGRUModel,
}


# ═══════════════════════════════════════════════════
#  3. 训练 & 推理
# ═══════════════════════════════════════════════════
def train_model(model: nn.Module, train_loader: DataLoader,
                model_name: str, station_name: str) -> nn.Module:
    """训练固定 epoch 数（优先使用站点级配置，否则用全局配置）。"""
    if station_name in STATION_EPOCHS and model_name in STATION_EPOCHS[station_name]:
        n_epochs = STATION_EPOCHS[station_name][model_name]
    else:
        n_epochs = MODEL_EPOCHS.get(model_name, 15)
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
            n += xb.size(0)
        mse = epoch_loss / max(n, 1)
        if epoch == 1 or epoch == n_epochs or epoch % 5 == 0:
            print(f"  [{station_name}/{model_name}] epoch {epoch:3d}/{n_epochs}  MSE={mse:.6f}")

    model.eval()
    return model


@torch.no_grad()
def predict_all(model: nn.Module, dataset: RunoffWindowDataset) -> np.ndarray:
    """对整个 dataset 进行推理，返回预测值数组。"""
    model.eval()
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    preds = []
    for xb, _ in loader:
        preds.append(model(xb.to(DEVICE)).cpu().numpy())
    return np.concatenate(preds)


# ═══════════════════════════════════════════════════
#  4. 主流程
# ═══════════════════════════════════════════════════
def main():
    # ---- 字体 ----
    rcParams["font.sans-serif"] = ["SimSun", "Songti SC", "STSong",
                                    "Noto Serif CJK SC", "AR PL SungtiL GB",
                                    "WenQuanYi Zen Hei"]
    rcParams["axes.unicode_minus"] = False
    # 提高全局基础字体大小
    rcParams["font.size"] = 13

    # ---- 读取数据 ----
    detail_df = pd.read_csv(DETAIL_CSV)
    metrics_df = pd.read_csv(METRICS_CSV)

    # 构建 MSE 查找表 (metrics CSV key)
    mse_lookup: dict = {}
    for _, r in metrics_df.iterrows():
        mse_lookup[(r["model"], r["station_id"])] = float(r["MSE"])

    # ---- 为帕本/曼德勒重新训练 ----
    # retrain_preds[station][model_key] = (actual, predicted)
    retrain_preds: dict = {}

    for station in RETRAIN_STATIONS:
        retrain_preds[station] = {}
        # 提取该站点真实径流序列 (取任一模型即可，actual_runoff 相同)
        sub = detail_df[
            (detail_df["station_id"] == station) &
            (detail_df["model"] == MODEL_KEY_MAP[MODELS[0]])
        ].sort_values("date")
        series = sub["actual_runoff"].values.astype(np.float32)
        print(f"\n=== 站点: {station}  序列长度: {len(series)} ===")

        ds_full = RunoffWindowDataset(series, WINDOW)
        loader = DataLoader(ds_full, batch_size=BATCH, shuffle=True)
        actuals = np.array([series[i + WINDOW] for i in range(len(series) - WINDOW)],
                           dtype=np.float32)

        for model_key in MODELS:
            model = MODEL_CLS[model_key]()
            model = train_model(model, loader, model_key, station)
            preds = predict_all(model, ds_full)
            retrain_preds[station][model_key] = (actuals, preds)

    # ---- 组装所有站点的绘图数据 ----
    # plot_data[station][model_key] = (actual_array, pred_array)
    plot_data: dict = {}

    for station in STATIONS:
        plot_data[station] = {}
        for model_key in MODELS:
            if station in RETRAIN_STATIONS:
                # 使用重训数据
                plot_data[station][model_key] = retrain_preds[station][model_key]
            else:
                # 使用原始 detail CSV
                detail_model_name = MODEL_KEY_MAP[model_key]
                sub = detail_df[
                    (detail_df["station_id"] == station) &
                    (detail_df["model"] == detail_model_name)
                ].sort_values("date")
                act = sub["actual_runoff"].values.astype(np.float32)
                prd = sub["predicted_runoff"].values.astype(np.float32)
                plot_data[station][model_key] = (act, prd)

    # ---- 绘图 ----
    n_rows = len(STATIONS)
    n_cols = len(MODELS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.8 * n_cols, 3.8 * n_rows))

    # 每个站点(行)独立的坐标范围 —— 同一行共享 xlim/ylim
    row_lims: dict = {}
    for i, station in enumerate(STATIONS):
        row_vals = []
        for m in MODELS:
            a, p = plot_data[station][m]
            row_vals.extend(a.tolist())
            row_vals.extend(p.tolist())
        r_min, r_max = min(row_vals), max(row_vals)
        pad = 0.05 * (r_max - r_min) if r_max > r_min else 0.01
        row_lims[station] = (r_min - pad, r_max + pad)

    panel_idx = 0
    for i, station in enumerate(STATIONS):
        lim = row_lims[station]
        for j, model_key in enumerate(MODELS):
            ax = axes[i, j]
            act, prd = plot_data[station][model_key]
            color = MODEL_COLORS[model_key]

            # 浅色网格
            ax.grid(True, ls=":", lw=0.4, color="#cccccc", zorder=0)
            ax.set_axisbelow(True)

            # 若样本足够，拟合回归直线并绘制 95% 置信区间蒙版
            if len(act) >= 5 and np.std(act) > 0 and np.std(prd) > 0:
                x_arr = act.astype(float)
                y_arr = prd.astype(float)
                # 最小二乘直线 y = a + b x
                A = np.vstack([x_arr, np.ones_like(x_arr)]).T
                b_slope, a_intercept = np.linalg.lstsq(A, y_arr, rcond=None)[0]
                # 拟合残差标准差
                y_fit = a_intercept + b_slope * x_arr
                resid = y_arr - y_fit
                dof = max(len(x_arr) - 2, 1)
                sigma = float(np.sqrt(np.sum(resid ** 2) / dof))

                # 在当前站点范围内生成平滑曲线
                x_line = np.linspace(lim[0], lim[1], 200)
                y_line = a_intercept + b_slope * x_line
                delta = 1.96 * sigma
                y_low = y_line - delta
                y_up = y_line + delta

                # 置信区间蒙版（使用模型颜色、较低透明度）
                ax.fill_between(
                    x_line,
                    y_low,
                    y_up,
                    color=color,
                    alpha=0.15,
                    zorder=1.5,
                    linewidth=0,
                )

                # 上下边界线（略深一些）
                ax.plot(x_line, y_low, color=color, lw=0.8, ls="-", alpha=0.8, zorder=2)
                ax.plot(x_line, y_up, color=color, lw=0.8, ls="-", alpha=0.8, zorder=2)

            # 拟合点
            ax.scatter(act, prd, s=10, alpha=0.5, color=color,
                       edgecolors="white", linewidths=0.2, zorder=3)
            # y = x 参考线
            ax.plot(lim, lim, ls="--", color="#555555", lw=1.0, zorder=2)

            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_aspect("equal", adjustable="box")

            # 减少刻度数量，避免拥挤
            ax.locator_params(axis="both", nbins=5)
            ax.tick_params(labelsize=9, direction="in", length=3)

            # 美化边框
            for spine in ax.spines.values():
                spine.set_linewidth(0.6)
                spine.set_color("#444444")

            # 坐标轴标签
            if j == 0:
                ax.set_ylabel("预测值", fontsize=14)
            else:
                ax.tick_params(labelleft=False)
            if i == n_rows - 1:
                ax.set_xlabel("观测值", fontsize=14)
            else:
                ax.tick_params(labelbottom=False)

            # MSE 标注 —— 半透明底色
            station_met = STATION_NAME_MAP_TO_METRICS.get(station, station)
            mse_val = mse_lookup.get((model_key, station_met), float("nan"))
            if np.isfinite(mse_val):
                if mse_val < 0.01:
                    mse_str = f"MSE = {mse_val:.4f}"
                else:
                    mse_str = f"MSE = {mse_val:.3f}"
                ax.text(0.96, 0.06, mse_str, transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                  ec="#aaaaaa", alpha=0.85))

            # (a)(b)... 标注
                label = f"({chr(ord('a') + panel_idx)})"
                ax.text(0.04, 0.94, label, transform=ax.transAxes,
                    ha="left", va="top", fontsize=13, fontweight="bold")
            panel_idx += 1

    # 行/列外部标签 —— 紧贴子图
    fig.tight_layout(rect=[0.04, 0.035, 0.99, 0.96])
    fig.subplots_adjust(hspace=0.12, wspace=0.12)

    for j, m in enumerate(MODELS):
        # 列标签：取该列第一个子图的中心 x
        bbox_j = axes[0, j].get_position()
        # 稍微加大列名与子图之间的竖直间距
        fig.text((bbox_j.x0 + bbox_j.x1) / 2, bbox_j.y1 + 0.012, m,
                 ha="center", va="bottom", fontsize=15, fontweight="bold")
    for i, s in enumerate(STATIONS):
        # 行标签：取该行第一个子图的中心 y，并让行名远离子图
        bbox_i = axes[i, 0].get_position()
        fig.text(bbox_i.x0 - 0.26, (bbox_i.y0 + bbox_i.y1) / 2, s,
             ha="right", va="center", fontsize=15, fontweight="bold", rotation=90)

    # 全局图例
    handles = [
        Line2D([], [], marker="o", ls="", color=MODEL_COLORS[m], label=m,
               markersize=6, markeredgecolor="white", markeredgewidth=0.3)
        for m in MODELS
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.52, 0.001),
               ncol=len(MODELS), frameon=False, fontsize=13)
    OUTPUT_IMG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_IMG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✓ 图片已保存至 {OUTPUT_IMG}")


if __name__ == "__main__":
    main()
