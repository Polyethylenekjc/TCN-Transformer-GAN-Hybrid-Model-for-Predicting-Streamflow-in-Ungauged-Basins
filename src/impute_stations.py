import os
import sys
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

# Ensure project root on sys.path so 'src.*' imports work when running this file directly
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.config_loader import ConfigLoader
from src.utils.grdc_parser import load_grdc_directory
from src.utils.glofas_station_features import (
    list_available_dates,
    extract_glofas_series,
    add_doy_features,
    lonlat_to_pixel,
)
try:
    from rich.progress import track
except ImportError:
    def track(iter, description=""):
        print(description)
        return iter


@dataclass
class ImputeConfig:
    image_dir: str
    station_dir: str
    region: List[float]
    image_size: Tuple[int, int]
    window_size: int
    kernel_size: int
    use_doy: bool
    hidden_dim: int
    num_layers: int
    num_epochs: int
    lr: float
    device: str
    output_dir: str


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):  # x: (B, T, F)
        out, _ = self.lstm(x)
        # take last time step
        y = self.fc(out[:, -1, :])  # (B,1)
        return y.squeeze(-1)


def build_sequences(features: np.ndarray, window: int) -> np.ndarray:
    """Build rolling windows features of shape (N, window, F)."""
    T, F = features.shape
    if T < 1:
        return np.zeros((0, window, F), dtype=np.float32)
    seqs: List[np.ndarray] = []
    for t in range(T):
        start = max(0, t - window + 1)
        pad_len = window - (t - start + 1)
        chunk = features[start : t + 1]
        if pad_len > 0:
            pad = np.repeat(chunk[:1, :], pad_len, axis=0)
            chunk = np.concatenate([pad, chunk], axis=0)
        seqs.append(chunk[-window:])
    return np.stack(seqs, axis=0).astype(np.float32)


def standardize(train_vals: np.ndarray) -> Tuple[float, float]:
    mu = float(np.nanmean(train_vals))
    sigma = float(np.nanstd(train_vals))
    if not np.isfinite(sigma) or sigma == 0:
        sigma = 1.0
    return mu, sigma


def run_imputation(origin_dir: str, cfg_path: str, use_doy: bool = True, kernel_size: int = 3, hidden_dim: int = 64):
    import time

    start_time = time.time()

    print(f"[IMPUTE] Loading config: {cfg_path}")
    cfg_raw = ConfigLoader.load_config(cfg_path)
    image_dir = cfg_raw['data']['image_dir']
    station_dir = cfg_raw['data']['station_dir']
    output_dir = cfg_raw['data'].get('output_dir', os.path.join(Path(station_dir).parent, 'output'))
    region = cfg_raw['data']['region']
    window = int(cfg_raw['data'].get('window_size', 5))
    img_h = int(cfg_raw['data'].get('image_size', [256, 256])[0])
    img_w = int(cfg_raw['data'].get('image_size', [256, 256])[1])
    # resolve device safely
    cfg_device = str(cfg_raw['train'].get('device', 'cpu')).lower()
    if cfg_device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'
    else:
        device = 'cuda' if cfg_device.startswith('cuda') else 'cpu'
    lr = float(cfg_raw['train'].get('lr', 1e-3))
    num_epochs = int(cfg_raw['train'].get('num_epochs', 50))
    num_layers = int(cfg_raw['model'].get('num_layers', 1))

    Path(station_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"[IMPUTE] Image dir: {image_dir}\n[IMPUTE] Station dir: {station_dir}\n[IMPUTE] Output dir: {output_dir}")
    print(f"[IMPUTE] Device: {device}  window={window}  kernel_size={kernel_size}  use_doy={use_doy}")

    dates = list_available_dates(image_dir)
    if not dates:
        raise RuntimeError(f"No NPY images found in {image_dir}")
    print(f"[IMPUTE] Found {len(dates)} image dates. Range: {dates[0]} - {dates[-1]}")

    # 1) Parse GRDC
    print(f"[IMPUTE] Parsing GRDC files in: {origin_dir}")
    stations = load_grdc_directory(origin_dir)
    print(f"[IMPUTE] Parsed {len(stations)} station files from origin directory")
    # filter by bbox region
    lon_min, lon_max, lat_min, lat_max = region
    stations = [s for s in stations if np.isfinite(s['lon']) and np.isfinite(s['lat']) and (lon_min <= s['lon'] <= lon_max) and (lat_min <= s['lat'] <= lat_max)]
    print(f"[IMPUTE] {len(stations)} stations remain after bbox filtering")
    if not stations:
        raise RuntimeError("No stations within the configured region.")

    # 2) Build per-station frames and features
    print(f"[IMPUTE] Pre-calculating pixel coordinates for {len(stations)} stations...")
    station_pixels = []
    for s in stations:
        lon, lat = float(s['lon']), float(s['lat'])
        px, py = lonlat_to_pixel(lon, lat, region, (img_h, img_w))
        station_pixels.append((px, py))

    print(f"[IMPUTE] Extracting GloFAS data for all stations across {len(dates)} dates...")
    # Shape: (num_dates, num_stations)
    all_glofas_values = np.full((len(dates), len(stations)), np.nan, dtype=np.float32)
    
    half = max(0, kernel_size // 2)
    
    for di, d in track(enumerate(dates), description="Extracting GloFAS Data...", total=len(dates)):
        path = Path(image_dir) / f"{d}.npy"
        if not path.exists():
            continue
        try:
            arr = np.load(path)
        except Exception:
            continue
            
        if arr.ndim == 2:
            ch0 = arr
        else:
            ch0 = arr[0]
            
        H, W = ch0.shape
        
        for si, (px, py) in enumerate(station_pixels):
            y0 = max(0, py - half)
            y1 = min(H, py + half + 1)
            x0 = max(0, px - half)
            x1 = min(W, px + half + 1)
            patch = ch0[y0:y1, x0:x1]
            if patch.size > 0:
                all_glofas_values[di, si] = float(np.nanmean(patch))

    per_station: List[Dict] = []
    if use_doy:
        doy_sin, doy_cos = add_doy_features(dates)

    for si, s in track(enumerate(stations), description="Building DataFrames...", total=len(stations)):
        lon, lat = float(s['lon']), float(s['lat'])
        # print(f"[IMPUTE] Building dataframe for station {si+1}/{len(stations)}")
        
        df = pd.DataFrame({'timestamp': dates})
        # merge observed runoff
        obs = s['df'].copy()
        # station df already in YYYYMMDD
        df = df.merge(obs, on='timestamp', how='left')  # adds 'runoff'
        df['lon'] = lon
        df['lat'] = lat
        df['glofas'] = all_glofas_values[:, si]
        if use_doy:
            df['doy_sin'] = doy_sin
            df['doy_cos'] = doy_cos
        per_station.append({**s, 'frame': df})

    # 3) Assemble training samples (only where runoff observed)
    # features order: [glofas, doy_sin, doy_cos] (when enabled)
    feat_cols = ['glofas'] + (['doy_sin', 'doy_cos'] if use_doy else [])
    all_X = []
    all_y = []
    all_indices = []  # (station_idx, t)
    for si, s in track(enumerate(per_station), description="Assembling Training Data...", total=len(per_station)):
        df = s['frame']
        feats = df[feat_cols].to_numpy(dtype=np.float32)
        seqs = build_sequences(feats, window)
        y = df['runoff'].to_numpy(dtype=np.float32)
        for t in range(len(df)):
            if np.isfinite(y[t]):
                all_X.append(seqs[t])
                all_y.append(y[t])
                all_indices.append((si, t))

    if not all_X:
        raise RuntimeError("No observed runoff samples found for training.")

    X = np.stack(all_X).astype(np.float32)  # (N, W, F)
    y = np.array(all_y, dtype=np.float32)   # (N,)

    print(f"[IMPUTE] Assembled training set: {len(all_X)} samples. Feature dim={X.shape[-1]}")

    # standardize features and target using training set
    feat_mu = np.nanmean(X, axis=(0, 1), keepdims=True)
    feat_std = np.nanstd(X, axis=(0, 1), keepdims=True)
    feat_std = np.where(feat_std == 0, 1.0, feat_std)
    Xn = (X - feat_mu) / feat_std
    y_mu = float(np.nanmean(y))
    y_std = float(np.nanstd(y))
    if y_std == 0 or not np.isfinite(y_std):
        y_std = 1.0
    yn = (y - y_mu) / y_std

    # train/val split: last 10% as val for temporal realism
    N = Xn.shape[0]
    val_size = max(1, int(0.1 * N))
    train_idx = np.arange(0, N - val_size)
    val_idx = np.arange(N - val_size, N)

    Xtr, ytr = Xn[train_idx], yn[train_idx]
    Xva, yva = Xn[val_idx], yn[val_idx]

    # 4) Train model
    input_dim = Xtr.shape[-1]
    model = LSTMRegressor(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    def _to_t(x):
        return torch.from_numpy(x).to(device)

    best_val = float('inf')
    best_state = None
    patience = 8
    bad = 0
    batch_size = 256

    def batches(Xa, ya, bs):
        for i in range(0, len(Xa), bs):
            yield Xa[i:i+bs], ya[i:i+bs]

    for epoch in track(range(num_epochs), description="Training LSTM..."):
        model.train()
        tr_loss = 0.0
        for bx, by in batches(Xtr, ytr, batch_size):
            bx_t = _to_t(bx)
            by_t = _to_t(by)
            pred = model(bx_t)
            loss = loss_fn(pred, by_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += float(loss.item()) * len(bx)
        tr_loss /= max(1, len(Xtr))

        model.eval()
        with torch.no_grad():
            pv = model(_to_t(Xva))
            vl = float(loss_fn(pv, _to_t(yva)).item())

        # print(f"[IMPUTE][Epoch {epoch+1:03d}] train_loss={tr_loss:.6f} val_loss={vl:.6f}")

        if vl < best_val:
            best_val = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[IMPUTE] Loaded best model state with val_loss={best_val:.6f}")

    # 5) Inference and export per station
    report: Dict[str, Dict] = {}
    for si, s in track(enumerate(per_station), description="Imputing & Saving...", total=len(per_station)):
        df = s['frame'].copy()
        feats = df[feat_cols].to_numpy(dtype=np.float32)
        seqs_all = build_sequences(feats, window)
        seqs_all = (seqs_all - feat_mu) / feat_std
        with torch.no_grad():
            yhat_n = model(_to_t(seqs_all)).cpu().numpy()
        yhat = yhat_n * y_std + y_mu

        # Fill only missing
        filled = df['runoff'].to_numpy(dtype=np.float32)
        is_nan = ~np.isfinite(filled)
        filled[is_nan] = yhat[is_nan]
        df['runoff'] = filled
        df['is_imputed'] = is_nan.astype(int)
        df['source'] = np.where(df['is_imputed'] == 1, 'lstm_glofas', 'observed')
        # columns: timestamp, lon, lat, runoff
        out_csv = Path(station_dir) / f"station_{si:02d}.csv"
        df[['timestamp', 'lon', 'lat', 'runoff', 'is_imputed', 'source']].to_csv(out_csv, index=False)
        # print(f"[IMPUTE] Saved station CSV: {out_csv}  (imputed {int(is_nan.sum())} days)")

        # station metrics (only where observation exists in validation slice for this station)
        obs_mask = s['frame']['runoff'].notna().to_numpy()
        # align with global val set by checking which samples belonged to this station's observed indices
        # approximate metric: compute MAE/RMSE over observed days using model predictions on those dates
        if obs_mask.any():
            y_true = s['frame']['runoff'].to_numpy(dtype=np.float32)
            y_pred_obs = yhat[obs_mask]
            y_true_obs = y_true[obs_mask]
            mae = float(np.nanmean(np.abs(y_pred_obs - y_true_obs)))
            rmse = float(np.sqrt(np.nanmean((y_pred_obs - y_true_obs) ** 2)))
        else:
            mae = float('nan')
            rmse = float('nan')

        report[s.get('id') or f'station_{si:02d}'] = {
            'name': s.get('name', ''),
            'lon': float(s['lon']),
            'lat': float(s['lat']),
            'area': float(s.get('area') or float('nan')),
            'n_days': int(len(df)),
            'obs_days': int(int(np.isfinite(s['frame']['runoff']).sum())),
            'imputed_days': int(int(is_nan.sum())),
            'mae': mae,
            'rmse': rmse,
        }

    # Save summary report
    rep_path = Path(output_dir) / 'imputation_report.json'
    with open(rep_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    elapsed = time.time() - start_time
    print(f"[IMPUTE] Report saved: {rep_path}")
    print(f"[IMPUTE] Completed imputation for {len(per_station)} stations in {elapsed:.1f}s")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Impute station runoff using GloFAS (channel 0) + LSTM')
    parser.add_argument('--origin-dir', type=str, default='/mnt/d/Store/TTF/stations_origin', help='Directory of raw GRDC .txt files')
    parser.add_argument('--config', type=str, default='data/config.yaml', help='Path to YAML config')
    parser.add_argument('--kernel-size', type=int, default=3, help='Kernel size for neighborhood average in GloFAS channel')
    parser.add_argument('--no-doy', action='store_true', help='Disable day-of-year sin/cos features')
    parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dim for LSTM')
    args = parser.parse_args()

    # make config path relative to repo root if not absolute
    cfg_path = args.config
    if not os.path.isabs(cfg_path):
        cfg_path = str((_REPO_ROOT / cfg_path).resolve())

    run_imputation(
        origin_dir=args.origin_dir,
        cfg_path=cfg_path,
        use_doy=not args.no_doy,
        kernel_size=args.kernel_size,
        hidden_dim=args.hidden_dim,
    )
