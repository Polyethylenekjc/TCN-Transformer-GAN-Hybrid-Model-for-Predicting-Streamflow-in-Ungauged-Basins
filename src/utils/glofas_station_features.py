from typing import List, Tuple, Optional
from pathlib import Path
import numpy as np
import pandas as pd


def lonlat_to_pixel(lon: float, lat: float, region: List[float], image_size: Tuple[int, int]) -> Tuple[int, int]:
    """Map lon/lat to pixel indices consistent with StreamflowDataset.

    region: [lon_min, lon_max, lat_min, lat_max]
    image_size: (H, W)
    """
    lon_min, lon_max, lat_min, lat_max = region
    H, W = image_size
    lon_span = max(1e-9, (lon_max - lon_min))
    lat_span = max(1e-9, (lat_max - lat_min))
    x_ratio = (lon - lon_min) / lon_span
    y_ratio = (lat_max - lat) / lat_span
    px = int(round(x_ratio * (W - 1)))
    py = int(round(y_ratio * (H - 1)))
    px = max(0, min(px, W - 1))
    py = max(0, min(py, H - 1))
    return px, py


def list_available_dates(image_dir: str) -> List[str]:
    """List available date strings (YYYYMMDD) from .npy filenames in directory."""
    p = Path(image_dir)
    dates = [f.stem for f in p.glob('*.npy') if not f.name.endswith('_mask.npy')]
    dates = [d for d in dates if d.isdigit() and len(d) == 8]
    return sorted(set(dates))


def extract_glofas_series(
    image_dir: str,
    dates: List[str],
    lon: float,
    lat: float,
    region: List[float],
    image_size: Tuple[int, int] = (256, 256),
    kernel_size: int = 1,
) -> pd.Series:
    """Extract per-day GloFAS (channel 0) values at station location.

    Returns a pandas Series indexed by date string YYYYMMDD.
    Missing files will yield NaN for that date.
    """
    px, py = lonlat_to_pixel(lon, lat, region, image_size)
    half = max(0, kernel_size // 2)
    vals = []
    for d in dates:
        path = Path(image_dir) / f"{d}.npy"
        if not path.exists():
            vals.append(np.nan)
            continue
        try:
            arr = np.load(path)
        except Exception:
            vals.append(np.nan)
            continue
        if arr.ndim == 2:
            # assume single-channel, take as is
            ch0 = arr
        else:
            ch0 = arr[0]

        y0 = max(0, py - half)
        y1 = min(ch0.shape[0], py + half + 1)
        x0 = max(0, px - half)
        x1 = min(ch0.shape[1], px + half + 1)
        patch = ch0[y0:y1, x0:x1]
        if patch.size == 0:
            vals.append(np.nan)
        else:
            vals.append(float(np.nanmean(patch)))
    return pd.Series(vals, index=dates, name='glofas')


def add_doy_features(dates: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute day-of-year sin/cos features aligned with dates list."""
    dts = pd.to_datetime(dates, format='%Y%m%d')
    doy = dts.dayofyear.to_numpy().astype(float)
    # handle leap year by mapping 366->365
    doy = np.where(doy > 365, 365, doy)
    sin = np.sin(2 * np.pi * (doy / 365.0))
    cos = np.cos(2 * np.pi * (doy / 365.0))
    return sin.astype(np.float32), cos.astype(np.float32)
