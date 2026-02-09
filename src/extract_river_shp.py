"""Extract river network shapefile from test-set runoff images.

This script does NOT use model predictions. It operates directly on the
runoff channel of the test dataset (image .npy files), computes an
annual-mean (time-mean) runoff map, selects pixels whose runoff is not
practically zero, groups them, and converts each connected river segment
into a polyline shapefile.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
from scipy.ndimage import label
import shapefile  # pyshp

from src.utils.config_loader import ConfigLoader


def pixel_to_lonlat(px: int, py: int, region: List[float], image_size: Tuple[int, int]) -> Tuple[float, float]:
    """Convert pixel indices back to lon/lat using proportional mapping.

    region: [lon_min, lon_max, lat_min, lat_max]
    image_size: (H, W)
    """
    lon_min, lon_max, lat_min, lat_max = region
    H, W = image_size

    lon_span = max(1e-9, (lon_max - lon_min))
    lat_span = max(1e-9, (lat_max - lat_min))

    x_ratio = px / max(1, (W - 1))
    y_ratio = py / max(1, (H - 1))

    lon = lon_min + x_ratio * lon_span
    lat = lat_max - y_ratio * lat_span

    return float(lon), float(lat)


def _mask_path_for(npy_path: Path) -> Path:
    """Return expected mask path for a given npy file (same rule as dataset)."""
    return npy_path.with_name(npy_path.stem + "_mask.npy")


def compute_annual_mean_from_images(
    image_dir: str,
    flow_threshold: float,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute time-mean runoff and active-day fraction directly from images.

    - Uses first channel of each .npy file as runoff.
    - Optionally restricts to dates in [date_start, date_end] if filenames are
      date strings like YYYYMMDD.
    - Applies `flow_threshold` to consider near-zero runoff as 0 before
      computing statistics.

    Returns:
        mean_runoff: (H, W) time-mean runoff
        active_frac: (H, W) fraction of days with runoff > `flow_threshold`
    """
    img_dir = Path(image_dir)
    files = sorted(
        f for f in img_dir.iterdir()
        if f.suffix == ".npy" and not f.name.endswith("_mask.npy")
    )
    if not files:
        raise RuntimeError(f"No .npy files found in {image_dir}")

    # Optionally filter by date range if stems look like dates
    def _in_range(path: Path) -> bool:
        if date_start is None and date_end is None:
            return True
        stem = path.stem
        if not (stem.isdigit() and len(stem) == 8):
            return True  # keep non-date filenames
        if date_start is not None and stem < date_start:
            return False
        if date_end is not None and stem > date_end:
            return False
        return True

    files = [f for f in files if _in_range(f)]
    if not files:
        raise RuntimeError("No image files remain after applying date filter.")

    # Determine spatial size from first file
    first_arr = np.load(files[0])
    if first_arr.ndim == 2:
        H, W = first_arr.shape
    else:
        H, W = first_arr.shape[1], first_arr.shape[2]

    sum_runoff = np.zeros((H, W), dtype=np.float64)
    count_days = np.zeros((H, W), dtype=np.int32)
    active_days = np.zeros((H, W), dtype=np.int32)

    for path in files:
        arr = np.load(path)
        if arr.ndim == 2:
            runoff = arr.astype(np.float64)
        else:
            runoff = arr[0].astype(np.float64)  # first channel as flow

        if runoff.shape != (H, W):
            raise RuntimeError(
                f"Inconsistent image size: {path} has shape {runoff.shape}, expected {(H, W)}"
            )

        # Apply mask if present (same rule as dataset)
        mpath = _mask_path_for(path)
        if mpath.exists():
            mask_np = np.load(mpath).astype(bool)
            if mask_np.shape != (H, W):
                raise RuntimeError(
                    f"Mask size mismatch for {mpath}: {mask_np.shape} vs {(H, W)}"
                )
        else:
            mask_np = np.ones_like(runoff, dtype=bool)

        # Replace NaN/Inf and apply mask
        runoff = np.nan_to_num(runoff, nan=0.0, posinf=0.0, neginf=0.0)
        runoff[~mask_np] = 0.0

        # Threshold small values to 0 ("不显著为0" 的判定基础)
        runoff[np.abs(runoff) < flow_threshold] = 0.0

        sum_runoff += runoff
        count_days[mask_np] += 1
        active_days[mask_np & (np.abs(runoff) > flow_threshold)] += 1

    # Avoid division by zero
    safe_count = count_days.copy().astype(np.float64)
    safe_count[safe_count == 0] = 1.0

    mean_runoff = sum_runoff / safe_count
    active_frac = active_days.astype(np.float64) / safe_count

    return mean_runoff.astype(np.float32), active_frac.astype(np.float32)


def extract_river_shapefile(
    config: Dict[str, Any],
    image_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
) -> Path:
    """Main pipeline to build river-network shapefile from test images.

    Steps:
      1. 从 image_dir 中所有测试集 .npy 图像提取径流通道，计算时间平均值和活跃频率。
      2. 选出 "全年径流值不显著为 0" 的像元：同时满足
         - 时间平均径流 > mean_threshold
         - 活跃频率 > min_active_fraction
      3. 对这些像元做 8 邻域连通分组，每个连通分量对应一段河流。
      4. 对每个连通分量按 (row, col) 排序，将像元中心串联成一条 polyline，
         避免同一段河流被表示为大量散点。
    """
    data_cfg = config.get("data", {})
    river_cfg = config.get("river_extraction", {})

    img_dir = image_dir or data_cfg.get("image_dir", "./data/images")
    out_dir = Path(output_dir or data_cfg.get("output_dir", "./output"))
    out_dir.mkdir(parents=True, exist_ok=True)

    region = data_cfg.get("region", [0.0, 1.0, 0.0, 1.0])
    flow_threshold = float(data_cfg.get("flow_threshold", 0.1))

    # 河网提取参数，可在 config.yaml 的 river_extraction 段中覆写
    mean_threshold = float(river_cfg.get("mean_threshold", flow_threshold))
    min_active_fraction = float(river_cfg.get("min_active_fraction", 0.1))
    min_component_size = int(river_cfg.get("min_component_size", 5))

    mean_runoff, active_frac = compute_annual_mean_from_images(
        img_dir,
        flow_threshold=flow_threshold,
        date_start=date_start,
        date_end=date_end,
    )

    H, W = mean_runoff.shape

    # 条件：时间平均值不小 + 在一年内经常有流量
    mask = (mean_runoff > mean_threshold) & (active_frac > min_active_fraction)

    # 保存中间栅格，方便调试
    np.save(out_dir / "annual_mean_runoff_obs.npy", mean_runoff)
    np.save(out_dir / "river_mask.npy", mask.astype(np.uint8))

    # 连通组件标记，8 邻域
    structure = np.ones((3, 3), dtype=np.int32)
    labeled, num = label(mask, structure=structure)

    shp_path = out_dir / "river_network_from_obs.shp"

    writer = shapefile.Writer(str(shp_path), shapeType=shapefile.POLYLINE)
    writer.autoBalance = 1
    writer.field("id", "N")
    writer.field("npts", "N")
    writer.field("mean_q", "F", decimal=6)
    writer.field("max_q", "F", decimal=6)

    region_list = [float(x) for x in region]

    seg_id = 0
    for label_id in range(1, num + 1):
        comp_mask = labeled == label_id
        n_pixels = int(comp_mask.sum())
        if n_pixels < min_component_size:
            continue  # 过滤掉小斑点

        # 提前拿出该分量的径流值用于属性
        comp_vals = mean_runoff[comp_mask]
        comp_mean = float(comp_vals.mean())
        comp_max = float(comp_vals.max())

        # 像元索引 (py, px)，按 (row, col) 排序，粗略保证线的连续性
        coords_idx = np.argwhere(comp_mask)  # (N, 2) -> (py, px)
        # np.lexsort: 先按第二个键排序，所以给 (px, py)
        order = np.lexsort((coords_idx[:, 1], coords_idx[:, 0]))
        coords_sorted = coords_idx[order]

        polyline: List[Tuple[float, float]] = []
        for py, px in coords_sorted:
            lon, lat = pixel_to_lonlat(int(px), int(py), region_list, (H, W))
            polyline.append((lon, lat))

        if len(polyline) < 2:
            continue  # 至少两个点才能构成线

        seg_id += 1
        writer.line([polyline])
        writer.record(seg_id, len(polyline), comp_mean, comp_max)

    writer.close()

    return shp_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract river-network shapefile from test-set runoff images"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./data/config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Override data.image_dir (test image directory)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override data.output_dir for outputs",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional start date (YYYYMMDD) to restrict images",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Optional end date (YYYYMMDD) to restrict images",
    )

    args = parser.parse_args()

    config = ConfigLoader.load_config(args.config)

    shp_path = extract_river_shapefile(
        config,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        date_start=args.start_date,
        date_end=args.end_date,
    )

    print(f"\n✓ River-network shapefile (from obs/test images) saved to: {shp_path}")


if __name__ == "__main__":
    main()
