import os
import shutil
import zipfile
from typing import Dict, List

import cdsapi

try:
    import xarray as xr
except Exception:
    xr = None
import numpy as np
import pandas as pd


# ========== 认证：分别为 GLOFAS 与 ERA5 创建客户端 ==========
def get_cds_client(service: str) -> cdsapi.Client:
    """Create cdsapi client for a specific service (glofas|era5)."""
    service = service.lower()
    if service == "glofas":
        return cdsapi.Client(url="https://ewds.climate.copernicus.eu/api", key="360a5389-ec27-4bc7-ad71-d521c1995e4a")
    elif service == "era5":
        return cdsapi.Client(url="https://cds.climate.copernicus.eu/api", key="360a5389-ec27-4bc7-ad71-d521c1995e4a")
    else:
        raise ValueError("service must be 'glofas' or 'era5'")


# ========== 区域/路径配置 ==========
OUTPUT_ROOT = "/mnt/d/store/TTF"
ERA5_DIR = os.path.join(OUTPUT_ROOT, "ERA5")
GLOFAS_DIR = os.path.join(OUTPUT_ROOT, "GLOFAS")
IMAGES_DIR = os.path.join(OUTPUT_ROOT, "images")
STATIONS_DIR = os.path.join(OUTPUT_ROOT, "stations")  # 可能为空
os.makedirs(ERA5_DIR, exist_ok=True)
os.makedirs(GLOFAS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(STATIONS_DIR, exist_ok=True)

# Pakistan + India region (N, W, S, E) per CDS convention
# Exact 0.1° grid coverage for 256×256:
# latMin=7.0, latMax=32.6, lonMin=77.5, lonMax=103.1
CDS_AREA_SEASIA = [32.6, 77.5, 7.0, 103.1]


# ========== 1. 下载 GLOFAS ==========
def download_glofas(year: int):
    target_zip = os.path.join(GLOFAS_DIR, f"{year}.zip")
    target_dir = os.path.join(GLOFAS_DIR, str(year))
    if os.path.exists(os.path.join(target_dir, f"{year}.nc")):
        print(f"[GLOFAS] {year} exists, skip download.")
        return
    
    client = get_cds_client("glofas")
    dataset = "cems-glofas-historical"
    request = {
        "system_version": ["version_4_0"],
        "hydrological_model": ["lisflood"],
        "product_type": ["consolidated"],
        "variable": ["river_discharge_in_the_last_24_hours"],
        "hyear": [str(year)],
        "hmonth": [f"{m:02d}" for m in range(1, 13)],
        "hday": [f"{d:02d}" for d in range(1, 32)],
        "data_format": "netcdf",
        "download_format": "zip",
        "area": CDS_AREA_SEASIA,  # [N, W, S, E]
    }

    print(f"[GLOFAS] Downloading {year} ...")
    client.retrieve(dataset, request, target_zip)
    print(f"[GLOFAS] Extracting {target_zip} -> {target_dir}")
    with zipfile.ZipFile(target_zip, 'r') as zf:
        zf.extractall(target_dir)

    # 重命名为 {year}.nc
    renamed = False
    for root, _, files in os.walk(target_dir):
        for fn in files:
            if fn.endswith('.nc'):
                old = os.path.join(root, fn)
                new = os.path.join(target_dir, f"{year}.nc")
                shutil.move(old, new)
                renamed = True
                print(f"[GLOFAS] {old} -> {new}")
                break
    if os.path.exists(target_zip):
        os.remove(target_zip)
    if not renamed:
        raise RuntimeError(f"[GLOFAS] No .nc found after extracting {target_zip}")


# ========== 2. 下载 ERA5 ==========
ERA5_VARIABLES = [
    "2m_dewpoint_temperature",
    "skin_temperature",
    "surface_latent_heat_flux",
    "surface_net_thermal_radiation",
    "surface_solar_radiation_downwards",
    "potential_evaporation",
    "runoff",
    "sub_surface_runoff",
    "total_evaporation",
    "total_precipitation",
]


def download_era5(year: int, month: int, hour: str = "12:00"):
    target = os.path.join(ERA5_DIR, f"{year}-{month:02d}.nc")
    if os.path.exists(target):
        print(f"[ERA5] Exists: {target}")
        return
    print(f"[ERA5] Downloading {year}-{month:02d} ...")

    client = get_cds_client("era5")
    dataset = "reanalysis-era5-land"
    request = {
        "variable": ERA5_VARIABLES,
        "year": f"{year}",
        "month": f"{month:02d}",
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [hour],  # 每日一次，可调整
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": CDS_AREA_SEASIA,  # [N, W, S, E]
    }
    client.retrieve(dataset, request).download(target=target)


# ========== 3. 合并为多通道 NPY ==========
def align_and_save_npy(years: List[int],
    out_dir: str = IMAGES_DIR,
    region: List[float] = [77.5, 103.1, 7.0, 32.6],
    resolution: float = 0.1,
    grid_h: int = 256,
    grid_w: int = 256):
    """
        读取 ERA5 与 GLOFAS NetCDF，裁剪到区域并按日期合并为 (C,H,W) NPY。
        通道顺序（10 通道）：
            [ GLOFAS_discharge,
                2m_dewpoint_temperature, skin_temperature, total_precipitation,
                total_evaporation, surface_solar_radiation_downwards(优先，若无则runoff),
                sub_surface_runoff, surface_latent_heat_flux, surface_net_thermal_radiation,
                potential_evaporation ]
        说明：优先使用 ERA5 的 ssrd（短波向下辐射）替代 ro（runoff）。
    """
    if xr is None:
        raise RuntimeError("xarray is required. Please install xarray and netCDF4.")

    lon_min, lon_max, lat_min, lat_max = region

    # 构建目标网格：使用半格偏移以得到严格的 256×256 中心点
    # 中心点覆盖 [lon_min+0.05, ..., lon_max-0.05], [lat_max-0.05, ..., lat_min+0.05]
    lons = lon_min + (resolution * 0.5) + resolution * np.arange(grid_w)
    lats = lat_max - (resolution * 0.5) - resolution * np.arange(grid_h)
    H, W = grid_h, grid_w

    # 打开 ERA5 多文件
    era5_files = [os.path.join(ERA5_DIR, f"{y}-{m:02d}.nc") for y in years for m in range(1, 13)]
    era5_files = [p for p in era5_files if os.path.exists(p)]
    if not era5_files:
        raise RuntimeError("No ERA5 files found to merge.")
    # Disable time decoding to avoid date2num dependency; will parse manually
    ds_era5 = xr.open_mfdataset(era5_files, combine='by_coords', decode_times=False)
    # ERA5 可能包含 expver/number 维度，这里取第一个成员以保证二维空间
    if 'expver' in ds_era5.dims:
        ds_era5 = ds_era5.isel(expver=0)
    if 'number' in ds_era5.dims:
        ds_era5 = ds_era5.isel(number=0)
    ds_era5 = ds_era5.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))

    # 打开 GLOFAS 多年
    glofas_files = [os.path.join(GLOFAS_DIR, str(y), f"{y}.nc") for y in years]
    glofas_files = [p for p in glofas_files if os.path.exists(p)]
    if not glofas_files:
        raise RuntimeError("No GLOFAS files found to merge.")
    ds_glofas = xr.open_mfdataset(glofas_files, combine='by_coords', decode_times=False)
    # GLOFAS 坐标可能叫 lat/latitude, lon/longitude
    lat_name = 'lat' if 'lat' in ds_glofas.coords else 'latitude'
    lon_name = 'lon' if 'lon' in ds_glofas.coords else 'longitude'
    ds_glofas = ds_glofas.sel({lat_name: slice(lat_max, lat_min), lon_name: slice(lon_min, lon_max)})

    # 通用：时间坐标名解析 + 日期提取
    def _get_time_name(ds):
        candidates = ['time', 'valid_time', 'forecast_time']
        for c in candidates:
            if c in ds.coords or c in ds.variables:
                return c
        for k in list(ds.coords) + list(ds.variables):
            if 'time' in k.lower():
                return k
        raise RuntimeError("Cannot find time coordinate in dataset")

    # 按日期交集输出
    def _extract_days(time_var):
        units = getattr(time_var, 'attrs', {}).get('units', '')
        calendar = getattr(time_var, 'attrs', {}).get('calendar', 'standard')
        raw = time_var.values
        try:
            import cftime
            decoded = cftime.num2date(raw, units, calendar=calendar)
            return [pd.to_datetime(str(d)).strftime('%Y%m%d') for d in decoded]
        except Exception:
            try:
                from netCDF4 import num2date
                decoded = num2date(raw, units, calendar=calendar)
                return [pd.to_datetime(str(d)).strftime('%Y%m%d') for d in decoded]
            except Exception:
                # Fallback: direct conversion
                return [pd.to_datetime(r, errors='coerce').strftime('%Y%m%d') for r in raw]

    era5_time_name = _get_time_name(ds_era5)
    glofas_time_name = _get_time_name(ds_glofas)

    era5_times = set(_extract_days(ds_era5[era5_time_name]))
    glofas_times = set(_extract_days(ds_glofas[glofas_time_name]))
    common_days = sorted(era5_times.intersection(glofas_times))
    if not common_days:
        raise RuntimeError("No overlapping days between ERA5 and GLOFAS.")

    # 插值到统一网格（如二者经纬度完全一致为 0.1°，则可直接重索引）
    ds_era5_interp = ds_era5.interp(latitude=lats, longitude=lons)
    ds_glofas_interp = ds_glofas.interp({lat_name: lats, lon_name: lons})

    def _pick(ds, candidates):
        for c in candidates:
            if c in ds:
                return c
        return None

    # 变量映射：
    # - GLOFAS 在 NetCDF 中常见短名为 dis24（也可能是 river_discharge_in_the_last_24_hours/dis）
    # - ERA5 使用一组同义名优先级选择
    var_map = {
        'glofas': None,  # 将在下方通过候选列表自动选择
        'era5_vars': []
    }
    era5_synonyms = [
        ['d2m', '2m_dewpoint_temperature'],
        ['skt', 'skin_temperature'],
        ['tp', 'total_precipitation'],
        ['e', 'evap', 'total_evaporation'],
        # 优先 ssrd（surface_solar_radiation_downwards），若不存在再回退到 ro/runoff
        ['ssrd', 'surface_solar_radiation_downwards', 'ro', 'runoff'],
        ['ssro', 'sub_surface_runoff'],
        ['slhf', 'surface_latent_heat_flux'],
        ['str', 'surface_net_thermal_radiation'],
        ['pev', 'potential_evaporation'],
    ]
    for group in era5_synonyms:
        chosen = _pick(ds_era5_interp, group)
        if chosen is None:
            raise RuntimeError(f"Missing ERA5 variable (any of): {group}")
        var_map['era5_vars'].append(chosen)

    # GLOFAS discharge 同义名选择
    glofas_synonyms = [
        'dis24',
        'river_discharge_in_the_last_24_hours',
        'river_discharge',
        'dis',
    ]
    var_map['glofas'] = _pick(ds_glofas_interp, glofas_synonyms)
    if var_map['glofas'] is None:
        raise RuntimeError(
            "Missing GLOFAS variable: none of " + ", ".join(glofas_synonyms) +
            f". Available: {list(ds_glofas_interp.data_vars)}"
        )

    # Build mapping day->index for fast selection
    day_to_idx_era5 = {}
    for idx, d in enumerate(_extract_days(ds_era5[era5_time_name])):
        if d not in day_to_idx_era5:
            day_to_idx_era5[d] = idx
    day_to_idx_glofas = {}
    for idx, d in enumerate(_extract_days(ds_glofas[glofas_time_name])):
        if d not in day_to_idx_glofas:
            day_to_idx_glofas[d] = idx

    for day in common_days:
        target_path = os.path.join(out_dir, f"{day}.npy")
        if os.path.exists(target_path):
            print(f"[MERGE] Skip existing {day}.npy")
            continue

        sel_era5 = ds_era5_interp.isel({era5_time_name: day_to_idx_era5[day]})
        sel_glofas = ds_glofas_interp.isel({glofas_time_name: day_to_idx_glofas[day]})

        channels: List[np.ndarray] = []
        # GLOFAS discharge
        if var_map['glofas'] not in sel_glofas:
            raise RuntimeError(f"Missing GLOFAS variable: {var_map['glofas']}")
        channels.append(sel_glofas[var_map['glofas']].values.astype(np.float32))

        for vn in var_map['era5_vars']:
            if vn not in sel_era5:
                raise RuntimeError(f"Missing ERA5 variable: {vn}")
            channels.append(sel_era5[vn].values.astype(np.float32))

        arr = np.stack(channels, axis=0)  # (C,H,W)
        assert arr.shape[1] == H and arr.shape[2] == W, f"Spatial shape mismatch: {arr.shape} vs {(H,W)}"
        np.save(target_path, arr)
        print(f"[MERGE] Saved {day}.npy  shape={arr.shape}")


# ========== 4. 主程序：下载 -> 合并 ==========
if __name__ == "__main__":
    import argparse
    import pandas as pd
    import re

    parser = argparse.ArgumentParser("Download ERA5 & GLOFAS, then merge to NPY")
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2024)
    # 取消并发参数，强制串行下载
    parser.add_argument("--hour", type=str, default="14:00", help="ERA5 hour (e.g. 00:00, 12:00, 14:00)")
    args = parser.parse_args()

    years = list(range(args.start_year, args.end_year + 1))
    months = list(range(1, 13))

    # 串行下载 GLOFAS（按年）
    print("🚀 Downloading GLOFAS (serial)...")
    for y in years:
        try:
            download_glofas(y)
        except Exception as e:
            print(f"[ERROR] GLOFAS year {y} failed: {e}")

    # 再串行下载 ERA5（按月）
    print("🚀 Downloading ERA5 (serial, hour=" + args.hour + ") ...")
    for y in years:
        for m in months:
            try:
                download_era5(y, m, args.hour)
            except Exception as e:
                print(f"[ERROR] ERA5 {y}-{m:02d} failed: {e}")

    def _extract_nc_path_from_error(err_msg: str) -> str:
        # 形如: [Errno -101] NetCDF: HDF error: '/mnt/d/store/TTF/ERA5/2015-08.nc'
        m = re.search(r"'([^']+\.nc)'", err_msg)
        return m.group(1) if m else ""

    def _redownload_era5_nc(nc_path: str, hour: str):
        # 根据文件名解析年月
        base = os.path.basename(nc_path)
        m = re.match(r"(\d{4})-(\d{2})\.nc", base)
        if not m:
            print(f"[RE-DL] 未能从文件名解析年月: {nc_path}")
            return False
        year = int(m.group(1))
        month = int(m.group(2))
        try:
            if os.path.exists(nc_path):
                os.remove(nc_path)
                print(f"[RE-DL] 删除损坏文件: {nc_path}")
            download_era5(year, month, hour)
            print(f"[RE-DL] 重新下载完成: {nc_path}")
            return True
        except Exception as e:
            print(f"[RE-DL] 重新下载失败: {e}")
            return False

    print("🔧 Merging NetCDF -> NPY (multi-channel images)...")
    max_merge_retries = 100
    for attempt in range(1, max_merge_retries + 1):
        try:
            align_and_save_npy(years)
            print("✅ Merge complete. NPY images at:", IMAGES_DIR)
            break
        except Exception as e:
            err_msg = str(e)
            print(f"[ERROR] Merge attempt {attempt} failed: {err_msg}")
            if "NetCDF: HDF error" in err_msg:
                nc_path = _extract_nc_path_from_error(err_msg)
                if nc_path:
                    print(f"[INFO] 检测到 HDF 损坏，尝试单线程重新下载 {nc_path} ...")
                    ok = _redownload_era5_nc(nc_path, args.hour)
                    if not ok:
                        print("[FATAL] 无法重新下载损坏文件，终止。")
                        break
                    else:
                        print("[INFO] 重新下载完成，准备再次合并。")
                        continue  # 重试合并
                else:
                    print("[WARN] 未能提取损坏的 .nc 路径，无法自动恢复。")
                    break
            else:
                print("[WARN] 非 HDF 错误，不进行自动重试。")
                break
    else:
        print("[ERROR] 达到最大重试次数仍失败。")