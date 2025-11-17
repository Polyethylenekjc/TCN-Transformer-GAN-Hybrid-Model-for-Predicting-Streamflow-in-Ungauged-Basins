import os
import sys
import glob
import json
import argparse

try:
    import xarray as xr
except Exception as e:
    print("[FATAL] xarray import failed:", e)
    sys.exit(1)

DEFAULT_ERA5_DIR = "/mnt/d/store/TTF/ERA5"
DEFAULT_GLOFAS_DIR = "/mnt/d/store/TTF/GLOFAS"

def summarize_nc(path: str, max_vars: int = 8):
    print(f"\n=== Inspect: {path} ===")
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        return
    ds = xr.open_dataset(path, decode_times=False)
    info = {
        'file': path,
        'dims': {k: int(v) for k, v in ds.dims.items()},
        'coords': list(ds.coords),
        'data_vars': list(ds.data_vars),
        'time_like': [k for k in list(ds.coords) + list(ds.variables) if 'time' in k.lower()],
    }
    # time-like vars details
    for k in info['time_like']:
        v = ds[k]
        info[k] = {
            'dims': list(v.dims),
            'attrs': {kk: str(v.attrs[kk]) for kk in v.attrs},
            'dtype': str(v.dtype),
        }
    # show examples of data vars
    examples = {}
    for k in info['data_vars'][:max_vars]:
        v = ds[k]
        examples[k] = {
            'dims': list(v.dims),
            'shape': list(v.shape),
            'attrs_keys': list(v.attrs.keys()),
        }
    info['examples'] = examples
    print(json.dumps(info, ensure_ascii=False, indent=2))
    ds.close()


def main():
    ap = argparse.ArgumentParser("Inspect NetCDF (ERA5/GLOFAS) metadata")
    ap.add_argument('--file', type=str, default='', help='Specific NetCDF file to inspect')
    ap.add_argument('--kind', type=str, choices=['era5', 'glofas', 'auto'], default='auto', help='Select which dataset to scan when --file not set')
    ap.add_argument('--era5-dir', type=str, default=DEFAULT_ERA5_DIR)
    ap.add_argument('--glofas-dir', type=str, default=DEFAULT_GLOFAS_DIR)
    args = ap.parse_args()

    if args.file:
        summarize_nc(args.file)
        return

    if args.kind in ('era5', 'auto'):
        era5_files = sorted(glob.glob(os.path.join(args.era5_dir, '*.nc')))
        print(f"ERA5 files found: {len(era5_files)} under {args.era5_dir}")
        if era5_files:
            summarize_nc(era5_files[0])
    if args.kind in ('glofas', 'auto'):
        year_dirs = [d for d in glob.glob(os.path.join(args.glofas_dir, '*')) if os.path.isdir(d)]
        print(f"GLOFAS year dirs: {len(year_dirs)} under {args.glofas_dir}")
        if year_dirs:
            ydir = sorted(year_dirs)[0]
            f = os.path.join(ydir, os.path.basename(ydir) + '.nc')
            if os.path.exists(f):
                summarize_nc(f)
            else:
                print(f"[WARN] no .nc found in {ydir}")

if __name__ == '__main__':
    main()
