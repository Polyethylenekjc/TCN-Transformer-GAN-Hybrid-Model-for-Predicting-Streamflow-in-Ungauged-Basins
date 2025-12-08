import os
import re
from typing import Dict, List, Tuple
import pandas as pd

try:
    from rich.progress import track
except ImportError:
    def track(iter, description=""):
        return iter

HEADER_PATTERN = re.compile(r"^#\s*([^:]+):\s*(.*)$")


def _parse_header(lines: List[str]) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    for ln in lines:
        if not ln.startswith('#'):
            continue
        m = HEADER_PATTERN.match(ln.rstrip('\n'))
        if not m:
            continue
        key = m.group(1).strip()
        val = m.group(2).strip()
        meta[key] = val
    return meta


def _find_data_start(lines: List[str]) -> int:
    # data table starts after a line 'YYYY-MM-DD;hh:mm; Value' or '# DATA' followed by header
    for i, ln in enumerate(lines):
        if ln.strip().lower().startswith('yyyy-mm-dd;hh:mm;'):
            return i
        if ln.strip().upper().startswith('# DATA'):
            # next lines contain header; try to find the header row
            for j in range(i + 1, min(i + 10, len(lines))):
                if lines[j].strip().lower().startswith('yyyy-mm-dd;hh:mm;'):
                    return j
    # fallback: search for the first non-comment and containing ';'
    for i, ln in enumerate(lines):
        if not ln.startswith('#') and ';' in ln:
            return i
    return -1


def parse_grdc_file(path: str) -> Tuple[Dict[str, str], pd.DataFrame]:
    """Parse a GRDC station text file to metadata and daily dataframe.

    Returns:
        meta: dict of header fields (strings)
        df: DataFrame with columns [timestamp (YYYYMMDD), runoff]
    """
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    meta = _parse_header(lines)

    start_idx = _find_data_start(lines)
    if start_idx < 0:
        raise RuntimeError(f"No data table found in {path}")

    records = []
    for ln in lines[start_idx + 1:]:
        if not ln.strip() or ln.startswith('#'):
            continue
        parts = [p.strip() for p in ln.strip().split(';')]
        if len(parts) < 3:
            continue
        date_str, _time_str, val_str = parts[0], parts[1], parts[2]
        # Normalize date to YYYYMMDD
        try:
            date_norm = pd.to_datetime(date_str, errors='coerce').strftime('%Y%m%d')
        except Exception:
            date_norm = None
        if not date_norm or date_norm == 'NaT':
            continue
        try:
            val = float(val_str.replace(',', '.'))
        except Exception:
            continue
        if val <= -999:  # missing indicator per file header
            val = float('nan')
        records.append((date_norm, val))

    df = pd.DataFrame(records, columns=['timestamp', 'runoff'])
    # drop duplicates keeping the first occurrence
    if not df.empty:
        df = df.drop_duplicates(subset=['timestamp'], keep='first').reset_index(drop=True)
    return meta, df


def load_grdc_directory(dir_path: str) -> List[Dict]:
    """Load all GRDC station .txt files under a directory.

    Returns a list of dicts with keys: id, name, lon, lat, area, meta, df
    where df has columns [timestamp, runoff].
    """
    stations: List[Dict] = []
    files = [fn for fn in sorted(os.listdir(dir_path)) if fn.lower().endswith('.txt')]
    
    for fn in track(files, description="Parsing GRDC files...", total=len(files)):
        full = os.path.join(dir_path, fn)
        try:
            meta, df = parse_grdc_file(full)
        except Exception as e:
            print(f"[GRDC] Parse failed {full}: {e}")
            continue

        def _get_float(meta: Dict[str, str], key_candidates: List[str]) -> float:
            for k in key_candidates:
                if k in meta:
                    try:
                        return float(str(meta[k]).replace(',', '.'))
                    except Exception:
                        pass
            return float('nan')

        def _get_str(meta: Dict[str, str], key_candidates: List[str]) -> str:
            for k in key_candidates:
                if k in meta:
                    return str(meta[k]).strip()
            return ''

        lon = _get_float(meta, ['Longitude (DD)', 'Longitude'])
        lat = _get_float(meta, ['Latitude (DD)', 'Latitude'])
        area = _get_float(meta, ['Catchment area (km²)', 'Catchment area (km?', 'Catchment area'])
        station_id = _get_str(meta, ['GRDC-No.', 'GRDC-No'])
        name = _get_str(meta, ['Station'])

        stations.append({
            'id': station_id,
            'name': name,
            'lon': lon,
            'lat': lat,
            'area': area,
            'meta': meta,
            'df': df,
        })
    return stations
