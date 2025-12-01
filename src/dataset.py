"""Dataset module for streamflow prediction."""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import os


class StreamflowDataset(Dataset):
    """Dataset for streamflow prediction with sliding window."""
    
    def __init__(
        self,
        image_dir: str,
        station_dir: str,
        config: Dict,
        normalize: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            image_dir: Directory containing image .npy files
            station_dir: Directory containing station .csv files
            config: Configuration dictionary
            normalize: Whether to normalize data
        """
        self.image_dir = Path(image_dir)
        self.station_dir = Path(station_dir)
        self.config = config
        self.normalize = normalize
        self.normalization_type = config.get('data', {}).get('normalization_type', 'z_score')
        
        # Configuration parameters
        self.window_size = config.get('data', {}).get('window_size', 5)
        self.upscale_factor = config.get('data', {}).get('upscale_factor', 2)
        self.region = config.get('data', {}).get('region', [100, 110, 25, 35])
        self.resolution = config.get('data', {}).get('resolution', 0.05)
        self.flow_threshold = config.get('data', {}).get('flow_threshold', 0.1)
        # Desired input image size, can be single int or [H, W]
        img_size_cfg = config.get('data', {}).get('image_size', [128, 128])
        if isinstance(img_size_cfg, int):
            self.input_height = int(img_size_cfg)
            self.input_width = int(img_size_cfg)
        else:
            self.input_height = int(img_size_cfg[0])
            self.input_width = int(img_size_cfg[1])
        
        # Load image filenames (sorted by date), ignore mask files like *_mask.npy
        self.image_files = sorted([
            f for f in self.image_dir.iterdir()
            if f.suffix == '.npy' and not f.name.endswith('_mask.npy')
        ])
        
        if len(self.image_files) == 0:
            raise RuntimeError(f"No .npy files found in {image_dir}")
        
        # Load station data
        self.stations = self._load_stations()
        
        # Create sliding window samples
        self.samples = self._create_samples()
        
        # Statistics for normalization
        self.image_mean = None
        self.image_std = None
        self.image_min = None
        self.image_max = None
        self.runoff_mean = None
        self.runoff_std = None
        self.runoff_min = None
        self.runoff_max = None
        
        if self.normalize:
            self._compute_statistics()
    
    def _load_stations(self) -> Dict[str, pd.DataFrame]:
        """Load station data from CSV files."""
        stations = {}
        
        for csv_file in self.station_dir.glob('*.csv'):
            station_name = csv_file.stem
            df = pd.read_csv(csv_file)
            stations[station_name] = df
        
        return stations
    
    def _create_samples(self) -> List[Dict]:
        """Create sliding window samples."""
        samples = []
        
        # Get dates from image filenames
        dates = [f.stem for f in self.image_files]
        
        # Create samples with sliding window
        for i in range(len(dates) - self.window_size):
            input_dates = dates[i:i + self.window_size]
            output_date = dates[i + self.window_size]
            
            samples.append({
                'input_dates': input_dates,
                'output_date': output_date,
                'input_indices': list(range(i, i + self.window_size)),
                'output_index': i + self.window_size
            })
        
        return samples
    
    def _mask_path_for(self, npy_path: Path) -> Path:
        """Return expected mask path for a given npy file."""
        return npy_path.with_name(npy_path.stem + '_mask.npy')

    def _compute_statistics(self) -> None:
        """Compute mean and std per channel for normalization in a memory-efficient way."""
        
        # Initialize with first image to get channels
        if not self.image_files:
            return
            
        first_img = np.load(self.image_files[0])
        if first_img.ndim == 2:
            self.num_channels = 1
        else:
            self.num_channels = first_img.shape[0]
            
        C = self.num_channels
        
        total_pixels = 0
        channel_sum = np.zeros(C, dtype=np.float64)
        channel_sumsq = np.zeros(C, dtype=np.float64)
        channel_min = np.full(C, np.inf, dtype=np.float64)
        channel_max = np.full(C, -np.inf, dtype=np.float64)

        # Accumulators for station runoffs
        total_runoff_count = 0
        total_runoff_sum = 0.0
        total_runoff_sumsq = 0.0
        runoff_min = float('inf')
        runoff_max = float('-inf')

        for img_file in self.image_files:
            img = np.load(img_file)
            img_t = torch.from_numpy(img).float()
            if img_t.dim() == 2:
                img_t = img_t.unsqueeze(0)

            if img_t.shape[1] != self.input_height or img_t.shape[2] != self.input_width:
                img_t = F.interpolate(
                    img_t.unsqueeze(0),
                    size=(self.input_height, self.input_width),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)

            mask_path = self._mask_path_for(Path(img_file))
            if mask_path.exists():
                mask_np = np.load(mask_path)
                mask_t = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0)
                if mask_t.shape[1] != self.input_height or mask_t.shape[2] != self.input_width:
                    mask_t = F.interpolate(
                        mask_t.unsqueeze(0),
                        size=(self.input_height, self.input_width),
                        mode='nearest'
                    ).squeeze(0)
                mask_bin = (mask_t.squeeze(0) > 0.5)
                # Select masked pixels for each channel: (C, N_masked)
                vals = img_t[:, mask_bin].cpu().numpy()
            else:
                # Flatten spatial dims: (C, H*W)
                vals = img_t.reshape(C, -1).cpu().numpy()

            if vals.shape[1] > 0:
                # Replace NaNs and Infs
                if not np.all(np.isfinite(vals)):
                    print(f"Warning: NaNs or Infs found in image file {img_file}. Replacing with 0.")
                    vals = np.nan_to_num(vals)
                
                total_pixels += vals.shape[1]
                channel_sum += vals.sum(axis=1)
                channel_sumsq += (vals ** 2).sum(axis=1)
                
                # Update min/max
                current_min = vals.min(axis=1)
                current_max = vals.max(axis=1)
                channel_min = np.minimum(channel_min, current_min)
                channel_max = np.maximum(channel_max, current_max)

        for station_name, station_data in self.stations.items():
            if 'runoff' in station_data.columns:
                arr = station_data['runoff'].values.astype(np.float64)
                
                # Replace NaNs and Infs
                if not np.all(np.isfinite(arr)):
                    print(f"Warning: NaNs or Infs found in station {station_name}. Replacing with 0.")
                    arr = np.nan_to_num(arr)

                total_runoff_count += arr.size
                total_runoff_sum += float(arr.sum())
                total_runoff_sumsq += float((arr ** 2).sum())
                
                # Update runoff min/max
                if arr.size > 0:
                    current_r_min = float(arr.min())
                    current_r_max = float(arr.max())
                    if current_r_min < runoff_min:
                        runoff_min = current_r_min
                    if current_r_max > runoff_max:
                        runoff_max = current_r_max

        if total_pixels == 0:
            raise RuntimeError("No valid pixels found to compute statistics.")

        self.image_mean = channel_sum / total_pixels
        img_var = channel_sumsq / total_pixels - (self.image_mean ** 2)
        self.image_std = np.sqrt(np.maximum(img_var, 0.0))
        self.image_std[self.image_std == 0] = 1.0
            
        self.image_min = channel_min
        self.image_max = channel_max
        # Avoid division by zero for constant channels
        mask_const = (self.image_max == self.image_min)
        self.image_max[mask_const] += 1.0

        if total_runoff_count > 0:
            self.runoff_mean = total_runoff_sum / total_runoff_count
            run_var = total_runoff_sumsq / total_runoff_count - (self.runoff_mean ** 2)
            self.runoff_std = float(np.sqrt(max(run_var, 0.0)))
            if self.runoff_std == 0:
                self.runoff_std = 1.0
            
            self.runoff_min = runoff_min
            self.runoff_max = runoff_max
            if self.runoff_max == self.runoff_min:
                self.runoff_max = self.runoff_min + 1.0
        else:
            self.runoff_mean = 0.0
            self.runoff_std = 1.0
            self.runoff_min = 0.0
            self.runoff_max = 1.0
    
    def _get_station_target(self, date: str) -> Tuple[np.ndarray, float]:
        """
        Get station target for a specific date.
        
        Returns:
            (pixel_positions, runoff_value) for all stations on that date
        """
        date_runoffs = []
        date_positions = []
        
        for station_name, df in self.stations.items():
            # Find row matching date
            matching = df[df['timestamp'].astype(str) == date]
            
            if len(matching) == 0:
                continue
            
            row = matching.iloc[0]
            lon, lat = row['lon'], row['lat']
            runoff = row['runoff']
            
            # Convert lon/lat to pixel position
            px, py = self._lonlat_to_pixel(lon, lat)
            
            date_runoffs.append(runoff)
            date_positions.append((px, py))
        
        if len(date_runoffs) == 0:
            # No station data for this date
            return np.array([], dtype=np.int32).reshape(-1, 2), np.array([])
        
        return np.array(date_positions, dtype=np.int32), np.array(date_runoffs)
    
    def _lonlat_to_pixel(self, lon: float, lat: float) -> Tuple[int, int]:
        """Convert lon/lat to pixel position using proportional mapping.
        This keeps mapping correct even when image_size != (region span / resolution).
        """
        lon_min, lon_max, lat_min, lat_max = self.region

        # Avoid division by zero
        lon_span = max(1e-9, (lon_max - lon_min))
        lat_span = max(1e-9, (lat_max - lat_min))

        # Proportional mapping: west->east maps to 0..W-1, north->south maps to 0..H-1
        x_ratio = (lon - lon_min) / lon_span
        y_ratio = (lat_max - lat) / lat_span

        px = int(round(x_ratio * (self.input_width - 1)))
        py = int(round(y_ratio * (self.input_height - 1)))

        # Clamp to image bounds
        px = max(0, min(px, self.input_width - 1))
        py = max(0, min(py, self.input_height - 1))

        return px, py
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
                - 'images': (T, C, H, W) input image sequence
                - 'output_image': (1, H_out, W_out) target flow map
                - 'stations': (num_stations,) station runoff values
                - 'station_positions': (num_stations, 2) pixel positions
        """
        sample = self.samples[idx]
        
        # Load input images
        input_images = []
        for idx_img in sample['input_indices']:
            img = np.load(self.image_files[idx_img])
            img_t = torch.from_numpy(img).float()
            if img_t.dim() == 2:
                img_t = img_t.unsqueeze(0)

            # Resize to configured input size if needed
            if img_t.shape[1] != self.input_height or img_t.shape[2] != self.input_width:
                img_t = F.interpolate(
                    img_t.unsqueeze(0),
                    size=(self.input_height, self.input_width),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)

            # Apply mask if present
            mask_path = self._mask_path_for(Path(self.image_files[idx_img]))
            if mask_path.exists():
                mask_np = np.load(mask_path)
                mask_t = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0)
                if mask_t.shape[1] != self.input_height or mask_t.shape[2] != self.input_width:
                    mask_t = F.interpolate(
                        mask_t.unsqueeze(0),
                        size=(self.input_height, self.input_width),
                        mode='nearest'
                    ).squeeze(0)
                img_t = img_t * mask_t  # broadcast (C,H,W) * (1,H,W)
            
            # Apply flow threshold to first channel (runoff/flow)
            img_np = img_t.numpy()
            
            # Replace NaNs/Infs in input image
            if not np.all(np.isfinite(img_np)):
                img_np = np.nan_to_num(img_np, nan=0.0, posinf=0.0, neginf=0.0)
                
            if img_np.shape[0] > 0:  # If there's at least one channel
                img_np[0][np.abs(img_np[0]) < self.flow_threshold] = 0.0

            input_images.append(img_np)
        
        # Stack: (T, C, H, W)
        input_images = np.stack(input_images, axis=0)
        
        # Load output image (first channel)
        output_img = np.load(self.image_files[sample['output_index']])
        out_t = torch.from_numpy(output_img).float()
        if out_t.dim() == 2:
            out_t = out_t.unsqueeze(0)

        # Use first channel as flow and resize to target input size if needed
        out_t = out_t[0:1, :, :]
        if out_t.shape[1] != self.input_height or out_t.shape[2] != self.input_width:
            out_t = F.interpolate(
                out_t.unsqueeze(0),
                size=(self.input_height, self.input_width),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        # Apply mask to output if present
        out_mask_path = self._mask_path_for(Path(self.image_files[sample['output_index']]))
        if out_mask_path.exists():
            mask_np = np.load(out_mask_path)
            mask_t = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0)
            if mask_t.shape[1] != self.input_height or mask_t.shape[2] != self.input_width:
                mask_t = F.interpolate(
                    mask_t.unsqueeze(0),
                    size=(self.input_height, self.input_width),
                    mode='nearest'
                ).squeeze(0)
            out_t = out_t * mask_t

        output_flow = out_t.numpy()
        
        # Apply flow threshold to output
        output_flow[np.abs(output_flow) < self.flow_threshold] = 0.0
        
        # Normalize
        if self.normalize:
            # Reshape stats for broadcasting: (1, C, 1, 1)
            # input_images is (T, C, H, W)
            img_min = self.image_min.reshape(1, -1, 1, 1)
            img_max = self.image_max.reshape(1, -1, 1, 1)
            img_mean = self.image_mean.reshape(1, -1, 1, 1)
            img_std = self.image_std.reshape(1, -1, 1, 1)

            if self.normalization_type == 'minmax':
                # MinMax normalization to [0, 1]
                input_images = (input_images - img_min) / (img_max - img_min)
                # output_flow is channel 0
                output_flow = (output_flow - self.image_min[0]) / (self.image_max[0] - self.image_min[0])
            else:
                # Z-score normalization (default)
                input_images = (input_images - img_mean) / img_std
                output_flow = (output_flow - self.image_mean[0]) / self.image_std[0]
        
        # Get station data
        output_date = sample['output_date']
        station_positions, station_runoffs = self._get_station_target(
            output_date
        )
        
        if self.normalize and len(station_runoffs) > 0:
            if self.normalization_type == 'minmax':
                station_runoffs = (station_runoffs - self.runoff_min) / (self.runoff_max - self.runoff_min)
            else:
                station_runoffs = (station_runoffs - self.runoff_mean) / self.runoff_std
        
        return {
            'images': torch.from_numpy(np.array(input_images)).float(),
            'output_image': torch.from_numpy(output_flow).float(),
            'stations': torch.from_numpy(station_runoffs).float() if len(station_runoffs) > 0 else torch.from_numpy(np.array([])).float(),
            'station_positions': torch.from_numpy(station_positions).long() if station_positions.size else torch.from_numpy(np.array([], dtype=np.int64)).long(),
            'date': output_date
        }


class DataLoader:
    """Custom data loader with batch collation."""
    
    def __init__(
        self,
        dataset: StreamflowDataset,
        batch_size: int = 4,
        shuffle: bool = True,
        num_workers: int = 0
    ):
        """Initialize data loader."""
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        
        self.indices = list(range(len(dataset)))
        if shuffle:
            np.random.shuffle(self.indices)
    
    def __iter__(self):
        """Iterate through batches."""
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        for batch_start in range(0, len(self.indices), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(self.indices))
            batch_indices = self.indices[batch_start:batch_end]
            
            # Collect samples
            batch_images = []
            batch_outputs = []
            batch_stations = []
            batch_positions = []
            
            for idx in batch_indices:
                sample = self.dataset[idx]
                batch_images.append(sample['images'])
                batch_outputs.append(sample['output_image'])
                batch_stations.append(sample['stations'])
                batch_positions.append(sample['station_positions'])
            
            # Stack and pad
            images = torch.stack(batch_images, dim=0)
            outputs = torch.stack(batch_outputs, dim=0)
            
            # Stations may have varying lengths, so we store as list
            yield {
                'images': images,
                'output_image': outputs,
                'stations': batch_stations,
                'station_positions': batch_positions
            }
    
    def __len__(self):
        """Get number of batches."""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
