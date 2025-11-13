"""Dataset module for streamflow prediction."""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
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
        
        # Configuration parameters
        self.window_size = config.get('data', {}).get('window_size', 5)
        self.upscale_factor = config.get('data', {}).get('upscale_factor', 2)
        self.region = config.get('data', {}).get('region', [100, 110, 25, 35])
        self.resolution = config.get('data', {}).get('resolution', 0.05)
        
        # Load image filenames (sorted by date)
        self.image_files = sorted([
            f for f in self.image_dir.iterdir() if f.suffix == '.npy'
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
        self.runoff_mean = None
        self.runoff_std = None
        
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
    
    def _compute_statistics(self) -> None:
        """Compute mean and std for normalization."""
        all_images = []
        all_runoffs = []
        
        for img_file in self.image_files:
            img = np.load(img_file)
            all_images.append(img)
        
        for station_data in self.stations.values():
            if 'runoff' in station_data.columns:
                all_runoffs.extend(station_data['runoff'].values)
        
        # Compute statistics
        all_images_array = np.concatenate(
            [im.flatten() for im in all_images]
        )
        all_runoffs_array = np.array(all_runoffs)
        
        self.image_mean = np.mean(all_images_array)
        self.image_std = np.std(all_images_array)
        self.runoff_mean = np.mean(all_runoffs_array)
        self.runoff_std = np.std(all_runoffs_array)
        
        # Avoid division by zero
        if self.image_std == 0:
            self.image_std = 1.0
        if self.runoff_std == 0:
            self.runoff_std = 1.0
    
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
        """Convert lon/lat to pixel position."""
        lon_min, lon_max, lat_min, lat_max = self.region
        
        px = int((lon - lon_min) / self.resolution)
        py = int((lat_min - lat) / self.resolution)  # Note: lat inverted
        
        # Clamp to image bounds
        px = max(0, min(px, 127))
        py = max(0, min(py, 127))
        
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
            input_images.append(img)
        
        # Stack: (T, C, H, W)
        input_images = np.stack(input_images, axis=0)
        
        # Load output image (first channel)
        output_img = np.load(self.image_files[sample['output_index']])
        output_flow = output_img[0:1, :, :]  # (1, H, W)
        
        # Normalize
        if self.normalize:
            input_images = (input_images - self.image_mean) / self.image_std
            output_flow = (output_flow - self.image_mean) / self.image_std
        
        # Get station data
        output_date = sample['output_date']
        station_positions, station_runoffs = self._get_station_target(
            output_date
        )
        
        if self.normalize and len(station_runoffs) > 0:
            station_runoffs = (
                (station_runoffs - self.runoff_mean) / self.runoff_std
            )
        
        return {
            'images': torch.from_numpy(input_images).float(),
            'output_image': torch.from_numpy(output_flow).float(),
            'stations': torch.from_numpy(station_runoffs).float(),
            'station_positions': torch.from_numpy(station_positions).long(),
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
