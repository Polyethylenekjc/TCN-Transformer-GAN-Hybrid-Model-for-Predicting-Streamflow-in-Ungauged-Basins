"""Generate improved dummy data with spatial/temporal correlation and site-image coupling."""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime, timedelta
import yaml
from scipy.ndimage import gaussian_filter


def generate_gaussian_blob_field(height, width, blob_centers, blob_amps, blob_sigmas):
    """
    Generate a smooth 2D field by summing Gaussian blobs.
    
    Args:
        height, width: Field dimensions
        blob_centers: List of (y, x) center coordinates
        blob_amps: List of amplitudes for each blob
        blob_sigmas: List of sigma (width) for each blob
        
    Returns:
        (height, width) array
    """
    field = np.zeros((height, width), dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    
    for (cy, cx), amp, sigma in zip(blob_centers, blob_amps, blob_sigmas):
        dist_sq = (yy - cy) ** 2 + (xx - cx) ** 2
        blob = amp * np.exp(-dist_sq / (2 * sigma ** 2))
        field += blob
    
    return field


def generate_dummy_data(
    days: int = 30,
    stations: int = 5,
    channels: int = 10,
    height: int = 128,
    width: int = 128,
    num_blobs: int = 4,
    blob_sigma_range: tuple = (5, 20),
    blob_speed_range: tuple = (0.5, 3),
    runoff_scale: float = 1.0,
    runoff_noise_std: float = 10.0,
    lag_days: int = 0,
    output_dir: str = './data'
):
    """
    Generate realistic dummy data with spatial/temporal structure.
    
    Args:
        days: Number of days to generate
        stations: Number of stations
        channels: Number of image channels
        height, width: Image spatial dimensions
        num_blobs: Number of Gaussian blobs per frame
        blob_sigma_range: (min, max) sigma for blobs
        blob_speed_range: (min, max) pixels/day movement
        runoff_scale: Scale factor for runoff derivation from images
        runoff_noise_std: Std dev of noise added to runoff
        lag_days: Delay between flow image and runoff observation (for testing)
        output_dir: Output directory
    """
    
    output_path = Path(output_dir)
    image_dir = output_path / 'images'
    station_dir = output_path / 'stations'
    
    # Create directories
    image_dir.mkdir(parents=True, exist_ok=True)
    station_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration
    config = {
        'data': {
            'image_dir': str(image_dir),
            'station_dir': str(station_dir),
            'output_dir': str(output_path / 'output'),
            'image_size': [height, width],
            'resolution': 0.05,
            'region': [100, 110, 25, 35],
            'window_size': 5,
            'upscale_factor': 2,
            'normalize': True
        },
        'model': {
            'input_channels': channels,
            'hidden_dim': 128,
            'num_layers': 2,
            'backbone': 'CNN',
            'temporal_module': 'LSTM',
            'use_gan': False
        },
        'train': {
            'batch_size': 4,
            'lr': 1e-4,
            'num_epochs': 100,
            'device': 'cuda'
        },
        'loss': {
            'metric': 'MSE',
            'weights': {
                'image': 1.0,
                'station': 0.5
            }
        },
        'stations': {
            'kernel_size': 3,
            'metric': 'MSE'
        },
        'system': {
            'mixed_precision': True,
            'grad_accum_steps': 2,
            'log_interval': 10,
            'use_rich': True,
            'debug': False
        }
    }
    
    # Save configuration
    config_path = output_path / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Config saved to {config_path}")
    
    # Random station locations (fixed across days)
    lon_min, lon_max, lat_min, lat_max = config['data']['region']
    station_lons = np.random.uniform(lon_min, lon_max, stations)
    station_lats = np.random.uniform(lat_min, lat_max, stations)
    
    # Convert lon/lat to pixel coordinates (using resolution in config)
    resolution = config['data']['resolution']
    station_pixels = []
    for lon, lat in zip(station_lons, station_lats):
        px = int((lon - lon_min) / resolution)
        py = int((lat_min - lat) / resolution)
        px = max(0, min(px, width - 1))
        py = max(0, min(py, height - 1))
        station_pixels.append((px, py))
    
    # Generate base flow field images with temporal/spatial structure
    print(f"Generating {days} days of images with temporal correlation...")
    start_date = datetime(2020, 1, 1)
    images_array = np.zeros((days, channels, height, width), dtype=np.float32)
    
    # Initialize blob states
    np.random.seed(42)  # for reproducibility of blob behavior
    blob_states = []
    for blob_id in range(num_blobs):
        y = np.random.uniform(height * 0.1, height * 0.9)
        x = np.random.uniform(width * 0.1, width * 0.9)
        speed = np.random.uniform(*blob_speed_range)
        direction = np.random.uniform(0, 2 * np.pi)
        sigma = np.random.uniform(*blob_sigma_range)
        amp = np.random.uniform(50, 150)
        blob_states.append({
            'x': x, 'y': y, 'vx': speed * np.cos(direction), 'vy': speed * np.sin(direction),
            'sigma': sigma, 'amp': amp
        })
    
    for day_idx in range(days):
        # Update blob positions (with wrapping)
        for blob in blob_states:
            blob['x'] = (blob['x'] + blob['vx']) % width
            blob['y'] = (blob['y'] + blob['vy']) % height
        
        # Generate base field from blobs
        blob_centers = [(blob['y'], blob['x']) for blob in blob_states]
        blob_amps = [blob['amp'] for blob in blob_states]
        blob_sigmas = [blob['sigma'] for blob in blob_states]
        
        base_field = generate_gaussian_blob_field(height, width, blob_centers, blob_amps, blob_sigmas)
        
        # Smooth the field
        base_field = gaussian_filter(base_field, sigma=2.0)
        
        # Create multiple channels with correlations
        # Channel 0: base precipitation field
        # Channel 1: smoothed variant
        # Channels 2+: linear combinations with noise
        for ch in range(channels):
            if ch == 0:
                images_array[day_idx, ch, :, :] = base_field.astype(np.float32)
            elif ch == 1:
                images_array[day_idx, ch, :, :] = gaussian_filter(base_field, sigma=3.0).astype(np.float32)
            else:
                # Linear combination of base field + scaled/shifted versions + noise
                weight = 0.5 + 0.5 * np.sin(2 * np.pi * ch / channels)
                combined = weight * base_field + (1 - weight) * gaussian_filter(base_field, sigma=5.0)
                noise = np.random.randn(height, width) * 5
                images_array[day_idx, ch, :, :] = (combined + noise).astype(np.float32)
        
        if (day_idx + 1) % max(1, days // 10) == 0:
            print(f"  Generated {day_idx + 1} images")
    
    # Clip and scale to reasonable range
    images_array = np.clip(images_array, 0, 500)
    
    # Save images
    print(f"Saving {days} images...")
    for day_idx in range(days):
        date = start_date + timedelta(days=day_idx)
        date_str = date.strftime('%Y%m%d')
        image_file = image_dir / f'{date_str}.npy'
        np.save(image_file, images_array[day_idx])
    
    # Generate station data coupled to image fields
    print(f"Generating {stations} stations coupled to flow fields...")
    
    for station_idx in range(stations):
        station_name = f'station_{station_idx:02d}'
        px, py = station_pixels[station_idx]
        
        dates = []
        runoffs = []
        
        for day_idx in range(days):
            date = start_date + timedelta(days=day_idx)
            dates.append(date.strftime('%Y%m%d'))
            
            # Get image for runoff (with lag)
            img_idx = max(0, min(days - 1, day_idx - lag_days))
            
            # Extract local neighborhood around station (kernel 5x5)
            kernel_half = 2
            y_start = max(0, py - kernel_half)
            y_end = min(height, py + kernel_half + 1)
            x_start = max(0, px - kernel_half)
            x_end = min(width, px + kernel_half + 1)
            
            # Average of channel 0 (main flow) in neighborhood
            local_avg = images_array[img_idx, 0, y_start:y_end, x_start:x_end].mean()
            
            # Convert to runoff: scale and add noise
            runoff = runoff_scale * local_avg + np.random.randn() * runoff_noise_std
            runoff = max(0, runoff)  # ensure non-negative
            runoffs.append(runoff)
        
        # Create dataframe with fixed location
        df = pd.DataFrame({
            'timestamp': dates,
            'lon': [station_lons[station_idx]] * days,
            'lat': [station_lats[station_idx]] * days,
            'runoff': runoffs
        })
        
        # Save station CSV
        station_file = station_dir / f'{station_name}.csv'
        df.to_csv(station_file, index=False)
        
        print(f"  Generated {station_name} (location: {station_lons[station_idx]:.2f}, {station_lats[station_idx]:.2f})")
    
    print(f"\n✓ Data generation complete!")
    print(f"  Images: {image_dir}")
    print(f"  Stations: {station_dir}")
    print(f"  Config: {config_path}")
    print(f"  Total samples for training: {days - 5} (window_size=5)")
    print(f"  Image stats:")
    print(f"    Min: {images_array.min():.2f}, Max: {images_array.max():.2f}, Mean: {images_array.mean():.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate realistic dummy data for streamflow prediction'
    )
    parser.add_argument('--days', type=int, default=30, help='Number of days to generate')
    parser.add_argument('--stations', type=int, default=5, help='Number of stations')
    parser.add_argument('--channels', type=int, default=10, help='Number of image channels')
    parser.add_argument('--height', type=int, default=128, help='Image height')
    parser.add_argument('--width', type=int, default=128, help='Image width')
    parser.add_argument('--num-blobs', type=int, default=4, help='Number of Gaussian blobs per frame')
    parser.add_argument('--blob-sigma-min', type=float, default=5, help='Min blob sigma')
    parser.add_argument('--blob-sigma-max', type=float, default=20, help='Max blob sigma')
    parser.add_argument('--blob-speed-min', type=float, default=0.5, help='Min blob speed (pixels/day)')
    parser.add_argument('--blob-speed-max', type=float, default=3, help='Max blob speed (pixels/day)')
    parser.add_argument('--runoff-scale', type=float, default=1.0, help='Scale factor for runoff from flow')
    parser.add_argument('--runoff-noise', type=float, default=10.0, help='Std dev of runoff noise')
    parser.add_argument('--lag-days', type=int, default=0, help='Lag between flow image and runoff (days)')
    parser.add_argument('--output-dir', type=str, default='./data', help='Output directory')
    
    args = parser.parse_args()
    
    generate_dummy_data(
        days=args.days,
        stations=args.stations,
        channels=args.channels,
        height=args.height,
        width=args.width,
        num_blobs=args.num_blobs,
        blob_sigma_range=(args.blob_sigma_min, args.blob_sigma_max),
        blob_speed_range=(args.blob_speed_min, args.blob_speed_max),
        runoff_scale=args.runoff_scale,
        runoff_noise_std=args.runoff_noise,
        lag_days=args.lag_days,
        output_dir=args.output_dir
    )
