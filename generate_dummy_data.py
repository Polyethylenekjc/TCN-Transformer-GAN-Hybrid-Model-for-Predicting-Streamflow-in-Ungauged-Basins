"""Generate dummy data for quick system validation."""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime, timedelta
import yaml


def generate_dummy_data(
    days: int = 30,
    stations: int = 5,
    channels: int = 10,
    height: int = 128,
    width: int = 128,
    output_dir: str = './data'
):
    """
    Generate dummy data for testing.
    
    Args:
        days: Number of days to generate
        stations: Number of stations
        channels: Number of image channels
        height: Image height
        width: Image width
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
            'upscale_factor': 2
        },
        'model': {
            'input_channels': channels,
            'hidden_dim': 128,
            'num_layers': 2,
            'backbone': 'CNN',
            'temporal_module': 'LSTM',  # Can be changed to 'Transformer'
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
            'log_interval': 10
        }
    }
    
    # Save configuration
    config_path = output_path / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Config saved to {config_path}")
    
    # Determine image size from config
    img_size = config['data'].get('image_size', [128, 128])
    if isinstance(img_size, int):
        height = int(img_size)
        width = int(img_size)
    else:
        height = int(img_size[0])
        width = int(img_size[1])

    # Generate images
    print(f"Generating {days} days of images...")
    start_date = datetime(2020, 1, 1)
    
    for day_idx in range(days):
        date = start_date + timedelta(days=day_idx)
        date_str = date.strftime('%Y%m%d')
        
        # Generate random image (C, H, W)
        image = np.random.randn(channels, height, width).astype(np.float32)
        # Scale to reasonable range (0-500 for precipitation/runoff data)
        image = np.abs(image) * 100 + 50
        
        image_file = image_dir / f'{date_str}.npy'
        np.save(image_file, image)
        
        if (day_idx + 1) % 10 == 0:
            print(f"  Generated {day_idx + 1} images")
    
    # Generate station data
    print(f"Generating {stations} stations...")
    lon_min, lon_max, lat_min, lat_max = config['data']['region']
    
    for station_idx in range(stations):
        station_name = f'station_{station_idx:02d}'
        
        # Random location within region
        lons = np.random.uniform(lon_min, lon_max, days)
        lats = np.random.uniform(lat_min, lat_max, days)
        
        # Random runoff values (0-500)
        runoffs = np.random.uniform(0, 500, days)
        
        # Create dataframe
        dates = [
            (start_date + timedelta(days=d)).strftime('%Y%m%d')
            for d in range(days)
        ]
        
        df = pd.DataFrame({
            'timestamp': dates,
            'lon': lons,
            'lat': lats,
            'runoff': runoffs
        })
        
        # Save station CSV
        station_file = station_dir / f'{station_name}.csv'
        df.to_csv(station_file, index=False)
        
        print(f"  Generated {station_name}")
    
    print(f"\n✓ Data generation complete!")
    print(f"  Images: {image_dir}")
    print(f"  Stations: {station_dir}")
    print(f"  Config: {config_path}")
    print(f"  Total samples for training: {days - 5} (window_size=5)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate dummy data for streamflow prediction'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Number of days to generate'
    )
    parser.add_argument(
        '--stations',
        type=int,
        default=5,
        help='Number of stations'
    )
    parser.add_argument(
        '--channels',
        type=int,
        default=10,
        help='Number of image channels'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=128,
        help='Image height'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=128,
        help='Image width'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    generate_dummy_data(
        days=args.days,
        stations=args.stations,
        channels=args.channels,
        height=args.height,
        width=args.width,
        output_dir=args.output_dir
    )
