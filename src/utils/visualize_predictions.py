"""
Visualization script for model predictions.

This script loads predicted and target images, randomly samples a few,
and creates comparison visualizations.

Usage:
    python -m src.utils.visualize_predictions --output-dir ./output --num-samples 6
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import random
from typing import List, Tuple


def load_image_pair(pred_file: Path, target_dir: Path = None) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Load a prediction and its corresponding target image.
    
    Args:
        pred_file: Path to prediction .npy file
        target_dir: Directory containing target images (optional, used as fallback)
        
    Returns:
        (prediction, target, date) tuple
    """
    # Extract date from filename (pred_YYYYMMDD.npy)
    date = pred_file.stem.replace('pred_', '')
    
    # Load prediction
    prediction = np.load(pred_file)
    
    # Try to load normalized target saved by evaluator first
    norm_target_file = pred_file.parent / f'target_{date}.npy'
    
    if norm_target_file.exists():
        target = np.load(norm_target_file)
    elif target_dir:
        # Fallback to raw target if normalized one not found
        target_file = target_dir / f'{date}.npy'
        if not target_file.exists():
            raise FileNotFoundError(f"Target file not found: {target_file}")
        target = np.load(target_file)
        # Use first channel if multi-channel
        if target.ndim == 3:
            target = target[0]
    else:
        raise FileNotFoundError(f"Could not find target for {date}")
    
    return prediction, target, date


def create_comparison_plot(
    samples: List[Tuple[np.ndarray, np.ndarray, str]],
    output_path: Path,
    figsize: Tuple[int, int] = (20, 15),
    denormalize: bool = False,
    norm_stats: dict = None
):
    """
    Create a comparison plot with predictions, targets, and errors.
    
    Args:
        samples: List of (prediction, target, date) tuples
        output_path: Path to save the figure
        figsize: Figure size
        denormalize: Whether to denormalize the data
        norm_stats: Dictionary containing normalization statistics (mean, std, min, max)
    """
    n_samples = len(samples)
    # Increase figure size per sample
    fig, axes = plt.subplots(n_samples, 4, figsize=(figsize[0], figsize[1] * n_samples / 2))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, (pred, target, date) in enumerate(samples):
        # Resize target to match prediction if needed
        if pred.shape != target.shape:
            from scipy.ndimage import zoom
            zoom_factors = (pred.shape[0] / target.shape[0], 
                          pred.shape[1] / target.shape[1])
            target = zoom(target, zoom_factors, order=1)
        
        # Denormalize if requested
        pred_orig = pred.copy()
        target_orig = target.copy()
        
        if denormalize and norm_stats:
            if 'min' in norm_stats and 'max' in norm_stats:
                # MinMax denormalization
                pred_denorm = pred * (norm_stats['max'] - norm_stats['min']) + norm_stats['min']
                target_denorm = target * (norm_stats['max'] - norm_stats['min']) + norm_stats['min']
            elif 'mean' in norm_stats and 'std' in norm_stats:
                # Z-score denormalization
                pred_denorm = pred * norm_stats['std'] + norm_stats['mean']
                target_denorm = target * norm_stats['std'] + norm_stats['mean']
            else:
                pred_denorm = pred
                target_denorm = target
        else:
            pred_denorm = pred
            target_denorm = target

        # Calculate error on denormalized data
        error = np.abs(pred_denorm - target_denorm)
        
        # Determine common color scale for pred and target
        vmin = min(pred_denorm.min(), target_denorm.min())
        vmax = max(pred_denorm.max(), target_denorm.max())
        
        # Plot prediction
        im1 = axes[i, 0].imshow(pred_denorm, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[i, 0].set_title(f'Prediction (Denorm)\n{date}', fontsize=12)
        axes[i, 0].axis('off')
        plt.colorbar(im1, ax=axes[i, 0], fraction=0.046)
        
        # Plot target
        im2 = axes[i, 1].imshow(target_denorm, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[i, 1].set_title(f'Target (Denorm)\n{date}', fontsize=12)
        axes[i, 1].axis('off')
        plt.colorbar(im2, ax=axes[i, 1], fraction=0.046)
        
        # Plot absolute error
        im3 = axes[i, 2].imshow(error, cmap='Reds')
        axes[i, 2].set_title(f'Absolute Error\n{date}', fontsize=12)
        axes[i, 2].axis('off')
        plt.colorbar(im3, ax=axes[i, 2], fraction=0.046)
        
        # Plot histogram comparison
        axes[i, 3].hist(pred_denorm.flatten(), bins=50, alpha=0.5, label='Prediction', density=True)
        axes[i, 3].hist(target_denorm.flatten(), bins=50, alpha=0.5, label='Target', density=True)
        axes[i, 3].set_title(f'Distribution (Denorm)\n{date}', fontsize=12)
        axes[i, 3].legend()
        axes[i, 3].set_xlabel('Value')
        axes[i, 3].set_ylabel('Density')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight') # Increased DPI
    print(f"Visualization saved to {output_path}")
    plt.close()


def create_statistics_plot(
    samples: List[Tuple[np.ndarray, np.ndarray, str]],
    output_path: Path
):
    """
    Create a scatter plot showing prediction vs target correlation.
    
    Args:
        samples: List of (prediction, target, date) tuples
        output_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    all_preds = []
    all_targets = []
    
    for pred, target, date in samples:
        # Resize target to match prediction if needed
        if pred.shape != target.shape:
            from scipy.ndimage import zoom
            zoom_factors = (pred.shape[0] / target.shape[0], 
                          pred.shape[1] / target.shape[1])
            target = zoom(target, zoom_factors, order=1)
        
        all_preds.extend(pred.flatten())
        all_targets.extend(target.flatten())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Scatter plot
    axes[0].scatter(all_targets, all_preds, alpha=0.1, s=1)
    axes[0].plot([all_targets.min(), all_targets.max()], 
                 [all_targets.min(), all_targets.max()], 
                 'r--', label='Perfect prediction')
    axes[0].set_xlabel('Target Value')
    axes[0].set_ylabel('Predicted Value')
    axes[0].set_title('Prediction vs Target')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Error distribution
    errors = all_preds - all_targets
    axes[1].hist(errors, bins=100, edgecolor='black', alpha=0.7)
    axes[1].axvline(0, color='r', linestyle='--', linewidth=2, label='Zero error')
    axes[1].set_xlabel('Prediction Error')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Error Distribution\nMean: {errors.mean():.4f}, Std: {errors.std():.4f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Statistics plot saved to {output_path}")
    plt.close()


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(
        description="Visualize model predictions vs targets"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output',
        help='Output directory containing predictions/'
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        default=None,
        help='Directory containing target images (if None, will try to infer from config)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=6,
        help='Number of random samples to visualize'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--denormalize',
        action='store_true',
        help='Denormalize predictions using stats from dataset'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='data/config.yaml',
        help='Path to config file (needed for denormalization)'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Load config and compute stats if denormalization is requested
    norm_stats = {}
    if args.denormalize:
        try:
            from src.utils.config_loader import ConfigLoader
            from src.dataset import StreamflowDataset
            
            config = ConfigLoader.load(args.config)
            # Initialize dataset to compute stats
            # We only need the stats, so we can use a dummy dataset or just initialize it
            # Note: This might take a moment as it computes stats over the dataset
            print("Computing dataset statistics for denormalization...")
            dataset = StreamflowDataset(
                image_dir=config['data']['image_dir'],
                station_dir=config['data']['station_dir'],
                config=config,
                normalize=True
            )
            
            if config['data'].get('normalization_type', 'z_score') == 'minmax':
                norm_stats = {
                    'min': dataset.image_min,
                    'max': dataset.image_max
                }
                print(f"Loaded MinMax stats: min={norm_stats['min']:.4f}, max={norm_stats['max']:.4f}")
            else:
                norm_stats = {
                    'mean': dataset.image_mean,
                    'std': dataset.image_std
                }
                print(f"Loaded Z-score stats: mean={norm_stats['mean']:.4f}, std={norm_stats['std']:.4f}")
                
        except Exception as e:
            print(f"Warning: Failed to load config or compute stats: {e}")
            print("Proceeding without denormalization.")
            args.denormalize = False
    
    # Paths
    output_dir = Path(args.output_dir)
    pred_dir = output_dir / 'predictions'
    
    if not pred_dir.exists():
        print(f"Error: Predictions directory not found at {pred_dir}")
        print("Please run evaluation first to generate predictions.")
        return
    
    # Find target image directory
    if args.image_dir:
        target_dir = Path(args.image_dir)
    else:
        # Try to infer from common locations
        possible_dirs = [
            Path('./data/images'),
            Path('/mnt/d/store/TTF/images'),
        ]
        target_dir = None
        for pdir in possible_dirs:
            if pdir.exists():
                target_dir = pdir
                break
        
        if target_dir is None:
            print("Error: Could not find target image directory.")
            print("Please specify it with --image-dir")
            return
    
    print(f"Using target directory: {target_dir}")
    print(f"Using prediction directory: {pred_dir}")
    
    # Get all prediction files
    pred_files = list(pred_dir.glob('pred_*.npy'))
    
    if len(pred_files) == 0:
        print(f"Error: No prediction files found in {pred_dir}")
        return
    
    print(f"Found {len(pred_files)} prediction files")
    
    # Randomly sample
    num_samples = min(args.num_samples, len(pred_files))
    sampled_files = random.sample(pred_files, num_samples)
    
    # Load samples
    samples = []
    for pred_file in sampled_files:
        try:
            pred, target, date = load_image_pair(pred_file, target_dir)
            samples.append((pred, target, date))
            print(f"Loaded sample: {date}")
        except Exception as e:
            print(f"Warning: Failed to load {pred_file.name}: {e}")
    
    if len(samples) == 0:
        print("Error: No valid samples could be loaded")
        return
    
    print(f"\nCreating visualizations for {len(samples)} samples...")
    
    # Create comparison plot
    comparison_path = output_dir / 'visualization_comparison.png'
    create_comparison_plot(
        samples, 
        comparison_path, 
        denormalize=args.denormalize,
        norm_stats=norm_stats
    )
    
    # Create statistics plot
    stats_path = output_dir / 'visualization_statistics.png'
    create_statistics_plot(samples, stats_path)
    
    print(f"\n✓ Visualization complete!")
    print(f"  - Comparison plot: {comparison_path}")
    print(f"  - Statistics plot: {stats_path}")


if __name__ == '__main__':
    main()
