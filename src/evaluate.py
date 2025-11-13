"""Inference and evaluation script."""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import logging

from src.model import StreamflowPredictionModel
from src.dataset import StreamflowDataset
from src.loss import calculate_metrics
from src.utils.config_loader import ConfigLoader


class Evaluator:
    """Model evaluator for predictions and metrics calculation."""
    
    def __init__(self, config: Dict, model_path: str):
        """
        Initialize evaluator.
        
        Args:
            config: Configuration dictionary
            model_path: Path to trained model
        """
        self.config = config
        self.device = torch.device(config.get('train', {}).get('device', 'cpu'))
        
        # Load model
        self.model = StreamflowPredictionModel(config)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()
        
        # Setup logging
        self.logger = self._setup_logging()
        self.logger.info(f"Model loaded from {model_path}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger('Evaluator')
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def predict_batch(
        self,
        images: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict on a batch.
        
        Args:
            images: (B, T, C, H, W) image sequence
            
        Returns:
            (B, 1, H_out, W_out) predictions
        """
        images = images.to(self.device)
        
        with torch.no_grad():
            predictions = self.model(images)
        
        return predictions.cpu()
    
    def evaluate_dataset(
        self,
        dataset: StreamflowDataset,
        output_dir: str = './output'
    ) -> Dict:
        """
        Evaluate on entire dataset.
        
        Args:
            dataset: Dataset to evaluate
            output_dir: Directory to save results
            
        Returns:
            Dictionary with metrics and results
        """
        output_path = Path(output_dir)
        pred_dir = output_path / 'predictions'
        pred_dir.mkdir(parents=True, exist_ok=True)
        
        all_predictions = []
        all_targets = []
        station_results = []
        
        self.logger.info(f"Evaluating on {len(dataset)} samples...")
        
        for idx in range(len(dataset)):
            sample = dataset[idx]
            
            # Get prediction
            images = sample['images'].unsqueeze(0)  # (1, T, C, H, W)
            prediction = self.predict_batch(images)  # (1, 1, H_out, W_out)
            
            target = sample['output_image'].unsqueeze(0)  # (1, 1, H, W)
            
            all_predictions.append(prediction)
            all_targets.append(target)
            
            # Save prediction image
            pred_image = prediction[0, 0].numpy()
            pred_file = pred_dir / f'pred_{sample["date"]}.npy'
            np.save(pred_file, pred_image)
            
            # Station predictions
            stations = sample['stations']
            positions = sample['station_positions']
            
            if len(stations) > 0:
                for pos, actual_runoff in zip(positions, stations):
                    px, py = pos[0].item(), pos[1].item()
                    
                    # Get neighborhood average
                    kernel_size = self.config.get('stations', {}).get('kernel_size', 3)
                    half = kernel_size // 2
                    
                    y_start = max(0, py - half)
                    y_end = min(pred_image.shape[0], py + half + 1)
                    x_start = max(0, px - half)
                    x_end = min(pred_image.shape[1], px + half + 1)
                    
                    pred_runoff = pred_image[y_start:y_end, x_start:x_end].mean()
                    
                    station_results.append({
                        'date': sample['date'],
                        'position_x': px,
                        'position_y': py,
                        'actual_runoff': actual_runoff.item(),
                        'predicted_runoff': float(pred_runoff)
                    })
            
            if (idx + 1) % 10 == 0:
                self.logger.info(f"Evaluated {idx + 1}/{len(dataset)} samples")
        
        # Calculate metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = calculate_metrics(all_predictions, all_targets)
        
        self.logger.info(f"Image-level metrics:")
        for metric_name, value in metrics.items():
            self.logger.info(f"  {metric_name}: {value:.4f}")
        
        # Save station results
        if station_results:
            df = pd.DataFrame(station_results)
            csv_path = output_path / 'stations_eval.csv'
            df.to_csv(csv_path, index=False)
            
            # Calculate station metrics
            actual_vals = df['actual_runoff'].values
            pred_vals = df['predicted_runoff'].values
            
            station_metrics = calculate_metrics(
                torch.from_numpy(pred_vals).unsqueeze(-1),
                torch.from_numpy(actual_vals).unsqueeze(-1)
            )
            
            self.logger.info(f"Station-level metrics:")
            for metric_name, value in station_metrics.items():
                self.logger.info(f"  {metric_name}: {value:.4f}")
            
            metrics['station_metrics'] = station_metrics
        
        return metrics


def main(
    config_path: str = './data/config.yaml',
    model_path: str = './output/best_model.pt',
    output_dir: str = './output'
):
    """
    Main evaluation function.
    
    Args:
        config_path: Path to configuration
        model_path: Path to trained model
        output_dir: Output directory
    """
    # Load configuration
    config = ConfigLoader.load_config(config_path)
    
    # Create dataset
    image_dir = config.get('data', {}).get('image_dir', './data/images')
    station_dir = config.get('data', {}).get('station_dir', './data/stations')
    
    dataset = StreamflowDataset(
        image_dir=image_dir,
        station_dir=station_dir,
        config=config,
        normalize=True
    )
    
    # Evaluate
    evaluator = Evaluator(config, model_path)
    metrics = evaluator.evaluate_dataset(dataset, output_dir)
    
    print(f"\n✓ Evaluation complete!")
    print(f"  Results saved to {output_dir}")


if __name__ == '__main__':
    import sys
    
    config_path = './data/config.yaml'
    model_path = './output/best_model.pt'
    output_dir = './output'
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    if len(sys.argv) > 2:
        model_path = sys.argv[2]
    if len(sys.argv) > 3:
        output_dir = sys.argv[3]
    
    main(config_path, model_path, output_dir)
