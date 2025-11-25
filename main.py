"""Main entry point for streamflow prediction system."""

import argparse
import sys
from pathlib import Path

from src.train import run_training
from src.evaluate import main as eval_main


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description='Temporal Streamflow Forecasting System'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Generate data command
    gen_parser = subparsers.add_parser(
        'generate',
        help='Generate dummy data for testing'
    )
    gen_parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Number of days to generate'
    )
    gen_parser.add_argument(
        '--stations',
        type=int,
        default=5,
        help='Number of stations'
    )
    gen_parser.add_argument(
        '--channels',
        type=int,
        default=10,
        help='Number of image channels'
    )
    gen_parser.add_argument(
        '--output-dir',
        type=str,
        default='./data',
        help='Output directory'
    )
    
    # Train command
    train_parser = subparsers.add_parser(
        'train',
        help='Train the model'
    )
    train_parser.add_argument(
        '--config',
        type=str,
        default='./data/config.yaml',
        help='Path to config file'
    )
    train_parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    train_parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Override number of epochs'
    )
    
    # Evaluate command
    eval_parser = subparsers.add_parser(
        'evaluate',
        help='Evaluate trained model'
    )
    eval_parser.add_argument(
        '--config',
        type=str,
        default='./data/config.yaml',
        help='Path to config file'
    )
    eval_parser.add_argument(
        '--model',
        type=str,
        default='./output/best_model.pt',
        help='Path to trained model'
    )
    eval_parser.add_argument(
        '--output-dir',
        type=str,
        default='./output',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    if args.command == 'generate':
        print("Generating dummy data...")
        from generate_dummy_data import generate_dummy_data
        generate_dummy_data(
            days=args.days,
            stations=args.stations,
            channels=args.channels,
            output_dir=args.output_dir
        )
    
    elif args.command == 'train':
        print(f"Training model from config: {args.config}")
        run_training(args.config, args.resume, args.epochs)
    
    elif args.command == 'evaluate':
        print(f"Evaluating model: {args.model}")
        eval_main(args.config, args.model, args.output_dir)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
