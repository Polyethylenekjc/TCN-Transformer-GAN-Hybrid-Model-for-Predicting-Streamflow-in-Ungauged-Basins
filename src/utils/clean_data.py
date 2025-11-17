"""
This script cleans .npy files in a specified data directory by replacing
NaN (Not a Number) and Inf (Infinity) values with 0.

It is designed to be run as a standalone utility to preprocess data before
training a model, ensuring that the dataset is free of non-finite values
that can cause training instability (e.g., NaN loss).

Usage:
    python -m src.utils.clean_data /path/to/your/data_folder
"""
import numpy as np
from pathlib import Path
import argparse
import sys

def clean_npy_file(file_path: Path):
    """
    Loads a .npy file, checks for NaNs and Infs, and replaces them with 0.
    The file is overwritten if changes are made.

    Args:
        file_path: The path to the .npy file to clean.
    """
    try:
        data = np.load(file_path, allow_pickle=True)
        
        # Check for non-finite values (NaNs or Infs)
        if not np.all(np.isfinite(data)):
            print(f"Found non-finite values in {file_path}. Cleaning file...")
            # Replace NaNs and Infs with 0
            cleaned_data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            # Overwrite the original file
            np.save(file_path, cleaned_data)
            print(f"Successfully cleaned and saved {file_path}")
        else:
            # No cleaning needed, file is already clean.
            pass
            
    except Exception as e:
        print(f"Error processing file {file_path}: {e}", file=sys.stderr)

def main():
    """
    Main function to parse command-line arguments and initiate the cleaning process.
    """
    parser = argparse.ArgumentParser(
        description="Clean .npy files in a directory by replacing NaNs and Infs with 0.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "data_dir",
        type=str,
        help="Path to the root data directory which contains the 'images' subdirectory.",
    )
    
    args = parser.parse_args()
    
    # The script expects the .npy files to be in a subdirectory named 'images'
    image_dir = Path(args.data_dir) / "images"
    
    if not image_dir.is_dir():
        print(f"Error: Image directory not found at '{image_dir}'", file=sys.stderr)
        print("Please ensure the provided path is the root data directory.", file=sys.stderr)
        sys.exit(1)
        
    print(f"Scanning for .npy files in '{image_dir}'...")
    
    npy_files = list(image_dir.glob("*.npy"))
    
    if not npy_files:
        print("No .npy files found to clean.")
        return
        
    cleaned_count = 0
    for file_path in npy_files:
        # Store original hash/checksum if needed to detect change, but for now just check
        original_mtime = file_path.stat().st_mtime
        clean_npy_file(file_path)
        new_mtime = file_path.stat().st_mtime
        if new_mtime != original_mtime:
            cleaned_count += 1
            
    print(f"\nData cleaning process finished. Scanned {len(npy_files)} files.")

if __name__ == "__main__":
    main()
