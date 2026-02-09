import h5py
import pandas as pd
import os
import glob
from pathlib import Path

# Directories
PATCHES_DIR = 'patches'
OUTPUT_DIR = 'csv_patches'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Find all .h5 files in subdirectories of PATCHES_DIR
# Assuming structure: patches/M-X/[file].h5
h5_files = glob.glob(os.path.join(PATCHES_DIR, '**', '*.h5'), recursive=True)

print(f"Found {len(h5_files)} .h5 files.")

for h5_path in h5_files:
    try:
        # Get filename without extension to use for CSV
        # e.g. patches\M-1\M-1_patches.h5 -> M-1_patches
        filename = Path(h5_path).stem
        if filename.endswith('_patches'):
            filename = filename[:-8]
        
        with h5py.File(h5_path, 'r') as f:
            if 'coords' in f:
                coords = f['coords'][:]
                
                df = pd.DataFrame(coords, columns=['x', 'y'])
                df['width'] = 224  # Size of your patches
                df['height'] = 224
                
                output_path = os.path.join(OUTPUT_DIR, f"{filename}.csv")
                df.to_csv(output_path, index=False)
                print(f"Saved: {output_path}")
            else:
                print(f"Skipping {h5_path}: 'coords' dataset not found.")
                
    except Exception as e:
        print(f"Error processing {h5_path}: {e}")

print("Processing complete.")