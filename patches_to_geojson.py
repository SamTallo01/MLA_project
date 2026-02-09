import h5py
import json
import os
import glob
from pathlib import Path

# --- Configuration ---
PATCHES_DIR = 'patches'
OUTPUT_DIR = 'geojson_patches'
PATCH_SIZE = 224  # Size of your patches in pixels

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Find all .h5 files
h5_files = glob.glob(os.path.join(PATCHES_DIR, '**', '*.h5'), recursive=True)
print(f"Found {len(h5_files)} .h5 files.")

for h5_path in h5_files:
    try:
        # Clean up filename (e.g. M-1_patches.h5 -> M-1)
        filename = Path(h5_path).stem
        if filename.endswith('_patches'):
            filename = filename[:-8]
        
        with h5py.File(h5_path, 'r') as f:
            if 'coords' in f:
                coords = f['coords'][:]
                features = []
                
                print(f"Processing {filename}: {len(coords)} patches...")
                
                for i, coord in enumerate(coords):
                    # coord usually contains [x, y]
                    x, y = float(coord[0]), float(coord[1])
                    w, h = float(PATCH_SIZE), float(PATCH_SIZE)
                    
                    # Create the 5-point polygon (closing the loop)
                    polygon_coords = [
                        [x, y],
                        [x + w, y],
                        [x + w, y + h],
                        [x, y + h],
                        [x, y]
                    ]
                    
                    # Create the GeoJSON feature
                    feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [polygon_coords]
                        },
                        "properties": {
                            "objectType": "detection", # Imported as Detection
                            "name": f"Patch_{i}"
                        }
                    }
                    features.append(feature)

                # Wrap in FeatureCollection
                geojson_data = {
                    "type": "FeatureCollection",
                    "features": features
                }
                
                # Save as GeoJSON
                output_path = os.path.join(OUTPUT_DIR, f"{filename}.geojson")
                with open(output_path, 'w') as out_file:
                    json.dump(geojson_data, out_file)
                
                print(f"Saved: {output_path}")
            else:
                print(f"Skipping {h5_path}: 'coords' dataset not found.")
                
    except Exception as e:
        print(f"Error processing {h5_path}: {e}")

print("All conversions complete.")