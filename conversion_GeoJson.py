import h5py
import json
import os
import glob
from pathlib import Path


PATCHES_DIR = r"patches"   
OUTPUT_DIR = r"geojson_patches"      
PATCH_SIZE = 512                   

os.makedirs(OUTPUT_DIR, exist_ok=True)

h5_files = glob.glob(os.path.join(PATCHES_DIR, '**', '*.h5'), recursive=True)
print(f"Found {len(h5_files)} .h5 files.")


for h5_path in h5_files:
    try:
        filename = Path(h5_path).stem
        if filename.endswith('_patches'):
            filename = filename[:-8]

        geojson_out = os.path.join(OUTPUT_DIR, f"{filename}.geojson")

        features = []

        with h5py.File(h5_path, "r") as f:
            if 'coords' not in f:
                print(f"Skipping {h5_path}: 'coords' dataset not found.")
                continue

            coords = f['coords'][:]

            for i, (x, y) in enumerate(coords):
                x, y = int(x), int(y)
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [x, y],
                            [x + PATCH_SIZE, y],
                            [x + PATCH_SIZE, y + PATCH_SIZE],
                            [x, y + PATCH_SIZE],
                            [x, y]
                        ]]
                    },
                    "properties": {
                        "name": f"Patch_{i:05d}",
                        "classification": "Patch"
                    }
                })

        geojson_data = {
            "type": "FeatureCollection",
            "features": features
        }

        with open(geojson_out, "w") as f:
            json.dump(geojson_data, f)

        print(f"Saved: {geojson_out} ({len(features)} patches)")

    except Exception as e:
        print(f"Error processing {h5_path}: {e}")

print("All conversions complete.")
