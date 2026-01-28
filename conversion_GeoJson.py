import h5py
import json
import os
import glob
from pathlib import Path

# ==========================
# Configura qui i percorsi
# ==========================
PATCHES_DIR = r"patches"             # cartella contenente tutti i file .h5
OUTPUT_DIR = r"geojson_patches"      # cartella dove salvare i GeoJSON
PATCH_SIZE = 512                      # dimensione delle patch

# Crea la cartella di output se non esiste
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Trova tutti i file .h5 nella cartella (anche nelle sottocartelle)
h5_files = glob.glob(os.path.join(PATCHES_DIR, '**', '*.h5'), recursive=True)
print(f"Found {len(h5_files)} .h5 files.")

# ==========================
# Processa ciascun file
# ==========================
for h5_path in h5_files:
    try:
        # Nome base del file (senza "_patches" e senza estensione)
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
                x, y = int(x), int(y)  # usa coordinate intere
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

        # Salva il GeoJSON
        with open(geojson_out, "w") as f:
            json.dump(geojson_data, f)

        print(f"Saved: {geojson_out} ({len(features)} patches)")

    except Exception as e:
        print(f"Error processing {h5_path}: {e}")

print("All conversions complete.")
