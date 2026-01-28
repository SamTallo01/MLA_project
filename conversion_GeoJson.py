import h5py
import json

h5_path = "patches/M-2/M-2_patches.h5"
geojson_out = "slide_patches.geojson"
PATCH_SIZE = 512  # args.patch_size

features = []

with h5py.File(h5_path, "r") as f:
    coords = f["coords"][:]   # (N, 2)

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

geojson = {
    "type": "FeatureCollection",
    "features": features
}

with open(geojson_out, "w") as f:
    json.dump(geojson, f)
