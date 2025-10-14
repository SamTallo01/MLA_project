# KimiaNet Feature Extractor

This module provides a modular wrapper for extracting feature embeddings from histopathology tiles using the KimiaNet pretrained model. The extracted features are designed for downstream integration with digital pathology pipelines (e.g. MIL, WSI classification).

## Directory Structure

```
feature_extractor/
│
├── kimianet_features.py         # Main feature extraction module
├── external/
│   └── KimiaNet/                # KimiaNet repo, weights, helper scripts (NOT tracked by git)
└── README.md
```

## Setup

1. **Clone KimiaNet**
   - Download [KimiaNet](https://github.com/KimiaLabMayo/KimiaNet) into `external/KimiaNet`
2. **Install Dependencies**
   ```
   pip install torch torchvision numpy pillow
   ```
3. **Add Weights**
   - Place KimiaNet model weights into `external/KimiaNet/KimiaNet_Weights` as described in the official repo.

## Usage

```
from kimianet_features import KimiaNetFeatureExtractor

extractor = KimiaNetFeatureExtractor(device="cuda")  # or "cpu"
features = extractor.extract_tile_features(list_of_tile_paths)
```
- **Input:** List of image tile paths (e.g., 224x224 or 1000x1000 px .png/.jpg files)
- **Output:** NumPy array of 1024-dimensional embeddings per tile; metadata can be associated for downstream tasks

## Best Practices

- `external/KimiaNet/` is **not version-controlled** (see `.gitignore`)—clone or update it manually as needed.
- Only feature extraction is included here; keep downstream classification or aggregation modules separate for clarity and extensibility.
- Track the KimiaNet version you use for reproducibility.
- Store output embeddings and metadata in a standard format (e.g., NumPy, CSV, or HDF5).

## References

- [KimiaNet GitHub](https://github.com/KimiaLabMayo/KimiaNet)
- Based on current best practices in weakly supervised digital pathology pipelines.

## License

Refer to the KimiaNet repository for code/model license information.
