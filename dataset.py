import os
import csv
import h5py
import torch
from torch.utils.data import Dataset

class MILFeatureDataset(Dataset):
    """
    Dataset MIL da CSV:
    - csv_path: percorso al CSV con due colonne: slide_id,label
    - features_dir: cartella con i file .h5 denominati <slide_id>.h5
    """
    def __init__(self, csv_path, features_dir):
        self.features_dir = features_dir
        self.samples = []

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                slide_id = row["slide_id"]
                label = int(row["label"])
                feat_path = os.path.join(features_dir, f"{slide_id}.h5")
                if os.path.exists(feat_path):
                    self.samples.append((feat_path, label))
                else:
                    print(f"[WARNING] Feature file not found: {feat_path}")

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid .h5 feature files found for CSV {csv_path} in {features_dir}")

    def __len__(self):
        return len(self.samples)

    def _read_features_from_h5(self, path):
        with h5py.File(path, "r") as f:
            # prova nomi comuni
            for candidate in ("features", "feats", "embeddings", "data", "X", "features_raw"):
                if candidate in f:
                    arr = f[candidate][:]
                    return arr
            # fallback: primo dataset
            for k in f.keys():
                arr = f[k][:]
                return arr
        raise RuntimeError(f"No datasets found in {path}")

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        arr = self._read_features_from_h5(path)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        arr = arr.astype("float32")
        tensor = torch.from_numpy(arr)
        return tensor, torch.tensor(label, dtype=torch.long)
