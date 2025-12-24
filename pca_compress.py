import os
import h5py
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA


def load_all_features(features_dir, max_patches_per_wsi=5000):
    """
    Load patch-level features from all h5 files
    (optionally subsample patches to limit memory)
    """
    all_feats = []

    h5_files = [
        f for f in os.listdir(features_dir) if f.endswith(".h5")
    ]

    for fname in tqdm(h5_files, desc="Loading features"):
        path = os.path.join(features_dir, fname)

        with h5py.File(path, "r") as f:
            feats = f["features"][:]  # (N, D)

        if max_patches_per_wsi and feats.shape[0] > max_patches_per_wsi:
            idx = np.random.choice(
                feats.shape[0], max_patches_per_wsi, replace=False
            )
            feats = feats[idx]

        all_feats.append(feats)

    return np.vstack(all_feats)


def fit_pca(features_dir, target_dim=512):
    """
    Fit PCA on patch features
    """
    X = load_all_features(features_dir)

    print(f"[INFO] Fitting PCA on {X.shape}")

    pca = PCA(n_components=target_dim, svd_solver="randomized")
    pca.fit(X)

    return pca


def apply_pca_to_h5(features_dir, out_dir, pca):
    os.makedirs(out_dir, exist_ok=True)

    h5_files = [
        f for f in os.listdir(features_dir) if f.endswith(".h5")
    ]

    for fname in tqdm(h5_files, desc="Compressing"):
        in_path = os.path.join(features_dir, fname)
        out_path = os.path.join(out_dir, fname)

        with h5py.File(in_path, "r") as f:
            feats = f["features"][:]  # (N, D)

        feats_c = pca.transform(feats).astype("float32")

        with h5py.File(out_path, "w") as f:
            f.create_dataset("features", data=feats_c)

    print(f"[DONE] Compressed features saved in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", required=True,
                        help="Folder with original UNI features (.h5)")
    parser.add_argument("--out-dir", required=True,
                        help="Folder to save compressed features")
    parser.add_argument("--target-dim", type=int, default=512)
    parser.add_argument("--max-patches", type=int, default=5000,
                        help="Subsample patches per WSI to fit PCA")
    args = parser.parse_args()

    pca = fit_pca(
        args.features_dir,
        target_dim=args.target_dim,
    )

    apply_pca_to_h5(
        args.features_dir,
        args.out_dir,
        pca,
    )
