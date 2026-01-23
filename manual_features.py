import os
import h5py
import numpy as np
import cv2
from tqdm import tqdm
from openslide import OpenSlide
import mahotas
from scipy.stats import skew, kurtosis

PATCH_SIZE = 512
DOWNSAMPLE_SIZE = 128  # patch ridotta per feature extraction

def extract_manual_features(patch_array):
    """
    Estrazione patch-level RAM-friendly con feature extra.
    Se non ci sono valori calcolabili, riempie con 0.
    """

    # Downsample patch
    patch_small = cv2.resize(
        patch_array,
        (DOWNSAMPLE_SIZE, DOWNSAMPLE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    # Grayscale + tissue mask
    gray = cv2.cvtColor(patch_small, cv2.COLOR_RGB2GRAY)
    _, tissue_mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    tissue_mask = tissue_mask > 0
    tissue_area = tissue_mask.sum()
    tissue_fraction = tissue_area / tissue_mask.size

    # Lunghezza array fissa
    FEATURE_LEN = 43
    features = np.zeros(FEATURE_LEN, dtype=np.float32)

    if tissue_area == 0:
        return features  # patch completamente bianca → tutto 0

    # -----------------------
    # Intensity e RGB stats
    # -----------------------
    R, G, B = patch_small[:,:,0], patch_small[:,:,1], patch_small[:,:,2]

    def stats_channel(ch):
        vals = ch[tissue_mask]
        if len(vals) == 0:
            return [0, 0, 0, 0, 0, 0]  # 6 feature
        return [
            vals.mean(), vals.std(),
            vals.min(), vals.max(),
            skew(vals), kurtosis(vals)
        ]

    R_stats = stats_channel(R)
    G_stats = stats_channel(G)
    B_stats = stats_channel(B)
    gray_stats = stats_channel(gray)

    # -----------------------
    # Haralick grayscale
    # -----------------------
    try:
        img_masked = gray.copy()
        img_masked[~tissue_mask] = 0
        img_norm = cv2.normalize(img_masked, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        har_feat = mahotas.features.haralick(img_norm).mean(axis=0)
    except:
        har_feat = np.zeros(13, dtype=np.float32)

    # -----------------------
    # Edge density
    # -----------------------
    try:
        edges = cv2.Canny(img_norm, 100, 200)
        edge_density = edges[tissue_mask].sum() / tissue_area
    except:
        edge_density = 0

    # -----------------------
    # Gradient magnitude
    # -----------------------
    try:
        gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx**2 + gy**2)
        grad_mean = grad_mag[tissue_mask].mean()
        grad_std = grad_mag[tissue_mask].std()
    except:
        grad_mean = 0
        grad_std = 0

    # -----------------------
    # Nuclear proxies
    # -----------------------
    try:
        nuclear_mask = (gray < 120) & tissue_mask
        nuclear_density = nuclear_mask.sum() / tissue_area
        nuclear_ratio = nuclear_mask.sum() / tissue_mask.sum()

        nuclear_blur = cv2.GaussianBlur(
            nuclear_mask.astype(np.float32),
            (9, 9),
            0
        )
        nuclear_cluster_std = nuclear_blur[tissue_mask].std()
    except:
        nuclear_density = 0
        nuclear_ratio = 0
        nuclear_cluster_std = 0

    # -----------------------
    # Concatenate features
    # -----------------------
    features = np.concatenate([
        [tissue_area, tissue_fraction],
        gray_stats,
        R_stats, G_stats, B_stats,
        har_feat,
        [edge_density, grad_mean, grad_std],
        [edge_density, grad_mean, grad_std,
        nuclear_density, nuclear_ratio, nuclear_cluster_std]
    ])

    # Assicurati lunghezza corretta
    if len(features) < FEATURE_LEN:
        features = np.pad(features, (0, FEATURE_LEN - len(features)), 'constant')
    elif len(features) > FEATURE_LEN:
        features = features[:FEATURE_LEN]

    return features.astype(np.float32)


def main(data_dir, patches_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    patch_folders = [f for f in os.listdir(patches_dir)
                     if os.path.isdir(os.path.join(patches_dir, f))]

    for slide_id in tqdm(patch_folders, desc="Slides"):
        output_path = os.path.join(output_dir, f"{slide_id}.h5")

        # --- Skip slide già processata ---
        if os.path.exists(output_path):
            print(f"Slide {slide_id} già processata, salto...")
            continue

        # Cerca WSI
        wsi_file = None
        for f in os.listdir(data_dir):
            if f.startswith(slide_id):
                wsi_file = os.path.join(data_dir, f)
                break

        if wsi_file is None:
            print(f"WSI not found for {slide_id}, skipping")
            continue

        patches_h5_path = os.path.join(
            patches_dir, slide_id, f"{slide_id}_patches.h5"
        )
        if not os.path.exists(patches_h5_path):
            print(f"Patches H5 not found for {slide_id}, skipping")
            continue

        # Leggi coordinate
        with h5py.File(patches_h5_path, 'r') as f:
            coords = np.array(f['coords'])

        slide = OpenSlide(wsi_file)
        all_features = []

        for (x, y) in tqdm(coords, desc=f"{slide_id}", leave=False):
            patch = slide.read_region(
                (int(x), int(y)), 0, (PATCH_SIZE, PATCH_SIZE)
            ).convert("RGB")
            patch_array = np.array(patch, dtype=np.uint8)

            feat = extract_manual_features(patch_array)
            all_features.append(feat)

            del patch, patch_array, feat

        slide.close()

        all_features = np.stack(all_features)

        # Salva HDF5
        with h5py.File(output_path, 'w') as f:
            f.create_dataset("coords", data=coords)
            f.create_dataset("features", data=all_features)

        print(f"Saved manual features to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--patches_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    main(args.data_dir, args.patches_dir, args.output_dir)
