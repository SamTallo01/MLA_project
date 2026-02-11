import os
import h5py
import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import numpy as np
import openslide
from tqdm import tqdm


def load_model(model_name="uni2-h"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}, loading model: {model_name}")

    if model_name == "uni":
        kwargs = {
            "model_name": "vit_large_patch16_224",
            "img_size": 224,
            "patch_size": 16,
            "num_classes": 0,
            "dynamic_img_size": True,
        }
    else:
        raise ValueError(f"Unknown model_name {model_name}")

    model = timm.create_model(**kwargs)
    model.eval()
    model.to(device)

    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    return model, transform, device


def extract_features_from_patches(
    patches_dir, wsi_dir, patch_size=512, batch_size=16, model_name="uni2-h"
):
    model, transform, device = load_model(model_name)

    out_features_dir = os.path.join(os.getcwd(), "features")
    os.makedirs(out_features_dir, exist_ok=True)

    wsi_dirs = [
        d for d in os.listdir(patches_dir)
        if os.path.isdir(os.path.join(patches_dir, d))
    ]

    for wsi_name in wsi_dirs:

        # Percorso output features
        out_h5 = os.path.join(out_features_dir, f"{wsi_name}_features.h5")

        # SE IL FILE ESISTE GIÀ → SALTA
        if os.path.exists(out_h5):
            print(f"[SKIP] Features già presenti")
            continue

        # Patch coordinate file
        h5_path = os.path.join(patches_dir, wsi_name, f"{wsi_name}_patches.h5")
        if not os.path.exists(h5_path):
            print(f"[WARN] H5 for {wsi_name} not found, skipping")
            continue

        with h5py.File(h5_path, "r") as f:
            coords = f["coords"][:]

        # Trova WSI associata
        wsi_candidates = [
            f for f in os.listdir(wsi_dir)
            if f.startswith(wsi_name)
            and f.lower().endswith((".svs", ".tif", ".tiff", ".ndpi"))
        ]

        if not wsi_candidates:
            print(f"[WARN] No matching WSI for {wsi_name} in {wsi_dir}")
            continue

        wsi_path = os.path.join(wsi_dir, wsi_candidates[0])
        print(f"\n[INFO] Processing WSI: {wsi_path}")

        slide = openslide.OpenSlide(wsi_path)
        features_list = []

        for i in tqdm(
            range(0, len(coords), batch_size),
            desc=f"Extracting features for {wsi_name}",
            unit="batch",
        ):
            batch_coords = coords[i : i + batch_size]
            batch_imgs = []

            for x, y in batch_coords:
                patch = slide.read_region(
                    (int(x), int(y)), 0, (patch_size, patch_size)
                ).convert("RGB")
                batch_imgs.append(transform(patch))

            batch_tensor = torch.stack(batch_imgs).to(device)

            with torch.no_grad():
                feats = model.forward_features(batch_tensor)

            # Usa il CLS token se output shape è N x 1 x D
            if feats.ndim == 3:
                feats = feats[:, 0, :]

            features_list.append(feats.cpu().numpy())

        slide.close()

        features_array = np.concatenate(features_list, axis=0)
        print("[INFO] Final feature shape:", features_array.shape)

        # Salva feature
        with h5py.File(out_h5, "w") as f:
            f.create_dataset("features", data=features_array)

        print(f"[OK] Saved features for {wsi_name} → {out_h5}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract features from WSI patches using UNI / UNI2-h"
    )
    parser.add_argument(
        "--patches", type=str, required=True, help="Folder containing patch HDF5 files"
    )
    parser.add_argument(
        "--wsi", type=str, required=True, help="Folder containing WSI files"
    )
    parser.add_argument(
        "--patch-size", type=int, default=512, help="Patch size in pixels"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for model inference"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="uni",
        choices=["uni", "uni2-h"],
        help="Feature extractor",
    )

    args = parser.parse_args()

    extract_features_from_patches(
        args.patches,
        args.wsi,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        model_name=args.model,
    )
