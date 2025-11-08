import os
import argparse
import numpy as np
import h5py
import openslide

from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.utils import bcolors
from wsi_core.wsi_utils import create_overlay

PATCH_SIZE = 512

def get_slide_mpp(slide):
    mpp_x = slide.properties.get("openslide.mpp-x")
    mpp_y = slide.properties.get("openslide.mpp-y")
    if mpp_x is not None and mpp_y is not None:
        return (float(mpp_x) + float(mpp_y)) / 2
    else:
        objective = slide.properties.get("openslide.objective-power", 40)
        return 10.0 / float(objective)

def patch(args):
    wsi_paths = [
        os.path.join(args.wsi, f)
        for f in os.listdir(args.wsi)
        if f.lower().endswith((".svs", ".tif", ".tiff", ".ndpi"))
    ]

    print(f"\n--- Found {len(wsi_paths)} WSI files ---\n")
    overlay_dir = os.path.join(args.out_dir, "overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    for wsi_path in wsi_paths:
        wsi_name = os.path.splitext(os.path.basename(wsi_path))[0]
        save_dir = os.path.join(args.out_dir, wsi_name)
        os.makedirs(save_dir, exist_ok=True)

        h5_path = os.path.join(save_dir, f"{wsi_name}_patches.h5")
        overlay_path = os.path.join(overlay_dir, f"{wsi_name}_overlay.png")
        example_patch_path = os.path.join(save_dir, f"{wsi_name}_example_patch.png")

        if os.path.exists(h5_path):
            print(f"{bcolors.WARNING}[SKIP] {wsi_name} already has HDF5 → {h5_path}{bcolors.ENDC}")
            continue

        print(f"\n{bcolors.OKCYAN}Processing {wsi_path}{bcolors.ENDC}")
        slide = openslide.OpenSlide(wsi_path)
        mpp = get_slide_mpp(slide)
        print(f"Detected MPP for {wsi_name}: {mpp:.4f} µm/pixel")

        wsi = WholeSlideImage(wsi_path)
        wsi.pixel_size_um = mpp

        print("Segmenting tissue...")
        wsi.segment_tissue(
            downsample_factor=args.downsample_factor,
            laplacian=args.laplacian,
            morph_disk_um=args.morph_disk_um,
            remove_small_um=args.remove_small_um,
            debug=args.debug,
            skip_laplacian_csv=args.skip_laplacian_csv,
        )

        mask = wsi.mask_small
        ds = wsi.mask_downsample
        h, w = mask.shape

        patch_size_ds = args.patch_size // ds
        stride_ds = args.stride // ds

        coords = []

        for y_ds in range(0, h - patch_size_ds + 1, stride_ds):
            for x_ds in range(0, w - patch_size_ds + 1, stride_ds):
                patch_mask = mask[y_ds:y_ds + patch_size_ds, x_ds:x_ds + patch_size_ds]
                if patch_mask.mean() < args.min_tissue_frac:
                    continue
                x = x_ds * ds
                y = y_ds * ds
                coords.append((x, y))

        coords = np.array(coords, dtype=np.int32)

        with h5py.File(h5_path, "w") as f:
            f.create_dataset("coords", data=coords)
        print(f"{bcolors.OKGREEN}Saved patch coordinates to {h5_path} ({len(coords)} patches){bcolors.ENDC}")

        create_overlay(
            wsi_input=wsi_path,
            h5_file=h5_path,
            output_path=overlay_path,
            downsample_factor=32,
            patch_size=args.patch_size,
            alpha=0.4,
        )

        if len(coords) > 0:
            debug_patch_dir = os.path.join("debug", wsi_name)
            os.makedirs(debug_patch_dir, exist_ok=True)

            x_rand, y_rand = coords[np.random.randint(len(coords))]
            patch_img = slide.read_region((int(x_rand), int(y_rand)), 0, (args.patch_size, args.patch_size)).convert("RGB")

            example_patch_path = os.path.join(debug_patch_dir, "example_patch.png")
            patch_img.save(example_patch_path)
            print(f"{bcolors.OKCYAN}Saved example patch → {example_patch_path}{bcolors.ENDC}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wsi", required=True)
    parser.add_argument("--out-dir", default="patches")
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--stride", type=int, default=PATCH_SIZE)
    parser.add_argument("--downsample-factor", type=int, default=16)
    parser.add_argument("--laplacian", type=bool, default=True)
    parser.add_argument("--morph_disk_um", type=float, default=50)
    parser.add_argument("--remove-small-um", type=float, default=100000)
    parser.add_argument("--min-tissue-frac", type=float, default=0.9)
    parser.add_argument("--debug", type=bool, default=True)
    parser.add_argument("--skip-laplacian-csv", type=str, default="skip_laplacian.csv")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    patch(args)
