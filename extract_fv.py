import os
import h5py
import torch
import openslide
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from models.resnet import get_feature_extractor
from wsi_core.utils import bcolors

PATCH_SIZE = 512

def extract_features_from_slide(slide_path, coords_h5_path, output_h5_path, batch_size=512, device="cuda"):

    slide = openslide.OpenSlide(slide_path)

    with h5py.File(coords_h5_path, 'r') as f:
        coords = np.array(f['coords'])

    model = get_feature_extractor(device=device)

    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

    features = []

    for i in tqdm(range(0, len(coords), batch_size), desc=f"Extracting {os.path.basename(slide_path)}"):
        batch_coords = coords[i:i+batch_size]
        batch_imgs = []

        for (x,y) in batch_coords:
            img = slide.read_region((int(x),int(y)), 0, (PATCH_SIZE, PATCH_SIZE)).convert("RGB")
            img = tf(img)
            batch_imgs.append(img)

        batch = torch.stack(batch_imgs).to(device)

        with torch.no_grad():
            feat = model(batch)

        features.append(feat.cpu()) 

    features = torch.cat(features, dim=0)
    features_np = features.numpy()

    # Save HDF5
    with h5py.File(output_h5_path, 'w') as f:
        f.create_dataset("coords", data=coords)
        f.create_dataset("features", data=features_np)

    print(f"{bcolors.OKGREEN}Saved {output_h5_path}{bcolors.ENDC}")

    # Save as PyTorch .pt file
    pt_path = os.path.splitext(output_h5_path)[0] + ".pt"
    torch.save(features, pt_path)
    print(f"{bcolors.OKGREEN}Saved PyTorch tensor → {pt_path}{bcolors.ENDC}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--wsi_dir", required=True)
    parser.add_argument("--patches_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Slide list
    slides = {os.path.splitext(f)[0]: os.path.join(args.wsi_dir, f)
              for f in os.listdir(args.wsi_dir)
              if f.lower().endswith((".svs",".tif",".tiff",".ndpi"))}

    # 2) Coord h5 search inside each sub-folder
    coords = {}
    for folder in os.listdir(args.patches_dir):
        folder_path = os.path.join(args.patches_dir, folder)
        if os.path.isdir(folder_path):
            for f in os.listdir(folder_path):
                if f.endswith("_patches.h5"):
                    slide_id = folder  # folder name is slide id
                    coords[slide_id] = os.path.join(folder_path, f)

    # 3) Match WSI to coords
    common = set(slides.keys()) & set(coords.keys())

    print(f"\nWSI found: {len(slides)}")
    print(f"Coords found: {len(coords)}")
    print(f"Matched cases: {len(common)}\n")

    for slide_id in sorted(common):
        slide_path = slides[slide_id]
        coords_path = coords[slide_id]
        out_path = os.path.join(args.output_dir, slide_id + ".h5")

        extract_features_from_slide(slide_path, coords_path, out_path,
                                    batch_size=args.batch_size)
