import os
import h5py
import torch
import openslide
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from wsi_core.utils import bcolors

PATCH_SIZE = 512

def get_resnet_feature_extractor(device="cuda"):
    from feature_extractor.models.resnet import get_feature_extractor
    model = get_feature_extractor(device=device)
    return model, transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    
def get_kimianet_feature_extractor(device="cuda"):
    from feature_extractor.models.kimianet import get_feature_extractor
    model = get_feature_extractor(device=device)
    return model, transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

def get_resnet18_feature_extractor(device="cuda"):
    from feature_extractor.models.resnet18 import get_feature_extractor
    model = get_feature_extractor(device=device)
    return model, transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

def extract_features(slide_path, coords, model, transform, device="cuda", batch_size=512):
    slide = openslide.OpenSlide(slide_path)
    features = []

    for i in tqdm(range(0, len(coords), batch_size), desc=f"Extracting {os.path.basename(slide_path)}"):
        batch_coords = coords[i:i+batch_size]
        batch_imgs = []

        for (x, y) in batch_coords:
            img = slide.read_region((int(x), int(y)), 0, (PATCH_SIZE, PATCH_SIZE)).convert("RGB")
            img = transform(img)
            batch_imgs.append(img)

        batch = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            feat = model(batch)
        features.append(feat.cpu())

    return torch.cat(features, dim=0)

def save_features(coords, features, output_h5_path):
    features_np = features.numpy()
    with h5py.File(output_h5_path, 'w') as f:
        f.create_dataset("coords", data=coords)
        f.create_dataset("features", data=features_np)
    print(f"{bcolors.OKGREEN}Saved {output_h5_path}{bcolors.ENDC}")

    pt_path = os.path.splitext(output_h5_path)[0] + ".pt"
    torch.save(features, pt_path)
    print(f"{bcolors.OKGREEN}Saved PyTorch tensor → {pt_path}{bcolors.ENDC}")

def main(wsi_dir, patches_dir, output_dir, batch_size=512, device="cuda", model_name="kimianet"):
    os.makedirs(output_dir, exist_ok=True)

    if model_name.lower() == "kimianet":
        feature_extractor_fn = get_kimianet_feature_extractor
        print(f"{bcolors.OKBLUE}Using KimiaNet extractor{bcolors.ENDC}")
    elif model_name.lower() == "resnet":
        feature_extractor_fn = get_resnet_feature_extractor
        print(f"{bcolors.OKBLUE}Using ResNet extractor{bcolors.ENDC}")
    elif model_name.lower() == "resnet18":
        feature_extractor_fn = get_resnet18_feature_extractor
        print(f"{bcolors.OKBLUE}Using ResNet18 extractor{bcolors.ENDC}")
    else:
        raise ValueError("Invalid --model option. Choose: resnet, resnet18, kimianet")

    slides = {os.path.splitext(f)[0]: os.path.join(wsi_dir, f)
            for f in os.listdir(wsi_dir)
            if f.lower().endswith((".svs", ".tif", ".tiff", ".ndpi"))}

    coords_dict = {}
    for folder in os.listdir(patches_dir):
        folder_path = os.path.join(patches_dir, folder)
        if os.path.isdir(folder_path):
            for f in os.listdir(folder_path):
                if f.endswith("_patches.h5"):
                    slide_id = folder
                    coords_dict[slide_id] = os.path.join(folder_path, f)

    common = set(slides.keys()) & set(coords_dict.keys())
    print(f"\nWSI found: {len(slides)}")
    print(f"Coords found: {len(coords_dict)}")
    print(f"Matched cases: {len(common)}\n")

    model, transform = feature_extractor_fn(device=device)

    for slide_id in sorted(common):
        slide_path = slides[slide_id]
        coords_path = coords_dict[slide_id]
        out_path = os.path.join(output_dir, slide_id + ".h5")

        with h5py.File(coords_path, 'r') as f:
            coords = np.array(f['coords'])

        features = extract_features(slide_path, coords, model, transform, device=device, batch_size=batch_size)
        save_features(coords, features, out_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--wsi_dir", required=True)
    parser.add_argument("--patches_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model", type=str, default="kimianet",
                        choices=["kimianet", "resnet", "resnet18"],
                        help="Select feature extractor backend")
    args = parser.parse_args()

    main(args.wsi_dir, args.patches_dir, args.output_dir,
        args.batch_size, model_name=args.model)
