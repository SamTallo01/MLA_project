import os
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import openslide
from models.clam import CLAM_SB, CLAM_MB
from train_CLAM import set_seed

def generate_heatmap(wsi_path, features_path, checkpoint_path, output_path, model_type="clam_sb", embed_dim=1024, n_classes=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 1. Load Model
    if model_type == "clam_sb":
        model = CLAM_SB(gate=True, size_arg="small", n_classes=n_classes, embed_dim=embed_dim)
    else:
        model = CLAM_MB(gate=True, size_arg="small", n_classes=n_classes, embed_dim=embed_dim)
    
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 2. Load Features and Coords
    print(f"[INFO] Loading features: {features_path}")
    with h5py.File(features_path, "r") as f:
        features = torch.from_numpy(f['features'][:]).float().to(device)
        coords = f['coords'][:] if 'coords' in f else None
    
    if coords is None:
        # If coords not in features file, try to find them in the patches folder
        slide_id = os.path.splitext(os.path.basename(wsi_path))[0]
        # Common path structure: patches/<slide_id>/<slide_id>_patches.h5
        # We'll check current directory logic
        patch_path = os.path.join("patches", slide_id, f"{slide_id}_patches.h5")
        if os.path.exists(patch_path):
            with h5py.File(patch_path, 'r') as f:
                coords = f['coords'][:]
        else:
            raise ValueError(f"Coordinates not found in {features_path} or {patch_path}")

    # 3. Model Inference
    print("[INFO] Running inference...")
    with torch.no_grad():
        logits, Y_prob, Y_hat, A, _ = model(features)
    
    # 4. Process Attention Scores
    # A has shape (1, N) for SB, or (n_classes, N) for MB
    # We take the attention for the predicted class
    pred_idx = Y_hat.item()
    if model_type == "clam_sb":
        attn = A[0].cpu().numpy()
    else:
        attn = A[pred_idx].cpu().numpy()
    
    # Normalize attention scores 0-1
    attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

    # 5. Build Heatmap
    print("[INFO] Generating visualization...")
    slide = openslide.OpenSlide(wsi_path)
    
    # Get thumbnail for background
    scale_factor = 32
    thumbnail = slide.get_thumbnail((slide.dimensions[0]//scale_factor, slide.dimensions[1]//scale_factor))
    thumb_np = np.array(thumbnail)
    
    # Create empty overlay
    overlay = np.zeros((thumb_np.shape[0], thumb_np.shape[1]))
    
    # Map attention back to thumb space
    # Assuming patch size 512
    patch_size = 512
    for i, (x, y) in enumerate(coords):
        tx, ty = int(x // scale_factor), int(y // scale_factor)
        tw, th = int(patch_size // scale_factor), int(patch_size // scale_factor)
        # Fill the patch area in the overlay
        overlay[ty:ty+th, tx:tx+tw] = np.maximum(overlay[ty:ty+th, tx:tx+tw], attn[i])

    # Plot
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 2, 1)

    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(overlay, cmap='jet', alpha=0.5)
    plt.title(f"Attention Heatmap (Pred: {pred_idx})")
    plt.colorbar(shrink=0.5)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"[OK] Heatmap saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--wsi-path", required=True)
    parser.add_argument("--features-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="heatmap.png")
    parser.add_argument("--model-type", choices=["clam_sb", "clam_mb"], default="clam_sb")
    parser.add_argument("--embed-dim", type=int, default=1024)
    args = parser.parse_args()


    generate_heatmap(
        args.wsi_path,
        args.features_path,
        args.checkpoint,
        args.output,
        model_type=args.model_type,
        embed_dim=args.embed_dim
    )