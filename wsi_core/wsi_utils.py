import os
import h5py
from PIL import Image, ImageDraw
import openslide

def create_overlay(
    wsi_input,
    h5_file,
    output_path,
    downsample_factor=32,
    patch_size=512,
    alpha=0.4 
):

    if not os.path.exists(wsi_input):
        print(f"[WARN] No WSI found: {wsi_input}")
        return

    if not os.path.exists(h5_file):
        print(f"[WARN] HDF5 not found: {h5_file}")
        return

    with h5py.File(h5_file, "r") as f:
        coords = f["coords"][:]

    if len(coords) == 0:
        print(f"[INFO] No valid patch in {h5_file}")
        return

    slide = openslide.OpenSlide(wsi_input)
    w, h = slide.dimensions
    w_ds, h_ds = w // downsample_factor, h // downsample_factor

    thumbnail = slide.get_thumbnail((w_ds, h_ds)).convert("RGBA")

    overlay = Image.new("RGBA", (w_ds, h_ds), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    patch_ds = patch_size // downsample_factor

    fill_color = (255, 0, 0, int(255 * alpha))

    for x, y in coords:
        x_ds = x // downsample_factor
        y_ds = y // downsample_factor
        draw.rectangle(
            [x_ds, y_ds, x_ds + patch_ds, y_ds + patch_ds],
            outline=None, 
            fill=fill_color
        )

    result = Image.alpha_composite(thumbnail, overlay)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.save(output_path)
    print(f"Overlay saved in: {output_path}")
