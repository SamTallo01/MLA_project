import h5py
import matplotlib.pyplot as plt
import numpy as np

def visualize_h5_patches(h5_path, n=10):
    with h5py.File(h5_path, 'r') as f:
        print("Datasets in HDF5:", list(f.keys()))
        imgs = f['imgs']       # immagini patch
        coords = f['coords']   # coordinate patch

        # Mostra le prime n patch
        for i in range(min(n, len(imgs))):
            img = imgs[i]        # shape: (H, W, C)
            coord = coords[i]    # eventualmente (x, y)
            plt.figure()
            plt.imshow(img)
            plt.title(f'Patch {i} at {coord}')
            plt.axis('off')
            plt.show()


if __name__ == "__main__":
    h5_path = "data_segmented/patches/M-1.h5"  # sostituisci con il tuo file
    visualize_h5_patches(h5_path)