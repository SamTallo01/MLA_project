import pandas as pd
import os
import matplotlib.pyplot as plt
from openslide import OpenSlide


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def plot_training_metrics():
    """
    Plots training metrics from the training log CSV file.
    Generates and saves plots for each fold and the mean across folds.
    """

    df = pd.read_csv("models/training_log.csv")
    df = df.sort_values(by=["fold", "epoch"])

    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)

    folds = df["fold"].unique()

    for f in folds:
        df_fold = df[df["fold"] == f]

        fig, axes = plt.subplots(3, 1, figsize=(8, 12))

        # --- Graph Loss ---
        axes[0].plot(df_fold["epoch"], df_fold["train_loss"], label="Train Loss")
        axes[0].plot(df_fold["epoch"], df_fold["val_loss"], label="Val Loss")
        axes[0].set_title(f"Fold {f} - Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()

        # --- Graph Accuracy ---
        axes[1].plot(df_fold["epoch"], df_fold["accuracy"], label="Accuracy")
        axes[1].set_title(f"Fold {f} - Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()

        # --- Graph Precision / Recall / F1 ---
        axes[2].plot(df_fold["epoch"], df_fold["precision"], label="Precision")
        axes[2].plot(df_fold["epoch"], df_fold["recall"], label="Recall")
        axes[2].plot(df_fold["epoch"], df_fold["F1"], label="F1 Score")
        axes[2].set_title(f"Fold {f} - Precision / Recall / F1")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Score")
        axes[2].legend()

        plt.tight_layout()

        save_path = os.path.join(output_dir, f"fold_{f}.png")
        plt.savefig(save_path)
        plt.close()

        print(f"Saved: {save_path}")

        df_mean = df.groupby("epoch").mean(numeric_only=True)

    fig, axes = plt.subplots(3, 1, figsize=(8, 12))

    # Loss (AVG)
    axes[0].plot(df_mean.index, df_mean["train_loss"], label="Train Loss (mean)")
    axes[0].plot(df_mean.index, df_mean["val_loss"], label="Val Loss (mean)")
    axes[0].set_title("Mean Across Folds - Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Accuracy (AVG)
    axes[1].plot(df_mean.index, df_mean["accuracy"], label="Accuracy (mean)")
    axes[1].set_title("Mean Across Folds - Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    # Precision / Recall / F1 (AVG)
    axes[2].plot(df_mean.index, df_mean["precision"], label="Precision (mean)")
    axes[2].plot(df_mean.index, df_mean["recall"], label="Recall (mean)")
    axes[2].plot(df_mean.index, df_mean["F1"], label="F1 Score (mean)")
    axes[2].set_title("Mean Across Folds - Precision / Recall / F1")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Score")
    axes[2].legend()

    plt.tight_layout()
    mean_path = os.path.join(output_dir, "fold_mean.png")
    plt.savefig(mean_path)
    plt.close()

    print(f"Saved avg graph in: {mean_path}")


def get_mpp(slide):
    """Retrieves the microns-per-pixel (mpp) from the slide properties."""
    props = slide.properties
    keys = ["openslide.mpp-x", "aperio.MPP", "hamamatsu.XResolution"]
    for k in keys:
        if k in props:
            try:
                return float(props[k])
            except:
                pass
    return None


def estimate_magnification(mpp):
    """Makes a rough estimate of the magnification based on the pixel size (mpp)."""
    if mpp is None:
        return "Unknown"
    if 0.23 <= mpp <= 0.28:
        return "40x"
    elif 0.45 <= mpp <= 0.55:
        return "20x"
    elif 0.90 <= mpp <= 1.10:
        return "10x"
    else:
        return f"Non-standard (~{mpp:.3f} µm/px)"


def check_wsi_folder(folder):
    """Scans a folder for WSI files and reports their pixel size and estimated magnification."""
    wsi_extensions = (".svs", ".tif", ".tiff", ".ndpi")
    results = []

    for fname in os.listdir(folder):
        if fname.lower().endswith(wsi_extensions):
            path = os.path.join(folder, fname)
            try:
                slide = OpenSlide(path)
                mpp = get_mpp(slide)
                mag = estimate_magnification(mpp)
                results.append(
                    {
                        "filename": fname,
                        "pixel_size_um": mpp,
                        "estimated_magnification": mag,
                    }
                )
                slide.close()
            except Exception as e:
                results.append(
                    {
                        "filename": fname,
                        "pixel_size_um": None,
                        "estimated_magnification": f"ERROR: {e}",
                    }
                )

    df = pd.DataFrame(results)
    print(df)
