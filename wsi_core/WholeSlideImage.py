import os
import cv2
import numpy as np
import openslide
from PIL import Image
from skimage import filters, morphology

Image.MAX_IMAGE_PIXELS = 933120000


class WholeSlideImage:
    def __init__(self, path, pixel_size_um=0.25):
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.wsi = openslide.open_slide(path)
        self.level_downsamples = self.wsi.level_downsamples
        self.level_dim = self.wsi.level_dimensions
        self.pixel_size_um = pixel_size_um

        self.mask_small = None
        self.mask_downsample = None

        self.stain_normalizer = None

    def segment_tissue(
        self,
        downsample_factor=8,
        laplacian=True,
        morph_disk_um=50,
        remove_small_um=100_000,
        debug=False,
        skip_laplacian_csv=None,
    ):

        force_second_attempt = False
        if skip_laplacian_csv and os.path.exists(skip_laplacian_csv):
            import csv

            with open(skip_laplacian_csv, newline="") as f:
                reader = csv.reader(f)
                slide_names = [row[0] for row in reader]
            if self.name in slide_names:
                force_second_attempt = True
                print(
                    f"{self.name} is in skip_laplacian CSV → using attempt 1 logic directly"
                )

        def _save_debug(array, filename):
            debug_dir = os.path.join("debug", self.name)
            os.makedirs(debug_dir, exist_ok=True)
            if array.dtype == bool or array.max() == 1:
                array = (array * 255).astype(np.uint8)
            Image.fromarray(array).save(os.path.join(debug_dir, filename))

        print(f"\n--- Tissue Segmentation for {self.name} ---")
        print(f"Downsample factor: {downsample_factor}")

        mpp = float(self.wsi.properties.get("openslide.mpp-x", 0.25))
        level0_w, level0_h = self.wsi.dimensions
        print(f"WSI Full Resolution: {level0_w} x {level0_h}, mpp={mpp} µm/px")

        target_w = level0_w // downsample_factor
        target_h = level0_h // downsample_factor

        img_thumb = self.wsi.get_thumbnail((target_w, target_h))
        img_np = np.array(img_thumb)

        if debug:
            debug_thumb = cv2.resize(
                img_np,
                (target_w // downsample_factor, target_h // downsample_factor),
                interpolation=cv2.INTER_AREA,
            )
            _save_debug(debug_thumb, "0_thumb.png")

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        if debug:
            _save_debug(gray, "1_gray.png")

        start_attempt = 0 if not force_second_attempt else 1

        for attempt in range(start_attempt, 2):
            if attempt == 0 and not force_second_attempt:
                use_laplacian_attempt = laplacian
            else:
                use_laplacian_attempt = False

            if use_laplacian_attempt:
                lap = cv2.Laplacian(gray, cv2.CV_64F)
                lap = cv2.convertScaleAbs(lap)
                gray_for_otsu = lap
                if debug:
                    _save_debug(gray_for_otsu, f"2_gray_laplacian_attempt{attempt}.png")
            else:
                gray_for_otsu = gray

            thresh = filters.threshold_otsu(gray_for_otsu)
            bw = gray_for_otsu > thresh

            if attempt == 1 or force_second_attempt:
                bw = np.logical_not(bw)

            if debug:
                _save_debug(bw, f"3_bw_attempt{attempt}.png")

            pixels_per_um = 1.0 / self.pixel_size_um / downsample_factor
            selem_radius = max(1, int(round(morph_disk_um * pixels_per_um)))
            selem = morphology.disk(selem_radius)

            bw_closed = morphology.binary_closing(bw, selem)
            if debug:
                _save_debug(bw_closed, f"4_closed_attempt{attempt}.png")

            area_thresh_px = remove_small_um * (pixels_per_um**2)
            bw_clean = morphology.remove_small_objects(
                bw_closed, min_size=area_thresh_px
            )
            bw_clean = morphology.remove_small_holes(
                bw_clean, area_threshold=area_thresh_px
            )
            if debug:
                _save_debug(bw_clean, f"5_clean_attempt{attempt}.png")

            tissue_percent = bw_clean.mean()
            print(f"Tissue percentage = {tissue_percent:.2%}")

            if (
                attempt == 0
                and not force_second_attempt
                and laplacian
                and (tissue_percent < 0.08 or tissue_percent > 0.70)
            ):
                print("Bad segmentation → retrying without Laplacian")
                continue

            break

        self.mask_small = bw_clean.astype(np.uint8)
        self.mask_downsample = downsample_factor
        print(
            f"Finished Tissue Mask: shape={self.mask_small.shape}, downsample={self.mask_downsample}"
        )

        return self.mask_small
