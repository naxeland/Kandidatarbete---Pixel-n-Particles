"""
rock_segmentation.py
--------------------
Loads rock images, runs the full segmentation pipeline, and saves:
  - debug_output/   — intermediate step images for inspection
  - masks_output/   — final binary masks (white = rock, black = background)
                      These are read by rock_measurements.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import io, exposure, morphology, restoration, segmentation, measure
from skimage.filters import threshold_sauvola
from skimage.morphology import disk
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from sklearn.cluster import KMeans
from skimage.feature import local_binary_pattern
from skimage.util import img_as_ubyte

# ---- Folders ----
image_folder  = Path("images")
output_folder = Path("debug_output")
masks_folder  = Path("masks_output")       # consumed by rock_measurements.py
output_folder.mkdir(exist_ok=True)
masks_folder.mkdir(exist_ok=True)

image_files = list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.png"))

# ---- Settings ----
# Tune TOPHAT_DISK to be clearly larger than your largest rock radius in pixels.
TOPHAT_DISK        = 60
MIN_OBJECT_SIZE    = 200
MAX_HOLE_SIZE      = 2000
SAUVOLA_WINDOW     = 51
WATERSHED_MIN_DIST = 12    # roughly the smallest rock radius you care about
FG_RATIO_THRESHOLD = 0.4   # fraction of a watershed region that must be KMeans-foreground


# ---- Helpers ----
def show_image(img, title, filename, cmap="gray"):
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_folder / filename, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()


def compute_lbp(image, P=24, R=3, method="uniform"):
    image = img_as_ubyte(image)
    return local_binary_pattern(image, P, R, method=method)


def clean_binary_mask(mask, min_size=200, max_hole_size=2000):
    mask = morphology.remove_small_objects(mask, min_size=min_size)
    mask = morphology.closing(mask, disk(3))
    mask = ndi.binary_fill_holes(mask)
    mask = morphology.remove_small_holes(mask, area_threshold=max_hole_size)
    mask = morphology.closing(mask, disk(2))
    return mask


def watershed_separate(binary_mask, min_dist=12):
    distance        = ndi.distance_transform_edt(binary_mask)
    distance_smooth = ndi.gaussian_filter(distance, sigma=2)
    coords          = peak_local_max(distance_smooth, min_distance=min_dist,
                                     labels=binary_mask)
    markers = np.zeros(binary_mask.shape, dtype=bool)
    if len(coords):
        markers[tuple(coords.T)] = True
    markers = measure.label(markers)
    return segmentation.watershed(-distance_smooth, markers, mask=binary_mask)


def norm01(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-8)


# ---- Main ----
if not image_files:
    print("No images found in 'images' folder")
else:
    for img_path in image_files:
        print(f"\nProcessing: {img_path.name}")
        base_name = img_path.stem

        # Load
        img = io.imread(str(img_path), as_gray=True)
        img = np.asarray(img, dtype=np.float32)
        if img.max() > 1.0:
            img = img / 255.0

        # Denoise
        denoised = restoration.denoise_nl_means(
            img, h=0.06, fast_mode=True, patch_size=5, patch_distance=6
        )

        # CLAHE
        clahe = exposure.equalize_adapthist(denoised, clip_limit=0.03)

        # Top-hat
        tophat = morphology.white_tophat(clahe, disk(TOPHAT_DISK))

        # Binary mask — threshold CLAHE and tophat separately, then union
        bw_clahe  = clahe  > threshold_sauvola(clahe,  window_size=SAUVOLA_WINDOW)
        bw_tophat = tophat > threshold_sauvola(tophat, window_size=SAUVOLA_WINDOW)
        bw = clean_binary_mask(bw_clahe | bw_tophat,
                               min_size=MIN_OBJECT_SIZE,
                               max_hole_size=MAX_HOLE_SIZE)

        # Watershed separation
        ws_labels = watershed_separate(bw, min_dist=WATERSHED_MIN_DIST)

        # LBP + KMeans
        lbp    = compute_lbp(clahe, P=8 * 3, R=3)
        feat   = np.stack([norm01(clahe).ravel(), norm01(lbp).ravel()], axis=1)
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
        kmeans.fit(feat)
        km_seg = kmeans.labels_.reshape(clahe.shape)

        # Select foreground cluster by overlap with binary mask (not brightness)
        overlap_scores = [
            float(np.logical_and(km_seg == i, bw).sum()) / ((km_seg == i).sum() + 1e-8)
            for i in range(2)
        ]
        fg_cluster            = int(np.argmax(overlap_scores))
        cluster_foreground_raw = (km_seg == fg_cluster) & bw

        # Final mask — keep watershed regions where KMeans agrees
        final_mask = np.zeros_like(bw, dtype=bool)
        for lab in np.unique(ws_labels):
            if lab == 0:
                continue
            region_mask = ws_labels == lab
            fg_ratio = (
                float(np.logical_and(region_mask, cluster_foreground_raw).sum())
                / region_mask.sum()
            )
            if fg_ratio >= FG_RATIO_THRESHOLD:
                final_mask[region_mask] = True

        final_mask = morphology.remove_small_objects(final_mask, min_size=MIN_OBJECT_SIZE)

        # Save final mask for rock_measurements.py
        mask_save_path = masks_folder / f"{base_name}_mask.png"
        io.imsave(str(mask_save_path), (final_mask.astype(np.uint8) * 255))
        print(f"  Mask saved → {mask_save_path}")

        # Debug visualisations
        show_image(img,                    "01 Original",            f"{base_name}_01_original.png")
        show_image(denoised,               "02 Denoised",            f"{base_name}_02_denoised.png")
        show_image(clahe,                  "03 CLAHE",               f"{base_name}_03_clahe.png")
        show_image(tophat,                 "04 Top-hat",             f"{base_name}_04_tophat.png")
        show_image(bw_clahe,               "05a Binary (CLAHE)",     f"{base_name}_05a_bw_clahe.png")
        show_image(bw_tophat,              "05b Binary (tophat)",    f"{base_name}_05b_bw_tophat.png")
        show_image(bw,                     "05c Binary (combined)",  f"{base_name}_05c_bw_combined.png")
        show_image(ws_labels,              "06 Watershed",           f"{base_name}_06_watershed.png",
                   cmap="nipy_spectral")
        show_image(lbp,                    "07 LBP texture",         f"{base_name}_07_lbp.png")
        show_image(km_seg,                 "08 KMeans classes",      f"{base_name}_08_kmeans.png")
        show_image(cluster_foreground_raw, "09 Foreground (KMeans)", f"{base_name}_09_foreground_raw.png")
        show_image(final_mask,             "10 Final mask",          f"{base_name}_10_final_mask.png")