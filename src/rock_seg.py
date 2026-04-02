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
from skimage import io, exposure, morphology, restoration, segmentation
from skimage.filters import threshold_sauvola
from skimage.morphology import disk
from scipy import ndimage as ndi
from sklearn.cluster import KMeans
from skimage.feature import local_binary_pattern
from skimage.util import img_as_ubyte
import time

# ---- Time ----
start = time.time()


# ---- Folders ----
image_folder  = Path("images")
output_folder = Path("debug_output")
masks_folder  = Path("masks_output")       # consumed by rock_measurements.py
output_folder.mkdir(exist_ok=True)
masks_folder.mkdir(exist_ok=True)

image_files = list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.png"))

# ---- Settings ----
# Tune TOPHAT_DISK to be clearly larger than your largest rock radius in pixels.
TOPHAT_DISK           = 10 
MIN_OBJECT_SIZE       = 200
MAX_HOLE_SIZE         = 2000
SAUVOLA_WINDOW        = 51
FELZENSZWALB_SCALE    = 200   # higher = larger segments
FELZENSZWALB_SIGMA    = 0.8   # Gaussian smoothing before segmentation
FELZENSZWALB_MIN_SIZE = 50    # minimum segment size in pixels
FG_RATIO_THRESHOLD    = 0.4   # fraction of a Felzenszwalb region that must be KMeans-foreground


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


def felzenszwalb_separate(image, binary_mask,
                          scale=FELZENSZWALB_SCALE,
                          sigma=FELZENSZWALB_SIGMA,
                          min_size=FELZENSZWALB_MIN_SIZE):
    labels = segmentation.felzenszwalb(image, scale=scale, sigma=sigma, min_size=min_size)
    labels = labels + 1          # shift so 0 can serve as background
    labels[~binary_mask] = 0
    return labels


def norm01(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-8)


# ---- Pipeline stages ----
def load_image(img_path):
    img = io.imread(str(img_path), as_gray=True)
    img = np.asarray(img, dtype=np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    return img


def denoise(img):
    return restoration.denoise_nl_means(
        img, h=0.06, fast_mode=True, patch_size=5, patch_distance=6
    )


def apply_clahe(denoised):
    return exposure.equalize_adapthist(denoised, clip_limit=0.03)


def apply_tophat(clahe):
    return morphology.white_tophat(clahe, disk(TOPHAT_DISK))


def build_binary_mask(clahe, tophat):
    bw_clahe  = clahe  > threshold_sauvola(clahe,  window_size=SAUVOLA_WINDOW)
    bw_tophat = tophat > threshold_sauvola(tophat, window_size=SAUVOLA_WINDOW)

    fz_clahe  = felzenszwalb_separate(clahe,  bw_clahe)
    fz_tophat = felzenszwalb_separate(tophat, bw_tophat)

    bw = clean_binary_mask(fz_clahe | fz_tophat,
                           min_size=MIN_OBJECT_SIZE,
                           max_hole_size=MAX_HOLE_SIZE)
    return bw_clahe, bw_tophat, fz_clahe, fz_tophat, bw


def run_kmeans(clahe):
    lbp  = compute_lbp(clahe, P=8 * 3, R=3)
    feat = np.stack([norm01(clahe).ravel(), norm01(lbp).ravel()], axis=1)
    km   = KMeans(n_clusters=2, random_state=0, n_init=10)
    km.fit(feat)
    km_seg = km.labels_.reshape(clahe.shape)
    return lbp, km_seg


def select_foreground(km_seg, bw):
    overlap_scores = [
        float(np.logical_and(km_seg == i, bw).sum()) / ((km_seg == i).sum() + 1e-8)
        for i in range(2)
    ]
    fg_cluster = int(np.argmax(overlap_scores))
    return (km_seg == fg_cluster) & bw


def build_final_mask(bw, cluster_foreground_raw):
    final_mask = np.zeros_like(bw, dtype=bool)
    for lab in np.unique(bw):
        if lab == 0:
            continue
        region_mask = bw == lab
        fg_ratio = (
            float(np.logical_and(region_mask, cluster_foreground_raw).sum())
            / region_mask.sum()
        )
        if fg_ratio >= FG_RATIO_THRESHOLD:
            final_mask[region_mask] = True
    return morphology.remove_small_objects(final_mask, min_size=MIN_OBJECT_SIZE)


def save_mask(final_mask, base_name):
    mask_save_path = masks_folder / f"{base_name}_mask.png"
    io.imsave(str(mask_save_path), (final_mask.astype(np.uint8) * 255))
    print(f"  Mask saved → {mask_save_path}")


def save_debug_images(base_name, img, denoised, clahe, tophat,
                      bw_clahe, bw_tophat, bw, fz_clahe, fz_tophat,
                      lbp, km_seg, cluster_foreground_raw, final_mask):
    show_image(img,                    "01 Original",                  f"{base_name}_01_original.png")
    show_image(denoised,               "02 Denoised",                  f"{base_name}_02_denoised.png")
    show_image(clahe,                  "03 CLAHE",                     f"{base_name}_03_clahe.png")
    show_image(tophat,                 "04 Top-hat",                   f"{base_name}_04_tophat.png")
    show_image(bw_clahe,               "05a Binary (CLAHE)",           f"{base_name}_05a_bw_clahe.png")
    show_image(bw_tophat,              "05b Binary (tophat)",          f"{base_name}_05b_bw_tophat.png")
    show_image(bw,                     "05c Binary (combined)",        f"{base_name}_05c_bw_combined.png")
    show_image(fz_clahe,               "06 Felzenszwalb (CLAHE)",      f"{base_name}_06_felzenszwalb_clahe.png",
               cmap="nipy_spectral")
    show_image(fz_tophat,              "06 Felzenszwalb (Top-hat)",    f"{base_name}_06_felzenszwalb_tophat.png",
               cmap="nipy_spectral")
    show_image(lbp,                    "07 LBP texture",               f"{base_name}_07_lbp.png")
    show_image(km_seg,                 "08 KMeans classes",            f"{base_name}_08_kmeans.png")
    show_image(cluster_foreground_raw, "09 Foreground (KMeans)",       f"{base_name}_09_foreground_raw.png")
    show_image(final_mask,             "10 Final mask",                f"{base_name}_10_final_mask.png")


def process_image(img_path):
    print(f"\nProcessing: {img_path.name}")
    base_name = img_path.stem

    t0 = time.time()
    img                   = load_image(img_path)
    t1 = time.time(); print(f"img Runtime: {t1-t0:.4f}s")
    denoised              = denoise(img)
    t2 = time.time(); print(f"denoised Runtime: {t2-t1:.4f}s")
    clahe                 = apply_clahe(denoised)
    t3 = time.time(); print(f"CLAHE Runtime: {t3-t2:.4f}s")
    tophat                = apply_tophat(clahe)
    t4 = time.time(); print(f"tophat Runtime: {t4-t3:.4f}s")
    lbp, km_seg           = run_kmeans(clahe)
    t5 = time.time(); print(f"KMeans Runtime: {t5-t4:.4f}s")
    bw_clahe, bw_tophat, fz_clahe, fz_tophat, bw = build_binary_mask(clahe, tophat)
    t6 = time.time(); print(f"Binary mask Runtime: {t6-t5:.4f}s")
    cluster_foreground_raw = select_foreground(km_seg, bw)
    t7 = time.time(); print(f"Foreground selection Runtime: {t7-t6:.4f}s")
    final_mask            = build_final_mask(bw, cluster_foreground_raw)
    t8 = time.time(); print(f"Final mask Runtime: {t8-t7:.4f}s")
    print(f"Total Runtime: {t8-t0:.4f}s")
    '''
    show_image(km_seg, "KMeans segmentation", f"{base_name}_kmeans.png", cmap="nipy_spectral")
    show_image(lbp, "LBP texture", f"{base_name}_lbp.png")
    show_image(cluster_foreground_raw, "Foreground (KMeans)", f"{base_name}_binary_mask.png")
    '''
    save_mask(cluster_foreground_raw, base_name)

    save_debug_images(base_name, img, denoised, clahe, tophat,
                      bw_clahe, bw_tophat, bw, fz_clahe, fz_tophat,
                      lbp, km_seg, cluster_foreground_raw, final_mask)

def move_files(image, src_folder, dst_folder):
    src = Path(src_folder) / image
    src.rename(Path(dst_folder) / image)

# ---- Main ----
while(1):
    if not image_files:
        print("No images found in 'images' folder")
    else:
        for img_path in image_files:
            process_image(img_path)
            move_files(img_path.name, "images", "images2")
    time.sleep(5)  # Check for new images every 5 seconds
        
        
