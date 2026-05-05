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
from skimage.color import rgb2hsv
from skimage.filters import threshold_sauvola, sobel
from skimage.morphology import disk
from scipy import ndimage as ndi
from sklearn.cluster import KMeans
from skimage.feature import local_binary_pattern, peak_local_max
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

# ---- Settings ----
# Tune TOPHAT_DISK to be clearly larger than your largest rock radius in pixels.
TOPHAT_DISK            = 100
USE_TOPHAT             = True   # set True to re-enable (slow)
MIN_OBJECT_SIZE        = 100
MAX_HOLE_SIZE          = 3000
SAUVOLA_WINDOW         = 61    # must be odd, should be larger than largest rock radius in pixels
WATERSHED_MIN_DISTANCE = 1    # minimum pixel distance between rock centers for watershed
WALL_BRIGHTNESS_THRESH = 0.15  # pixels darker than this are treated as wall/background
TOP_ROCK_MIN_INTENSITY = 0.5   # watershed regions with mean intensity below this are dropped as non-top rocks
MIN_ROCK_AREA          = 500   # watershed regions smaller than this (px) are dropped as rubble
status = False # set to False to disable infinite loop and just process existing images once


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


def watershed_label(binary_mask, min_distance=None):
    if min_distance is None:
        min_distance = WATERSHED_MIN_DISTANCE
    distance = ndi.distance_transform_edt(binary_mask)
    coords = peak_local_max(distance, min_distance=min_distance, labels=binary_mask)
    local_maxi = np.zeros(binary_mask.shape, dtype=bool)
    if len(coords):
        local_maxi[tuple(coords.T)] = True
    markers, _ = ndi.label(local_maxi)
    return segmentation.watershed(-distance, markers, mask=binary_mask)



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


def load_image_color(img_path):
    img = io.imread(str(img_path))
    if img.ndim < 3:
        return None
    return (img[:, :, :3].astype(np.float32) / 255.0)


def denoise(img):
    return restoration.denoise_nl_means(
        img, h=0.06, fast_mode=True, patch_size=5, patch_distance=6
    )


def apply_clahe(denoised):
    return exposure.equalize_adapthist(denoised, clip_limit=0.03)


def apply_tophat(img, downsample=4):
    from skimage.transform import resize
    h, w = img.shape
    small = resize(img, (h // downsample, w // downsample), anti_aliasing=True)
    th_small = morphology.white_tophat(small, disk(max(1, TOPHAT_DISK // downsample)))
    return resize(th_small, (h, w), anti_aliasing=False).astype(np.float32)

def tophat_(denoised):
    selem_bg = disk(TOPHAT_DISK)
    tophat = morphology.white_tophat(denoised, selem_bg)
    return tophat

def build_binary_mask(clahe, tophat):
    bw_clahe = clahe > threshold_sauvola(clahe, window_size=SAUVOLA_WINDOW)
    ws_clahe = watershed_label(bw_clahe, min_distance=WATERSHED_MIN_DISTANCE)

    if USE_TOPHAT and tophat is not None:
        bw_tophat = tophat > threshold_sauvola(tophat, window_size=SAUVOLA_WINDOW)
        ws_tophat = watershed_label(bw_tophat, min_distance=WATERSHED_MIN_DISTANCE)
        combined  = ws_clahe | ws_tophat
    else:
        bw_tophat = np.zeros_like(bw_clahe)
        combined  = bw_clahe

    bw = clean_binary_mask(combined,
                           min_size=MIN_OBJECT_SIZE,
                           max_hole_size=MAX_HOLE_SIZE)
    return bw_clahe, bw_tophat, bw


def run_kmeans(clahe):#, img_color=None):
    lbp      = compute_lbp(clahe, P=8 * 3, R=3)
    features = [norm01(clahe).ravel(), norm01(lbp).ravel()]
    '''
    if img_color is not None:
        hsv = rgb2hsv(img_color)
        features.append(norm01(hsv[:, :, 0]).ravel())  # Hue
        features.append(norm01(hsv[:, :, 1]).ravel())  # Saturation
    '''
    feat   = np.stack(features, axis=1)
    km     = KMeans(n_clusters=2, random_state=0, n_init=10)
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


def detect_container_roi(img, closing_radius=30):
    """Find the bright interior of the container, masking out the dark rubber rim.
    Uses a global Otsu threshold on the original image (before CLAHE) so the
    dark rubber — which is much darker than any rock — is cleanly excluded.
    """
    from skimage.filters import threshold_otsu
    thresh = threshold_otsu(img)
    roi = img > thresh
    # Keep only the largest connected bright region (the rock heap)
    labeled, _ = ndi.label(roi)
    component_sizes = np.bincount(labeled.ravel())
    component_sizes[0] = 0  # ignore the background label
    largest_label = int(component_sizes.argmax())
    roi = labeled == largest_label
    roi = ndi.binary_fill_holes(roi)
    roi = morphology.closing(roi, disk(closing_radius))
    return roi


def filter_top_rocks(labeled, intensity_img, min_intensity=None, min_area=None):
    if min_intensity is None:
        min_intensity = TOP_ROCK_MIN_INTENSITY
    if min_area is None:
        min_area = MIN_ROCK_AREA
    filtered = np.zeros_like(labeled)
    for label_id in range(1, int(labeled.max()) + 1):
        region_mask = labeled == label_id
        if region_mask.sum() >= min_area and float(intensity_img[region_mask].mean()) >= min_intensity:
            filtered[region_mask] = label_id
    return filtered


def fill_rock_interiors(mask, closing_radius=1, max_hole_size=1000):
    """Close small gaps within rocks, then fill enclosed holes."""
    filled = morphology.closing(mask, disk(closing_radius))
    #filled = ndi.binary_fill_holes(filled)
    filled = morphology.remove_small_holes(filled, area_threshold=max_hole_size)
    return filled


def save_mask(final_mask, base_name):
    mask_save_path = masks_folder / f"{base_name}_mask.png"
    io.imsave(str(mask_save_path), (final_mask.astype(np.uint8) * 255))
    print(f"  Mask saved → {mask_save_path}")


def save_debug_images(base_name, img, denoised, clahe, tophat,
                      bw_clahe, bw_tophat, bw,
                      lbp, km_seg, cluster_foreground_raw, labeled_rocks, final_mask):
    show_image(img,                    "01 Original",                  f"{base_name}_01_original.png")
    show_image(denoised,               "02 Denoised",                  f"{base_name}_02_denoised.png")
    show_image(clahe,                  "03 CLAHE",                     f"{base_name}_03_clahe.png")
    show_image(tophat,                 "04 Top-hat",                   f"{base_name}_04_tophat.png")
    show_image(bw_clahe,               "05a Binary (CLAHE)",           f"{base_name}_05a_bw_clahe.png")
    show_image(bw_tophat,              "05b Binary (tophat)",          f"{base_name}_05b_bw_tophat.png")
    show_image(bw,                     "05c Binary (combined)",        f"{base_name}_05c_bw_combined.png")
    show_image(lbp,                    "07 LBP texture",               f"{base_name}_07_lbp.png")
    show_image(km_seg,                 "08 KMeans classes",            f"{base_name}_08_kmeans.png")
    
    show_image(cluster_foreground_raw, "09 Foreground (KMeans)",       f"{base_name}_09_foreground_raw.png")
    show_image(labeled_rocks,          "10 Watershed labels",          f"{base_name}_10_watershed.png",
               cmap="nipy_spectral")
    show_image(final_mask,             "11 Surface rocks (final)",     f"{base_name}_11_final_mask.png")


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
    #tophat                = apply_tophat(denoised)
    tophat               = tophat_(denoised)
    t4 = time.time(); print(f"tophat Runtime: {t4-t3:.4f}s")
    lbp, km_seg           = run_kmeans(clahe) 
    t5 = time.time(); print(f"KMeans Runtime: {t5-t4:.4f}s")
    bw_clahe, bw_tophat, bw = build_binary_mask(clahe, tophat)
    t6 = time.time(); print(f"Binary mask Runtime: {t6-t5:.4f}s")
    '''
    bright_mask            = denoised > WALL_BRIGHTNESS_THRESH
    container_roi          = detect_container_roi(img)
    container_roi          = morphology.erosion(container_roi, disk(5))
    '''
    #bw                     = bw & bright_mask & container_roi
    cluster_ws = watershed_label(km_seg, min_distance=WATERSHED_MIN_DISTANCE)
    #bw_ws = watershed_label(bw, min_distance=WATERSHED_MIN_DISTANCE)
    cluster_foreground = select_foreground(cluster_ws > 0, bw)
    #cluster_foreground_raw = fill_rock_interiors(cluster_foreground)
    t7 = time.time(); print(f"Foreground selection Runtime: {t7-t6:.4f}s")
    #labeled_rocks = watershed_label(cluster_foreground_raw, min_distance=WATERSHED_MIN_DISTANCE)
    #labeled_rocks = filter_top_rocks(labeled_rocks, denoised)
    #final_mask    = labeled_rocks > 0
    t8 = time.time(); print(f"Final mask Runtime: {t8-t7:.4f}s")
    print(f"Total Runtime: {t8-t0:.4f}s")

    save_mask(cluster_foreground, base_name)

    show_image(km_seg,                 "08 KMeans classes",            f"{base_name}_08_kmeans.png")
    #show_image(labeled_rocks, "Watershedding", f"{base_name}_09_foreground.png")
    show_image(cluster_foreground, "Cluster foreground", f"{base_name}_bw_tophat.png")
    #show_image(final_mask,             "mask",     f"{base_name}_11_final_mask.png")
    '''
    tophat_img = tophat if USE_TOPHAT else np.zeros_like(clahe)
    bw_tophat_img = bw_tphat if USE_TOPHAT else np.zeros_like(bw_clahe)
    save_debug_images(base_name, img, denoised, clahe, tophat_img,
                      bw_clahe, bw_tophat_img, bw,
                      lbp, km_seg, cluster_foreground_raw, labeled_rocks, final_mask)
'''

def move_files(image, src_folder, dst_folder):
    src = Path(src_folder) / image
    src.rename(Path(dst_folder) / image)


def remove_image(img_path):
    """Delete an image file from the images folder after it has been processed."""
    try:
        Path(img_path).unlink()
        print(f"  Removed: {img_path}")
    except FileNotFoundError:
        print(f"  [warn] File already gone: {img_path}")


# ---- Main ----
image_files = list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.png"))
if status:
    while(1):
        if not image_files:
            print("No images found in 'images' folder")
        else:
            for img_path in image_files:
                process_image(img_path)
                #remove_image(img_path)
        time.sleep(5)  # Check for new images every 5 seconds
else:
    if not image_files:
        print("No images found in 'images' folder")
    else:
        for img_path in image_files:
            process_image(img_path)
            #remove_image(img_path)
    time.sleep(5)  # Check for new images every 5 seconds
        
        
