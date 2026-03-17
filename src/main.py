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
#from img_split import get_middle_fifth

# ---- Folders ----
image_folder = Path("images")
output_folder = Path("debug_output")
results_folder = Path("results_output")
output_folder.mkdir(exist_ok=True)
results_folder.mkdir(exist_ok=True)
image_files = list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.png"))

# ---- Settings ----
# FIX #1: Tune TOPHAT_DISK to roughly the radius of your rocks in pixels.
# If rocks are ~40px wide, set this to ~20. If they're ~100px wide, set to ~50.
# The white top-hat only captures features *smaller* than this disk — set it
# bigger than your largest rock, or you'll cut them off.
TOPHAT_DISK = 60

MIN_OBJECT_SIZE = 200       # Raised from 100 — tiny blobs are rarely real rocks
MAX_HOLE_SIZE = 2000        # Raised from 500 — allow filling larger pores
SAUVOLA_WINDOW = 51         # Keep as is, but see note below if still noisy

# FIX #4: was 10000, which cuts off any rock larger than ~113px diameter.
# Raise this substantially. 100000 means a ~357px-diameter rock still passes.
MIN_AREA = 200
MAX_AREA = 100000

# FIX #5: Minimum pixel distance between rock center peaks for watershed.
# Set this to roughly the smallest rock radius you care about.
WATERSHED_MIN_DIST = 12

# FIX #3: Minimum fraction of a watershed region that must be KMeans-foreground
# for the region to be kept. 0.4 = at least 40% of pixels must agree.
FG_RATIO_THRESHOLD = 0.4


# ---- Helper for showing and saving images ----
def show_image(img, title, filename, cmap="gray"):
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_folder / filename, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()


# ---- LBP ----
def compute_lbp(image, P=24, R=3, method="uniform"):
    image = img_as_ubyte(image)
    lbp = local_binary_pattern(image, P, R, method=method)
    return lbp


# ---- Binary cleanup ----
def clean_binary_mask(mask, min_size=200, max_hole_size=2000):
    mask = morphology.remove_small_objects(mask, min_size=min_size)
    mask = morphology.closing(mask, disk(3))
    mask = ndi.binary_fill_holes(mask)
    mask = morphology.remove_small_holes(mask, area_threshold=max_hole_size)
    mask = morphology.closing(mask, disk(2))
    return mask


# ---- Watershed separation for touching rocks ----
def watershed_separate(binary_mask, min_dist=12):
    """Use distance transform + watershed to split touching/merged rocks."""
    distance = ndi.distance_transform_edt(binary_mask)
    # Smooth the distance map to reduce over-splitting
    distance_smooth = ndi.gaussian_filter(distance, sigma=2)
    coords = peak_local_max(distance_smooth, min_distance=min_dist, labels=binary_mask)
    markers = np.zeros(binary_mask.shape, dtype=bool)
    if len(coords):
        markers[tuple(coords.T)] = True
    markers = measure.label(markers)
    ws = segmentation.watershed(-distance_smooth, markers, mask=binary_mask)
    return ws


# ---- Measurements from binary mask ----
def get_region_areas_and_diameters_from_binary(binary_image, min_area=0):
    labeled = measure.label(binary_image > 0, connectivity=2)
    regions = measure.regionprops(labeled)
    areas, diameters = [], []
    for region in regions:
        if region.area < min_area:
            continue
        areas.append(float(region.area))
        if hasattr(region, "equivalent_diameter_area"):
            diameters.append(float(region.equivalent_diameter_area))
        else:
            diameters.append(float(region.equivalent_diameter))
    return areas, diameters


def save_sorted_values(values, filepath, column_name):
    values = np.sort(np.array(values, dtype=float))
    np.savetxt(filepath, values, delimiter=",", header=column_name,
               comments="", fmt="%.6f")


# ---- Main processing ----
if not image_files:
    print("No images found in 'images' folder")
else:
    for img_path in image_files:
        print(f"\nProcessing: {img_path.name}")
        base_name = img_path.stem

        # ---- Load ----
        img = io.imread(str(img_path), as_gray=True)
        #img = get_middle_fifth(img)
        img = np.asarray(img, dtype=np.float32)
        if img.max() > 1.0:
            img = img / 255.0

        # ---- Denoise ----
        denoised = restoration.denoise_nl_means(
            img, h=0.06, fast_mode=True, patch_size=5, patch_distance=6
        )

        # ---- CLAHE ----
        clahe = exposure.equalize_adapthist(denoised, clip_limit=0.03)

        # ---- Top-hat ----
        selem_bg = disk(TOPHAT_DISK)
        tophat = morphology.white_tophat(clahe, selem_bg)

        # FIX #2: Apply Sauvola to CLAHE directly, not to the tophat image.
        # Tophat compresses most background pixels near zero, which breaks
        # Sauvola's local mean/std estimates. CLAHE has much better contrast
        # distribution for adaptive thresholding.
        # We also OR in a tophat-based threshold as a second opinion.
        thr_clahe = threshold_sauvola(clahe, window_size=SAUVOLA_WINDOW)
        bw_clahe = clahe > thr_clahe

        thr_tophat = threshold_sauvola(tophat, window_size=SAUVOLA_WINDOW)
        bw_tophat = tophat > thr_tophat

        # Union: a pixel is foreground if either method says so
        bw_combined = bw_clahe | bw_tophat
        bw = clean_binary_mask(bw_combined, min_size=MIN_OBJECT_SIZE,
                               max_hole_size=MAX_HOLE_SIZE)

        # FIX #5: Watershed to split touching rocks before measuring
        ws_labels = watershed_separate(bw, min_dist=WATERSHED_MIN_DIST)

        # ---- LBP + KMeans ----
        radius = 3
        n_points = 8 * radius
        lbp = compute_lbp(clahe, P=n_points, R=radius)

        # FIX #6: Normalize both features before stacking so neither dominates
        def norm01(x):
            lo, hi = x.min(), x.max()
            return (x - lo) / (hi - lo + 1e-8)

        feat = np.stack([norm01(clahe).ravel(), norm01(lbp).ravel()], axis=1)
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
        kmeans.fit(feat)
        km_seg = kmeans.labels_.reshape(clahe.shape)

        # FIX #3: Select the foreground cluster by overlap with the binary mask,
        # NOT by which cluster is brighter. Rocks can be dark or light —
        # choosing by brightness is unreliable and is likely your main error source.
        overlap_scores = []
        for i in range(2):
            ki = km_seg == i
            overlap = float(np.logical_and(ki, bw).sum()) / (ki.sum() + 1e-8)
            overlap_scores.append(overlap)
        fg_cluster = int(np.argmax(overlap_scores))
        cluster_foreground_raw = (km_seg == fg_cluster) & bw

        # ---- Final mask: keep watershed regions where KMeans agrees ----
        # For each watershed region, check what fraction of its pixels are
        # in the KMeans foreground. If enough agree, keep the whole region.
        # This avoids the KMeans boundary noise cutting through real rocks.
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

        # Remove anything too small
        final_mask = morphology.remove_small_objects(final_mask, min_size=MIN_OBJECT_SIZE)
        final_mask_u8 = final_mask.astype(np.uint8) * 255

        # ---- Measurements ----
        areas, diameters = get_region_areas_and_diameters_from_binary(
            final_mask_u8, min_area=MIN_AREA
        )

        # ---- Save results ----
        save_sorted_values(areas,
                           results_folder / f"{base_name}_pixel_areas.csv",
                           "pixel_area")
        save_sorted_values(diameters,
                           results_folder / f"{base_name}_diameters.csv",
                           "equivalent_diameter_pixels")

        # ---- Debug visualisations ----
        show_image(img,                  "01 Original",              f"{base_name}_01_original.png")
        show_image(denoised,             "02 Denoised",              f"{base_name}_02_denoised.png")
        show_image(clahe,                "03 CLAHE",                 f"{base_name}_03_clahe.png")
        show_image(tophat,               "04 Top-hat",               f"{base_name}_04_tophat.png")
        show_image(bw_clahe,             "05a Binary (CLAHE)",       f"{base_name}_05a_bw_clahe.png")
        show_image(bw_tophat,            "05b Binary (tophat)",      f"{base_name}_05b_bw_tophat.png")
        show_image(bw,                   "05c Binary (combined)",    f"{base_name}_05c_bw_combined.png")
        show_image(ws_labels,            "06 Watershed labels",      f"{base_name}_06_watershed.png",
                   cmap="nipy_spectral")
        show_image(lbp,                  "07 LBP texture",           f"{base_name}_07_lbp.png")
        show_image(km_seg,               "08 KMeans classes",        f"{base_name}_08_kmeans.png")
        show_image(cluster_foreground_raw, "09 Foreground (KMeans)", f"{base_name}_09_foreground_raw.png")
        show_image(final_mask,           "10 Final mask",            f"{base_name}_10_final_mask.png")

        print(f"  Objects detected : {len(diameters)}")
        print(f"  Diameters (px)   : {np.sort(np.array(diameters)).round(1).tolist()}")