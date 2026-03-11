from matplotlib import pyplot as plt
import numpy as np

from skimage import exposure, morphology, restoration, segmentation, color, measure
from skimage.filters import threshold_sauvola
from skimage.morphology import disk
from scipy import ndimage as ndi
from sklearn.cluster import KMeans
from skimage.feature import local_binary_pattern
from skimage.util import img_as_ubyte

from img_split import get_middle_fifth

def img_convert(img):
    img = np.asarray(img)
    if img.ndim == 3:
        img = img.mean(axis=2)
    if img.ndim != 2:
        raise ValueError(f"img_seg expects a 2D grayscale image, got shape {img.shape}")
    if img.max() > 1.0:
        img = img / 255.0

    img = get_middle_fifth(img)
    return img

# ---- 1) Denoise and contrast correction ----

def denoise_contrast(img):
    denoised = restoration.denoise_nl_means(
        img, h=0.08, fast_mode=True, patch_size=5, patch_distance=6
    )
    clahe = exposure.equalize_adapthist(denoised, clip_limit=0.03)
    selem_bg = disk(50)
    
    tophat = morphology.white_tophat(clahe, selem_bg)
    return tophat, clahe

# ---- 2) Initial foreground via adaptive/local threshold ----
def adaptive_threshold(tophat, clahe):
    '''
    window_size = 51
    thr = threshold_sauvola(tophat, window_size=window_size)
    bw = tophat > thr   
        # ---- 3) Morphological clean-up ----
    
    bw = morphology.remove_small_objects(bw, max_size=100)
    bw = morphology.closing(bw, disk(3))
    bw = morphology.opening(bw, disk(2))
    bw = ndi.binary_fill_holes(bw)
    bw = morphology.remove_small_holes(bw, max_size=100)
    bw = morphology.closing(bw, disk(2)) 
    

    labels[~bw] = 0
    '''
    
    labels = segmentation.felzenszwalb(clahe, scale=100, sigma=0.5, min_size=100)
        # ---- 4b) Per-rock interior cleanup ----
    clean_labels = np.zeros_like(labels, dtype=np.int32)
    new_id = 1
    for lab in np.unique(labels):
        if lab == 0:
            continue
        region = labels == lab
        region = ndi.binary_fill_holes(region)
        region = morphology.remove_small_holes(region, max_size=500)
        area = int(region.sum())
        if 100 < area < 10000:
            clean_labels[region] = new_id
            new_id += 1
    return clean_labels

# ---- 5) LBP texture + k-means clustering ----
# Use image-derived features (CLAHE intensity + LBP texture), not label IDs.
def k_cluster(clahe, labels):
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(img_as_ubyte(labels), n_points, radius, method='uniform')
    feat = np.stack([clahe.ravel(), lbp.ravel()], axis=1)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(feat)
    km_seg = kmeans.labels_.reshape(clahe.shape)

    valid = labels > 0
    cluster_scores = []
    for i in range(2):
        mask = (km_seg == i) & valid
        if np.any(mask):
            cluster_scores.append(float(labels[mask].mean()))
        else:
            cluster_scores.append(float("-inf"))

    cluster_foreground = km_seg == int(np.argmax(cluster_scores))
    cluster_foreground &= valid

    cluster_foreground = morphology.remove_small_objects(cluster_foreground, max_size=100)
    cluster_foreground = morphology.remove_small_holes(cluster_foreground, max_size=500)
    cluster_foreground = ndi.binary_fill_holes(cluster_foreground)
    return cluster_foreground.astype(np.uint8) * 255


def draw_border(img_seg):
    unique_values = np.unique(img_seg)
    is_binary_like = unique_values.size <= 2

    if is_binary_like:
        labeled = measure.label(img_seg > 0, connectivity=2)
    else:
        labeled = img_seg.astype(np.int32, copy=False)

    base = color.label2rgb(labeled, bg_label=0, bg_color=(0, 0, 0))
    boundaries = segmentation.find_boundaries(labeled, mode="outer")
    output_image = base.copy()
    output_image[boundaries] = [1.0, 0.0, 0.0]  # Red border
    return output_image

def get_object_diameters(segmented_image):
    unique_values = np.unique(segmented_image)
    is_binary_like = unique_values.size <= 2

    if is_binary_like:
        binary = segmented_image > 0
        labeled = measure.label(binary, connectivity=2)
    else:
        labeled = segmented_image.astype(np.int32, copy=False)

    regions = measure.regionprops(labeled)
    diameters = []
    for region in regions:
        if hasattr(region, "equivalent_diameter_area"):
            diameters.append(float(region.equivalent_diameter_area))
        else:
            diameters.append(float(region.equivalent_diameter))
    return diameters

def measure_dia(img):
    mask = np.any(img > 0, axis=2)   # anything not black
    labels = measure.label(mask)
    props = measure.regionprops(labels)
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(img)

    for p in props:
        min_row, min_col, max_row, max_col = p.bbox

        x_diameter = max_col - min_col
        y_diameter = max_row - min_row

        # Draw bounding box
        rect = plt.Rectangle(
            (min_col, min_row),
            x_diameter,
            y_diameter,
            fill=False,
            edgecolor='yellow',
            linewidth=2
        )
        ax.add_patch(rect)

        # Label position (center of object)
        y0, x0 = p.centroid

        # Add text with measurements
        ax.text(
            x0, y0,
            f"x={x_diameter}\ny={y_diameter}",
            color='white',
            fontsize=10,
            ha='center',
            va='center',
            bbox=dict(facecolor='black', alpha=0.6)
        )

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
