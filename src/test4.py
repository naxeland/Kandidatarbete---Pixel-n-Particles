import numpy as np

from skimage import exposure, morphology, restoration, segmentation
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
    window_size = 51
    thr = threshold_sauvola(tophat, window_size=window_size)
    bw = tophat > thr
    
    labels = segmentation.felzenszwalb(clahe, scale=100, sigma=0.5, min_size=100)
    labels[~bw] = 0
    
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
def k_cluster(img, labels):
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(img_as_ubyte(labels), n_points, radius, method='uniform')
    feat = np.stack([img.ravel(), lbp.ravel()], axis=1)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(feat)
    km_seg = kmeans.labels_.reshape(img.shape)

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

    return (k_cluster(img, labels).astype(np.uint8) * 255)
