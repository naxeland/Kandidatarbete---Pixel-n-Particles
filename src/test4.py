import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from skimage import io, exposure, morphology, restoration, segmentation, measure, color
from skimage.filters import threshold_sauvola
from skimage.morphology import disk
from scipy import ndimage as ndi
from sklearn.cluster import KMeans
from skimage.feature import local_binary_pattern
from skimage.util import img_as_ubyte

from img_split import get_middle_fifth


# ---- Load images from folder ----
image_folder = Path("images")
image_files = list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.png"))


def compute_lbp(image, P=24, R=3, method="uniform"):
    image = img_as_ubyte(image)
    lbp = local_binary_pattern(image, P, R, method=method)
    return lbp


def clean_binary_mask(mask, min_size=100, max_size=500):
    mask = morphology.remove_small_objects(mask, min_size=min_size)
    mask = morphology.closing(mask, disk(3))
    mask = morphology.opening(mask, disk(2))
    mask = ndi.binary_fill_holes(mask)
    mask = morphology.remove_small_holes(mask, area_threshold=max_size)
    mask = morphology.closing(mask, disk(2))
    return mask


def clean_labeled_regions(labels, min_area=100, max_area=10000, max_size=500):
    clean_labels = np.zeros_like(labels, dtype=np.int32)
    new_id = 1

    for lab in np.unique(labels):
        if lab == 0:
            continue

        region = labels == lab
        region = ndi.binary_fill_holes(region)
        region = morphology.remove_small_holes(region, area_threshold=max_size)

        area = int(region.sum())
        if min_area < area < max_area:
            clean_labels[region] = new_id
            new_id += 1

    return clean_labels


def get_object_diameters(segmented_image, min_area=0):
    unique_values = np.unique(segmented_image)
    is_binary_like = set(unique_values).issubset({0, 1, 255})

    if is_binary_like:
        labeled = measure.label(segmented_image > 0, connectivity=2)
    else:
        labeled = segmented_image.astype(np.int32, copy=False)

    regions = measure.regionprops(labeled)
    diameters = []

    for region in regions:
        if region.area < min_area:
            continue

        if hasattr(region, "equivalent_diameter_area"):
            diameters.append(float(region.equivalent_diameter_area))
        else:
            diameters.append(float(region.equivalent_diameter))

    return diameters


def draw_border(segmented_image):
    unique_values = np.unique(segmented_image)
    is_binary_like = set(unique_values).issubset({0, 1, 255})

    if is_binary_like:
        labeled = measure.label(segmented_image > 0, connectivity=2)
    else:
        labeled = segmented_image.astype(np.int32, copy=False)

    base = color.label2rgb(labeled, bg_label=0, bg_color=(0, 0, 0))
    boundaries = segmentation.find_boundaries(labeled, mode="outer")
    output = base.copy()
    output[boundaries] = [1.0, 0.0, 0.0]
    return output


if not image_files:
    print("No images found in 'images' folder")
else:
    for img_path in image_files:
        print(f"Processing: {img_path.name}")

        # ---- Read and normalize image ----
        img = io.imread(str(img_path), as_gray=True)
        img = get_middle_fifth(img)
        img = np.asarray(img, dtype=np.float32)

        if img.max() > 1.0:
            img = img / 255.0

        # ---- 1) Denoise and contrast correction ----
        denoised = restoration.denoise_nl_means(
            img,
            h=0.08,
            fast_mode=True,
            patch_size=5,
            patch_distance=6
        )

        clahe = exposure.equalize_adapthist(denoised, clip_limit=0.03)
        selem_bg = disk(50)
        tophat = morphology.white_tophat(clahe, selem_bg)

        # ---- 2) Initial foreground via adaptive/local threshold ----
        window_size = 51
        thr = threshold_sauvola(tophat, window_size=window_size)
        bw = tophat > thr

        # ---- 3) Morphological clean-up ----
        bw = clean_binary_mask(bw, min_size=100, max_size=500)

        # ---- 4) Felzenszwalb segmentation inside binary mask ----
        labels = segmentation.felzenszwalb(
            clahe,
            scale=100,
            sigma=1,
            min_size=100
        )
        labels[~bw] = 0

        clean_labels = clean_labeled_regions(
            labels,
            min_area=100,
            max_area=10000,
            max_size=500
        )

        # ---- 5) LBP texture + k-means clustering ----
        radius = 3
        n_points = 8 * radius

        lbp = compute_lbp(clahe, P=n_points, R=radius, method="uniform")
        feat = np.stack([clahe.ravel(), lbp.ravel()], axis=1)

        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
        kmeans.fit(feat)
        km_seg = kmeans.labels_.reshape(clahe.shape)

        valid = clean_labels > 0
        cluster_scores = []

        for i in range(2):
            mask = (km_seg == i) & valid
            if np.any(mask):
                cluster_scores.append(float(clahe[mask].mean()))
            else:
                cluster_scores.append(float("-inf"))

        cluster_foreground = km_seg == int(np.argmax(cluster_scores))
        cluster_foreground &= valid

        cluster_foreground = morphology.remove_small_objects(
            cluster_foreground,
            min_size=100
        )
        cluster_foreground = morphology.remove_small_holes(
            cluster_foreground,
            area_threshold=500
        )
        cluster_foreground = ndi.binary_fill_holes(cluster_foreground)
        cluster_foreground = cluster_foreground.astype(np.uint8) * 255

        # ---- Measurements ----
        diameters = get_object_diameters(cluster_foreground, min_area=100)
        border_image = draw_border(cluster_foreground)

        # ---- Visualization ----
        fig, ax = plt.subplots(3, 3, figsize=(15, 13))

        ax[0, 0].imshow(img, cmap="gray")
        ax[0, 0].set_title("Original")
        ax[0, 0].axis("off")

        ax[0, 1].imshow(denoised, cmap="gray")
        ax[0, 1].set_title("Denoised")
        ax[0, 1].axis("off")

        ax[0, 2].imshow(tophat, cmap="gray")
        ax[0, 2].set_title("Tophat / Corrected")
        ax[0, 2].axis("off")

        ax[1, 0].imshow(bw, cmap="gray")
        ax[1, 0].set_title("Binary cleaned")
        ax[1, 0].axis("off")

        ax[1, 1].imshow(clean_labels, cmap="nipy_spectral")
        ax[1, 1].set_title("Felzenszwalb + cleanup")
        ax[1, 1].axis("off")

        ax[1, 2].imshow(lbp, cmap="gray")
        ax[1, 2].set_title("LBP texture")
        ax[1, 2].axis("off")

        ax[2, 0].imshow(km_seg, cmap="gray")
        ax[2, 0].set_title("K-means classes")
        ax[2, 0].axis("off")

        ax[2, 1].imshow(cluster_foreground, cmap="gray")
        ax[2, 1].set_title("Final foreground")
        ax[2, 1].axis("off")

        ax[2, 2].imshow(border_image)
        ax[2, 2].set_title(f"Borders ({len(diameters)} objekt)")
        ax[2, 2].axis("off")

        plt.tight_layout()
        plt.show()

        print(f"Antal objekt: {len(diameters)}")
        print(f"Diametrar: {diameters}")