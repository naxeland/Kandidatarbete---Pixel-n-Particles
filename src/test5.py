import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from skimage import io, exposure, morphology, restoration, segmentation, measure, color, feature
from skimage.filters import threshold_sauvola
from skimage.morphology import disk
from scipy import ndimage as ndi
from sklearn.cluster import KMeans
from skimage.feature import local_binary_pattern
from skimage.util import img_as_ubyte

from img_split import get_middle_fifth


# ---- Folders ----
image_folder = Path("images")
output_folder = Path("debug_output")
output_folder.mkdir(exist_ok=True)

image_files = list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.png"))


# ---- Helper for showing and saving images ----
def show_image(img, title, filename, cmap="gray"):
    plt.figure(figsize=(6, 6))

    if cmap is None:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap=cmap)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    save_path = output_folder / filename
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()


# ---- LBP ----
def compute_lbp(image, P=24, R=3, method="uniform"):
    image = img_as_ubyte(image)
    lbp = local_binary_pattern(image, P, R, method=method)
    return lbp


# ---- Binary cleanup ----
def clean_binary_mask(mask, min_size=100, max_size=500):
    mask = morphology.remove_small_objects(mask, min_size=min_size)
    mask = morphology.closing(mask, disk(3))
    mask = morphology.opening(mask, disk(2))
    mask = ndi.binary_fill_holes(mask)
    mask = morphology.remove_small_holes(mask, area_threshold=max_size)
    mask = morphology.closing(mask, disk(2))
    return mask


# ---- Label cleanup ----
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


# ---- Diameter measurement ----
def get_object_diameters(segmented_image, min_area=0):
    labeled = measure.label(segmented_image > 0, connectivity=2)
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


# ---- Draw borders from binary image ----
def draw_border(segmented_image):
    labeled = measure.label(segmented_image > 0, connectivity=2)
    base = color.label2rgb(labeled, bg_label=0, bg_color=(0, 0, 0))

    boundaries = segmentation.find_boundaries(labeled, mode="outer")

    output = base.copy()
    output[boundaries] = [1, 0, 0]

    return output


# ---- Draw borders from label image ----
def draw_border_from_labels(label_image):
    base = color.label2rgb(label_image, bg_label=0, bg_color=(0, 0, 0))
    boundaries = segmentation.find_boundaries(label_image, mode="outer")

    output = base.copy()
    output[boundaries] = [1, 0, 0]

    return output


# ---- Watershed split for touching objects ----
def split_touching_objects(binary_mask, min_distance=12, min_area=100, max_area=10000):
    """
    Delar ihopklumpade objekt med distance transform + watershed.
    Returnerar:
        labeled_ws      : label-bild efter watershed
        distance        : distance transform
        local_maxima    : binär bild med toppar
    """
    mask = binary_mask > 0

    # Distance transform
    distance = ndi.distance_transform_edt(mask)

    # Lokala maxima som frön för watershed
    local_maxima = feature.peak_local_max(
        distance,
        min_distance=min_distance,
        labels=mask,
        footprint=np.ones((3, 3)),
        exclude_border=False
    )

    marker_mask = np.zeros_like(mask, dtype=bool)
    if len(local_maxima) > 0:
        marker_mask[tuple(local_maxima.T)] = True

    markers = measure.label(marker_mask)

    # Om inga toppar hittas, använd connected components som fallback
    if markers.max() == 0:
        labeled_ws = measure.label(mask, connectivity=2)
    else:
        labeled_ws = segmentation.watershed(-distance, markers, mask=mask)

    # Städning av label-resultatet
    clean_labels = np.zeros_like(labeled_ws, dtype=np.int32)
    new_id = 1

    for lab in np.unique(labeled_ws):
        if lab == 0:
            continue

        region = labeled_ws == lab
        area = int(region.sum())

        if min_area < area < max_area:
            clean_labels[region] = new_id
            new_id += 1

    return clean_labels, distance, marker_mask


# ---- Main processing ----
if not image_files:
    print("No images found in 'images' folder")

else:
    for img_path in image_files:
        print(f"Processing: {img_path.name}")

        base_name = img_path.stem

        img = io.imread(str(img_path), as_gray=True)
        img = get_middle_fifth(img)
        img = np.asarray(img, dtype=np.float32)

        if img.max() > 1.0:
            img = img / 255.0

        # ---- Denoise ----
        denoised = restoration.denoise_nl_means(
            img,
            h=0.08,
            fast_mode=True,
            patch_size=5,
            patch_distance=6
        )

        # ---- CLAHE ----
        clahe = exposure.equalize_adapthist(denoised, clip_limit=0.03)

        # ---- Top-hat ----
        selem_bg = disk(50)
        tophat = morphology.white_tophat(clahe, selem_bg)

        # ---- Adaptive threshold ----
        thr = threshold_sauvola(tophat, window_size=51)
        bw = tophat > thr
        bw = clean_binary_mask(bw, min_size=100, max_size=500)

        # ---- Felzenszwalb ----
        labels = segmentation.felzenszwalb(
            clahe,
            scale=100,
            sigma=0.5,
            min_size=100
        )
        labels[~bw] = 0
        clean_labels = clean_labeled_regions(
            labels,
            min_area=100,
            max_area=10000,
            max_size=500
        )

        # ---- LBP + KMeans ----
        radius = 3
        n_points = 8 * radius

        lbp = compute_lbp(clahe, P=n_points, R=radius)
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

        # ---- Watershed split ----
        watershed_labels, distance_map, markers = split_touching_objects(
            cluster_foreground,
            min_distance=12,
            min_area=100,
            max_area=10000
        )

        final_binary = (watershed_labels > 0).astype(np.uint8) * 255

        # ---- Measurements ----
        diameters_before = get_object_diameters(
            cluster_foreground.astype(np.uint8) * 255,
            min_area=100
        )
        diameters_after = [
            float(r.equivalent_diameter_area) if hasattr(r, "equivalent_diameter_area")
            else float(r.equivalent_diameter)
            for r in measure.regionprops(watershed_labels)
            if r.area >= 100
        ]

        border_image_before = draw_border(cluster_foreground.astype(np.uint8) * 255)
        border_image_after = draw_border_from_labels(watershed_labels)

        marker_vis = markers.astype(np.uint8) * 255

        # ---- Show + Save steps ----
        show_image(img, "Original", f"{base_name}_01_original.png")
        show_image(denoised, "Denoised", f"{base_name}_02_denoised.png")
        show_image(clahe, "CLAHE", f"{base_name}_03_clahe.png")
        show_image(tophat, "Tophat", f"{base_name}_04_tophat.png")
        show_image(bw, "Binary mask", f"{base_name}_05_binary.png")
        show_image(clean_labels, "Felzenszwalb", f"{base_name}_06_labels.png", cmap="nipy_spectral")
        show_image(lbp, "LBP texture", f"{base_name}_07_lbp.png")
        show_image(km_seg, "KMeans classes", f"{base_name}_08_kmeans.png")
        show_image(cluster_foreground, "Foreground before watershed", f"{base_name}_09_foreground_before_ws.png")
        show_image(distance_map, "Distance transform", f"{base_name}_10_distance_transform.png")
        show_image(marker_vis, "Watershed markers", f"{base_name}_11_markers.png")
        show_image(watershed_labels, "Watershed labels", f"{base_name}_12_watershed_labels.png", cmap="nipy_spectral")
        show_image(border_image_before, "Borders before watershed", f"{base_name}_13_borders_before_ws.png", cmap=None)
        show_image(border_image_after, "Borders after watershed", f"{base_name}_14_borders_after_ws.png", cmap=None)
        show_image(final_binary, "Final binary after watershed", f"{base_name}_15_final_binary_ws.png")

        print(f"Antal objekt före watershed: {len(diameters_before)}")
        print(f"Diametrar före watershed: {diameters_before}")
        print(f"Antal objekt efter watershed: {len(diameters_after)}")
        print(f"Diametrar efter watershed: {diameters_after}")