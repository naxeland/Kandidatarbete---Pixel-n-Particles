import os
import numpy as np
import matplotlib.pyplot as plt
from image_segmenter import segment_rocks_instances
from skimage import io, morphology, measure

from test4 import img_seg


# -------- Core Processing Function --------
def white_borders_area_multi(
    img,
    min_object_size=100,
    min_hole_size=500,
    smooth_radius=1,
    min_peak_distance=8,
    backend="kmeans",
    predicted_mask=None,
    border_thickness_px=1,
    use_otsu=True,
    manual_threshold=None,
):
    segmentation_result = segment_rocks_instances(
        image=img,
        predicted_mask=predicted_mask,
        backend=backend,
        min_object_size=min_object_size,
        min_hole_size=min_hole_size,
        smooth_radius=smooth_radius,
        min_peak_distance=min_peak_distance,
    )
    segmented_img = segmentation_result["mask_u8"]
    instance_labels = segmentation_result["instance_labels"]
    rock_measurements = segmentation_result["measurements"]
    segmentation_mask = segmented_img > 0
    white = segmentation_mask.copy()

    # Cleanup
    white = morphology.remove_small_objects(white, max_size=min_object_size)
    white = morphology.remove_small_holes(white, max_size=min_object_size)

    # Extract borders (morphological gradient)
    selem = morphology.disk(max(1, border_thickness_px))
    dil = morphology.dilation(white, footprint=selem)
    ero = morphology.erosion(white, footprint=selem)
    border = np.logical_and(dil, np.logical_not(ero))

    # Label each border component
    labels = measure.label(border, connectivity=2)

    total_area = int(border.sum())

    per_shape = {}
    for region in measure.regionprops(labels):
        per_shape[region.label] = int(region.area)

    return (
        total_area,
        per_shape,
        border,
        labels,
        segmented_img,
        segmentation_mask,
        instance_labels,
        rock_measurements,
    )


def _compute_size_distribution(areas, bins=5):
    if not areas:
        return []
    counts, edges = np.histogram(areas, bins=bins)
    dist = []
    for i, count in enumerate(counts):
        low, high = edges[i], edges[i + 1]
        dist.append((float(low), float(high), int(count)))
    return dist


def plot_border_areas(segmented_img, border, title, segmentation_mask, instance_labels):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # Show image after image_segmenter processing
    axes[0].imshow(img_seg(img), cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Segmented image")
    axes[0].axis("off")

    # Show processed image with border overlay
    axes[1].imshow(segmentation_mask, cmap="gray")

    overlay = np.zeros((*border.shape, 4), dtype=np.float32)
    overlay[border, 0] = 1.0
    overlay[border, 3] = 0.65
    axes[1].imshow(overlay)
    axes[1].set_title(title)
    axes[1].axis("off")

    axes[2].imshow(instance_labels, cmap="nipy_spectral")
    axes[2].set_title("Rock instances")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.show()


# -------- Batch Processing from Folder --------
def process_folder(folder_path="images"):
    valid_extensions = (".png", ".jpg", ".jpeg", ".tif", ".bmp")

    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' not found.")
        return

    files = [f for f in os.listdir(folder_path)
             if f.lower().endswith(valid_extensions)]

    if not files:
        print("No images found in folder.")
        return

    for filename in files:
        path = os.path.join(folder_path, filename)
        print(f"\nProcessing: {filename}")

        img = io.imread(path)

        (
            total_area,
            per_shape,
            border,
            labels,
            segmented_img,
            segmentation_mask,
            instance_labels,
            rock_measurements,
        ) = white_borders_area_multi(
            img,
            min_object_size=100,
            min_hole_size=500,
            smooth_radius=1,
            min_peak_distance=8,
            backend="kmeans",
            border_thickness_px=1,
            use_otsu=True,
        )

        print(f"  Total border area: {total_area} px")

        for label, area in sorted(per_shape.items(), key=lambda x: x[1], reverse=True):
            print(f"    Shape {label}: {area} px")

        rock_count = len(rock_measurements)
        areas = [m["area_px"] for m in rock_measurements]
        print(f"  Rock count: {rock_count}")
        if areas:
            print(
                f"  Rock area stats (px): min={min(areas)}, max={max(areas)}, "
                f"mean={np.mean(areas):.1f}, median={np.median(areas):.1f}"
            )
            print("  Size distribution (area bins, px):")
            for low, high, count in _compute_size_distribution(areas, bins=5):
                print(f"    [{low:.1f}, {high:.1f}): {count}")

        plot_border_areas(
            segmented_img=segmented_img,
            border=border,
            title=f"{filename} - White border areas",
            segmentation_mask=segmentation_mask,
            instance_labels=instance_labels,
        )


# -------- Run --------
if __name__ == "__main__":
    process_folder("images")
