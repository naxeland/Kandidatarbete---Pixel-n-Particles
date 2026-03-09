import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage import color, io

from image_segmenter import segment_rocks_instances
from img_split import get_middle_fifth

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def process_and_plot(
    image_path,
    min_object_size=100,
    min_hole_size=500,
    smooth_radius=1,
    min_peak_distance=8,
):
    """Run segmentation and plot each segmented object's pixel area."""
    image = io.imread(str(image_path))

    result = segment_rocks_instances(
        image=image,
        backend="kmeans",
        min_object_size=min_object_size,
        min_hole_size=min_hole_size,
        smooth_radius=smooth_radius,
        min_peak_distance=min_peak_distance,
    )

    instance_labels = result["instance_labels"]
    measurements = result["measurements"]

    base = color.label2rgb(instance_labels, bg_label=0, image=np.zeros_like(instance_labels), alpha=0.7)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(get_middle_fifth(image), cmap="gray" if image.ndim == 2 else None)
    axes[0].set_title(f"Original ({Path(image_path).name})")
    axes[0].axis("off")

    axes[1].imshow(base)
    axes[1].set_title("Segmented (No Area Labels)")
    axes[1].axis("off")

    axes[2].imshow(base)
    axes[2].set_title("Segmented with Pixel Areas")
    axes[2].axis("off")

    for m in measurements:
        row = m["centroid_row"]
        col = m["centroid_col"]
        area = m["area_px"]
        axes[2].text(
            col,
            row,
            f"{area} px",
            color="white",
            fontsize=8,
            ha="center",
            va="center",
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 1, "edgecolor": "none"},
        )

    print(f"Detected objects: {len(measurements)}")
    for m in measurements:
        print(f"Object {m['id']}: {m['area_px']} px")

    plt.tight_layout()
    plt.show()


def process_folder(folder_path, min_object_size, min_hole_size, smooth_radius, min_peak_distance):
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    image_paths = sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No supported image files in folder: {folder}")

    for image_path in image_paths:
        process_and_plot(
            image_path=image_path,
            min_object_size=min_object_size,
            min_hole_size=min_hole_size,
            smooth_radius=smooth_radius,
            min_peak_distance=min_peak_distance,
        )


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Segment an image and show pixel area for each segmented object."
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        type=str,
        help="Path to input image. If omitted, all images in --folder are processed.",
    )
    parser.add_argument("--folder", type=str, default="images", help="Folder containing input images")
    parser.add_argument("--min-object-size", type=int, default=100)
    parser.add_argument("--min-hole-size", type=int, default=500)
    parser.add_argument("--smooth-radius", type=int, default=1)
    parser.add_argument("--min-peak-distance", type=int, default=8)
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.image:
        process_and_plot(
            image_path=args.image,
            min_object_size=args.min_object_size,
            min_hole_size=args.min_hole_size,
            smooth_radius=args.smooth_radius,
            min_peak_distance=args.min_peak_distance,
        )
    else:
        process_folder(
            folder_path=args.folder,
            min_object_size=args.min_object_size,
            min_hole_size=args.min_hole_size,
            smooth_radius=args.smooth_radius,
            min_peak_distance=args.min_peak_distance,
        )


if __name__ == "__main__":
    main()
