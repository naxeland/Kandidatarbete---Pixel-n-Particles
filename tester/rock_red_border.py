from pathlib import Path
import argparse
import numpy as np
from skimage import io, segmentation
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt

from image_segmenter import segment_image

VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _middle_fifth_slice(width: int) -> tuple[int, int]:
    """Return start/end x indices for the middle fifth of an image width."""
    col_width = width // 5
    start = 2 * col_width
    end = min(3 * col_width, width)
    return start, end


def _align_mask_to_image(mask: np.ndarray, image_shape: tuple[int, ...]) -> np.ndarray:
    """
    Align a 2D mask to full image size.
    If mask is already full-size it is returned as-is.
    If mask is middle-fifth width, place it in that slice of full-size canvas.
    """
    h, w = image_shape[:2]
    if mask.shape == (h, w):
        return mask

    if mask.shape[0] != h:
        raise ValueError(
            f"Mask/image height mismatch: mask={mask.shape}, image={(h, w)}"
        )

    start, end = _middle_fifth_slice(w)
    expected_w = end - start
    if mask.shape[1] != expected_w:
        raise ValueError(
            f"Mask width does not match image middle-fifth width: "
            f"mask={mask.shape}, expected width={expected_w} for image width={w}"
        )

    aligned = np.zeros((h, w), dtype=bool)
    aligned[:, start:end] = mask
    return aligned


def _to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """Convert grayscale/RGB/RGBA image to uint8 RGB for drawing."""
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    elif image.ndim != 3 or image.shape[2] not in (3,):
        raise ValueError("Unsupported image shape. Expected (H, W), (H, W, 3), or (H, W, 4).")

    if image.dtype != np.uint8:
        image = img_as_ubyte(image)

    return image.copy()


def rocks_with_red_border(image: np.ndarray) -> np.ndarray:
    """
    Segment rocks and return an image with red borders around all rocks.

    Parameters:
        image: np.ndarray
            Input image array.

    Returns:
        np.ndarray
            RGB uint8 image with red rock borders.
    """
    rock_mask = segment_image(image) > 0
    border_mask = segmentation.find_boundaries(rock_mask, mode="outer")
    border_mask = _align_mask_to_image(border_mask, image.shape)

    out = _to_rgb_uint8(image)
    out[border_mask] = [255, 0, 0]
    return out


def process_image(input_path: str, output_path: str | None = None) -> str:
    """
    Read an input image, draw red rock borders, and save result.
    """
    input_file = Path(input_path)
    image = io.imread(str(input_file))
    output_image = rocks_with_red_border(image)

    if output_path is None:
        output_file = input_file.with_name(f"{input_file.stem}_red_borders.png")
    else:
        output_file = Path(output_path)

    io.imsave(str(output_file), output_image)
    return str(output_file)


def process_folder(folder_path: str = "images", save_output: bool = False) -> None:
    """
    Process all images in a folder, plot bordered results, and optionally save them.
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder.resolve()}")

    image_files = sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    )
    if not image_files:
        raise ValueError(f"No images found in folder: {folder.resolve()}")

    for img_path in image_files:
        image = io.imread(str(img_path))
        output_image = rocks_with_red_border(image)

        if save_output:
            output_file = img_path.with_name(f"{img_path.stem}_red_borders.png")
            io.imsave(str(output_file), output_image)
            print(f"Saved bordered image to: {output_file}")

        plt.figure(figsize=(8, 6))
        plt.imshow(output_image)
        plt.title(f"{img_path.name} - Red rock borders")
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add red borders around rocks for one image or all images in a folder."
    )
    parser.add_argument(
        "input_image",
        nargs="?",
        default=None,
        help="Path to input rock image (optional if using --folder)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output image path (default: <input_name>_red_borders.png)",
    )
    parser.add_argument(
        "--folder",
        default=None,
        help="Process all images in a folder (example: images)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="When using --folder, save processed images to disk.",
    )
    args = parser.parse_args()

    if args.folder:
        process_folder(args.folder, save_output=args.save)
    elif args.input_image:
        saved_to = process_image(args.input_image, args.output)
        print(f"Saved bordered image to: {saved_to}")

        output_image = io.imread(saved_to)
        plt.figure(figsize=(8, 6))
        plt.imshow(output_image)
        plt.title(f"{Path(saved_to).name} - Red rock borders")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    else:
        process_folder("images", save_output=args.save)
