import numpy as np
from pathlib import Path
from skimage import io


VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def load_images_from_folder(folder_name: str = "images") -> list[np.ndarray]:
    """
    Load images from a folder and return them as numpy arrays.

    Parameters:
        folder_name: str
            Folder name or path to load from. Defaults to "images".

    Returns:
        list[np.ndarray]
            Loaded image arrays.
    """
    folder = Path(folder_name)
    if not folder.is_absolute():
        folder = Path.cwd() / folder

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    image_paths = sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    )

    if not image_paths:
        raise ValueError(f"No images found in folder: {folder}")

    return [io.imread(str(path)) for path in image_paths]


def get_middle_fifth(image: np.ndarray) -> np.ndarray:
    """
    Split an image into 5 vertical columns and return the middle column.

    Parameters:
        image: np.ndarray
            Input image array with shape (H, W) or (H, W, C).

    Returns:
        np.ndarray
            The middle fifth of the image.
    """
    if image.ndim < 2:
        raise ValueError("image must be at least 2D with shape (H, W) or (H, W, C)")

    width = image.shape[1]
    col_width = width // 5

    if col_width == 0:
        raise ValueError("image width must be at least 5 pixels")

    start = 2 * col_width
    end = 3 * col_width if 3 * col_width <= width else width
    return image[:, start:end]
