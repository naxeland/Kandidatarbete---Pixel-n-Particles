import numpy as np
import matplotlib.pyplot as plt
from skimage import measure

try:
    from .pixel_diameter import get_object_diameters
except ImportError:
    from pixel_diameter import get_object_diameters


def plot_segmented_with_diameters(
    segmented_image,
    decimals=2,
    cmap="gray",
    line_color="cyan",
    text_color="white",
    ax=None,
    show=True,
):
    """
    Plot a 2D segmented image and annotate each object with its diameter in pixels.

    Parameters
    ----------
    segmented_image : array-like
        2D segmented image (binary mask or labeled image).
    decimals : int, optional
        Number of decimals shown for each diameter.
    cmap : str, optional
        Matplotlib colormap used for the displayed mask.
    line_color : str, optional
        Color of connector lines.
    text_color : str, optional
        Color of annotation text.
    ax : matplotlib.axes.Axes, optional
        Existing axis to draw on. If None, a new figure/axis is created.
    show : bool, optional
        If True, calls plt.show() when done.

    Returns
    -------
    tuple
        (fig, ax, diameters), where diameters is list[float].
    """
    img = np.asarray(segmented_image)
    if img.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {img.shape}")

    unique_values = np.unique(img)
    is_binary_like = unique_values.size <= 2

    if is_binary_like:
        mask = img > 0
        labeled = measure.label(mask, connectivity=2)
        display_img = mask
    else:
        labeled = img.astype(np.int32, copy=False)
        display_img = labeled > 0

    regions = measure.regionprops(labeled)
    diameters = get_object_diameters(img)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        created_fig = True
    else:
        fig = ax.figure

    ax.imshow(display_img, cmap=cmap)
    ax.set_title("Segmented objects with pixel diameters")
    ax.axis("off")

    n = max(len(regions), 1)
    for i, (region, diameter) in enumerate(zip(regions, diameters)):
        cy, cx = region.centroid
        angle = (2 * np.pi * i) / n
        offset = max(12, int(diameter * 0.9))
        tx = cx + offset * np.cos(angle)
        ty = cy + offset * np.sin(angle)

        ax.annotate(
            f"{diameter:.{decimals}f} px",
            xy=(cx, cy),
            xytext=(tx, ty),
            color=text_color,
            fontsize=9,
            ha="center",
            va="center",
            arrowprops={"arrowstyle": "-", "color": line_color, "lw": 1.2},
            bbox={"facecolor": "black", "alpha": 0.6, "edgecolor": "none", "pad": 2},
        )

    if show and created_fig:
        plt.tight_layout()
        plt.show()

    return fig, ax, diameters
