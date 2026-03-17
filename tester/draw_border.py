import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, segmentation, color


def plot_segmented_with_red_borders(segmented_image, ax=None, show=True):
    """
    Draw red borders around each segmented object in a 2D image and plot it.

    Parameters
    ----------
    segmented_image : array-like
        2D segmented image (binary mask or labeled image).
    ax : matplotlib.axes.Axes, optional
        Existing axis to draw on. If None, a new figure/axis is created.
    show : bool, optional
        If True, calls plt.show() when done.

    Returns
    -------
    tuple
        (fig, ax, output_image), where output_image is RGB with red borders.
    """
    img = np.asarray(segmented_image)
    if img.ndim != 2:
        raise ValueError(f"Expected a 2D segmented image, got shape {img.shape}")

    unique_values = np.unique(img)
    is_binary_like = unique_values.size <= 2

    if is_binary_like:
        labeled = measure.label(img > 0, connectivity=2)
    else:
        labeled = img.astype(np.int32, copy=False)

    base = color.label2rgb(labeled, bg_label=0, bg_color=(0, 0, 0))
    boundaries = segmentation.find_boundaries(labeled, mode="outer")
    output_image = base.copy()
    output_image[boundaries] = [1.0, 0.0, 0.0]  # Red border

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        created_fig = True
    else:
        fig = ax.figure

    ax.imshow(output_image)
    ax.set_title("Segmented objects with red borders")
    ax.axis("off")

    if show and created_fig:
        plt.tight_layout()
        plt.show()

    return fig, ax, output_image
