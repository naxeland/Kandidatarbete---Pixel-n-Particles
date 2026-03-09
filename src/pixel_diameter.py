import numpy as np
from skimage import measure


def get_object_diameters(segmented_image):
    """
    Calculate equivalent diameters (in pixels) for all segmented objects in a 2D image.

    Parameters
    ----------
    segmented_image : array-like
        2D segmented image. Can be:
        - binary mask (0/1, False/True, or 0/255), or
        - labeled image with integer object IDs (>0 as foreground).

    Returns
    -------
    list[float]
        Equivalent diameters in pixels for each detected object.
    """
    img = np.asarray(segmented_image)

    if img.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {img.shape}")

    # If image is binary-like, label connected components first.
    unique_values = np.unique(img)
    is_binary_like = unique_values.size <= 2

    if is_binary_like:
        binary = img > 0
        labeled = measure.label(binary, connectivity=2)
    else:
        labeled = img.astype(np.int32, copy=False)

    regions = measure.regionprops(labeled)
    diameters = []
    for region in regions:
        if hasattr(region, "equivalent_diameter_area"):
            diameters.append(float(region.equivalent_diameter_area))
        else:
            diameters.append(float(region.equivalent_diameter))
    return diameters
