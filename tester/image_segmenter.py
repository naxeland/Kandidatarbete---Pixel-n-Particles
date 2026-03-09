import numpy as np
from pathlib import Path

from scipy import ndimage as ndi
from skimage import io, measure, morphology, segmentation
from skimage.feature import local_binary_pattern, peak_local_max
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from sklearn.cluster import KMeans
from img_split import get_middle_fifth


def _load_as_grayscale_middle_fifth(image):
    """
    Load input as grayscale float image in range [0, 1] using the middle-fifth crop.
    """
    imge = get_middle_fifth(image)

    if isinstance(imge, (str, Path)):
        img = io.imread(str(imge), as_gray=True)
    else:
        img = np.asarray(imge)

    if img.ndim == 3:
        img = img.mean(axis=2)

    img = img.astype(np.float32, copy=False)
    if img.max() > 1.0:
        img = img / 255.0

    return img


def _segment_foreground_kmeans(img, radius=3):
    """
    Create a coarse foreground mask from intensity + LBP texture using KMeans.
    """
    n_points = 8 * radius
    lbp = local_binary_pattern(img_as_ubyte(img), n_points, radius, method="uniform")
    feat = np.stack([img.ravel(), lbp.ravel()], axis=1)

    kmeans = KMeans(n_clusters=2, random_state=0).fit(feat)
    km_seg = kmeans.labels_.reshape(img.shape)

    means = [img[km_seg == i].mean() if np.any(km_seg == i) else 0.0 for i in range(2)]
    cluster_foreground = km_seg == int(np.argmax(means))
    return cluster_foreground


def _cleanup_mask(mask, min_object_size=100, min_hole_size=500, smooth_radius=1):
    """
    Remove noise, fill holes, and lightly smooth edges.
    """
    cleaned = morphology.remove_small_objects(mask, max_size=min_object_size)
    cleaned = morphology.remove_small_holes(cleaned, max_size=min_hole_size)

    if smooth_radius > 0:
        selem = disk(smooth_radius)
        cleaned = morphology.opening(cleaned, footprint=selem)
        cleaned = morphology.closing(cleaned, footprint=selem)

    return cleaned


def _normalize_mask(mask):
    """
    Normalize model/binary masks to boolean foreground.
    """
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.dtype == np.bool_:
        return mask
    return mask > 0


def split_touching_rocks(mask, min_peak_distance=8):
    """
    Split touching foreground regions into instances using marker-controlled watershed.

    Parameters:
        mask: np.ndarray (bool)
            Foreground rock mask.
        min_peak_distance: int
            Minimum distance between watershed markers in pixels.

    Returns:
        np.ndarray
            Instance label image, where 0 is background and each rock has a unique id.
    """
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return np.zeros(mask.shape, dtype=np.int32)

    distance = ndi.distance_transform_edt(mask)
    peak_coords = peak_local_max(
        distance,
        labels=mask,
        min_distance=max(1, int(min_peak_distance)),
        exclude_border=False,
    )

    markers = np.zeros(mask.shape, dtype=np.int32)
    if len(peak_coords) == 0:
        return measure.label(mask, connectivity=2)

    for idx, (r, c) in enumerate(peak_coords, start=1):
        markers[r, c] = idx

    labels = segmentation.watershed(-distance, markers, mask=mask)
    return labels.astype(np.int32, copy=False)


def measure_rock_instances(instance_labels):
    """
    Measure per-rock geometry from an instance label image.

    Parameters:
        instance_labels: np.ndarray
            Label image with one id per rock.

    Returns:
        list[dict]
            Per-rock measurements (area, equivalent diameter, major/minor axis, etc).
    """
    measurements = []
    for region in measure.regionprops(instance_labels):
        measurements.append(
            {
                "id": int(region.label),
                "area_px": int(region.area),
                "equivalent_diameter_px": float(region.equivalent_diameter_area),
                "major_axis_length_px": float(region.axis_major_length),
                "minor_axis_length_px": float(region.axis_minor_length),
                "perimeter_px": float(region.perimeter),
                "centroid_row": float(region.centroid[0]),
                "centroid_col": float(region.centroid[1]),
            }
        )
    return measurements


def segment_rocks_from_mask(
    predicted_mask,
    min_object_size=100,
    min_hole_size=500,
    smooth_radius=1,
    min_peak_distance=8,
):
    """
    Post-process a predicted rock mask and split touching rocks into instances.

    Returns:
        dict with keys:
            - mask_u8: binary uint8 mask (0 background, 255 foreground)
            - instance_labels: int32 label map (0 background, 1..N rocks)
            - measurements: list of per-rock measurement dicts
    """
    coarse_mask = _normalize_mask(predicted_mask)
    cleaned_mask = _cleanup_mask(
        coarse_mask,
        min_object_size=min_object_size,
        min_hole_size=min_hole_size,
        smooth_radius=smooth_radius,
    )
    instance_labels = split_touching_rocks(cleaned_mask, min_peak_distance=min_peak_distance)
    measurements = measure_rock_instances(instance_labels)

    return {
        "mask_u8": cleaned_mask.astype(np.uint8) * 255,
        "instance_labels": instance_labels,
        "measurements": measurements,
    }


def segment_rocks_instances(
    image=None,
    predicted_mask=None,
    backend="kmeans",
    min_object_size=100,
    min_hole_size=500,
    smooth_radius=1,
    min_peak_distance=8,
):
    """
    Segment rocks and split touching rocks into per-instance labels.

    Supported backends:
        - "kmeans": legacy unsupervised mask from intensity+LBP
        - "mask": use `predicted_mask` from YOLO-seg / Mask R-CNN

    For Recommendation A in production, pass backend="mask" and supply
    your model's predicted foreground mask.
    """
    if backend == "mask":
        if predicted_mask is None:
            raise ValueError("backend='mask' requires predicted_mask.")
        return segment_rocks_from_mask(
            predicted_mask=predicted_mask,
            min_object_size=min_object_size,
            min_hole_size=min_hole_size,
            smooth_radius=smooth_radius,
            min_peak_distance=min_peak_distance,
        )

    if backend != "kmeans":
        raise ValueError(f"Unsupported backend: {backend}")
    if image is None:
        raise ValueError("backend='kmeans' requires image.")

    img = _load_as_grayscale_middle_fifth(image)
    coarse_mask = _segment_foreground_kmeans(img)
    return segment_rocks_from_mask(
        predicted_mask=coarse_mask,
        min_object_size=min_object_size,
        min_hole_size=min_hole_size,
        smooth_radius=smooth_radius,
        min_peak_distance=min_peak_distance,
    )


def segment_image(image):
    """
    Run the segmentation pipeline on a single image and return a binary mask.

    Parameters:
        image: numpy.ndarray or str or pathlib.Path
            Input image array or path to an image file.

    Returns:
        numpy.ndarray
            Binary uint8 mask (0 background, 255 foreground).
    """
    result = segment_rocks_instances(image=image, backend="kmeans")
    return result["mask_u8"]
