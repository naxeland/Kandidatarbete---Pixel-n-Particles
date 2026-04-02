"""
rock_measurements.py
--------------------
Reads the binary masks produced by rock_segmentation.py and computes
per-rock pixel areas and equivalent diameters, saved as CSV files in
results_output/.

Run this independently of rock_segmentation.py — useful for re-running
with different MIN_AREA thresholds without redoing the slow segmentation.
"""

import numpy as np
from pathlib import Path
from skimage import io, measure

# ---- Folders ----
masks_folder   = Path("masks_output")    # written by rock_segmentation.py
results_folder = Path("results_output")
results_folder.mkdir(exist_ok=True)

mask_files = sorted(masks_folder.glob("*_mask.png"))

# ---- Settings ----
# Adjust MIN_AREA freely here without re-running segmentation.
MIN_AREA = 200


# ---- Helpers ----
def get_region_areas_and_diameters(binary_image, min_area=0):
    """Label connected regions in a binary image and return their areas
    and equivalent diameters (in pixels)."""
    labeled  = measure.label(binary_image > 0, connectivity=2)
    regions  = measure.regionprops(labeled)
    areas, diameters = [], []
    for region in regions:
        if region.area < min_area:
            continue
        areas.append(float(region.area))
        if hasattr(region, "equivalent_diameter_area"):
            diameters.append(float(region.equivalent_diameter_area))
        else:
            diameters.append(float(region.equivalent_diameter))
    return areas, diameters


def save_sorted_values(values, filepath, column_name):
    values = np.sort(np.array(values, dtype=float))
    np.savetxt(filepath, values, delimiter=",",
               header=column_name, comments="", fmt="%.6f")


# ---- Main ----
if not mask_files:
    print("No mask files found in 'masks_output'. Run rock_segmentation.py first.")
else:
    for mask_path in mask_files:
        # Strip the '_mask' suffix to recover the original image name
        base_name = mask_path.stem.replace("_mask", "")
        print(f"\nMeasuring: {mask_path.name}  (source: {base_name})")

        mask = io.imread(str(mask_path), as_gray=True)

        areas, diameters = get_region_areas_and_diameters(mask, min_area=MIN_AREA)

        save_sorted_values(
            areas,
            results_folder / f"{base_name}_pixel_areas.csv",
            "pixel_area"
        )
        save_sorted_values(
            diameters,
            results_folder / f"{base_name}_diameters.csv",
            "equivalent_diameter_pixels"
        )

        print(f"  Objects detected : {len(diameters)}")
        print(f"  Diameters (px)   : {np.sort(np.array(diameters)).round(1).tolist()}")
        print(f"  Areas (px^2)    : {np.sort(np.array(areas)).round(1).tolist()}")
        print(f"  Results saved    → {results_folder / base_name}_*.csv")