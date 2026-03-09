import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from skimage import io, exposure, morphology, restoration, segmentation
from skimage.filters import threshold_sauvola
from skimage.morphology import disk
from scipy import ndimage as ndi
from sklearn.cluster import KMeans
from skimage.feature import local_binary_pattern
from skimage.util import img_as_ubyte

# ---- Load images from folder ----
image_folder = Path('images')
image_files = list(image_folder.glob('*.jpg')) + list(image_folder.glob('*.png'))

def compute_lbp(image, P=8, R=1, method='uniform'):
    # convert to integer dtype to suppress the warning
    image = img_as_ubyte(image)          # -> uint8
    # alternatively: image = image.astype(np.uint8, copy=False)

    lbp = local_binary_pattern(image, P, R, method=method)
    return lbp

if not image_files:
    print("No images found in 'images' folder")
else:
    for img_path in image_files:
        print(f"Processing: {img_path.name}")
        img = io.imread(str(img_path), as_gray=True)
        if img.max() > 1.0:
            img = img / 255.0

        # ---- 1) Denoise and contrast correction ----
        denoised = restoration.denoise_nl_means(
            img, h=0.08, fast_mode=True, patch_size=5, patch_distance=6
        )
        clahe = exposure.equalize_adapthist(denoised, clip_limit=0.03)
        selem_bg = disk(50)
        tophat = morphology.white_tophat(clahe, selem_bg)

        # ---- 2) Initial foreground via adaptive/local threshold ----
        window_size = 51
        thr = threshold_sauvola(tophat, window_size=window_size)
        bw = tophat > thr

        # ---- 3) Morphological clean-up ----
        bw = morphology.remove_small_objects(bw, max_size=100)
        bw = morphology.closing(bw, disk(3))
        bw = morphology.opening(bw, disk(2))
        bw = ndi.binary_fill_holes(bw)
        bw = morphology.remove_small_holes(bw, max_size=500)
        bw = morphology.closing(bw, disk(2))

        # ---- 4) Felzenszwalb segmentation ----
        labels = segmentation.felzenszwalb(clahe, scale=100, sigma=0.5, min_size=100)
        labels[~bw] = 0

        # ---- 4b) Per-rock interior cleanup ----
        clean_labels = np.zeros_like(labels, dtype=np.int32)
        new_id = 1
        for lab in np.unique(labels):
            if lab == 0:
                continue
            region = labels == lab
            region = ndi.binary_fill_holes(region)
            region = morphology.remove_small_holes(region, max_size=500)
            area = int(region.sum())
            if 100 < area < 10000:
                clean_labels[region] = new_id
                new_id += 1
        
        # ---- 5) LBP texture + k-means clustering ----
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(clahe, n_points, radius, method='uniform')
        feat = np.stack([clahe.ravel(), lbp.ravel()], axis=1)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(feat)
        km_seg = kmeans.labels_.reshape(clahe.shape)
        cluster_foreground = (km_seg == np.argmax([clahe[km_seg == i].mean() for i in range(2)]))
        new_id = 1
        for lab in np.unique(cluster_foreground):
            if lab == 0:
                continue
            region = cluster_foreground == lab
            region = ndi.binary_fill_holes(region)
            region = morphology.remove_small_holes(region, max_size=500)
            area = int(region.sum())
            if 100 < area < 10000:
                cluster_foreground[region] = new_id
                new_id += 1
            
        #Create outline by finding edges of segmented regions
        #km_seg_outline = np.zeros_like(km_seg)
        #for label_id in np.unique(km_seg):
        #    if label_id == 0:
        #        continue
        #    region = km_seg == label_id
        #    edges = ndi.binary_erosion(region) != region
        #    km_seg_outline[edges] = label_id

        #km_seg = km_seg_outline
        
        

        # ---- Visualization ----
        fig, ax = plt.subplots(3, 2, figsize=(12, 12))

        ax[0, 0].imshow(img, cmap='gray')
        ax[0, 0].set_title('Original')
        ax[0, 0].axis('off')

        ax[0, 1].imshow(tophat, cmap='gray')
        ax[0, 1].set_title('Tophat/Corrected')
        ax[0, 1].axis('off')

        ax[1, 0].imshow(bw, cmap='gray')
        ax[1, 0].set_title('Binary (filled/cleaned)')
        ax[1, 0].axis('off')

        ax[1, 1].imshow(clean_labels, cmap='nipy_spectral')
        ax[1, 1].set_title('Felzenszwalb + per-rock cleanup')
        ax[1, 1].axis('off')

        ax[2, 0].imshow(cluster_foreground, cmap='gray')
        ax[2, 0].set_title('LBP + k-means Clustering')
        ax[2, 0].axis('off')

        ax[2, 1].imshow(km_seg, cmap='gray')
        ax[2, 1].set_title('LBP + k-means (outline)')
        ax[2, 1].axis('off')

        plt.tight_layout()
        plt.show()
