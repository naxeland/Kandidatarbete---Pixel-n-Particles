import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from skimage import io, exposure, morphology, measure, restoration
from skimage.filters import threshold_sauvola
from skimage.morphology import remove_small_objects, disk, local_maxima
from scipy import ndimage as ndi
from sklearn.cluster import KMeans
from skimage.feature import local_binary_pattern
from skimage.segmentation import watershed

# ---- Load images from folder ----
image_folder = Path('images')
image_files = list(image_folder.glob('*.jpg')) + list(image_folder.glob('*.png'))

if not image_files:
    print("No images found in 'images' folder")
else:
    for img_path in image_files:
        print(f"Processing: {img_path.name}")
        img = io.imread(str(img_path), as_gray=True)
        if img.max() > 1.0:
            img = img / 255.0

        # ---- 1) Denoise and contrast correction ----
        denoised = restoration.denoise_nl_means(img, h=0.08, fast_mode=True, patch_size=5, patch_distance=6, channel_axis=None)
        clahe = exposure.equalize_adapthist(denoised, clip_limit=0.03)

        selem = disk(50)
        tophat = morphology.white_tophat(clahe, selem)

        # ---- 2) Initial foreground via adaptive threshold ----
        window_size = 51
        sauvola = threshold_sauvola(tophat, window_size=window_size)
        bw_sauvola = tophat > sauvola

        # ---- 3) Morphological clean-up ----
        bw = remove_small_objects(bw_sauvola, min_size=200)
        bw = morphology.closing(bw, morphology.disk(3))
        bw = morphology.opening(bw, morphology.disk(2))

        # ---- 4) Separate touching rocks: distance transform + watershed ----
        distance = ndi.distance_transform_edt(bw)
        local_maxi = local_maxima(distance)
        markers, _ = ndi.label(local_maxi)
        labels = watershed(-distance, markers, mask=bw)

        props = measure.regionprops(labels)
        clean_labels = np.zeros_like(labels)
        label_id = 1
        for p in props:
            if 100 < p.area < 10000:
                clean_labels[labels == p.label] = label_id
                label_id += 1

        # ---- 5) LBP texture + k-means clustering ----
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(clahe, n_points, radius, method='uniform')
        feat = np.stack([clahe.ravel(), lbp.ravel()], axis=1)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(feat)
        km_seg = kmeans.labels_.reshape(clahe.shape)
        cluster_foreground = (km_seg == np.argmax([clahe[km_seg == i].mean() for i in range(2)]))
        
        

        # ---- Visualization (Step 5 only) ----
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.suptitle(f"{img_path.name} - LBP + K-means")
        ax.imshow(km_seg, cmap='gray')
        ax.set_title('LBP + k-means Clustering')
        ax.axis('off')
        plt.tight_layout()
        plt.show()
