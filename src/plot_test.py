import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.feature import local_binary_pattern
from skimage.util import img_as_ubyte
from img_split import get_middle_fifth

import draw_border
import test4

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

        img = get_middle_fifth(img)
            
        denoise, clahe = test4.denoise_contrast(img)
        

        segmented = test4.k_cluster(clahe, test4.adaptive_threshold(denoise, clahe))
        #segmented = test4.k_cluster(clahe, denoise)

        borderedImage = test4.draw_border(segmented)

        #diameters = test4.get_object_diameters(borderedImage)

        dia = test4.measure_dia(borderedImage)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Original image")
        axes[0].axis("off")

        axes[1].imshow(borderedImage, cmap="gray")
        axes[1].set_title("Segmented image with borders")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()
        
        #plot_segmented_with_diameters(segmented, decimals=1, cmap="viridis", line_color="cyan", text_color="white", show=True)
        #draw_border.plot_segmented_with_red_borders(segmented, show=True)
        
        
        
