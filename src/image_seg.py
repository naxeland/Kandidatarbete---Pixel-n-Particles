
import matplotlib.pyplot as plt
import os
from skimage import io, segmentation, color, feature, filters
import scipy as sp
import scipy.ndimage as ski
import numpy as np
from skimage.feature import local_binary_pattern

# Load and segment all images
# Set the folder path containing images
image_folder = 'images'
# Get list of image files with supported extensions
image_files = [f for f in os.listdir(image_folder) 
               if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

# Create subplots for original and segmented images
fig, axes = plt.subplots(len(image_files), 2, figsize=(12, 5 * len(image_files)))
# Reshape axes if only one image to maintain consistent indexing
if len(image_files) == 1:
    axes = axes.reshape(1, -1)
elif len(image_files) > 1:
    pass
else:
    axes = axes.reshape(1, -1)
# Process each image in the folder
for idx, image_file in enumerate(image_files):
    # Construct full image path and read the image
    image_path = os.path.join(image_folder, image_file)
    image = io.imread(image_path)
    
    # Convert RGBA to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = color.rgba2rgb(image)
    
    # Convert image to grayscale for processing
    gray_image = color.rgb2gray(image)
    
    # Apply Felzenszwalb segmentation algorithm
    segments = segmentation.felzenszwalb(gray_image, scale=250, sigma=0.8, min_size=50)
    
    # Convert segments to colored image with averaged colors
    segmented = color.label2rgb(segments, gray_image, kind='avg')
    # Find segment boundaries
    boundaries = segmentation.find_boundaries(segments, mode='thick')
    # Add black borders to segmented image
    segmented[boundaries] = [0, 0, 0]
    
    # Detect edges using Canny edge detection
    edges = feature.canny(gray_image, sigma=1.0)
    # Fill holes in edge detection
    fill_rocks = sp.ndimage.binary_fill_holes(edges)
    # Label connected components
    label_objects, nb_labels = sp.ndimage.label(fill_rocks)
    # Count pixels in each label
    sizes = np.bincount(label_objects.ravel())
    # Create mask for labels with size > 20 pixels
    mask_sizes = sizes > 20
    mask_sizes[0] = 0
    # Apply size filter to cleaned rocks mask
    rocks_cleaned = mask_sizes[label_objects]
    
    # Display original grayscale image
    axes[idx, 0].imshow(gray_image, cmap='gray')
    axes[idx, 0].set_title(f'{image_file} - Original')
    # Display segmented image with cleaned rocks overlay
    axes[idx, 1].imshow(segmented)
    axes[idx, 1].set_title(f'{image_file} - Segmented')
    #axes[idx, 1].imshow(rocks_cleaned, cmap='gray', alpha=0.5)

# Adjust subplot spacing and display
plt.tight_layout()
plt.show()

