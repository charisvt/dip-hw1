import numpy as np
from PIL import Image

def calculate_hist_of_img(img_array, return_normalized):
    # Initialize histogram dictionary with all intensity levels
    hist = {i: 0 for i in range(256)}

    # Calculate histogram by iterating over image
    for pixel_value in img_array.flatten():
        intensity = int(pixel_value * 255) # Convert pixel value to intensity level
        hist[intensity] += 1 # Increment the count for the corresponding intensity level
    
    # If normalization is required
    if return_normalized:
        # Normalize the histogram by dividing by the total number of pixels
        total_pixels = img_array.size
        hist = {k: v / total_pixels for k, v in hist.items()}
    return hist

def apply_hist_modification_transform(img_array, modification_transform):
    transformed_img = np.copy(img_array)
    
    # Create a lookup table for all possible intensity values (0-255)
    # to avoid conditional checks inside the nested loops
    lookup = {}
    for i in range(256):
        if i in modification_transform:
            lookup[i] = modification_transform[i]
        else:
            # Find closest intensity level that has a mapping
            closest_level = min(modification_transform.keys(), key=lambda x: abs(x - i))
            lookup[i] = modification_transform[closest_level]
    
    # Iterate through the image array and apply the transformation
    for i in range(transformed_img.shape[0]):
        for j in range(transformed_img.shape[1]):
            # Convert pixel value to intensity (0-255 range) and apply the transformation
            pixel_value = int(transformed_img[i, j] * 255)  # Convert to 0-255 range
            transformed_img[i, j] = lookup[pixel_value] / 255.0  # Convert back to 0-1 range
    
    return transformed_img

# Image path
filename = "input_img.jpg"

# Read image into a PIL entity
img = Image.open(fp=filename)

# Keep only Luminance component of the image
bw_img = img.convert("L")

# create a numpy array from the image
img_array = np.array(bw_img).astype(float) / 255.0