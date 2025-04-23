import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from hist_utils import calculate_hist_of_img
from hist_modif import perform_hist_eq, perform_hist_matching

def plot_image_with_histogram(img_array, title, ax_img, ax_hist):
    """Plot an image and its histogram."""
    # Plot image
    ax_img.imshow(img_array, cmap='gray')
    ax_img.set_title(title)
    ax_img.axis('off')
    
    # Calculate and plot histogram
    hist = calculate_hist_of_img(img_array, return_normalized=True)
    ax_hist.bar(hist.keys(), hist.values(), width=1, color='gray')
    ax_hist.set_xlim(0, 255)
    ax_hist.set_title(f'Histogram of {title}')
    ax_hist.set_xlabel('Intensity')
    ax_hist.set_ylabel('Frequency')

def main():
    # Load input and reference images
    input_img = Image.open('input_img.jpg').convert('L')
    ref_img = Image.open('ref_img.jpg').convert('L')
    
    # Convert to numpy arrays and normalize to [0, 1]
    input_array = np.array(input_img).astype(float) / 255.0
    ref_array = np.array(ref_img).astype(float) / 255.0
    
    # List of processing modes
    modes = ['greedy', 'non-greedy', 'post-disturbance']
    
    # Part 1: Histogram Equalization
    print("Performing Histogram Equalization...")
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    fig.suptitle('Histogram Equalization', fontsize=16)
    
    # Plot original image
    plot_image_with_histogram(input_array, 'Original Image', axes[0, 0], axes[0, 1])
    
    # Process image with each mode
    for i, mode in enumerate(modes, 1):
        print(f"Processing with {mode} mode...")
        processed_img = perform_hist_eq(input_array, mode)
        plot_image_with_histogram(processed_img, f'Equalized ({mode})', axes[i, 0], axes[i, 1])
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig('histogram_equalization_results.png')
    
    # Part 2: Histogram Matching
    print("\nPerforming Histogram Matching...")
    fig, axes = plt.subplots(5, 2, figsize=(12, 20))
    fig.suptitle('Histogram Matching', fontsize=16)
    
    # Plot original and reference images
    plot_image_with_histogram(input_array, 'Original Image', axes[0, 0], axes[0, 1])
    plot_image_with_histogram(ref_array, 'Reference Image', axes[1, 0], axes[1, 1])
    
    # Process image with each mode
    for i, mode in enumerate(modes, 2):
        print(f"Processing with {mode} mode...")
        processed_img = perform_hist_matching(input_array, ref_array, mode)
        plot_image_with_histogram(processed_img, f'Matched ({mode})', axes[i, 0], axes[i, 1])
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig('histogram_matching_results.png')
    
    plt.show()

if __name__ == "__main__":
    main()
