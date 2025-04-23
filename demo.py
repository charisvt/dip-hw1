import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from hist_utils import calculate_hist_of_img
from hist_modif import perform_hist_eq, perform_hist_matching

def analyze_histogram(img_array, title, verbose=False):
    """Analyze and print statistics about an image histogram."""
    # Calculate non-normalized histogram
    hist = calculate_hist_of_img(img_array, return_normalized=False)
    
    # Get image dimensions and total pixels
    total_pixels = img_array.size
    
    # Find min/max used intensity levels
    used_levels = [k for k, v in hist.items() if v > 0]
    if not used_levels:
        if verbose:
            print(f"{title}: No pixels detected!")
        return
    
    min_level = min(used_levels)
    max_level = max(used_levels)
    num_used_levels = len(used_levels)
    
    # Calculate percentiles
    sorted_intensities = []
    for intensity, count in hist.items():
        sorted_intensities.extend([intensity] * count)
    sorted_intensities.sort()
    
    p10 = sorted_intensities[int(0.1 * len(sorted_intensities))]
    p25 = sorted_intensities[int(0.25 * len(sorted_intensities))]
    p50 = sorted_intensities[int(0.5 * len(sorted_intensities))]
    p75 = sorted_intensities[int(0.75 * len(sorted_intensities))]
    p90 = sorted_intensities[int(0.9 * len(sorted_intensities))]
    
    # Find the density of the histogram
    highest_concentration = max(hist.values()) / total_pixels * 100
    highest_intensity = max(hist.items(), key=lambda x: x[1])[0]
    
    # Print analysis only if verbose mode is enabled
    if verbose:
        print(f"\n----- Histogram Analysis for {title} -----")
        print(f"Total unique intensity levels used: {num_used_levels}/256")
        print(f"Intensity range: {min_level} to {max_level}")
        print(f"Percentiles: P10={p10}, P25={p25}, P50={p50}, P75={p75}, P90={p90}")
        print(f"Highest concentration: {highest_concentration:.2f}% of pixels at intensity {highest_intensity}")
        
        # Print distribution by ranges
        ranges = [(0, 63), (64, 127), (128, 191), (192, 255)]
        for range_min, range_max in ranges:
            pixels_in_range = sum(hist.get(i, 0) for i in range(range_min, range_max+1))
            percent = pixels_in_range / total_pixels * 100
            print(f"Intensity {range_min}-{range_max}: {percent:.2f}% of pixels")
        
        print("------------------------------")

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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Histogram Equalization and Matching Demo')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose analysis output')
    args = parser.parse_args()
    
    # Load input and reference images
    input_img = Image.open('input_img.jpg').convert('L')
    ref_img = Image.open('ref_img.jpg').convert('L')
    
    # Convert to numpy arrays and normalize to [0, 1]
    input_array = np.array(input_img).astype(float) / 255.0
    ref_array = np.array(ref_img).astype(float) / 255.0
    
    # Analyze original image
    if args.verbose:
        print("\n===== ORIGINAL IMAGE ANALYSIS =====")
    analyze_histogram(input_array, "Original Image", args.verbose)
    if args.verbose:
        print("\n===== REFERENCE IMAGE ANALYSIS =====")
    analyze_histogram(ref_array, "Reference Image", args.verbose)
    
    # List of processing modes
    modes = ['greedy', 'non-greedy', 'post-disturbance']
    
    # Part 1: Histogram Equalization
    if args.verbose:
        print("\n===== HISTOGRAM EQUALIZATION RESULTS =====")
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
        
        # Analyze equalized image
        analyze_histogram(processed_img, f"Equalized Image ({mode})", args.verbose)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig('histogram_equalization_results.png')
    
    # Part 2: Histogram Matching
    if args.verbose:
        print("\n===== HISTOGRAM MATCHING RESULTS =====")
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
        
        # Analyze matched image
        analyze_histogram(processed_img, f"Matched Image ({mode})", args.verbose)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig('histogram_matching_results.png')
    
    plt.show()

if __name__ == "__main__":
    main()
