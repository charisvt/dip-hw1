import numpy as np
from hist_utils import calculate_hist_of_img, apply_hist_modification_transform

def perform_histogram_modification(img_array, hist_ref, mode):
    assert mode in {"greedy", "non-greedy", "post-disturbance"}, f"Invalid mode: {mode}"
    
    # Compute transformation based on mode
    modification_transform = {}
    
    if mode == "greedy":
        # Step 1: Get unnormalized histogram of the input
        hist_input = calculate_hist_of_img(img_array, return_normalized=False)
        
        # Step 2: Total number of pixels in the image
        total_pixels = int(np.prod(img_array.shape))
        
        # Step 3: Sorted output levels (assumed from hist_ref)
        output_levels = sorted(hist_ref.keys())  # g0, g1, ..., gLg-1
        Lg = len(output_levels)
        target_bin_count = total_pixels / Lg  # Number of pixels per output level
        
        # Step 4: Sorted input levels
        input_levels = sorted(hist_input.keys())
        
        # Step 5: Greedy assignment
        # Process input levels sequentially, assign to output levels based on accumulated counts
        current_output_idx = 0
        accumulated_count = 0

        for f_val in input_levels:
            if current_output_idx >= Lg:
                # If we've exhausted output levels, assign to last one
                modification_transform[f_val] = output_levels[-1]
                continue

            accumulated_count += hist_input[f_val]
            modification_transform[f_val] = output_levels[current_output_idx]

            if accumulated_count >= target_bin_count:
                current_output_idx += 1
                accumulated_count = 0

        # Step 6: Apply the transformation
        modified_img = apply_hist_modification_transform(img_array, modification_transform)

    # Non-greedy implementation
    elif mode == "non-greedy":
        # Step 1: Get unnormalized histogram of the input
        hist_input = calculate_hist_of_img(img_array, return_normalized=False)
        
        # Step 2: Total number of pixels in the image
        total_pixels = int(np.prod(img_array.shape))
        
        # Step 3: Sorted output levels (assumed from hist_ref)
        output_levels = sorted(hist_ref.keys())  # g0, g1, ..., gLg-1
        Lg = len(output_levels)
        target_bin_count = total_pixels / Lg  # Number of pixels per output level
        
        # Step 4: Sorted input levels
        input_levels = sorted(hist_input.keys())
        
        # Step 5: Non-greedy assignment
        # More balanced approach that considers bin count deficiency
        current_output_idx = 0
        accumulated_count = 0
        
        # Process each input level in order
        for f_val in input_levels:
            # If we've exhausted output levels, assign to the last one
            if current_output_idx >= Lg:
                modification_transform[f_val] = output_levels[-1]
                continue
                
            # Calculate deficiency for current output level
            deficiency = target_bin_count - accumulated_count
            
            # Check if current input level should be mapped to current output level
            if accumulated_count == 0 or deficiency >= hist_input[f_val] / 2:
                # Map this input level to current output level
                modification_transform[f_val] = output_levels[current_output_idx]
                accumulated_count += hist_input[f_val]
            else:
                # Move to next output level and map this input level to it
                current_output_idx += 1
                
                # Check if we've run out of output levels
                if current_output_idx >= Lg:
                    modification_transform[f_val] = output_levels[-1]
                else:
                    modification_transform[f_val] = output_levels[current_output_idx]
                    accumulated_count = hist_input[f_val]
        
        # Step 6: Apply the transformation
        modified_img = apply_hist_modification_transform(img_array, modification_transform)

    elif mode == "post-disturbance":
        # Step 1: Get unnormalized histogram of the input
        hist_input = calculate_hist_of_img(img_array, return_normalized=False)
        
        # Step 2: Sorted input levels to determine constant spacing d
        input_levels = sorted(hist_input.keys())
        
        # Calculate constant spacing d (assume it's consistent between levels)
        # If there's only one level, use a small default value
        if len(input_levels) > 1:
            d = (input_levels[-1] - input_levels[0]) / (len(input_levels) - 1)
        else:
            d = 1  # Default if only one level exists
        
        # Step 3: Create a disturbed version of the image by adding uniform noise
        # Create a copy of the original image
        f_hat = np.copy(img_array)
        
        # Generate random uniform noise in range [-d/2, d/2] (scaled to 0-1 range)
        noise = np.random.uniform(-d/2, d/2, f_hat.shape) / 255.0
        
        # Add noise to create disturbed image
        f_hat = np.clip(f_hat + noise, 0.0, 1.0)  # Keep values in valid range [0,1]
        
        # Step 4: Now apply the greedy approach on the disturbed image
        # Calculate histogram of the disturbed image
        hist_disturbed = calculate_hist_of_img(f_hat, return_normalized=False)
        
        # Total number of pixels
        total_pixels = int(np.prod(img_array.shape))
        
        # Sorted output levels
        output_levels = sorted(hist_ref.keys())
        Lg = len(output_levels)
        target_bin_count = total_pixels / Lg
        
        # Sorted input levels of the disturbed image
        disturbed_input_levels = sorted(hist_disturbed.keys())
        
        # Greedy assignment for the disturbed image
        current_output_idx = 0
        accumulated_count = 0
        
        for f_val in disturbed_input_levels:
            if current_output_idx >= Lg:
                # If we've exhausted output levels, assign to last one
                modification_transform[f_val] = output_levels[-1]
                continue
                
            accumulated_count += hist_disturbed[f_val]
            modification_transform[f_val] = output_levels[current_output_idx]
            
            if accumulated_count >= target_bin_count:
                current_output_idx += 1
                accumulated_count = 0
        
        # Apply the transformation to the original image (not the disturbed one)
        modified_img = apply_hist_modification_transform(img_array, modification_transform)

    return modified_img

def perform_hist_eq(img_array, mode):
    # For histogram equalization, create a uniform reference histogram
    hist_ref = {i: 1 for i in range(256)}
    
    # Call the histogram modification function with the uniform reference histogram
    return perform_histogram_modification(img_array, hist_ref, mode)

def perform_hist_matching(img_array, img_array_ref, mode):
    # Calculate the histogram of the reference image
    hist_ref = calculate_hist_of_img(img_array_ref, return_normalized=False)
    
    # Call the histogram modification function with the reference histogram
    return perform_histogram_modification(img_array, hist_ref, mode)