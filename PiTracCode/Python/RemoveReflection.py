import cv2
import numpy as np

def remove_reflections(original_image: np.ndarray,
                       filtered_image: np.ndarray,
                       mask: np.ndarray = None,
                       brightness_percentage: int = 99,
                       k_reflection_min_val: int = 200,
                       k_pixel_ignore_value: int = 128) -> np.ndarray:
    """
    Remove bright reflections from original_image by marking them in filtered_image.

    Parameters:
    - original_image:     8-bit grayscale image (H×W).
    - filtered_image:     same shape/type as original_image; pixels to be masked will be set to k_pixel_ignore_value.
    - mask:               optional binary mask; if provided, percentile is computed only over mask>0.
    - brightness_percentage: percentile for dynamic cutoff (unused below, kept for compatibility).
    - k_reflection_min_val: fixed lower bound for thresholding bright spots.
    - k_pixel_ignore_value: value with which to mark reflections in filtered_image.

    Returns:
    - filtered_image with reflections set to k_pixel_ignore_value.
    """
    # 1. Compute dynamic brightness cutoff (for logging or future use)
    if mask is not None and mask.any():
        vals = original_image[mask > 0]
    else:
        vals = original_image.ravel()
    lowest = int(vals.min())
    highest = int(vals.max())
    brightness_cutoff = float(np.percentile(vals, brightness_percentage))
    # (In C++ you actually override this with a constant, so we’ll do the same:)
    
    # 2. Threshold to find bright (reflection) pixels
    #    inRange will produce a binary image where pixels >= k_reflection_min_val are 255
    thresh = cv2.inRange(original_image,
                         k_reflection_min_val,  # lower 
                         255)                   # upper
    
    # 3. Morphological close then dilation to expand reflection regions
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph = cv2.morphologyEx(morph, cv2.MORPH_DILATE, dilate_kernel, iterations=1)
    
    # 4. Mark reflections in filtered_image as “ignore”
    #    assuming filtered_image is uint8; adjust if you have a different type
    filtered_image[morph == 255] = k_pixel_ignore_value
    
    return filtered_image
