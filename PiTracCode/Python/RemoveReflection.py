import cv2
import numpy as np

# Constants (match your C++ definitions)
kReflectionMinimumRGBValue = 245  # brightness threshold for reflections
kPixelIgnoreValue = 128           # value to mark ignored pixels

def remove_reflections(
    original_image: np.ndarray,
    filtered_image: np.ndarray,
    mask: np.ndarray = None
) -> None:
    """
    Detects and removes specular “bright” reflections from a grayscale image
    by marking those pixels in `filtered_image` with kPixelIgnoreValue.

    Args:
        original_image (np.ndarray): 8-bit single-channel (grayscale) input.
        filtered_image (np.ndarray): 8-bit single-channel image to modify in place.
        mask (np.ndarray, optional): Not used in this implementation (placeholder).
    """
    # 1. Dynamically compute brightness cutoff (optional – C++ version uses fixed kReflectionMinimumRGBValue)
    # brightness_cutoff, lowest_brightness, highest_brightness = get_image_characteristics(
    #     original_image, brightness_percentage=99
    # )
    # brightness_cutoff = max(brightness_cutoff - 1, 0)

    # 2. Threshold to find “very bright” pixels (reflections)
    lower = kReflectionMinimumRGBValue
    upper = 255
    # thresh will be 255 where original_image in [lower, upper], else 0
    thresh = cv2.inRange(original_image, lower, upper)

    # 3. Morphological closing to fill small holes, then dilation to expand reflection regions
    #    This helps mask out edges and neighboring pixels that could pollute later processing.
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph = cv2.morphologyEx(morph, cv2.MORPH_DILATE, dilate_kernel, iterations=1)

    # 4. Mark all detected reflection pixels in the filtered image as “ignore”
    #    Note: images are indexed as [row, col] == [y, x]
    reflection_mask = (morph == 255)
    filtered_image[reflection_mask] = kPixelIgnoreValue

    # (Optional) visualize for debugging
    # cv2.imshow("Reflections Mask", morph)
    # cv2.imshow("Filtered Image", filtered_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
