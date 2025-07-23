import numpy as np
import cv2

def compress_image(image: np.ndarray, scale: float) -> np.ndarray:
    """
    Compress a black-and-white image by a given scale factor.

    Parameters:
    - image (np.ndarray): 2D NumPy array of the image.
    - scale (float): Compression factor (e.g., 2.0 means 2x smaller).

    Returns:
    - np.ndarray: Compressed image as a 2D NumPy array.
    """
    if scale <= 0:
        raise ValueError("Scale must be a positive number.")

    height, width = image.shape
    new_size = (int(width / scale), int(height / scale))
    compressed = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return compressed
