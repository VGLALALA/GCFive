import cv2
import numpy as np

def get_image_characteristics(img: np.ndarray, brightness_percentage: int) -> tuple[int, int, int]:
    histSize = 256
    histRange = (0, 256)
    total_points = img.shape[0] * img.shape[1]

    # Calculate histogram
    b_hist = cv2.calcHist([img], [0], None, [histSize], histRange)
    b_hist = b_hist.flatten()

    accum = 0
    target_points = total_points * (100 - brightness_percentage) / 100.0
    i = histSize - 1
    highest_brightness = -1

    while i >= 0:
        num_pixels_in_bin = int(round(b_hist[i]))
        accum += num_pixels_in_bin

        if highest_brightness < 0 and num_pixels_in_bin > 0:
            highest_brightness = i

        if accum >= target_points:
            break

        i -= 1

    brightness_cutoff = i + 1

    # Find lowest non-zero brightness
    lowest_brightness = next((idx for idx, val in enumerate(b_hist) if val > 0), 0)

    return brightness_cutoff, lowest_brightness, highest_brightness
