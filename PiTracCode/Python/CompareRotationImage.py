import numpy as np


def compare_rotation_image(
    img1: np.ndarray,
    img2: np.ndarray,
    ignore_value: tuple[int, int, int] = (255, 255, 255)
) -> tuple[int, int, np.ndarray]:
    """
    Compare two same-size images pixel by pixel, ignoring pixels marked with ignore_value.
    Returns (score, total_pixels_examined, correspondence_image), where:
      - score is the count of matching pixels (p1 == p2)
      - total_pixels_examined is the count of pixels where both p1 and p2 != ignore_value
      - correspondence_image is a binary mask (255 for matches, ignore_value for ignored pixels, 0 otherwise)
    """
    assert img1.shape[0] == img2.shape[0] and img1.shape[1] == img2.shape[1], \
        "Input images must have the same dimensions"

    rows, cols = img1.shape[:2]
    # Create an output image for debugging correspondence
    correspondence = np.zeros((rows, cols), dtype=img1.dtype)

    score = 0
    total_examined = 0

    for y in range(rows):
        for x in range(cols):
            p1 = int(img1[y, x])
            # If img2 has multiple channels, take the second one; else assume single-channel
            if img2.ndim == 3 and img2.shape[2] > 1:
                p2 = int(img2[y, x, 1])
            else:
                p2 = int(img2[y, x])

            if p1 != ignore_value[0] and p2 != ignore_value[1]:
                total_examined += 1
                if p1 == p2:
                    score += 1
                    correspondence[y, x] = 255
            else:
                correspondence[y, x] = ignore_value[2]

    return score, total_examined, correspondence
