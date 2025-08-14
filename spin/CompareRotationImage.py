from typing import Tuple

import numpy as np

from utility.config_reader import CONFIG

PIXEL_IGNORE_VALUE = CONFIG.getint("Spin", "pixel_ignore_value", fallback=128)


def compare_rotation_image(
    img1: np.ndarray, img2: np.ndarray, index: int
) -> Tuple[int, int, np.ndarray]:
    """
    Port of:
      cv::Vec2i BallImageProc::CompareRotationImage(const cv::Mat& img1,
                                                   const cv::Mat& img2,
                                                   const int index)

    img1: H×W single-channel uint8
    img2: H×W×2 int32 (simulating Vec2i), where we compare the second channel [*,*,1]
    index: unused here, but kept for signature compatibility
    Returns: (score, total_pixels_examined, test_correspondence_img)
    """
    rows, cols = img1.shape
    assert (
        img2.shape[0] == rows and img2.shape[1] == cols
    ), "img1 and img2 must have the same dimensions"

    # Prepare the correspondence image (uint8)
    test_correspondence_img = np.zeros((rows, cols), dtype=np.uint8)

    score = 0
    total_pixels_examined = 0

    # Note: original C++ loops x over cols and y over rows, but numpy is [row, col]
    for x in range(cols):
        for y in range(rows):
            p1 = int(img1[y, x])
            p2 = int(img2[y, x, 1])  # second element of the Vec2i

            if p1 != PIXEL_IGNORE_VALUE and p2 != PIXEL_IGNORE_VALUE:
                total_pixels_examined += 1
                if p1 == p2:
                    score += 1
                    test_correspondence_img[y, x] = 255
            else:
                test_correspondence_img[y, x] = PIXEL_IGNORE_VALUE

    return score, total_pixels_examined, test_correspondence_img
