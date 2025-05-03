import numpy as np

def unproject_3d_ball_to_2d_image(
    src3d: np.ndarray,
    destination_image_gray: np.ndarray,
    ball: “GolfBall”  # your Python GolfBall class, not used here but kept for API parity
) -> None:
    """
    Copy the second channel from a (rows×cols×2) int32 src3d array back into the
    single‐channel 8‐bit destination_image_gray, ignoring the first channel.

    Args:
        src3d:                  np.ndarray of shape (H, W, 2), dtype=int32.
                                [:,:,0] is ignored; [:,:,1] is the pixel value.
        destination_image_gray: np.ndarray of shape (H, W), dtype=uint8;
                                this will be modified in place.
        ball:                   GolfBall instance (unused here).
    """
    rows, cols = destination_image_gray.shape
    # Iterate in (y, x) order
    for y in range(rows):
        for x in range(cols):
            # src3d[y, x] is a length‐2 array: [maxValueZ, pixelValue]
            pixel_value = int(src3d[y, x, 1])
            destination_image_gray[y, x] = pixel_value
