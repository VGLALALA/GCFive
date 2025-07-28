import numpy as np

from .GolfBall import GolfBall


def unproject_3d_ball_to_2d_image(src3D: np.ndarray, ball: GolfBall) -> np.ndarray:
    """
    Unprojects a 3D image back to a 2D grayscale image.

    Args:
        src3D: A numpy array of shape (H, W, 2) where [:, :, 0] is the Z-depth and [:, :, 1] is the pixel value.
        ball: An object representing the ball, used for additional operations if needed.

    Returns:
        A numpy array of shape (H, W) representing the 2D grayscale image.
    """
    rows, cols = src3D.shape[:2]
    destination_image_gray = np.zeros((rows, cols), dtype=np.uint8)

    for x in range(cols):
        for y in range(rows):
            # Extract Z-depth and pixel value
            maxValueZ = src3D[y, x, 0]
            pixelValue = src3D[y, x, 1]

            # Update the destination image with the pixel value
            destination_image_gray[y, x] = pixelValue

            # Debugging: Check if the pixel is an ignore pixel within the ball
            # if ball.PointIsInsideBall(x, y) and pixelValue == kPixelIgnoreValue:
            #     print(f"Unproject3dBallTo2dImage found ignore pixel within ball at ({x}, {y}).")

    # Optional: Apply morphology operations to fill holes (commented out as it may fuzz the image)
    # kernel = np.ones((3, 3), np.uint8)
    # destination_image_gray = cv2.morphologyEx(destination_image_gray, cv2.MORPH_CLOSE, kernel)

    return destination_image_gray
