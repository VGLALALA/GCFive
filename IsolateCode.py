import cv2
import numpy as np
from GolfBall import GolfBall

def isolate_ball(
    img: np.ndarray,
    ball: GolfBall
) -> np.ndarray:
    """
    Crops out a square region around the detected ball.
    If ball parameters not provided, auto-detects them.

    Args:
        img:   Full grayscale image.
        ball:  GolfBall instance with .x, .y, .measured_radius_pixels.

    Returns:
        ball_crop: Cropped image of the ball.
    """
    # Auto-detect if needed. The original implementation treated zero values as
    # missing which caused valid coordinates near the image origin to trigger
    # detection again. Only ``None`` should be considered "missing".
    if (
        getattr(ball, "x", None) is None
        or getattr(ball, "y", None) is None
        or getattr(ball, "measured_radius_pixels", None) is None
    ):
        circles = cv2.HoughCircles(
            img, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
            param1=100, param2=30, minRadius=30, maxRadius=70
        )
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            x, y, r = circles[0]
            ball.x = x
            ball.y = y 
            ball.measured_radius_pixels = r
        else:
            raise ValueError("No ball detected in image!")

    # Compute crop dimensions
    r = int(round(ball.measured_radius_pixels))
    x1 = int(round(ball.x - r))
    y1 = int(round(ball.y - r))
    w = h = int(2 * r)

    # Clip to image bounds
    x1 = max(0, min(x1, img.shape[1] - w - 1))
    y1 = max(0, min(y1, img.shape[0] - h - 1))

    # Crop and return
    ball_crop = img[y1 : y1 + h, x1 : x1 + w].copy()
    return ball_crop
