import cv2
import numpy as np
from spin.GolfBall import GolfBall
from image_processing.ballDetection import get_detected_balls_info

def isolate_ball(
    img: np.ndarray,
    ball: GolfBall
) -> np.ndarray:
    """
    Crops out a square region around the detected ball using YOLO model.
    If ball parameters not provided, auto-detects them using YOLO.

    Args:
        img:   Full grayscale image.
        ball:  Optional GolfBall instance with .x, .y, .measured_radius_pixels.

    Returns:
        ball_crop: Cropped image of the ball.
    """
    
    r = int(round(ball.measured_radius_pixels))
    x1 = int(round(ball.x - r))
    y1 = int(round(ball.y - r))
    w = int(2 * r)

    # Clip to image bounds
    x1 = max(0, min(x1, img.shape[1] - w - 1))
    y1 = max(0, min(y1, img.shape[0] - w - 1))

    # Crop and return
    ball_crop = img[y1 : y1 + w, x1 : x1 + w].copy()
    return ball_crop
