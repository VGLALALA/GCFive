import cv2
import numpy as np
from typing import Tuple
from GolfBall import GolfBall
def match_ball_image_sizes(
    img1: np.ndarray,
    img2: np.ndarray,
    ball1: GolfBall,
    ball2: GolfBall
) -> Tuple[np.ndarray,np.ndarray,GolfBall,GolfBall]:
    """
    Upscale the smaller of img1/img2 so both have the same shape.
    Adjusts ball1 or ball2's measured_radius_pixels and x,y accordingly.

    Returns:
        (new_img1, new_img2, new_ball1, new_ball2)
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Clone so we don't overwrite originals
    out1, out2 = img1.copy(), img2.copy()
    b1, b2 = ball1, ball2

    if (h1 > h2) or (w1 > w2):
        # img2 is smaller — scale it up to match img1
        scale = float(h1) / float(h2)
        new_size = (w1, h1)
        out2 = cv2.resize(img2, new_size, interpolation=cv2.INTER_LINEAR)

        # Update ball2 metadata
        b2 = GolfBall(
            x = ball2.x * scale,
            y = ball2.y * scale,
            measured_radius_pixels = ball2.measured_radius_pixels * scale,
            angles_camera_ortho_perspective = ball2.angles_camera_ortho_perspective
        )
        # If you store ball_circle_[2], multiply that too:
        # b2.ball_circle_[2] *= scale

    elif (h2 > h1) or (w2 > w1):
        # img1 is smaller — scale it up to match img2
        scale = float(h2) / float(h1)
        new_size = (w2, h2)
        out1 = cv2.resize(img1, new_size, interpolation=cv2.INTER_LINEAR)

        # Update ball1 metadata
        b1 = GolfBall(
            x = ball1.x * scale,
            y = ball1.y * scale,
            measured_radius_pixels = ball1.measured_radius_pixels * scale,
            angles_camera_ortho_perspective = ball1.angles_camera_ortho_perspective
        )
        # b1.ball_circle_[2] *= scale

    # If they’re already the same size, we just return copies
    return out1, out2, b1, b2
