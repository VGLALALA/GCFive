import cv2
import numpy as np
import copy
from typing import Tuple
from MaskAreaOutsideBall import mask_area_outside_ball
from GetBallRotation import GolfBall

def isolate_ball(
    img: np.ndarray,
    ball: GolfBall
) -> Tuple[np.ndarray, GolfBall]:
    """
    Crops out a square region around the detected ball, recenters the ball metadata,
    equalizes (optional), then masks out pixels just outside the ball edge.

    Args:
        img:   Full grayscale image.
        ball:  GolfBall instance with .x, .y, .measured_radius_pixels.

    Returns:
        ball_crop:   Cropped & masked image of the ball.
        ball_local:  A copy of `ball` whose x,y coords are relative to the crop.
    """
    # Make a local copy so we don't mutate the caller’s ball
    ball_local = copy.deepcopy(ball)

    # 1) Compute a slightly larger radius to include a tiny border
    surround_mult = 1.05
    r1  = int(round(ball_local.measured_radius_pixels * surround_mult))
    rInc = r1 - ball_local.measured_radius_pixels

    # 2) Determine top-left corner of crop
    x1 = int(ball_local.x - r1)
    y1 = int(ball_local.y - r1)
    w  = h = 2 * r1

    # 3) Clip to image bounds
    x1 = max(0, min(x1, img.shape[1] - w - 1))
    y1 = max(0, min(y1, img.shape[0] - h - 1))

    # 4) Crop
    ball_crop = img[y1 : y1 + h, x1 : x1 + w].copy()

    # 5) Recenter ball coordinates in the cropped image
    new_center = int(round(rInc + ball_local.measured_radius_pixels))
    ball_local.x = new_center
    ball_local.y = new_center

    # 6) (Optional) equalize histogram if desired
    # ball_crop = cv2.equalizeHist(ball_crop)

    # 7) Mask out just outside the ball to remove edge artifacts
    #    reference reduction factor from C++
    reference_mask_factor = 0.995
    ball_crop = mask_area_outside_ball(
        ball_crop,
        ball_local,
        reference_mask_factor,
        ignore_value=0
    )

    return ball_crop, ball_local

