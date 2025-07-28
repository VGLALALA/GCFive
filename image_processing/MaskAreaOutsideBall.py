from typing import Union

import cv2
import numpy as np

from spin.GolfBall import GolfBall


def mask_area_outside_ball(
    ball_image: np.ndarray,
    ball: GolfBall,
    mask_reduction_factor: float,
    mask_value: Union[int, tuple] = 128,
) -> np.ndarray:
    """
    Masks everything outside a reduced‐radius circle around the ball,
    painting the outside region with mask_value.

    :param ball_image:            Input image (H×W or H×W×C)
    :param ball:                  GolfBall with .measured_radius_pixels, .x, .y
    :param mask_reduction_factor: Fraction to shrink the mask circle (e.g. 0.92)
    :param mask_value:            Int or tuple (length = channels) to paint outside
    :returns:                     New image with outside masked
    """
    # Copy the input so we don't overwrite it
    out = ball_image.copy()

    # Compute reduced radius and center from the GolfBall
    reduced_radius = int(ball.measured_radius_pixels * mask_reduction_factor)
    center = (int(ball.x), int(ball.y))

    # Build a single‐channel mask
    h, w = ball_image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, reduced_radius, color=255, thickness=-1)

    # If color image, broaden mask dims to broadcast across channels
    if out.ndim == 3:
        mask = mask[:, :, None]

    # Wherever mask==0, paint with mask_value
    out[mask == 0] = mask_value

    return out
