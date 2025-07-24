"""Ball speed estimation utilities."""

import math
from typing import Tuple

import cv2

from .ballDetection import detect_golfballs


def calculate_ball_speed(
    frame_start: "cv2.Mat",
    frame_end: "cv2.Mat",
    fps: float,
) -> Tuple[float, float]:
    """Return ball speed between two frames.

    Parameters
    ----------
    frame_start, frame_end : ndarray
        Video frames containing the ball.
    fps : float
        Capture frame rate.

    Returns
    -------
    (speed_mps, speed_mph)
        Estimated speed in metres per second and miles per hour.
    """

    balls1 = detect_golfballs(frame_start.copy(), display=False)
    balls2 = detect_golfballs(frame_end.copy(), display=False)

    if not balls1 or not balls2:
        raise ValueError("Ball could not be detected in the supplied frames")

    x1, y1, r1 = balls1[0]
    x2, y2, r2 = balls2[0]

    # convert pixel displacement to metres
    px_per_mm = ((r1 + r2) / 2) * 2 / 42.67
    distance_mm = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 / px_per_mm
    distance_m = distance_mm / 1000.0

    dt = 1.0 / fps
    speed_mps = distance_m / dt
    speed_mph = speed_mps * 2.237
    return speed_mps, speed_mph

