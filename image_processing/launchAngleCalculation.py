"""Launch angle estimation utilities."""

import math
from typing import Tuple

import cv2

from .ballDetection import detect_golfballs


def calculate_launch_angle(
    frame_start: "cv2.Mat",
    frame_end: "cv2.Mat",
) -> float:
    """Return the vertical launch angle between two frames in degrees."""

    balls1 = detect_golfballs(frame_start.copy(), display=False)
    balls2 = detect_golfballs(frame_end.copy(), display=False)

    if not balls1 or not balls2:
        raise ValueError("Ball could not be detected in the supplied frames")

    x1, y1, r1 = balls1[0]
    x2, y2, r2 = balls2[0]

    px_per_mm = ((r1 + r2) / 2) * 2 / 42.67
    dx_mm = (x2 - x1) / px_per_mm
    dy_mm = -(y2 - y1) / px_per_mm

    angle_rad = math.atan2(dy_mm, dx_mm)
    return math.degrees(angle_rad)

