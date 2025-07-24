
import math
from typing import Tuple

BALL_DIAMETER_MM = 42.67


def calculate_launch_angle(det1: Tuple[int, int, int], det2: Tuple[int, int, int]) -> float:
    """Return launch angle in degrees from two detections."""
    x1, y1, r1 = det1
    x2, y2, r2 = det2
    pixels_per_mm = ((r1 + r2) / 2 * 2) / BALL_DIAMETER_MM
    dx_mm = (x2 - x1) / pixels_per_mm
    dy_mm = -(y2 - y1) / pixels_per_mm
    angle_rad = math.atan2(dy_mm, dx_mm)
    return math.degrees(angle_rad)
