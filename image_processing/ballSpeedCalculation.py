import math
from typing import Tuple

BALL_DIAMETER_MM = 42.67


def calculate_ball_speed(det1: Tuple[int, int, int], det2: Tuple[int, int, int], fps: float) -> float:
    """Return ball speed in meters per second given two detections and frame rate."""
    x1, y1, r1 = det1
    x2, y2, r2 = det2
    pixels_per_mm = ((r1 + r2) / 2 * 2) / BALL_DIAMETER_MM
    dx_mm = (x2 - x1) / pixels_per_mm
    dy_mm = (y2 - y1) / pixels_per_mm
    dt = 1.0 / fps if fps else 1.0
    speed_mm_per_s = math.sqrt(dx_mm ** 2 + dy_mm ** 2) / dt
    return speed_mm_per_s / 1000.0
