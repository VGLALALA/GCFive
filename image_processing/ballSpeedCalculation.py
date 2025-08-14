import math

import cv2

from image_processing.ballDetection import detect_golfballs


def calculate_ball_speed(
    frame1, frame2, delta_t, ball_diameter_mm=42.67, return_mph=True
):
    """
    Estimate the ball’s speed based on two image frames and the time between them.

    Args:
      frame1, frame2    : two consecutive BGR frames (numpy arrays)
      delta_t           : time between frames (seconds)
      detect_fn         : function(frame) → List of (cx, cy, r_px)
      ball_diameter_mm  : real golf‑ball diameter (default 42.67 mm)
      return_mph        : if True, also convert m/s → mph

    Returns:
      speed_m_per_s     : float, speed in meters per second
      (optionally) speed_mph : float, speed in miles per hour
    """
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    det1 = detect_golfballs(frame1)
    det2 = detect_golfballs(frame2)
    if not det1 or not det2:
        raise ValueError("Ball not detected in one or both frames.")

    # Use the first detection’s radius (px) to get scale: mm per pixel
    _, _, r1 = det1[0]
    mm_per_px = ball_diameter_mm / (2 * r1)

    # Compute pixel displacement
    cx1, cy1, _ = det1[0]
    cx2, cy2, _ = det2[0]
    dx = cx2 - cx1
    dy = cy2 - cy1
    px_dist = math.hypot(dx, dy)

    # Convert to mm then to meters
    dist_m = (px_dist * mm_per_px) / 1000.0

    # Speed in m/s (guard against zero/negative delta_t)
    if delta_t is None or delta_t <= 0:
        raise ValueError(
            f"Invalid delta_t: {delta_t}. It must be positive and non-zero."
        )
    speed_m_per_s = dist_m / max(delta_t, 1e-6)

    if return_mph:
        speed_mph = speed_m_per_s * 2.2369362920544
        return speed_mph

    return speed_m_per_s
