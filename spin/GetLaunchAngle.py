import math
from image_processing.ballDetection import detect_golfballs
import cv2
def calculate_launch_angle(frame1, frame2):
    """
    Given two image frames and the time between them, return the
    launch angle (in degrees) of the ball’s motion relative to
    the horizontal in the image plane.

    Args:
      frame1, frame2: two consecutive BGR frames (e.g. numpy arrays)
      delta_t: time between frames (seconds)
      detect_fn: function(frame) → List[(cx, cy, r)] as you described

    Returns:
      launch_angle_deg: positive means “upward” movement (towards
                        top of image), negative means “downward.”
    """
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_GRAY2BGR)
    det1 = detect_golfballs(frame1)
    det2 = detect_golfballs(frame2)
    if not det1 or not det2:
        raise ValueError("Ball not detected in one or both frames.")

    # take the first (or best) detection from each
    cx1, cy1, _ = det1[0]
    cx2, cy2, _ = det2[0]

    # pixel displacements
    dx = cx2 - cx1
    # invert dy so that upward motion (towards smaller y in image)
    # becomes positive
    dy = cy1 - cy2

    # angle of motion vector in the image plane
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)

    return angle_deg
