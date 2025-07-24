"""Backspin estimation using two sequential ball images."""

from typing import Tuple
import numpy as np
import cv2

from spin.GetBallRotation import get_fine_ball_rotation


def calculate_backspin(img1: "cv2.Mat", img2: "cv2.Mat", delta_t: float) -> float:
    """Return estimated backspin in RPM using :mod:`spin.GetBallRotation`."""
    rot_x, rot_y, rot_z = get_fine_ball_rotation(img1, img2, compress_candidates=False)
    backspin_rpm = (rot_y / delta_t) * (60.0 / 360.0)
    return backspin_rpm

