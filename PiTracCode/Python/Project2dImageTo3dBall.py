from typing import Tuple
import numpy as np
from GolfBall import GolfBall
from ProjectOp import ProjectionOp
import math
kPixelIgnoreValue = 128
def project_2d_image_to_3d_ball(
    image_gray: np.ndarray,
    ball: GolfBall,
    rotation_angles_degrees: Tuple[int, int, int]
) -> np.ndarray:
    """
    Projects a 2D grayscale image of a golf ball onto a virtual 3D hemisphere,
    rotates it by the specified Euler angles, and unprojects back to 2D.
    Returns a (H, W, 2) image where [:, :, 0] is the Z-depth and [:, :, 1] is
    the grayscale/ignore flags.
    """
    rows, cols = image_gray.shape
    # output: depth + pixel value/ignore
    projectedImg = np.zeros((rows, cols, 2), dtype=np.int32)
    projectedImg[..., 1] = kPixelIgnoreValue
    # convert degrees to radians (note sign flip on X)
    x_rad = -math.radians(rotation_angles_degrees[0])
    y_rad = math.radians(rotation_angles_degrees[1])
    z_rad = math.radians(rotation_angles_degrees[2])
    op = ProjectionOp(ball, projectedImg, x_rad, y_rad, z_rad)
    # iterate over pixels
    for y in range(rows):
        for x in range(cols):
            pixel = int(image_gray[y, x])
            op(pixel, x, y)
    return projectedImg