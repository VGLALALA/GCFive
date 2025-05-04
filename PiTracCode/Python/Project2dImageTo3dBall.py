import numpy as np
import math
from typing import Tuple
from GolfBall import GolfBall
from ProjectionOp import ProjectionOp
# Match your C++ ignore value
K_PIXEL_IGNORE_VALUE = 128

def project_to_3d_ball(
    image_gray: np.ndarray,
    ball: GolfBall,                       # your Python GolfBall class
    rotation_angles_deg: Tuple[int, int, int]
) -> np.ndarray:
    """
    Projects each pixel of the 2D grayscale image onto a 3D hemispherical surface
    at the specified Euler rotations, producing a (rows x cols x 2) array where
    each entry is (projected_value, mask_flag).

    Args:
        image_gray:          HxW uint8 array of the ball crop.
        ball:                GolfBall instance (provides geometry/calibration).
        rotation_angles_deg: (x_deg, y_deg, z_deg) in degrees.

    Returns:
        projected_img: HxW x 2 int32 array.  
                       [:,:,0] = projected pixel value  
                       [:,:,1] = mask (kPixelIgnoreValue for “no data”)
    """
    rows, cols = image_gray.shape[:2]

    # initialize to (0, ignore)
    projected_img = np.full(
        (rows, cols, 2),
        fill_value=(0, K_PIXEL_IGNORE_VALUE),
        dtype=np.int32
    )

    # Convert rotations to radians (negate X as in C++)
    rot_x = -math.radians(rotation_angles_deg[0])
    rot_y =  math.radians(rotation_angles_deg[1])
    rot_z =  math.radians(rotation_angles_deg[2])

    # Instantiate your projection functor (you must implement this)
    proj_op = ProjectionOp(ball, projected_img, rot_x, rot_y, rot_z)

    # Serial loop (analogous to C++ forEach when serialization is on)
    for y in range(rows):
        for x in range(cols):
            pixel = int(image_gray[y, x])
            proj_op(pixel, (x, y))

    return projected_img
