import numpy as np
from GolfBall import GolfBall
from Project2dImageTo3dBall import project_to_3d_ball
from Unproject3Dto2D import unproject_3d_ball_to_2d_image
from typing import Tuple

def get_rotated_image(
    gray_2d_input_image: np.ndarray,
    ball: GolfBall,         # your Python GolfBall class
    rotation: Tuple[int,int,int]
) -> np.ndarray:
    """
    Mimics BallImageProc::GetRotatedImage in C++.

    Args:
        gray_2d_input_image: single-channel 2D NumPy array.
        ball:                GolfBall instance with metadata for unprojection.
        rotation:            (x_deg, y_deg, z_deg) Euler angles in degrees.

    Returns:
        output_gray_img:     same shape as input, after 3D→2D re-projection.
    """
    # 1) Project the 2D input onto a 3D hemisphere at the given rotation
    ball_3d_image = project_to_3d_ball(
        gray_2d_input_image,
        ball,
        rotation
    )

    # 2) Prepare an empty output image
    output_gray_img = np.zeros_like(gray_2d_input_image)

    # 3) Unproject the 3D hemisphere back to a 2D grayscale image
    unproject_3d_ball_to_2d_image(
        ball_3d_image,
        output_gray_img,
        ball
    )

    return output_gray_img
