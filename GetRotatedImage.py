import numpy as np
from GolfBall import GolfBall
from Project2dImageTo3dBall import project_2d_image_to_3d_ball
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
    print("shape1: " + str(gray_2d_input_image.shape))
    # 1) Project the 2D input onto a 3D hemisphere at the given rotation
    ball_3d_image = project_2d_image_to_3d_ball(
        gray_2d_input_image,
        ball,
        rotation
    )
    print("shape2: " + str(ball_3d_image.shape))
    # 2) Prepare an empty output image
    output_gray_img = np.zeros_like(gray_2d_input_image)
    print("shape3: " + str(output_gray_img.shape))
    # 3) Unproject the 3D hemisphere back to a 2D grayscale image
    ball_2d_image = unproject_3d_ball_to_2d_image(
        ball_3d_image,
        ball
    )
    print("shape4: " + str(ball_3d_image.shape))
    return ball_2d_image

def test():
    full_gray_image = "data/Images/log_cam2_last_strobed_img.png"
    from ROI import run_hough_with_radius
    from IsolateCode import isolate_ball
    import cv2
    test_img = cv2.imread(full_gray_image, cv2.IMREAD_GRAYSCALE)
    best_ball1, _ = run_hough_with_radius(test_img)
    ballimg = isolate_ball(test_img,best_ball1)
    from ImageCompressor import compress_image
    #compress_ballimg = compress_image(ballimg,4.0)
    best_ball1.x = ballimg.shape[1] // 2
    best_ball1.y = ballimg.shape[0] // 2
    adjustedimg2 = get_rotated_image(ballimg, best_ball1, (0,30,-30))

    cv2.imshow("Original", ballimg)
    cv2.imshow("Predicted", adjustedimg2)
    cv2.waitKey(0)
