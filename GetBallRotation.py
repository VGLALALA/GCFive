import cv2
from typing import Tuple
import numpy as np
from IsolateCode import isolate_ball
from RemoveReflection import remove_reflections
from MaskAreaOutsideBall import mask_area_outside_ball
from GetRotatedImage import get_rotated_image
from GenerateRotationCandidate import generate_rotation_candidates
from CompareRotationImage import compare_rotation_image
from matchBallSize import match_ball_image_sizes
from RotationSearchSpace import RotationSearchSpace
from CompareCandidateAngleImage import compare_candidate_angle_images
from ApplyGaborFilter import apply_gabor_filter_image
from ImageCompressor import compress_image
import time
import os
import multiprocessing
from GradientDescent import optimize_rotation
from GolfBall import GolfBall
from ROI import run_hough_with_radius

COARSE_X_INC   = 6
COARSE_X_START = -42
COARSE_X_END   = 42

COARSE_Y_INC   = 5
COARSE_Y_START = -30
COARSE_Y_END   = 30

COARSE_Z_INC   = 6
COARSE_Z_START = -50
COARSE_Z_END   = 60
    

def get_fine_ball_rotation(
    ball_image1: np.ndarray,
    ball_image2: np.ndarray,
    compress_candidates: bool = False
) -> Tuple[float, float, float]:
    """
    Returns (spin_x, spin_y, spin_z) in degrees, corresponding to side-, back-, and axial-spin.
    """
    
    # Create GolfBall instances for each image
    best_ball1, best_ball2 = run_hough_with_radius(full_gray_image) 

    # Isolate each ball into its own tight crop
    best_ball1 = isolate_ball(ball_image1, ball1)
    best_ball2 = isolate_ball(ball_image2, ball2)

    # Update the center coordinates of best_ball1 and best_ball2
    best_ball1.x = ball_image1.shape[1] // 2
    best_ball1.y = ball_image1.shape[0] // 2

    best_ball2.x = ball_image2.shape[1] // 2
    best_ball2.y = ball_image2.shape[0] // 2

    # Resize so both crops are the same size
    ball_image1, ball_image2 = match_ball_image_sizes(ball_image1, ball_image2)

    # Apply Gabor filters to pick out dimple edges
    ball_image1 = cv2.equalizeHist(ball_image1)
    ball_image2 = cv2.equalizeHist(ball_image2)
    edge1, calibrated_binary_threshold = apply_gabor_filter_image(ball_image1)
    edge2, calibrated_binary_threshold = apply_gabor_filter_image(ball_image2, calibrated_binary_threshold)

    # Remove specular reflections
    gaberRefRemoved1 = remove_reflections(ball_image1, edge1)
    gaberRefRemoved2 = remove_reflections(ball_image2, edge2)

    # Mask out everything outside the ball's circle
    FINAL_MASK_FACTOR = 0.92
    gaberRefRemoved1 = mask_area_outside_ball(gaberRefRemoved1, best_ball1, FINAL_MASK_FACTOR, (255, 255, 255))
    gaberRefRemoved2 = mask_area_outside_ball(gaberRefRemoved2, best_ball2, FINAL_MASK_FACTOR, (255, 255, 255))

    # De-rotate each image half the perspective offset so both appear "centered"
    offset1 = np.array(best_ball1.angles_camera_ortho_perspective[:2], dtype=np.float32)
    offset2 = np.array(best_ball2.angles_camera_ortho_perspective[:2], dtype=np.float32)
    delta_float = (offset2 - offset1) / 2.0
    delta_float[1] *= -1.0
    delta2d = np.round(delta_float).astype(int)
    delta = np.array([delta2d[0], delta2d[1], 0], dtype=int)

    adjustedimg1 = get_rotated_image(gaberRefRemoved1, best_ball1, tuple(delta))
    delta2d = np.round(-(offset2 - offset1 - delta_float)).astype(int)
    delta2d[1] = -delta2d[1]
    delta2 = np.array([delta2d[0], delta2d[1], 0], dtype=int)

    adjustedimg2 = get_rotated_image(gaberRefRemoved2, best_ball2, tuple(delta2))

    # Compress images if the flag is set
    if compress_candidates:
        edge1 = compress_image(edge1, 3)
        edge2 = compress_image(edge2, 3)

    # Coarse-search for best 3D rotation that aligns edges1 → edges2
    coarse_space = RotationSearchSpace(
        x_start=COARSE_X_START, x_end=COARSE_X_END, x_inc=COARSE_X_INC,
        y_start=COARSE_Y_START, y_end=COARSE_Y_END, y_inc=COARSE_Y_INC,
        z_start=COARSE_Z_START, z_end=COARSE_Z_END, z_inc=COARSE_Z_INC,
    )

    output_mat, mat_size, candidates = generate_rotation_candidates(edge1, coarse_space, best_ball1)

    comparison_csv_data = []
    best_candidate_index, comparison_csv_data = compare_candidate_angle_images(
        edge2, output_mat, candidates, mat_size
    )

    rotation_result = np.array([0.0, 0.0, 0.0])

    if best_candidate_index < 0:
        return rotation_result

    write_spin_analysis_CSV_files = True

    if write_spin_analysis_CSV_files:
        csv_dir = "./data/spin"
        os.makedirs(csv_dir, exist_ok=True)
        csv_fname_coarse = os.path.join(csv_dir, "spin_analysis_coarse.csv")
        with open(csv_fname_coarse, "w") as csv_file_coarse:
            for element in comparison_csv_data:
                csv_file_coarse.write(element)

    c = candidates[best_candidate_index]

    final_search_space = RotationSearchSpace(
        x_start=c.x_rotation_degrees - COARSE_X_INC // 2,
        x_end=c.x_rotation_degrees + COARSE_X_INC // 2,
        x_inc=1,
        y_start=c.y_rotation_degrees - COARSE_Y_INC // 2,
        y_end=c.y_rotation_degrees + COARSE_Y_INC // 2,
        y_inc=COARSE_Y_INC // 2,
        z_start=c.z_rotation_degrees - COARSE_Z_INC // 2,
        z_end=c.z_rotation_degrees + COARSE_Z_INC // 2,
        z_inc=1
    )

    foutput_mat, fmat_size, fcandidates  = generate_rotation_candidates(edge1, final_search_space, best_ball1)

    best_candidate_index, comparison_csv_data = compare_candidate_angle_images(
        edge2, foutput_mat, fcandidates, fmat_size
    )

    if write_spin_analysis_CSV_files:
        csv_fname_fine = os.path.join(csv_dir, "spin_analysis_fine.csv")
        with open(csv_fname_fine, "w") as csv_file_fine:
            for element in comparison_csv_data:
                csv_file_fine.write(element)

    best_rot_x, best_rot_y, best_rot_z = 0, 0, 0

    if best_candidate_index >= 0:
        final_c = fcandidates[best_candidate_index]
        best_rot_x = final_c.x_rotation_degrees
        best_rot_y = final_c.y_rotation_degrees
        best_rot_z = final_c.z_rotation_degrees
    else:
        rotation_result = np.array([0, 0, 0])
    
    result_bball2d_image = get_rotated_image(
        ball_image1,
        best_ball1,
        (best_rot_x,best_rot_y,best_rot_z)
    )
    cv2.imshow("Actual", ball_image2)
    cv2.imshow("Final rotated-by-best-angle originalBall1", result_bball2d_image)
    cv2.waitKey(0)

    rotation_result = np.array([best_rot_x, best_rot_y, best_rot_z])
    return rotation_result

if __name__ == '__main__':
    
    multiprocessing.freeze_support()
    
    import json
    with open('HyperParameter.json', 'r') as file:
        params = json.load(file)
    
    test_img_path1 = input("Enter the path to the first test image: ").strip()
    test_img_path2 = input("Enter the path to the second test image: ").strip()
    if not test_img_path1 or not test_img_path2:
        test_img_path1 = params.get("image_path1", "data/Images/Screenshot 2025-06-06 112413.png")
        test_img_path2 = params.get("image_path2", "data/Images/Screenshot 2025-06-06 112341.png")
    
    delta_t = params.get("delta_t", 1/3000)

    test_img1 = cv2.imread(test_img_path1, cv2.IMREAD_GRAYSCALE)
    test_img2 = cv2.imread(test_img_path2, cv2.IMREAD_GRAYSCALE)
    if test_img1 is not None and test_img2 is not None:
        best_rot_x, best_rot_y, best_rot_z = get_fine_ball_rotation(test_img1, test_img2, compress_candidates=True)

        side_spin_rpm = (best_rot_x / delta_t) * (60 / 360)
        back_spin_rpm = (best_rot_y / delta_t) * (60 / 360)

        print(f"Side Spin: {side_spin_rpm} rpm")
        print(f"Back Spin: {back_spin_rpm} rpm")
    else:
        print(f"Failed to load test images from: {test_img_path1} and/or {test_img_path2}. Please check the paths and ensure the images exist.")
