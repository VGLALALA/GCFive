import cv2
from typing import Tuple
import numpy as np
from typing import Tuple
from IsolateCode import isolate_ball
from RemoveReflection import remove_reflections
from MaskAreaOutsideBall import mask_area_outside_ball
from GetRotatedImage import get_rotated_image
from GenerateRotationCandidate import generate_rotation_candidates
from CompareRotationImage import compare_rotation_image
from matchBallSize import match_ball_image_sizes
from RotationSearchSpace import RotationSearchSpace
from ROI import run_hough_with_radius
from CompareCandidateAngleImage import compare_candidate_angle_images
from ApplyGaborFilter import apply_gabor_filter_image, apply_gabor_filter_to_ball
import time

COARSE_X_INC   = 6
COARSE_X_START = -42
COARSE_X_END   = 42

COARSE_Y_INC   = 5
COARSE_Y_START = -30
COARSE_Y_END   = 30

COARSE_Z_INC   = 6
COARSE_Z_START = -50
COARSE_Z_END   = 60

def get_ball_rotation(
    full_gray_image: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Returns (spin_x, spin_y, spin_z) in degrees, corresponding to side-, back-, and axial-spin.
    """
    # Display the full gray image
    cv2.imshow("Full Gray Image", full_gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("Step 1: Isolating each ball into its own tight crop")
    # 1) Isolate each ball into its own tight crop
    best_ball1, best_ball2 = run_hough_with_radius(full_gray_image)
    print("Balls isolated")

    # Show the isolated balls
    ball_image1 = isolate_ball(full_gray_image, best_ball1)
    ball_image2 = isolate_ball(full_gray_image, best_ball2)
    
    # Display the isolated ball images
    cv2.imshow("Isolated Ball 1", ball_image1)
    cv2.imshow("Isolated Ball 2", ball_image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow("Ball Image 1", ball_image1)
    cv2.imshow("Ball Image 2", ball_image2)
    cv2.waitKey(0)
    # Update the center coordinates of best_ball1 and best_ball2
    best_ball1.x = ball_image1.shape[1] // 2
    best_ball1.y = ball_image1.shape[0] // 2

    best_ball2.x = ball_image2.shape[1] // 2
    best_ball2.y = ball_image2.shape[0] // 2
    # --- END TEST ---
    print("step 2")
    # 2) Resize so both crops are the same size
    ball_image1, ball_image2 = match_ball_image_sizes(ball_image1, ball_image2)

    print("step 3")
    # 3) Apply Gabor filters to pick out dimple edges
    ball_image1 = cv2.equalizeHist(ball_image1)
    
    ball_image2 = cv2.equalizeHist(ball_image2)
    edge1 = apply_gabor_filter_image(ball_image1)
    edge2 = apply_gabor_filter_image(ball_image2)
    
    # Clean up the edge maps
    kernel = np.ones((1, 1), np.uint8)
    edge1_clean = cv2.morphologyEx(edge1, cv2.MORPH_OPEN, kernel)
    edge1_clean = cv2.morphologyEx(edge1_clean, cv2.MORPH_CLOSE, kernel)
    edge2_clean = cv2.morphologyEx(edge2, cv2.MORPH_OPEN, kernel)
    edge2_clean = cv2.morphologyEx(edge2_clean, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("Gabor Edges 1 (clean)", edge1_clean)
    cv2.imshow("Gabor Edges 2 (clean)", edge2_clean)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("step 4")

    # 4) Remove specular reflections
    gaberRefRemoved1 = remove_reflections(ball_image1, edge1_clean)
    gaberRefRemoved2 = remove_reflections(ball_image2, edge2_clean)
    cv2.imshow("Gabor Edges 1 (clean)", gaberRefRemoved1)
    cv2.imshow("Gabor Edges 2 (clean)", gaberRefRemoved2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 5) Mask out everything outside the ball's circle
    print("step 5")
    FINAL_MASK_FACTOR = 0.92
    gaberRefRemoved1 = mask_area_outside_ball(gaberRefRemoved1, best_ball1, FINAL_MASK_FACTOR, mask_value=(255, 255, 255))
    gaberRefRemoved2 = mask_area_outside_ball(gaberRefRemoved2, best_ball2, FINAL_MASK_FACTOR, mask_value=(255, 255, 255))
    cv2.imshow("Gabor Edges 1 (clean)", gaberRefRemoved1)
    cv2.imshow("Gabor Edges 2 (clean)", gaberRefRemoved2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 6) De-rotate each image half the perspective offset so both appear "centered"
    print("step 6")
    offset1 = np.array(best_ball1.angles_camera_ortho_perspective[:2], dtype=np.float32)
    offset2 = np.array(best_ball2.angles_camera_ortho_perspective[:2], dtype=np.float32)
    # Compute half-perspective offset in X/Y and pad Z-axis rotation with zero
    delta_float = (offset2 - offset1) / 2.0
    delta_float[1] *= -1.0  # Account for how our rotations are signed
    delta2d = np.round(delta_float).astype(int)
    delta = np.array([delta2d[0], delta2d[1], 0], dtype=int)

    adjustedimg1 = get_rotated_image(gaberRefRemoved1, best_ball1, tuple(delta))
    print(f"Adjusting rotation for camera view of ball 1 to offset (x,y,z)={delta[0]},{delta[1]},{delta[2]}")
    cv2.imshow("Adjusted Image", adjustedimg1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # time.sleep(909090)
    # Compute remaining offset, invert Y sign, and pad Z-axis with zero
    delta2d = np.round(-(offset2 - offset1 - delta_float)).astype(int)
    delta2d[1] = -delta2d[1]
    delta2 = np.array([delta2d[0], delta2d[1], 0], dtype=int)

    adjustedimg2 = get_rotated_image(gaberRefRemoved2, best_ball2, tuple(delta2))
    print(f"Adjusting rotation for camera view of ball 2 to offset (x,y,z)={delta2[0]},{delta2[1]},{delta2[2]}")
    cv2.imshow("Final perspective-de-rotated filtered ball_image1DimpleEdges", adjustedimg1)
    cv2.imshow("Final perspective-de-rotated filtered ball_image2DimpleEdges", adjustedimg2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 7) Coarse-search for best 3D rotation that aligns edges1 → edges2
    print("step 7")
    coarse_space = RotationSearchSpace(
        x_start=COARSE_X_START, x_end=COARSE_X_END, x_inc=COARSE_X_INC,
        y_start=COARSE_Y_START, y_end=COARSE_Y_END, y_inc=COARSE_Y_INC,
        z_start=COARSE_Z_START, z_end=COARSE_Z_END, z_inc=COARSE_Z_INC,
    )
    output_mat, mat_size, candidates = generate_rotation_candidates(edge1_clean, coarse_space, best_ball1)

    print("step 8")
    comparison_csv_data = []
    best_candidate_index, comparison_csv_data = compare_candidate_angle_images(
        adjustedimg2, output_mat, candidates, mat_size
    )

    rotation_result = np.array([0.0, 0.0, 0.0])

    if best_candidate_index < 0:
        print("Warning: No best candidate found.")
        return rotation_result

    write_spin_analysis_CSV_files = True

    if write_spin_analysis_CSV_files:
        csv_fname_coarse = "./data/spin/spin_analysis_coarse.csv"
        print(f"Writing CSV spin data to: {csv_fname_coarse}")
        with open(csv_fname_coarse, "w") as csv_file_coarse:
            for element in comparison_csv_data:
                csv_file_coarse.write(element)

    c = candidates[best_candidate_index]

    print(f"Best Coarse Initial Rotation Candidate was #{best_candidate_index} - Rot: ({c.x_rotation_degrees}, {c.y_rotation_degrees}, {c.z_rotation_degrees})")

    print("step 9")

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

    foutput_mat, fmat_size, fcandidates  = generate_rotation_candidates(edge1_clean, final_search_space, best_ball1)
    best_candidate_index, comparison_csv_data = compare_candidate_angle_images(
        adjustedimg2, foutput_mat, fcandidates, fmat_size
    )

    if write_spin_analysis_CSV_files:
        csv_fname_fine = "./data/spin/spin_analysis_fine.csv"
        print(f"Writing CSV spin data to: {csv_fname_fine}")
        with open(csv_fname_fine, "w") as csv_file_fine:
            for element in comparison_csv_data:
                csv_file_fine.write(element)

    best_rot_x, best_rot_y, best_rot_z = 0, 0, 0

    if best_candidate_index >= 0:
        final_c = fcandidates[best_candidate_index]
        best_rot_x = final_c.x_rotation_degrees
        best_rot_y = final_c.y_rotation_degrees
        best_rot_z = final_c.z_rotation_degrees

        print(f"Best Raw Fine (and final) Rotation Candidate was #{best_candidate_index} - Rot: ({best_rot_x}, {best_rot_y}, {best_rot_z})")
    else:
        print("Warning: No best final candidate found. Returning 0,0,0 spin results.")
        rotation_result = np.array([0, 0, 0])
    
    print("step 10")
    spin_offset = offset1 + delta
    spin_offset_rad = np.radians(spin_offset)
    bx, by, bz = best_rot_x, best_rot_y, best_rot_z

    norm_x = int(round( bx * np.cos(spin_offset_rad[1]) + bz * np.sin(spin_offset_rad[1]) ))
    norm_y = int(round( by * np.cos(spin_offset_rad[0]) - bz * np.sin(spin_offset_rad[0]) ))
    norm_z = int(round( bz * np.cos(spin_offset_rad[0]) * np.cos(spin_offset_rad[1])
                        - by * np.sin(spin_offset_rad[0])
                        - bx * np.sin(spin_offset_rad[1]) ))

    # C-golfer convention: side-spin positive when surface moves right→left
    norm_x = -norm_x

    return (norm_x, norm_y, norm_z)

    # Test with sample image
    # Test with sample image path
test_image_path = "./data/Images/log_cam2_last_strobed_img.png"
test_img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)


print(get_ball_rotation(test_img))
