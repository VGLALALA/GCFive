import cv2
from typing import Tuple
import numpy as np
from typing import Tuple
from dataclasses import dataclass
from IsolateCode import isolate_ball
from ApplyGaborFilter import apply_gabor_filter_to_ball
from RemoveReflection import remove_reflections
from MaskAreaOutsideBall import mask_area_outside_ball
from GetRotatedImage import get_rotated_image
from GenerateRotationCandidate import generate_rotation_candidates
from CompareRotationImage import compare_rotation_image
from matchBallSize import match_ball_image_sizes
from GenerateRotationCandidate import RotationSearchSpace
from GolfBall import GolfBall
from ROI import run_hough_with_radius
import time
from ApplyGaborFilter import apply_gabor_filter_image

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
) -> Tuple[float,float,float]:
    """
    Returns (spin_x, spin_y, spin_z) in degrees, corresponding to side-, back-, and axial-spin.
    """
    print("Step 1")
    # 1) Isolate each ball into its own tight crop
    best_ball1,best_ball2 = run_hough_with_radius(full_gray_image)
    print("test 1")

    # Show the isolated balls
    ball_image1 = isolate_ball(full_gray_image, best_ball1)
    ball_image2 = isolate_ball(full_gray_image, best_ball2)
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
    # cv2.imshow("Gabor Edges 1 (clean)", edge1_clean)
    # cv2.imshow("Gabor Edges 2 (clean)", edge2_clean)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    print("step 4")

    # 4) Remove specular reflections
    gaberRefRemoved1 = remove_reflections(ball_image1, edge1_clean)
    gaberRefRemoved2 = remove_reflections(ball_image2, edge2_clean)
    

    # 5) Mask out everything outside the ball's circle
    
    FINAL_MASK_FACTOR = 0.92
    gaberRefRemoved1 = mask_area_outside_ball(gaberRefRemoved1, best_ball1, FINAL_MASK_FACTOR, mask_value=(255, 255, 255))
    gaberRefRemoved2 = mask_area_outside_ball(gaberRefRemoved2, best_ball2, FINAL_MASK_FACTOR, mask_value=(255, 255, 255))
    # cv2.imshow("Gabor Edges 1 (clean)", gaberRefRemoved1)
    # cv2.imshow("Gabor Edges 2 (clean)", gaberRefRemoved2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # time.sleep(10222)

    # 6) De-rotate each image half the perspective offset so both appear "centered"
    print("step 6")
    offset1 = np.array(best_ball1.angles_camera_ortho_perspective[:2], dtype=np.float32)
    offset2 = np.array(best_ball2.angles_camera_ortho_perspective[:2], dtype=np.float32)
    delta = (offset2 - offset1) * 0.5
    delta[1] *= -1  # match your C++ sign convention
    print(offset1,offset2,delta)
    
    unrotatedBallImg1DimpleEdges = gaberRefRemoved1.copy()
    unrotatedBallImg2DimpleEdges = gaberRefRemoved2.copy()
    
    edges1 = get_rotated_image(unrotatedBallImg1DimpleEdges, best_ball1, tuple(delta.astype(int)))
    edges2 = get_rotated_image(unrotatedBallImg2DimpleEdges, best_ball2, tuple((-delta).astype(int)))
    cv2.imshow("Edges 1", edges1)
    cv2.imshow("Edges 2", edges2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 7) Coarse-search for best 3D rotation that aligns edges1 → edges2
    coarse_space = RotationSearchSpace(
        x_start=COARSE_X_START, x_end=COARSE_X_END, x_inc=COARSE_X_INC,
        y_start=COARSE_Y_START, y_end=COARSE_Y_END, y_inc=COARSE_Y_INC,
        z_start=COARSE_Z_START, z_end=COARSE_Z_END, z_inc=COARSE_Z_INC,
    )
    candidates = generate_rotation_candidates(edges1, coarse_space, local_ball1)
    best_idx = compare_rotation_image(edges2, candidates)
    best_coarse = candidates[best_idx]

    # 8) Fine-search around that best candidate
    fine_space = best_coarse.make_refined_search_space()
    final_candidates = generate_rotation_candidates(edges1, fine_space, local_ball1)
    best_idx = compare_rotation_image(edges2, final_candidates)
    best_final = final_candidates[best_idx]

    # 9) Normalize to real-world spin axes
    spin_offset = offset1 + delta
    spin_offset_rad = np.radians(spin_offset)
    bx, by, bz = best_final.x, best_final.y, best_final.z

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
test_image_path = "/home/vglalala/GCFive/Images/log_cam2_last_strobed_img.png"
test_img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)


get_ball_rotation(test_img)
