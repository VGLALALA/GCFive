import cv2
import numpy as np
from typing import Tuple
from dataclasses import dataclass
from GolfBall import GolfBall
from IsolateCode import isolate_ball
from ApplyGaborFilter import apply_gabor_filter_to_ball
from RemoveReflection import remove_reflections
from MaskAreaOutsideBall import mask_area_outside_ball
from GetRotatedImage import get_rotated_image
from GenerateRotationCandidate import generate_rotation_candidates
from CompareRotationImage import compare_rotation_image
from matchBallSize import match_ball_image_sizes


COARSE_X_INC   = 6
COARSE_X_START = -42
COARSE_X_END   = 42

COARSE_Y_INC   = 5
COARSE_Y_START = -30
COARSE_Y_END   = 30

COARSE_Z_INC   = 6
COARSE_Z_START = -50
COARSE_Z_END   = 60

@dataclass
class RotationSearchSpace:
    x_inc: int
    x_start: int
    x_end: int
    y_inc: int
    y_start: int
    y_end: int
    z_inc: int
    z_start: int
    z_end: int



def get_ball_rotation(
    full_gray_image1: np.ndarray,
    ball1: GolfBall,
    full_gray_image2: np.ndarray,
    ball2: GolfBall,
) -> Tuple[float,float,float]:
    """
    Returns (spin_x, spin_y, spin_z) in degrees, corresponding to side-, back-, and axial-spin.
    """

    # 1) Isolate each ball into its own tight crop
    ball_image1, local_ball1 = isolate_ball(full_gray_image1, ball1)
    ball_image2, local_ball2 = isolate_ball(full_gray_image2, ball2)

    # 2) Resize so both crops are the same size
    ball_image1, ball_image2, local_ball1, local_ball2 = \
        match_ball_image_sizes(ball_image1, ball_image2, local_ball1, local_ball2)

    # 3) Apply Gabor filters to pick out dimple edges
    edges1 = apply_gabor_filter_to_ball(ball_image1, local_ball1)
    edges2 = apply_gabor_filter_to_ball(ball_image2, local_ball2)

    # 4) Remove specular reflections
    remove_reflections(ball_image1, edges1)
    remove_reflections(ball_image2, edges2)

    # 5) Mask out everything outside the ball’s circle
    FINAL_MASK_FACTOR = 0.92
    edges1 = mask_area_outside_ball(edges1, local_ball1, FINAL_MASK_FACTOR, ignore_value=128)
    edges2 = mask_area_outside_ball(edges2, local_ball2, FINAL_MASK_FACTOR, ignore_value=128)

    # 6) De-rotate each image half the perspective offset so both appear “centered”
    offset1 = np.array(local_ball1.angles_camera_ortho_perspective[:2], dtype=np.float32)
    offset2 = np.array(local_ball2.angles_camera_ortho_perspective[:2], dtype=np.float32)
    delta = (offset2 - offset1) * 0.5
    delta[1] *= -1  # match your C++ sign convention

    edges1 = get_rotated_image(edges1, local_ball1, tuple(delta.astype(int)))
    edges2 = get_rotated_image(edges2, local_ball2, tuple((-delta).astype(int)))

    # 7) Coarse‐search for best 3D rotation that aligns edges1 → edges2
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
