import time
import math
from typing import List, Tuple

# Penalty constants (same as your C++)
_SPIN_LOW_COUNT_PENALTY_POWER           = 2.0
_SPIN_LOW_COUNT_PENALTY_SCALING_FACTOR  = 1000.0
_SPIN_LOW_COUNT_DIFF_WEIGHTING_FACTOR   = 500.0

class RotationCandidate:
    def __init__(self, index:int, x:int, y:int, z:int,
                 pixels_examined:float, pixels_matching:float, score:float):
        self.index               = index
        self.x_rotation_degrees  = x
        self.y_rotation_degrees  = y
        self.z_rotation_degrees  = z
        self.pixels_examined     = pixels_examined
        self.pixels_matching     = pixels_matching
        self.score               = score

def compare_candidate_angle_images(
    target_image,                     # np.ndarray, unused here but kept for API parity
    candidate_elements_mat,           # np.ndarray or similar, unused in Python port
    candidate_elements_mat_size,      # Tuple[int,int,int], e.g. (xSize,ySize,zSize)
    candidates: List[RotationCandidate],
    comparison_csv_data: List[str]
) -> int:
    """
    Returns the index into `candidates` of the best‐scaled‐score candidate.
    Also updates `comparison_csv_data` in place with any CSV lines you’ve generated upstream.
    """

    # Start timing (for debug/logging)
    t0 = time.perf_counter()

    xSize, ySize, zSize = candidate_elements_mat_size
    num_candidates = xSize * ySize * zSize

    # ----------------------------------------------------------------
    # Phase 1: find max pixels_examined and max pixels_matching
    max_pixels_examined = -1.0
    max_pixels_matching = -1.0
    max_pixels_examined_index = -1
    best_match_rot = (0,0,0)

    for c in candidates:
        if c.pixels_examined > max_pixels_examined:
            max_pixels_examined = c.pixels_examined
            max_pixels_examined_index = c.index

        if c.pixels_matching > max_pixels_matching:
            max_pixels_matching = c.pixels_matching
            best_match_rot = (c.x_rotation_degrees,
                              c.y_rotation_degrees,
                              c.z_rotation_degrees)

    # ----------------------------------------------------------------
    # Phase 2: compute scaled scores and pick the best
    max_scaled_score = -math.inf
    best_scaled_idx = -1
    best_scaled_rot = (0,0,0)

    for c in candidates:
        # penalty = ((maxExamined - thisExamined)/W)^P / S
        diff_ratio = (max_pixels_examined - c.pixels_examined) / _SPIN_LOW_COUNT_DIFF_WEIGHTING_FACTOR
        low_count_penalty = (diff_ratio ** _SPIN_LOW_COUNT_PENALTY_POWER) / _SPIN_LOW_COUNT_PENALTY_SCALING_FACTOR

        # scale the raw match score
        scaled_score = (c.score * 10.0) - low_count_penalty

        if scaled_score > max_scaled_score:
            max_scaled_score    = scaled_score
            best_scaled_idx     = c.index
            best_scaled_rot     = (c.x_rotation_degrees,
                                   c.y_rotation_degrees,
                                   c.z_rotation_degrees)

    # (Optional) print out debug info like in C++
    print(f"Best by pixels matching: rot=({best_match_rot[0]},"
          f"{best_match_rot[1]},{best_match_rot[2]})")
    print(f"Best by scaled score {max_scaled_score:.4f}: rot=({best_scaled_rot[0]},"
          f"{best_scaled_rot[1]},{best_scaled_rot[2]})")

    # Stop timing
    t1 = time.perf_counter()
    print(f"compare_candidate_angle_images: {t1-t0:.6f}s")

    # comparison_csv_data is assumed already populated upstream; return best index
    return best_scaled_idx
