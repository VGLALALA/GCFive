import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from GolfBall import GolfBall
from Project2dImageTo3dBall import project_2d_image_to_3d_ball

@dataclass
class RotationCandidate:
    index: int
    x_deg: int
    y_deg: int
    z_deg: int
    img_3d: np.ndarray

@dataclass
class RotationSearchSpace:
    x_start: int
    x_end:   int
    x_inc:   int
    y_start: int
    y_end:   int
    y_inc:   int
    z_start: int
    z_end:   int
    z_inc:   int

def generate_rotation_candidates(
    base_dimple_image: np.ndarray,
    search_space: RotationSearchSpace,
    ball: GolfBall
) -> List[np.ndarray]:
    """
    Generate all rotated ball candidates over the given search space.
    Returns a list of rotated images.
    """
    candidates = []

    for x_deg in range(
            search_space.x_start,
            search_space.x_end + 1,
            search_space.x_inc
        ):
        print(x_deg)
        for y_deg in range(
                search_space.y_start,
                search_space.y_end + 1,
                search_space.y_inc
            ):
            for z_deg in range(
                    search_space.z_start,
                    search_space.z_end + 1,
                    search_space.z_inc
                ):
                # Project the 2D dimple edges onto a 3D ball at these Euler angles
                rotated_img = project_2d_image_to_3d_ball(
                    base_dimple_image,
                    ball,
                    (x_deg, y_deg, z_deg)
                )
                candidates.append(rotated_img)

    return candidates
