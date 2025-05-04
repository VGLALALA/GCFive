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
    ball: GolfBall  # your existing Python GolfBall class
) -> Tuple[
    List[RotationCandidate],
    np.ndarray,
    Tuple[int,int,int]
]:
    """
    Generate all rotated‐ball candidates over the given search_space.
    Returns (candidates, index_matrix, (xSize,ySize,zSize)).
    """
    # Compute grid dimensions
    x_size = math.ceil((search_space.x_end   - search_space.x_start) / search_space.x_inc) + 1
    y_size = math.ceil((search_space.y_end   - search_space.y_start) / search_space.y_inc) + 1
    z_size = math.ceil((search_space.z_end   - search_space.z_start) / search_space.z_inc) + 1

    # 3D index matrix
    index_mat = np.zeros((x_size, y_size, z_size), dtype=np.uint16)

    candidates: List[RotationCandidate] = []
    idx = 0

    for ix, x_deg in enumerate(range(
            search_space.x_start,
            search_space.x_end + 1,
            search_space.x_inc
        )):
        for iy, y_deg in enumerate(range(
                search_space.y_start,
                search_space.y_end + 1,
                search_space.y_inc
            )):
            for iz, z_deg in enumerate(range(
                    search_space.z_start,
                    search_space.z_end + 1,
                    search_space.z_inc
                )):
                # Project the 2D dimple edges onto a 3D ball at these Euler angles:
                img3d = project_2d_image_to_3d_ball(
                    base_dimple_image,
                    ball,
                    (x_deg, y_deg, z_deg)
                )

                # Record candidate
                c = RotationCandidate(
                    index=idx,
                    x_deg=x_deg,
                    y_deg=y_deg,
                    z_deg=z_deg,
                    img_3d=img3d
                )
                candidates.append(c)

                # Store index in our 3D grid
                index_mat[ix, iy, iz] = idx

                idx += 1

    return candidates, index_mat, (x_size, y_size, z_size)
