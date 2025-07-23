import numpy as np
from typing import Tuple
from .CompareRotationImage import compare_rotation_image
from .Project2dImageTo3dBall import project_2d_image_to_3d_ball
from .GolfBall import GolfBall
from .RotationSearchSpace import RotationSearchSpace
from scipy.optimize import minimize


def optimize_rotation(
    base_image: np.ndarray,
    target_image: np.ndarray,
    ball: GolfBall,
    search_space: RotationSearchSpace
) -> Tuple[np.ndarray, float]:
    """
    Global refinement using Powell's method within bounds defined by search_space.

    Args:
        base_image: H×W numpy array of the base dimple image.
        target_image: H×W numpy array of the target edge image.
        ball: GolfBall instance containing radius/center.
        search_space: defines start, end, and inc for x, y, z axes.

    Returns:
        best_angles: optimized (x, y, z) in degrees.
        best_score: matching score.
    """
    # Objective: negative matching score (we minimize)
    def objective(angles):
        # enforce bounds
        x = np.clip(angles[0], search_space.x_start, search_space.x_end)
        y = np.clip(angles[1], search_space.y_start, search_space.y_end)
        z = np.clip(angles[2], search_space.z_start, search_space.z_end)
        img3d = project_2d_image_to_3d_ball(base_image, ball, (x, y, z))
        pm, pe, _ = compare_rotation_image(target_image, img3d, 0)
        score = pm/pe if pe else 0.0
        return -score

    # Initial guess: grid center
    x0 = (search_space.x_start + search_space.x_end)/2.0
    y0 = (search_space.y_start + search_space.y_end)/2.0
    z0 = (search_space.z_start + search_space.z_end)/2.0
    initial = np.array([x0, y0, z0], dtype=float)

    # Bounds for Powell need custom wrapper, so we clamp inside objective
    res = minimize(
        objective,
        initial,
        method='Powell',
        options={'maxiter': 50, 'disp': True}
    )

    best_angles = np.clip(res.x,
                          [search_space.x_start, search_space.y_start, search_space.z_start],
                          [search_space.x_end, search_space.y_end, search_space.z_end])
    best_score = -res.fun
    print(f"Powell optimization result: angles={best_angles}, score={best_score:.6f}")
    return best_angles, best_score
