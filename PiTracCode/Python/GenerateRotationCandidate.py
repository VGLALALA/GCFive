from RotationSearchSpace import RotationSearchSpace
import numpy as np
import time
from Project2dImageTo3dBall import project_2d_image_to_3d_ball
from RotationCandidate import RotationCandidate

def generate_rotation_candidates(base_dimple_image: np.ndarray,
                                   search_space: RotationSearchSpace,
                                   ball) -> tuple[np.ndarray, tuple[int, int, int], list]:
    """
    Generate a set of rotated candidate images over the specified search space.

    Args:
        base_dimple_image: grayscale or color image of the ball’s dimples (H×W×C or H×W)
        search_space: object with attributes:
            anglex_rotation_degrees_increment, start, end
            angley_rotation_degrees_increment, start, end
            anglez_rotation_degrees_increment, start, end
        ball: GolfBall-like object passed through to projection functions

    Returns:
        output_mat:   uint16 numpy array of shape (xSize, ySize, zSize),
                      containing the index of each candidate in the flat list
        mat_size:    tuple (xSize, ySize, zSize)
        candidates:  list of RotationCandidate instances
    """
    t0 = time.perf_counter()

    # Unpack increments and ranges
    xi, xs, xe = (search_space.x_inc, search_space.x_start, search_space.x_end)
    yi, ys, ye = (search_space.y_inc, search_space.y_start, search_space.y_end)
    zi, zs, ze = (search_space.z_inc, search_space.z_start, search_space.z_end)

    # Compute grid sizes
    xSize = int(np.floor((xe - xs) / xi)) + 1
    ySize = int(np.floor((ye - ys) / yi)) + 1
    zSize = int(np.floor((ze - zs) / zi)) + 1
    mat_size = (xSize, ySize, zSize)

    # Prepare output structures
    output_mat = np.zeros(mat_size, dtype=np.uint16)
    candidates = []

    idx = 0
    for x_idx, x_deg in enumerate(range(xs, xe+1, xi)):
        print("X_deg: " + str(x_deg))
        for y_idx, y_deg in enumerate(range(ys, ye+1, yi)):
            for z_idx, z_deg in enumerate(range(zs, ze+1, zi)):
                # Project to 3D hemisphere at (x_deg, y_deg, z_deg)
                ball3d = project_2d_image_to_3d_ball(base_dimple_image,
                                                     ball,
                                                     (x_deg, y_deg, z_deg))
                # Build candidate record
                c = RotationCandidate(
                    index=idx,
                    img=ball3d,
                    x_rotation_degrees=x_deg,
                    y_rotation_degrees=y_deg,
                    z_rotation_degrees=z_deg,
                    score=0.0
                )
                candidates.append(c)
                output_mat[x_idx, y_idx, z_idx] = idx
                idx += 1

                # Optional debug:
                # dbg = np.zeros_like(base_dimple_image)
                # unproject_3d_ball_to_2d_image(ball3d, dbg, ball)
                # cv2.imshow(f"Candidate {idx}", dbg)
                # cv2.waitKey(1)

    t1 = time.perf_counter()
    elapsed = t1 - t0
    print(f"compute_candidate_angle_images took {elapsed:.4f} s, generated {idx} candidates")

    return output_mat, mat_size, candidates
