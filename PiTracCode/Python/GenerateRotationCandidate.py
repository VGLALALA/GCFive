from concurrent.futures import ProcessPoolExecutor, as_completed
from RotationSearchSpace import RotationSearchSpace
from RotationCandidate import RotationCandidate
from Project2dImageTo3dBall import project_2d_image_to_3d_ball
import numpy as np
import time
from functools import partial

def _worker_task(task, base_image, ball):
    """
    Helper for multiprocessing: projects one angle tuple into a 3D image.
    Args:
        task: (idx, x_idx, y_idx, z_idx, x_deg, y_deg, z_deg)
        base_image: input dimple image
        ball: GolfBall-like object
    Returns:
        (idx, x_idx, y_idx, z_idx, img3d)
    """
    idx, x_idx, y_idx, z_idx, x_deg, y_deg, z_deg = task
    img3d = project_2d_image_to_3d_ball(base_image, ball, (x_deg, y_deg, z_deg))
    return idx, x_idx, y_idx, z_idx, img3d

def generate_rotation_candidates(base_dimple_image: np.ndarray,
                                 search_space: RotationSearchSpace,
                                 ball) -> tuple[np.ndarray, tuple[int, int, int], list]:
    """
    Generate a set of rotated candidate images over the specified search space,
    in parallel using multiprocessing.
    Returns:
        output_mat:   uint16 numpy array of shape (xSize, ySize, zSize),
                      containing the index of each candidate in the flat list
        mat_size:     tuple (xSize, ySize, zSize)
        candidates:   list of RotationCandidate instances
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

    # Build flat list of tasks: each is (idx, x_idx, y_idx, z_idx, x_deg, y_deg, z_deg)
    tasks = []
    idx = 0
    for x_idx, x_deg in enumerate(range(xs, xe + 1, xi)):
        for y_idx, y_deg in enumerate(range(ys, ye + 1, yi)):
            for z_idx, z_deg in enumerate(range(zs, ze + 1, zi)):
                tasks.append((idx, x_idx, y_idx, z_idx, x_deg, y_deg, z_deg))
                idx += 1

    total = len(tasks)
    output_mat = np.zeros(mat_size, dtype=np.uint16)
    candidates = [None] * total

    # Partial to freeze base_image and ball
    worker = partial(_worker_task, base_image=base_dimple_image, ball=ball)

    # Spin up a pool of processes
    with ProcessPoolExecutor() as executor:
        futures = { executor.submit(worker, task): task[0] for task in tasks }
        for count, fut in enumerate(as_completed(futures), 1):
            idx, x_idx, y_idx, z_idx, img3d = fut.result()
            # record in output matrix
            output_mat[x_idx, y_idx, z_idx] = idx
            # build and store candidate
            x_deg, y_deg, z_deg = tasks[idx][4], tasks[idx][5], tasks[idx][6]
            candidates[idx] = RotationCandidate(
                index=idx,
                img=img3d,
                x_rotation_degrees=x_deg,
                y_rotation_degrees=y_deg,
                z_rotation_degrees=z_deg,
                score=0.0
            )
            # optional: simple progress indicator
            if count % (total // 10 or 1) == 0:
                print(f"Progress: {count/total*100:.1f}%")

    t1 = time.perf_counter()
    print(f"compute_candidate_angle_images took {t1 - t0:.3f}s, generated {total} candidates")

    return output_mat, mat_size, candidates
