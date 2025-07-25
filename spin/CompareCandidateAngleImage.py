import time
import math
from typing import List, Tuple
from .RotationCandidate import RotationCandidate
import time
import numpy as np
from typing import List, Tuple
from .CompareRotationImage import compare_rotation_image

def compare_candidate_angle_images(
    target_image: np.ndarray,
    candidate_elements_mat: np.ndarray,
    candidates: List[RotationCandidate],
    candidate_size: Tuple[int,int,int],
    serialize_debug: bool = False
) -> Tuple[int, List[str]]:
    """
    Port of BallImageProc::CompareCandidateAngleImages to Python.

    Args:
        target_image: H×W grayscale array.
        candidate_elements_mat: 3D array (X×Y×Z) of uint indices into `candidates`.
        candidates: pre-sized list of RotationCandidate objects.
        serialize_debug: if True, iterates explicitly rather than vectorized.

    Returns:
        best_index: index of the best candidate.
        comparison_csv_data: list of TSV strings for each candidate.
    """
    # Print the shape of the target image
    print(f"Shape of target_image: {target_image.shape}")

    # candidate_elements_mat is a 3D array of indices into candidates list
    # Get dimensions for iteration
    xSize, ySize, zSize = candidate_size
    num_candidates = xSize * ySize * zSize
    comparison_data = [""] * num_candidates

    start = time.perf_counter()

    # 1) Setup & iteration
    if serialize_debug:
        for x in range(xSize):
            for y in range(ySize):
                for z in range(zSize):
                    idx = int(candidate_elements_mat[x, y, z])
                    c = candidates[idx]
                    # print(f"Shape of c.img: {c.img.shape}")
                    pm, pe, _ = compare_rotation_image(target_image, c.img, c.index)
                    c.pixels_matching = pm
                    c.pixels_examined = pe
                    c.score = pm / pe if pe else 0.0
                    comparison_data[c.index] = (
                        f"{c.index}\t{c.x_rotation_degrees}\t"
                        f"{c.y_rotation_degrees}\t{c.z_rotation_degrees}\t"
                        f"{pm}\t{pe}\t{c.score:.6f}\n"
                    )
    else:
        for idx_flat in np.ndindex(candidate_elements_mat.shape):
            idx = int(candidate_elements_mat[idx_flat])
            c = candidates[idx]
            # print(f"Shape of c.img: {c.img.shape}")
            pm, pe, _ = compare_rotation_image(target_image, c.img, c.index)
            c.pixels_matching = pm
            c.pixels_examined = pe
            c.score = pm / pe if pe else 0.0
            comparison_data[c.index] = (
                f"{c.index}\t{c.x_rotation_degrees}\t"
                f"{c.y_rotation_degrees}\t{c.z_rotation_degrees}\t"
                f"{pm}\t{pe}\t{c.score:.6f}\n"
            )

    # 2) Find maxima
    max_pixels_examined = max(c.pixels_examined for c in candidates)
    max_pixels_matching = max(c.pixels_matching for c in candidates)

    # 3) Apply low-count penalty
    kPower = 2.0
    kScale = 1000.0
    kWeight = 500.0

    best_score = -np.inf
    best_index = -1
    for c in candidates:
        low_penalty = ((max_pixels_examined - c.pixels_examined) / kWeight) ** kPower / kScale
        final_score = (c.score * 10.0) - low_penalty
        if final_score > best_score:
            best_score = final_score
            best_index = c.index

    elapsed = time.perf_counter() - start
    print(f"CompareCandidateAngleImages: {elapsed:.6f}s")

    return best_index, comparison_data
