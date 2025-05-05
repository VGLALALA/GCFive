import time
import numpy as np
import cv2
from typing import List, Tuple
from RotationCandidate import RotationCandidate
from CompareRotationImage import compare_rotation_image

def ImgComparsionOp(
    target_image: np.ndarray,
    candidate_elements_mat: np.ndarray,
    candidates: List[RotationCandidate],
) -> Tuple[int, List[str]]:
    """
    :param target_image:        H×W array
    :param candidate_elements_mat: 3D array of shape (X,Y,Z) of ints indexing into `candidates`
    :param candidates:          list of RotationCandidate, pre-sized to X*Y*Z
    :returns: best_index, comparison_csv_data
    """
    xSize, ySize, zSize = candidate_elements_mat.shape
    num_candidates = xSize * ySize * zSize
    comparison_data: List[str] = [''] * num_candidates

    # 1) compute per-candidate raw scores & CSV lines
    for x in range(xSize):
        for y in range(ySize):
            for z in range(zSize):
                idx = int(candidate_elements_mat[x, y, z])
                c = candidates[idx]

                pm, pe = compare_rotation_image(target_image, c.img, c.index)
                c.pixels_matching = pm
                c.pixels_examined = pe
                c.score = pm / pe if pe else 0.0

                comparison_data[c.index] = (
                    f"{c.index}\t"
                    f"{c.x_rotation_degrees}\t"
                    f"{c.y_rotation_degrees}\t"
                    f"{c.z_rotation_degrees}\t"
                    f"{pm}\t{pe}\t"
                    f"{c.score:.6f}\n"
                )

    # 2) find maxima
    max_pixels_examined = max(c.pixels_examined for c in candidates)

    # 3) apply low-count penalty & pick best
    kDiffFactor = 500.0
    kPenaltyPower = 2.0
    kPenaltyScale = 1000.0

    best_score = -np.inf
    best_idx   = -1

    for c in candidates:
        penalty = (((max_pixels_examined - c.pixels_examined) / kDiffFactor) ** kPenaltyPower) / kPenaltyScale
        final_score = (c.score * 10.0) - penalty
        if final_score > best_score:
            best_score = final_score
            best_idx   = c.index

    return best_idx, comparison_data


# Example usage
if __name__ == "__main__":
    X, Y, Z = 4, 4, 4
    H, W    = 100, 100
    target  = np.random.randint(0,256,(H,W),dtype=np.uint8)
    candidates = [RotationCandidate(i, np.random.randint(0,256,(H,W),dtype=np.uint8),0,0,0)
                  for i in range(X*Y*Z)]
    indices = np.arange(X*Y*Z, dtype=np.int32).reshape((X,Y,Z))
    best, csv = ImgComparsionOp(target, indices, candidates)
    print("Best candidate:", best)
    with open("results.tsv","w") as f:
        f.writelines(csv)
