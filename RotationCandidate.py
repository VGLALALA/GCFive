from dataclasses import dataclass
import numpy as np

@dataclass
class RotationCandidate:
    index: int
    img: np.ndarray
    x_rotation_degrees: int
    y_rotation_degrees: int
    z_rotation_degrees: int
    score: float = 0.0