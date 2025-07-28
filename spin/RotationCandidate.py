from dataclasses import dataclass

import numpy as np


@dataclass
class RotationCandidate:
    index: int
    img: np.ndarray
    x_rotation_degrees: float
    y_rotation_degrees: float
    z_rotation_degrees: float
    score: float = 0.0
