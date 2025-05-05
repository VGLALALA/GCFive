from dataclasses import dataclass
@dataclass
class RotationSearchSpace:
    # X-axis rotation range
    x_start: int
    x_end: int
    x_inc: int
    # Y-axis rotation range
    y_start: int
    y_end: int
    y_inc: int
    # Z-axis rotation range
    z_start: int
    z_end: int
    z_inc: int