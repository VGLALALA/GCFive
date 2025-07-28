from dataclasses import dataclass


@dataclass
class RotationSearchSpace:
    x_inc: int
    x_start: int
    x_end: int
    y_inc: int
    y_start: int
    y_end: int
    z_inc: int
    z_start: int
    z_end: int
