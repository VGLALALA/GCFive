from dataclasses import dataclass

@dataclass
class GolfBall:
    x: float
    y: float
    measured_radius_pixels: float
    angles_camera_ortho_perspective: Tuple[float,float,float]