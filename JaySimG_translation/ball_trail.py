from dataclasses import dataclass, field
import numpy as np
from .vector import vec3


@dataclass
class BallTrail:
    points: list = field(default_factory=lambda: [vec3(), vec3()])

    def add_point(self, point):
        self.points.append(np.array(point, dtype=float))

    def clear_points(self):
        self.points = [vec3(), vec3()]
