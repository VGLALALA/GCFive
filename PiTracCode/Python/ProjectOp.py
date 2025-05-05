import math
from typing import Tuple

K_PIXEL_IGNORE_VALUE = 128

class ProjectionOp:
    def __init__(self, ball, projected_img, rotation: Tuple[int, int, int]):
        """
        ball: an object with attributes
            - measured_radius_pixels: float
            - x: float   (ball center column)
            - y: float   (ball center row)
        projected_img: numpy array of shape (rows, cols, 2), dtype=int32
        rotation: (angle_x, angle_y, angle_z)
        """
        self.ball = ball
        self.proj = projected_img

        # Convert to radians (note the negative on X to match your C++ code)
        ax, ay, az = rotation
        self.x_rad = -math.radians(ax)
        self.y_rad = math.radians(ay)
        self.z_rad = math.radians(az)

        # Precompute sines / cosines
        self.sinX, self.cosX = math.sin(self.x_rad), math.cos(self.x_rad)
        self.sinY, self.cosY = math.sin(self.y_rad), math.cos(self.y_rad)
        self.sinZ, self.cosZ = math.sin(self.z_rad), math.cos(self.z_rad)

        # Flags for whether to bother rotating around each axis
        self.rotX = abs(self.x_rad) > 1e-3
        self.rotY = abs(self.y_rad) > 1e-3
        self.rotZ = abs(self.z_rad) > 1e-3

    def get_ball_z(self, col, row):
        """
        Project (col,row) onto the hemisphere centered at (ball.x, ball.y)
        Returns (col_offset, row_offset, z) where z >= 0 on the visible hemisphere.
        """
        r = self.ball.measured_radius_pixels
        cx, cy = self.ball.x, self.ball.y
        dx = col - cx
        dy = row - cy

        # outside bounding box?
        if abs(dx) > r or abs(dy) > r:
            return dx, dy, 0.0

        diff = r * r - (dx * dx + dy * dy)
        if diff <= 0:
            return dx, dy, 0.0

        return dx, dy, math.sqrt(diff)

    def project_pixel(self, col, row, pixel_value):
        # 1) find 3D point on unrotated hemisphere
        dx, dy, dz = self.get_ball_z(col, row)
        prerot_invalid = (dz <= 1e-4)

        # if off-sphere, just mark ignore and bail
        if prerot_invalid:
            self.proj[row, col, 0] = int(dz)
            self.proj[row, col, 1] = K_PIXEL_IGNORE_VALUE
            return

        # 2) apply X-axis rotation (around horizontal axis)
        if self.rotX:
            dy, dz = dy * self.cosX - dz * self.sinX, dy * self.sinX + dz * self.cosX

        # 3) apply Y-axis rotation (around vertical axis)
        if self.rotY:
            dx, dz = dx * self.cosY + dz * self.sinY, dz * self.cosY - dx * self.sinY

        # 4) apply Z-axis rotation (in-plane)
        if self.rotZ:
            dx, dy = dx * self.cosZ - dy * self.sinZ, dx * self.sinZ + dy * self.cosZ

        # 5) map back to image coords
        new_col = dx + self.ball.x
        new_row = dy + self.ball.y

        # only write if it lands inside the image and still on front hemisphere
        rows, cols = self.proj.shape[:2]
        if 0 <= new_col < cols and 0 <= new_row < rows and dz > 0:
            rc = min(max(int(new_col + 0.5), 0), cols - 1)
            rr = min(max(int(new_row + 0.5), 0), rows - 1)
            self.proj[rr, rc, 0] = int(dz)
            self.proj[rr, rc, 1] = (K_PIXEL_IGNORE_VALUE if prerot_invalid else pixel_value)
