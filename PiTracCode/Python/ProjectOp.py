import cv2
import numpy as np
import math
from typing import Tuple
from GolfBall import GolfBall


class ProjectionOp:
    def __init__(
        self,
        ball: GolfBall,
        projected_img: np.ndarray,
        x_rad: float,
        y_rad: float,
        z_rad: float
    ):
        self.currentBall = ball
        self.projectedImg = projected_img
        self.rows, self.cols = projected_img.shape[:2]
        # store rotations in radians
        self.x_rot = x_rad
        self.y_rot = y_rad
        self.z_rot = z_rad
        # precompute trigs
        self.sinX = math.sin(x_rad)
        self.cosX = math.cos(x_rad)
        self.sinY = math.sin(y_rad)
        self.cosY = math.cos(y_rad)
        self.sinZ = math.sin(z_rad)
        self.cosZ = math.cos(z_rad)
        # thresholds for no-ops
        self.rotatingOnX = abs(x_rad) > 0.001
        self.rotatingOnY = abs(y_rad) > 0.001
        self.rotatingOnZ = abs(z_rad) > 0.001

    def get_ball_z(self, imageX: float, imageY: float) -> Tuple[float, float, float]:
        # Map 2D image coords to hemisphere Z coordinate
        r = self.currentBall.measured_radius_pixels
        cx = self.currentBall.x
        cy = self.currentBall.y
        x_c = imageX - cx
        y_c = imageY - cy
        # outside circle -> ignore
        if abs(x_c) > r or abs(y_c) > r:
            return x_c, y_c, 0.0
        diff = r*r - (x_c**2 + y_c**2)
        if diff < 0:
            return x_c, y_c, 0.0
        z = math.sqrt(diff)
        return x_c, y_c, z

    def __call__(self, pixelValue: int, imageX: int, imageY: int):
        x_c, y_c, z = self.get_ball_z(imageX, imageY)
        prerotated_invalid = z <= 1e-4
        # write ignore for off-sphere
        if prerotated_invalid:
            self.projectedImg[imageY, imageX, 0] = int(z)
            self.projectedImg[imageY, imageX, 1] = 128  # ignore value
            return
        # rotate point in 3D
        imgXc, imgYc, imgZ = x_c, y_c, z
        if self.rotatingOnX:
            tmp = imgYc
            imgYc = imgYc * self.cosX - imgZ * self.sinX
            imgZ = tmp * self.sinX + imgZ * self.cosX
        if self.rotatingOnY:
            tmp = imgXc
            imgXc = imgXc * self.cosY + imgZ * self.sinY
            imgZ = imgZ * self.cosY - tmp * self.sinY
        if self.rotatingOnZ:
            tmp = imgXc
            imgXc = imgXc * self.cosZ - imgYc * self.sinZ
            imgYc = tmp * self.sinZ + imgYc * self.cosZ
        # shift back
        newX = imgXc + self.currentBall.x
        newY = imgYc + self.currentBall.y
        _, _, z2 = self.get_ball_z(newX, newY)
        # valid rotated point
        if (0 <= newX < self.cols and 0 <= newY < self.rows and z2 > 0.0):
            rx = int(newX + 0.5)
            ry = int(newY + 0.5)
            self.projectedImg[ry, rx, 0] = int(z2)
            self.projectedImg[ry, rx, 1] = 128 if prerotated_invalid else pixelValue



