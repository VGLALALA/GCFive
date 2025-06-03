import numpy as np
import cv2
from IsolateCode import isolate_ball
from GolfBall import GolfBall

def test_isolate_ball_synthetic_center():
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(img, (50, 50), 20, 255, -1)
    ball = GolfBall(x=50, y=50, measured_radius_pixels=20, angles_camera_ortho_perspective=(0.0, 0.0, 0.0))
    crop = isolate_ball(img, ball)
    assert crop.shape == (40, 40)
    assert crop[20, 20] == 255

def test_isolate_ball_zero_coordinates():
    img = np.zeros((30, 30), dtype=np.uint8)
    cv2.circle(img, (5, 5), 4, 255, -1)
    ball = GolfBall(x=5, y=5, measured_radius_pixels=4, angles_camera_ortho_perspective=(0.0, 0.0, 0.0))
    crop = isolate_ball(img, ball)
    assert crop.shape == (8, 8)
    assert crop[4, 4] == 255
