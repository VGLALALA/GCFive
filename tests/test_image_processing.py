import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

import spin.GetLaunchAngle as GetLaunchAngle
from image_processing.ballSpeedCalculation import calculate_ball_speed
from image_processing.Convert_Canny import convert_to_canny
from image_processing.Convert_GrayScale import convert_to_grayscale
from image_processing.ImageCompressor import compress_image
from image_processing.launchAngleCalculation import \
    calculate_launch_angle as calc_launch
from image_processing.movementDetection import has_ball_moved
from image_processing.RemoveReflection import remove_reflections


def test_convert_to_grayscale(tmp_path):
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    path = tmp_path / "img.png"
    cv2.imwrite(str(path), img)
    gray = convert_to_grayscale(str(path))
    assert gray.shape == (10, 10)
    assert gray.dtype == np.uint8


def test_convert_to_canny(tmp_path):
    img = np.zeros((10, 10), dtype=np.uint8)
    img[4:6, 4:6] = 255
    path = tmp_path / "edge.png"
    cv2.imwrite(str(path), img)
    edges = convert_to_canny(str(path))
    assert edges.shape == img.shape
    assert edges.sum() > 0


def test_compress_image():
    img = np.ones((10, 10), dtype=np.uint8) * 255
    out = compress_image(img, 2)
    assert out.shape == (5, 5)
    with pytest.raises(ValueError):
        compress_image(img, -1)


def test_remove_reflections():
    orig = np.array([[0, 255], [0, 0]], dtype=np.uint8)
    filtered = orig.copy()
    out = remove_reflections(orig, filtered)
    assert out[0, 1] == 128


def test_has_ball_moved(monkeypatch):
    def fake_delta(a, b):
        return 0.0, 0.02, 0.0

    monkeypatch.setattr(
        "image_processing.movementDetection.delta_similarity", fake_delta
    )
    prev = np.zeros((5, 5, 3), dtype=np.uint8)
    curr = np.zeros((5, 5, 3), dtype=np.uint8)
    moved, delta = has_ball_moved(prev, curr, (0, 0, 5, 5))
    assert moved is True
    assert delta == 0.02


def test_has_ball_moved_empty():
    prev = np.zeros((5, 5, 3), dtype=np.uint8)
    curr = np.zeros((5, 5, 3), dtype=np.uint8)
    moved, delta = has_ball_moved(prev, curr, (6, 6, 8, 8))
    assert moved is False
    assert delta == 0.0


def test_launch_angle_calculation():
    det1 = (0, 10, 10)
    det2 = (10, 0, 10)
    angle = calc_launch(det1, det2)
    assert pytest.approx(angle, rel=1e-5) == 45.0


def test_ball_speed(monkeypatch):
    def fake_detect(img):
        if fake_detect.calls == 0:
            fake_detect.calls += 1
            return [(5, 5, 10)]
        return [(15, 5, 10)]

    fake_detect.calls = 0
    monkeypatch.setattr(
        "image_processing.ballSpeedCalculation.detect_golfballs", fake_detect
    )
    frame1 = np.zeros((20, 20, 3), dtype=np.uint8)
    frame2 = np.zeros((20, 20, 3), dtype=np.uint8)
    speed = calculate_ball_speed(frame1, frame2, 0.01, return_mph=True)
    assert pytest.approx(speed, rel=1e-2) == 4.77


def test_get_launch_angle(monkeypatch):
    def fake_detect(img):
        if fake_detect.calls == 0:
            fake_detect.calls += 1
            return [(10, 20, 5)]
        return [(20, 10, 5)]

    fake_detect.calls = 0
    monkeypatch.setattr(GetLaunchAngle, "detect_golfballs", fake_detect)
    frame1 = np.zeros((30, 30), dtype=np.uint8)
    frame2 = np.zeros((30, 30), dtype=np.uint8)
    angle = GetLaunchAngle.calculate_launch_angle(frame1, frame2)
    assert pytest.approx(angle, rel=1e-5) == 45.0
