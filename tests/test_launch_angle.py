import pytest

from image_processing.launchAngleCalculation import calculate_launch_angle


def test_calculate_launch_angle():
    det1 = (0, 0, 20)
    det2 = (10, -10, 20)
    assert calculate_launch_angle(det1, det2) == pytest.approx(45.0)
