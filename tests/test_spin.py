import numpy as np

from spin.spinAxis import calculate_spin_axis
from spin.Vector2RPM import calculate_spin_components


def test_calculate_spin_components():
    sidespin, backspin, total = calculate_spin_components([360, 0, 0], 1000)
    assert np.isclose(sidespin, 60)
    assert np.isclose(backspin, 0)
    assert np.isclose(total, 60)


def test_calculate_spin_axis():
    assert calculate_spin_axis(2000, 1000) == 22.5
    assert calculate_spin_axis(2000, -1000) == -22.5
