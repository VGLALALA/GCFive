import pytest

from spin.spinAxis import calculate_spin_axis


def test_spin_axis_positive():
    assert calculate_spin_axis(1000, 500) == pytest.approx(22.5)


def test_spin_axis_negative():
    assert calculate_spin_axis(1000, -500) == pytest.approx(-22.5)
