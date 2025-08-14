import numpy as np
import pytest

from trajectory_simulation.ball import Ball
from trajectory_simulation.vector import length, vec3


def test_ball_reset():
    b = Ball()
    b.position[:] = vec3(1, 2, 3)
    b.velocity[:] = vec3(1, 1, 1)
    b.omega[:] = vec3(1, 1, 1)
    b.position_list.append(vec3(1, 2, 3))
    b.total_position_list.append(vec3(1, 2, 3))
    b.reset()
    assert np.allclose(b.position, vec3(0.0, 0.1, 0.0))
    assert np.allclose(b.velocity, vec3())
    assert b.position_list == []
    assert b.total_position_list == []


def test_ball_hit_from_data():
    b = Ball()
    data = {"Speed": 100, "VLA": 10, "HLA": 5, "TotalSpin": 3000, "SpinAxis": 3}
    b.hit_from_data(data)
    assert np.allclose(b.position, vec3(0.0, 0.05, 0.0))
    assert length(b.velocity) == pytest.approx(data["Speed"] * 0.44704, rel=1e-4)
    assert length(b.omega) == pytest.approx(data["TotalSpin"] * 0.10472, rel=1e-4)
