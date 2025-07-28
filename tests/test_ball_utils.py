import numpy as np
from trajectory_simulation.vector import vec3, length
from trajectory_simulation.ball import project, angle_between, limit_length, rotation_matrix


def test_project():
    v = vec3(1, 2, 0)
    n = vec3(0, 1, 0)
    p = project(v, n)
    np.testing.assert_allclose(p, vec3(0, 2, 0))


def test_angle_between():
    a = vec3(1, 0, 0)
    b = vec3(0, 1, 0)
    angle = angle_between(a, b)
    assert np.isclose(angle, np.pi / 2)


def test_limit_length():
    v = vec3(3, 4, 0)
    limited = limit_length(v, 2)
    assert np.isclose(length(limited), 2)
    np.testing.assert_array_equal(limit_length(vec3(1, 1, 0), 5), vec3(1, 1, 0))


def test_rotation_matrix():
    R = rotation_matrix(vec3(0, 0, 1), np.pi / 2)
    rotated = R @ vec3(1, 0, 0)
    np.testing.assert_allclose(np.round(rotated, 5), vec3(0, 1, 0))
