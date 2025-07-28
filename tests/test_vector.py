import numpy as np

from trajectory_simulation.vector import cross, dot, length, normalized, vec3


def test_vec3_creation():
    v = vec3(1, 2, 3)
    assert isinstance(v, np.ndarray)
    np.testing.assert_array_equal(v, np.array([1.0, 2.0, 3.0]))


def test_length_and_normalized():
    v = vec3(3, 4, 0)
    assert length(v) == 5.0
    n = normalized(v)
    np.testing.assert_allclose(n, v / 5.0)
    np.testing.assert_array_equal(normalized(vec3()), vec3())


def test_cross_and_dot():
    a = vec3(1, 0, 0)
    b = vec3(0, 1, 0)
    np.testing.assert_array_equal(cross(a, b), vec3(0, 0, 1))
    assert dot(a, b) == 0.0
    assert dot(a, a) == 1.0
