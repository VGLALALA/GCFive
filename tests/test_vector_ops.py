import numpy as np
import pytest

from trajectory_simulation.vector import cross, dot, length, normalized, vec3


def test_vector_operations():
    v = vec3(3, 4, 0)
    assert length(v) == pytest.approx(5.0)
    n = normalized(v)
    assert length(n) == pytest.approx(1.0)
    assert np.allclose(cross(vec3(1, 0, 0), vec3(0, 1, 0)), vec3(0, 0, 1))
    assert dot(vec3(1, 2, 3), vec3(4, 5, 6)) == pytest.approx(32.0)
