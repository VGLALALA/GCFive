import numpy as np
import pytest
cv2 = pytest.importorskip("cv2")

from trajectory_simulation.ball import Ball
from trajectory_simulation.ball_trail import BallTrail
from trajectory_simulation.range import RangeSim
from trajectory_simulation.flightDataCalculation import calculate_descending_angle, get_trajectory_metrics


def test_ball_trail():
    trail = BallTrail()
    trail.add_point([1, 2, 3])
    np.testing.assert_array_equal(trail.points[-1], np.array([1, 2, 3]))
    trail.clear_points()
    assert len(trail.points) == 2


def test_range_properties():
    rs = RangeSim.__new__(RangeSim)
    rs.ball = Ball()
    rs.ball.position = np.array([10.0, 2.0, 0.0])
    rs._apex = 5.0
    assert pytest.approx(rs.distance_yards, rel=1e-5) == np.linalg.norm([10.0, 0.0]) * 1.09361
    assert rs.apex_feet == 15.0


def test_calculate_descending_angle():
    positions = np.array([[0, 1, 0], [1, 2, 0], [2, 0, 0]], dtype=float)
    angle = calculate_descending_angle(positions, 2)
    assert pytest.approx(angle, rel=1e-5) == -45.0


def test_get_trajectory_metrics(monkeypatch):
    import trajectory_simulation.flightDataCalculation as fdc

    def fake_simulate(data, delta=0.01, max_time=20.0):
        return np.array([[0, 0, 0], [10, 5, 0], [20, 0, 0]]), 2.0

    monkeypatch.setattr(fdc, 'simulate_shot', fake_simulate)
    metrics, pos = get_trajectory_metrics({})
    assert metrics['carry_distance'] > 0
    assert metrics['total_distance'] > 0
    assert metrics['apex'] == pytest.approx(5 * 3.28084)
    assert metrics['time_of_flight'] == 2.0
    assert 'descending_angle' in metrics
    assert pos.shape[0] == 3
