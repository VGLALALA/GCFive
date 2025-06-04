import numpy as np
from JaySimG_translation.ball import Ball


def test_ball_update_gravity():
    b = Ball()
    b.reset()
    start_y = b.position[1]
    b.update(0.1)
    assert b.position[1] < start_y


def test_ball_hit_velocity():
    b = Ball()
    b.hit()
    assert np.linalg.norm(b.velocity) > 0


def test_ball_hit_from_data_move():
    b = Ball()
    data = {
        "Speed": 100.0,
        "VLA": 20.0,
        "HLA": 5.0,
        "TotalSpin": 3000.0,
        "SpinAxis": 2.0,
    }
    b.hit_from_data(data)
    b.update(0.1)
    assert not np.allclose(b.position[[0, 2]], 0.0)
