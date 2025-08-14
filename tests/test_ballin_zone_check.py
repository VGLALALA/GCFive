import pytest

from image_processing import ballinZoneCheck as bzc


def test_order_polygon_and_point_in_poly():
    points = [(1, 0), (0, 0), (0, 1), (1, 1)]
    ordered = bzc._order_polygon(points)
    assert set(ordered) == set(points)
    poly = [(0, 0), (1, 0), (1, 1), (0, 1)]
    assert bzc._point_in_poly(0.5, 0.5, poly)
    assert not bzc._point_in_poly(1.5, 1.5, poly)


def test_is_point_in_zone(monkeypatch):
    samples = [(0, 0), (1, 0), (1, 1), (0, 1)]
    monkeypatch.setattr(
        bzc,
        "load_hitting_zone_samples",
        lambda file_path="hitting_zone_calibration.json": samples,
    )
    assert bzc.is_point_in_zone(0.2, 0.2)
    assert not bzc.is_point_in_zone(2, 2)
