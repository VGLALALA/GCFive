import math

def _order_polygon(points):
    """
    Given a list of (x, y) points, return them sorted around their centroid
    so they form a non‑self‑intersecting polygon.
    """
    cx = sum(x for x, y in points) / len(points)
    cy = sum(y for x, y in points) / len(points)
    return sorted(points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

def _point_in_poly(x, y, poly):
    """
    Ray‑casting algorithm to test if (x,y) is inside the polygon defined by poly.
    poly must be an ordered list of (x, y) vertices.
    Returns True if inside, False otherwise.
    """
    inside = False
    n = len(poly)
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[i - 1]
        # Check if edge crosses horizontal ray at y
        intersect = ((yi > y) != (yj > y)) and \
                    (x < (xj - xi) * (y - yi) / (yj - yi) + xi)
        if intersect:
            inside = not inside
    return inside
import json

def load_hitting_zone_samples(file_path="hitting_zone_calibration.json"):
    """
    Load the hitting zone samples from a JSON file.
    
    :param file_path: Path to the JSON file containing the samples.
    :return: List of samples if successful, None otherwise.
    """
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
            return data.get("samples", None)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading hitting zone samples: {e}")
        return None

def is_point_in_zone(x, y):
    samples=load_hitting_zone_samples()
    """
    samples: list of four [x, y] corner coordinates, in any order
    x, y:   the point to test
    Returns True if (x,y) lies inside the quadrilateral, False otherwise.
    """
    if len(samples) != 4:
        raise ValueError("Need exactly four corner points")
    # Order them into a proper polygon
    poly = _order_polygon(samples)
    return _point_in_poly(x, y, poly)

