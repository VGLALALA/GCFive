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

def is_point_in_zone(samples, x, y):
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

# --- Example usage ---
if __name__ == "__main__":
    samples = [
        [-94.53612068965518, 448.22758620689655],
        [-14.464406779661017, 440.6305084745763],
        [-125.22717391304347, 565.1565217391304],
        [-12.445416666666667, 541.6083333333333]
    ]
    test_point = (-50, 500)
    inside = is_point_in_zone(samples, *test_point)
    print(f"Point {test_point} is inside zone? {inside}")
