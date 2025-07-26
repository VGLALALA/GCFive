import math
def calculate_spin_axis(backspin_rpm, sidespin_rpm):
    if backspin_rpm == 0:
        return 90.0 if sidespin_rpm > 0 else -90.0 if sidespin_rpm < 0 else 0.0
    spin_axis_rad = math.atan2(sidespin_rpm, backspin_rpm)
    spin_axis_deg = math.degrees(spin_axis_rad)
    return spin_axis_deg