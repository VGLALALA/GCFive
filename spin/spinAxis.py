import math


def calculate_spin_axis(backspin_rpm, sidespin_rpm):
    axis_angle = abs(sidespin_rpm) / abs(backspin_rpm) * 45
    if sidespin_rpm > 0:
        axis_angle = axis_angle
    else:
        axis_angle = -axis_angle
    return axis_angle
