import numpy as np
from typing import Tuple, Union

def calculate_spin_components(rotation_vector: Union[np.ndarray, list], delta_t_ms: float) -> Tuple[float, float, float]:
    """
    Calculate backspin and sidespin (in RPM) from a rotation vector and frame interval.
    
    Parameters:
        rotation_vector: Union[np.ndarray, list]
            Rotation vector [rx, ry, rz] in degrees between two frames.
            Assumed coordinate system:
                - X = right (sidespin)
                - Y = up (ignored / tilt)
                - Z = forward toward target (backspin)
        delta_t_ms: float
            Time difference between two frames in milliseconds.
            
    Returns:
        Tuple[float, float, float]
            (sidespin_rpm, backspin_rpm, total_spin_rpm)
    """
    # Convert rotation_vector to np.ndarray if it's a list
    if isinstance(rotation_vector, list):
        rotation_vector = np.array(rotation_vector)

    # Convert time to seconds
    delta_t_sec = delta_t_ms / 1000.0

    # Angular velocity in deg/sec
    angular_velocity_deg_per_s = rotation_vector / delta_t_sec

    # Convert deg/sec to RPM: (deg/s ÷ 360) × 60 = deg/s ÷ 6
    angular_velocity_rpm = angular_velocity_deg_per_s / 6.0

    # Extract spin components
    sidespin_rpm = angular_velocity_rpm[0]   # X-axis
    backspin_rpm = angular_velocity_rpm[2]   # Z-axis
    total_spin_rpm = np.linalg.norm(angular_velocity_rpm)

    return sidespin_rpm, backspin_rpm, total_spin_rpm
