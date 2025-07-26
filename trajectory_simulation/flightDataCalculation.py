# Cell 1: Imports and Setup
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


from trajectory_simulation.ball import Ball  # Ensure this module is accessible
# Cell 2: Trajectory Simulation Functions
def simulate_shot(data, delta=0.01, max_time=20.0):
    
    ball = Ball()
    ball.hit_from_data(data)
    positions = [ball.position.copy()]
    time = 0.0
    while time < max_time:
        ball.update(delta)
        positions.append(ball.position.copy())
        time += delta
        if ball.position[1] <= 0.0 and np.linalg.norm(ball.velocity) < 0.01:
            break
    return np.array(positions), time

def calculate_descending_angle(positions, landing_index):
    """Calculate descending angle using trajectory just *before* first bounce.

    Parameters
    ----------
    positions : np.ndarray
        Array of ball positions over time.
    landing_index : int
        Index of the first sample where the ball touches the ground.

    Returns
    -------
    float
        Descending angle in degrees.
    """

    # Need at least two samples prior to impact to compute a slope.
    if landing_index < 2:
        return 0.0

    # Use the last step before ground contact to avoid using the post-bounce
    # position which results in an artificially shallow angle.
    prev_pos = positions[landing_index - 1]
    prev_prev_pos = positions[landing_index - 2]

    delta_xz = prev_pos[[0, 2]] - prev_prev_pos[[0, 2]]
    delta_y = prev_pos[1] - prev_prev_pos[1]

    angle_rad = np.arctan2(-delta_y, np.linalg.norm(delta_xz))
    return np.degrees(angle_rad)

def get_trajectory_metrics(data):
    """
    Simulate the shot and calculate carry distance, total distance, apex height,
    time of flight, and descending angle.

    Parameters
    ----------
    data : dict
        Dictionary containing shot data with keys "Speed", "VLA", "HLA", "TotalSpin", and "SpinAxis".
    delta : float, optional
        Time step for the simulation, by default 0.01.
    max_time : float, optional
        Maximum simulation time, by default 20.0.

    Returns
    -------
    dict
        Dictionary containing carry distance, total distance, apex height, time of flight, and descending angle.
    """
    positions, flight_time = simulate_shot(data)
    before_rolling_index = next(i for i, pos in enumerate(positions) if pos[1] <= 0.0)
    carry_distance = np.linalg.norm(positions[before_rolling_index][[0, 2]]) * 1.09361
    total_distance = np.linalg.norm(positions[-1][[0, 2]]) * 1.09361
    apex = positions[:, 1].max() * 3.28084
    descending_angle = calculate_descending_angle(positions, before_rolling_index)

    return {
        "carry_distance": carry_distance,
        "total_distance": total_distance,
        "apex": apex,
        "time_of_flight": flight_time,
        "descending_angle": descending_angle
    },positions




if __name__ == "__main__":
    # Cell 4: Run Main Simulation and Save Outputs
    data = {
        "Speed": 171,
        "VLA": 10.4,
        "HLA": 2,
        "TotalSpin": 2545.0,
        "SpinAxis": -3.52,
    }



    positions, flight_time = simulate_shot(data)
    before_rolling_index = next(i for i, pos in enumerate(positions) if pos[1] <= 0.0)
    carry_distance = np.linalg.norm(positions[before_rolling_index][[0, 2]]) * 1.09361
    total_distance = np.linalg.norm(positions[-1][[0, 2]]) * 1.09361
    apex = positions[:, 1].max() * 3.28084
    descending_angle = calculate_descending_angle(positions, before_rolling_index)

    print(f"Carry Distance: {carry_distance:.2f} yd")
    print(f"Total Distance: {total_distance:.2f} yd")
    print(f"Time of Flight: {flight_time:.2f} s")
    print(f"Apex Height: {apex:.2f} ft")
    print(f"Descending Angle: {descending_angle:.2f} degrees")

