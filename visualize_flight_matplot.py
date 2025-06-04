import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from datetime import datetime

from trajectory_simulation.ball import Ball


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


def calculate_descending_angle(positions, before_rolling_index):
    if len(positions) < 2:
        return 0.0
    # Calculate the angle of descent using the last two positions before rolling
    delta_xz = positions[before_rolling_index][[0, 2]] - positions[before_rolling_index - 1][[0, 2]]
    delta_y = positions[before_rolling_index][1] - positions[before_rolling_index - 1][1]
    angle_rad = np.arctan2(-delta_y, np.linalg.norm(delta_xz))
    return np.degrees(angle_rad)


def main():
    data = {
        "Speed": 80.8,  # mph
        "VLA": 38.8,     # vertical launch angle
        "HLA": -1.7,      # horizontal launch angle
        "TotalSpin": 3980.0,  # rpm
        "SpinAxis": -4.1,      # degrees
    }
    positions, flight_time = simulate_shot(data)
    
    # Determine the index where the ball starts rolling
    before_rolling_index = next(i for i, pos in enumerate(positions) if pos[1] <= 0.0)
    
    # Calculate carry distance (before rolling)
    carry_distance = np.linalg.norm(positions[before_rolling_index][[0, 2]]) * 1.09361
    
    # Calculate total distance (after rolling)
    total_distance = np.linalg.norm(positions[-1][[0, 2]]) * 1.09361
    
    # Convert meters to feet (1 meter = 3.28084 feet)
    apex = positions[:, 1].max() * 3.28084
    
    # Calculate descending angle
    descending_angle = calculate_descending_angle(positions, before_rolling_index)
    
    print(f"Carry Distance: {carry_distance:.2f} yd")
    print(f"Total Distance: {total_distance:.2f} yd")
    print(f"Time of Flight: {flight_time:.2f} s")
    print(f"Apex Height: {apex:.2f} ft")
    print(f"Descending Angle: {descending_angle:.2f} degrees")

    # 3D Trajectory Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(positions[:, 0] * 1.09361, positions[:, 2] * 1.09361, positions[:, 1] * 3.28084)
    ax.set_xlabel('X (yd)')
    ax.set_ylabel('Z (yd)')
    ax.set_zlabel('Y (ft)')
    ax.set_title('Shot Trajectory')

    # Add text with data to the plot
    textstr = '\n'.join((
        f"Carry Distance: {carry_distance:.2f} yd",
        f"Total Distance: {total_distance:.2f} yd",
        f"Time of Flight: {flight_time:.2f} s",
        f"Apex Height: {apex:.2f} ft",
        f"Descending Angle: {descending_angle:.2f} degrees",
        f"Speed: {data['Speed']} mph",
        f"VLA: {data['VLA']} degrees",
        f"HLA: {data['HLA']} degrees",
        f"Total Spin: {data['TotalSpin']} rpm",
        f"Spin Axis: {data['SpinAxis']} degrees"
    ))
    plt.gcf().text(0.02, 0.95, textstr, fontsize=10, verticalalignment='top')

    plt.tight_layout()
    
    # Create results directory if it doesn't exist
    os.makedirs('results/trajectory', exist_ok=True)
    
    # Save the 3D image with the current time
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_3d = f'results/trajectory/trajectory_3d_{current_time}.png'
    plt.savefig(filename_3d)
    print(f"Saved 3D trajectory to {filename_3d}")

    # 2D Side View Plot
    fig, ax = plt.subplots()
    ax.plot(positions[:, 0] * 1.09361, positions[:, 1] * 3.28084)
    ax.set_xlabel('X (yd)')
    ax.set_ylabel('Y (ft)')
    ax.set_title('Side View of Shot Trajectory')
    plt.tight_layout()

    # Save the 2D side view image
    filename_2d = f'results/trajectory/trajectory_side_{current_time}.png'
    plt.savefig(filename_2d)
    print(f"Saved side view trajectory to {filename_2d}")


if __name__ == "__main__":
    main()
