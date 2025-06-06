import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from datetime import datetime
import cv2
from io import BytesIO
from PIL import Image

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


def render_frame(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGR)
    buf.close()
    plt.close(fig)
    return frame


def simulate_realtime_view(data, view='3d', delta=0.01, max_time=20.0, fps=30, z_limits=(-6, 6), y_limit=60, x_max=None):

    ball = Ball()
    ball.hit_from_data(data)
    positions_list = [ball.position.copy()]
    time = 0.0
    frame_interval = 1.0 / fps
    frames = []

    while time < max_time:
        ball.update(delta)
        positions_list.append(ball.position.copy())
        time += delta
        if ball.position[1] <= 0.0 and np.linalg.norm(ball.velocity) < 0.01:
            break

        if time >= len(frames) * frame_interval:
            positions = np.array(positions_list)
            if view == '3d':
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.plot(positions[:, 0] * 1.09361, positions[:, 2] * 1.09361, positions[:, 1] * 3.28084)
                ax.set_xlim([0, 100])
                ax.set_ylim([-50, 50])
                apex = positions[:, 1].max() * 3.28084
                x_max = positions[:, 0].max() * 1.09361

                ax.set_xlim([0, x_max + 5])
                ax.set_zlim([0, apex + 10])
                ax.set_xlabel('X (yd)')
                ax.set_ylabel('Z (yd)')
                ax.set_zlabel('Y (ft)')
                ax.set_title('Realtime Shot Trajectory')

            elif view == 'side':
                fig, ax = plt.subplots()
                ax.plot(positions[:, 0] * 1.09361, positions[:, 1] * 3.28084)
                if x_max is not None:
                    ax.set_xlim([0, x_max + 5])
                else:
                    ax.set_xlim([0, 100])
                ax.set_ylim([0, y_limit])
                ax.set_xlabel('X (yd)')
                ax.set_ylabel('Y (ft)')
                ax.set_title('Realtime Side View')

            elif view == 'down':
                fig, ax = plt.subplots()
                ax.plot(positions[:, 2] * 1.09361, positions[:, 1] * 3.28084)
                ax.set_xlim(z_limits)
                ax.set_ylim([0, y_limit])
                ax.set_xlabel('Z (yd)')
                ax.set_ylabel('Y (ft)')
                ax.set_title('Realtime Down-the-Line View')

            plt.tight_layout()
            frames.append(render_frame(fig))

    os.makedirs('results/trajectory', exist_ok=True)
    height, width, _ = frames[0].shape
    filename = f'results/trajectory/trajectory_realtime_{view}.mp4'
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()
    print(f"Saved realtime {view} trajectory video to {filename}")


def main():
    data = {
        "Speed": 171,
        "VLA": 10.4,
        "HLA": 0,
        "TotalSpin": 2545.0,
        "SpinAxis": 0,
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

    x_max = positions[:, 0].max() * 1.09361
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(positions[:, 0] * 1.09361, positions[:, 2] * 1.09361, positions[:, 1] * 3.28084)
    ax.set_xlabel('X (yd)')
    ax.set_ylabel('Z (yd)')
    ax.set_zlabel('Y (ft)')
    ax.set_title('Shot Trajectory')
    ax.set_zlim([0, apex + 10])

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

    os.makedirs('results/trajectory', exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'results/trajectory/trajectory_3d_{current_time}.png')

    fig, ax = plt.subplots()
    ax.plot(positions[:, 0] * 1.09361, positions[:, 1] * 3.28084)
    ax.set_xlim([0, x_max + 5])
    ax.set_xlabel('X (yd)')
    ax.set_ylabel('Y (ft)')
    ax.set_title('Side View of Shot Trajectory')
    ax.set_ylim([0, apex + 10])
    plt.tight_layout()
    plt.savefig(f'results/trajectory/trajectory_side_{current_time}.png')

    # Save Down-the-Line (DTL) static image
    fig, ax = plt.subplots()
    ax.plot(positions[:, 2] * 1.09361, positions[:, 1] * 3.28084)
    z_min = positions[:, 2].min() * 1.09361
    z_max = positions[:, 2].max() * 1.09361
    ax.set_xlim([z_min - 2, z_max + 2])
    ax.set_ylim([0, apex + 10])
    ax.set_xlabel('Z (yd)')
    ax.set_ylabel('Y (ft)')
    ax.set_title('Down-the-Line View of Shot Trajectory')
    plt.tight_layout()
    plt.savefig(f'results/trajectory/trajectory_dtl_{current_time}.png')

    simulate_realtime_view(data, view='3d', z_limits=(-50, 50), y_limit=apex + 10)
    simulate_realtime_view(data, view='side', y_limit=apex + 10, z_limits=(0, x_max + 5))
    z_min = positions[:, 2].min() * 1.09361
    z_max = positions[:, 2].max() * 1.09361
    simulate_realtime_view(data, view='down', z_limits=(z_min - 2, z_max + 2), y_limit=apex + 10)


if __name__ == "__main__":
    main()
