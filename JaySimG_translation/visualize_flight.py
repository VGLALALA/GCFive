import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .ball import Ball


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


def main():
    data = {
        "Speed": 100.0,  # mph
        "VLA": 20.8,     # vertical launch angle
        "HLA": 1.7,      # horizontal launch angle
        "TotalSpin": 7494.0,  # rpm
        "SpinAxis": 2.7,      # degrees
    }
    positions, flight_time = simulate_shot(data)
    distance = np.linalg.norm(positions[-1][[0, 2]])
    apex = positions[:, 1].max()
    print(f"Carry Distance: {distance:.2f} m")
    print(f"Time of Flight: {flight_time:.2f} s")
    print(f"Apex Height: {apex:.2f} m")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(positions[:, 0], positions[:, 2], positions[:, 1])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_zlabel('Y (m)')
    ax.set_title('Shot Trajectory')
    plt.tight_layout()
    plt.savefig('trajectory.png')
    print("Saved trajectory to trajectory.png")


if __name__ == "__main__":
    main()
