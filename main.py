
import cv2
import numpy as np
import socket
import json
from image_processing.ballDetection import detect_golfballs
from image_processing.movementDetection import has_ball_moved
from image_processing.ballSpeedCalculation import calculate_ball_speed
from image_processing.launchAngleCalculation import calculate_launch_angle
from spin.GetBallRotation import get_fine_ball_rotation
from trajectory_simulation.range import RangeSim


VIS_HOST = "localhost"
VIS_PORT = 49152
DISTANCE_THRESHOLD_MM = 200  # 20 cm


def send_to_visualization(speed_mps: float, launch_angle: float, backspin_rpm: float, carry: float, total: float):
    data = {
        "ShotDataOptions": {"ContainsBallData": True},
        "BallData": {
            "Speed": speed_mps * 2.237,
            "VLA": launch_angle,
            "HLA": 0.0,
            "TotalSpin": backspin_rpm,
            "SpinAxis": 0.0,
        },
    }
    try:
        sock = socket.create_connection((VIS_HOST, VIS_PORT), timeout=1)
        sock.sendall(json.dumps(data).encode())
        sock.close()
    except OSError as exc:
        print(f"Failed to send to visualization: {exc}")


def simulate_trajectory(speed_mps: float, launch_angle: float, backspin_rpm: float):
    sim = RangeSim()
    sim.ball.hit_from_data(
        {
            "Speed": speed_mps * 2.237,
            "VLA": launch_angle,
            "HLA": 0.0,
            "TotalSpin": backspin_rpm,
            "SpinAxis": 0.0,
        }
    )
    dt = 0.01
    for _ in range(2000):
        sim.ball.update(dt)
    return sim.distance_yards, sim.distance_yards


def main():
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print("Starting camera feed...")
    tracking = False
    prev_frame = None
    prev_det = None
    bbox = None



if __name__ == "__main__":
    main()

