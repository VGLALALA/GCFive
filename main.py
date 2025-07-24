"""Example pipeline for detecting and tracking a golf ball.

This script attempts to detect a stationary golf ball using the YOLO model
provided in :mod:`image_processing.ballDetection`.  Once the ball is detected
and fully visible on the left side of the frame it monitors the region for
movement.  When movement is detected it captures frames until the ball has
travelled roughly 20 cm and then estimates speed, launch angle and backspin.

The resulting shot data is printed and could be sent to a visualisation web
front‑end using :mod:`trajectory_simulation`.
"""

from __future__ import annotations

import json
import socket
import time
from typing import Tuple

import cv2

from image_processing.ballDetection import detect_golfballs
from image_processing.movementDetection import detect_ball_movement
from image_processing.ballSpeedCalculation import calculate_ball_speed
from image_processing.launchAngleCalculation import calculate_launch_angle
from image_processing.backspinCalculation import calculate_backspin
from trajectory_simulation.range import RangeSim


def send_to_visualisation(data: dict, host: str = "localhost", port: int = 49152) -> None:
    """Send shot data to the visualisation server if available."""

    try:
        with socket.create_connection((host, port), timeout=1) as sock:
            sock.sendall(json.dumps(data).encode())
    except OSError:
        # Server might not be running – fall back to console output
        print(json.dumps(data, indent=2))


def wait_for_ball(cap: cv2.VideoCapture, conf: float = 0.25) -> Tuple[Tuple[int, int, int], cv2.Mat]:
    """Wait until a ball is fully visible on the left half of the image."""

    while True:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to read from camera")

        detections = detect_golfballs(frame.copy(), conf=conf, display=False)
        if detections:
            x, y, r = detections[0]
            h, w = frame.shape[:2]
            if x - r < 0 or y - r < 0 or x + r > w or y + r > h:
                print("Move ball fully into frame")
            elif x > w // 2:
                print("Move ball to left side")
            else:
                return (x, y, r), frame

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise KeyboardInterrupt


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    try:
        while True:
            bbox_data, start_frame = wait_for_ball(cap)
            x, y, r = bbox_data
            bbox = (x - r, y - r, 2 * r, 2 * r)
            print("Ball locked. Waiting for movement...")

            while True:
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError("Failed to read from camera")

                delta, moved = detect_ball_movement(start_frame, frame, bbox)
                if moved:
                    print("Movement detected")
                    start_time = time.time()
                    start_img = start_frame.copy()
                    end_img = frame.copy()
                    travelled_mm = 0.0
                    start_center = (x, y)

                    while travelled_mm < 200.0:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        detections = detect_golfballs(frame.copy(), conf=0.25, display=False)
                        if not detections:
                            continue
                        x2, y2, r2 = detections[0]
                        px_per_mm = ((r + r2) / 2) * 2 / 42.67
                        travelled_mm = (((x2 - start_center[0]) ** 2 + (y2 - start_center[1]) ** 2) ** 0.5) / px_per_mm
                        end_img = frame.copy()

                    end_time = time.time()

                    speed_mps, speed_mph = calculate_ball_speed(start_img, end_img, fps)
                    launch_angle = calculate_launch_angle(start_img, end_img)
                    backspin = calculate_backspin(start_img, end_img, end_time - start_time)

                    sim = RangeSim()
                    shot = {
                        "ShotDataOptions": {"ContainsBallData": True},
                        "BallData": {
                            "Speed": speed_mph,
                            "VLA": launch_angle,
                            "HLA": 0.0,
                            "TotalSpin": backspin,
                            "SpinAxis": 0.0,
                        },
                    }
                    sim.ball.hit_from_data(shot["BallData"])
                    for _ in range(2000):
                        sim.step(1 / 240.0)

                    shot["Carry"] = sim.distance_yards
                    shot["Apex"] = sim.apex_feet

                    send_to_visualisation(shot)
                    cv2.destroyAllWindows()
                    break

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    raise KeyboardInterrupt

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

