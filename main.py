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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if not tracking:
            detections = detect_golfballs(frame, conf=0.25, imgsz=640, display=False)
            if not detections:
                print("No ball detected. Place ball in view on left side.")
                cv2.imshow("frame", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            x, y, r = detections[0]
            if x > frame.shape[1] // 2:
                print("Move ball to the left side of the frame.")
                cv2.imshow("frame", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            bbox = (x - r, y - r, x + r, y + r)
            prev_frame = frame.copy()
            prev_det = detections[0]
            tracking = True
            print("Ball detected. Waiting for movement...")
            continue

        moved, delta = has_ball_moved(prev_frame, frame, bbox)
        if not moved:
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        print(f"Ball movement detected (delta={delta:.4f}). Recording...")
        frames = [prev_frame]
        detections = [prev_det]
        total_distance = 0.0
        last_center = (prev_det[0], prev_det[1])
        pixels_per_mm = (prev_det[2] * 2) / 42.67

        while total_distance < DISTANCE_THRESHOLD_MM:
            ret, f = cap.read()
            if not ret:
                break
            dets = detect_golfballs(f, conf=0.25, imgsz=640, display=False)
            if not dets:
                break
            frames.append(f.copy())
            detections.append(dets[0])
            cx, cy, _ = dets[0]
            dx = cx - last_center[0]
            dy = cy - last_center[1]
            total_distance += (dx**2 + dy**2) ** 0.5 / pixels_per_mm
            last_center = (cx, cy)

        if len(frames) >= 2:
            speed = calculate_ball_speed(detections[0], detections[1], fps)
            angle = calculate_launch_angle(detections[0], detections[1])
            gray0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
            gray1 = cv2.cvtColor(frames[1], cv2.COLOR_BGR2GRAY)
            rot = get_fine_ball_rotation(gray0, gray1)
            backspin_rpm = (rot[1] / (1.0 / fps)) * (60.0 / 360.0)
            carry, total = simulate_trajectory(speed, angle, backspin_rpm)
            print(f"Speed: {speed:.2f} m/s, Angle: {angle:.1f} deg, Backspin: {backspin_rpm:.0f} rpm")
            print(f"Carry: {carry:.1f} yd, Total: {total:.1f} yd")
            send_to_visualization(speed, angle, backspin_rpm, carry, total)
        else:
            print("Insufficient frames captured to compute metrics.")

        tracking = False
        prev_frame = None
        prev_det = None
        bbox = None

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
