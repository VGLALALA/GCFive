# coding=utf-8
import json
import math
import os
import queue
import threading
import time

import cv2
import numpy as np

import camera.cv_grab_callback as cv_grab_callback  # Import the monitoring module
from camera.hittingZoneCalibration import calibrate_hitting_zone_stream
from image_processing.ballDetectionyolo import (  # Import YOLO detection function
    detect_golfballs,
)
from image_processing.ballinZoneCheck import (  # Import the zone check function
    is_point_in_zone,
)
from image_processing.ballSpeedCalculation import calculate_ball_speed
from image_processing.get2Dcoord import get_ball_xz
from spin.GetBallRotation import get_fine_ball_rotation
from spin.GetLaunchAngle import calculate_launch_angle
from spin.spinAxis import calculate_spin_axis
from spin.Vector2RPM import calculate_spin_components
from storage.database import init_db, insert_shot_record
from trajectory_simulation.flightDataCalculation import get_trajectory_metrics
from utility.config_reader import CONFIG

YOLO_CONF = CONFIG.getfloat("YOLO", "conf", fallback=0.25)
YOLO_IMGSZ = CONFIG.getint("YOLO", "imgsz", fallback=640)
RECALIBRATE_HITTING_ZONE = CONFIG.getboolean(
    "Calibration", "recalibrate_hitting_zone", fallback=False
)
USE_HITTING_ZONE = CONFIG.getboolean("Detection", "use_hitting_zone", fallback=True)


def main():
    initial_frame, best_match_frame = None, None
    initial_ts, best_ts, delta_t = None, None, 0.0
    # Setup camera using the helper function from cv_grab_callback
    cam = cv_grab_callback.setup_camera_and_buffer()
    if cam is None:
        print("Failed to set up camera.")
        return

    monoCamera = cam.mono
    if RECALIBRATE_HITTING_ZONE:
        # Perform hitting zone calibration using the same camera settings
        calibrate_hitting_zone_stream(cam=cam)

    print("Searching for ball... Press 'q' to exit.")

    ball_detected = False
    detected_circle = None
    original_cropped_roi = None
    stationary_start_time = None

    # --- Detection Loop ---
    # Capture frames until ball detected or 'q' pressed
    while not ball_detected and (cv2.waitKey(1) & 0xFF) != ord("q"):
        try:
            # Grab a frame from the camera
            frame = cam.grab()

            # Convert to BGR for YOLO detection if it's a grayscale image
            if monoCamera:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                frame_bgr = frame

            # Use YOLO to detect golf balls
            detected_balls = detect_golfballs(
                frame_bgr, conf=YOLO_CONF, imgsz=YOLO_IMGSZ, display=False
            )
            ballx, ballz = get_ball_xz(frame_bgr, detected_balls)
            if detected_balls:
                # Take the first detected ball
                center_x, center_y, radius = detected_balls[0]
                print(
                    f"Ball detected at position: ({ballx}, {ballz}) with radius: {radius}"
                )
                detected_circle = (center_x, center_y, radius)

                # Optionally check if the detected ball is within the predefined zone
                if USE_HITTING_ZONE and not is_point_in_zone(ballx, ballz):
                    print("Ball is outside the zone.")
                    stationary_start_time = None
                    continue
                else:
                    if USE_HITTING_ZONE:
                        print("Ball is within the zone.")
                    else:
                        print("Zone check disabled.")
                    if stationary_start_time is None:
                        stationary_start_time = time.time()
                    elif time.time() - stationary_start_time >= 3:
                        ball_detected = True
                        print("Ball has been stationary for over 3 seconds.")

                # Calculate crop coordinates
                crop_size = 100  # Size of the square crop around the ball
                half_crop = crop_size // 2
                x1 = max(0, center_x - half_crop)
                y1 = max(0, center_y - half_crop)
                x2 = min(frame_bgr.shape[1], center_x + half_crop)
                y2 = min(frame_bgr.shape[0], center_y + half_crop)

                # Crop the region around the ball and make a deep copy
                if monoCamera:
                    original_cropped_roi = frame[y1:y2, x1:x2].copy()
                else:
                    original_cropped_roi = cv2.cvtColor(
                        frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY
                    ).copy()

                # Draw the detected circle
                cv2.circle(frame_bgr, (center_x, center_y), radius, (0, 255, 0), 2)
                cv2.circle(frame_bgr, (center_x, center_y), 2, (0, 0, 255), 3)

                # Display the frame with detection and the cropped ROI
                cv2.imshow("Ball Detection - Press q to exit", frame_bgr)
                cv2.imshow("Detected Ball (Cropped)", original_cropped_roi)

                if ball_detected:
                    print(
                        "Ball detected and stationary! Press any key in a display window to start monitoring."
                    )
                    cv2.waitKey(0)  # Wait indefinitely for a key press
                    cv2.destroyAllWindows()  # Close detection windows

                    # Break the detection loop
                    break

        except Exception as e:
            print(f"Camera grab failed: {e}")
            pass  # Continue loop on errors

    # --- Start Monitoring if ball was detected ---
    if ball_detected and cam:
        print("Starting monitoring...")

        # Create queue and stop event
        frame_queue = queue.Queue(maxsize=10)  # Limit queue size
        stop_event = threading.Event()

        # Create and start acquisition thread
        acquire_thread = threading.Thread(
            target=cv_grab_callback.acquire_frames, args=(cam, frame_queue, stop_event)
        )
        acquire_thread.start()

        # Create and start processing thread
        process_thread = threading.Thread(
            target=cv_grab_callback.process_frames,
            args=(cam, detected_circle, original_cropped_roi, frame_queue, stop_event),
        )
        process_thread.start()

        # Main thread loop to keep program alive and handle events
        print("Press 'q' to stop monitoring.")
        while not stop_event.is_set() and (cv2.waitKey(1) & 0xFF) != ord("q"):
            time.sleep(0.01)  # Small sleep to prevent busy waiting

        # Signal threads to stop and wait for them to finish
        stop_event.set()
        acquire_thread.join()
        frame_queue.put((None, None))
        process_thread.join()

        # Retrieve the initial and best match frames and their timestamps
        initial_frame, best_match_frame, initial_ts, best_ts = (
            cv_grab_callback.retriveData()
        )
        delta_t = (
            (best_ts - initial_ts)
            if (best_ts is not None and initial_ts is not None)
            else 0.0
        )
        if delta_t <= 0:
            print(
                f"Warning: non-positive delta_t ({delta_t}). Attempting fallback using acquisition order."
            )
            # As a fallback, assume uniform spacing and approximate with small positive dt
            # Use a conservative small dt to avoid divide-by-zero; user should improve timestamping
            delta_t = max(delta_t, 1e-6)

        print("Monitoring stopped.")

        # Save the two frames used for processing into data/Images with timestamps
        try:
            images_dir = os.path.join("data", "Images")
            os.makedirs(images_dir, exist_ok=True)
            if initial_frame is not None and best_match_frame is not None:
                initial_path = os.path.join(
                    images_dir, f"initial_{initial_ts:.6f}s.png"
                )
                best_path = os.path.join(images_dir, f"best_{best_ts:.6f}s.png")
                cv2.imwrite(initial_path, initial_frame)
                cv2.imwrite(best_path, best_match_frame)
                print(f"Saved frames to: {initial_path} and {best_path}")
        except Exception as e:
            print(f"Warning: failed to save frames: {e}")

    # --- Release camera and buffer ---
    cv_grab_callback.release_camera_and_buffer(cam)
    print("Camera and buffer released.")

    best_rot_x, best_rot_y, best_rot_z = get_fine_ball_rotation(
        initial_frame, best_match_frame
    )
    delta_ms = delta_t * 1000
    side_spin_rpm, back_spin_rpm, total_spin_rpm = calculate_spin_components(
        [best_rot_x, best_rot_y, best_rot_z], delta_ms
    )
    spin_axis = calculate_spin_axis(back_spin_rpm, side_spin_rpm)
    launch_angle = calculate_launch_angle(initial_frame, best_match_frame)
    ball_speed_mph = calculate_ball_speed(
        initial_frame, best_match_frame, delta_t, return_mph=True
    )
    data = {
        "Speed": ball_speed_mph,
        "VLA": launch_angle,
        "HLA": 0,
        "TotalSpin": total_spin_rpm,
        "SpinAxis": spin_axis,
    }
    trajectory_data, postitions = get_trajectory_metrics(data)
    carry = trajectory_data["carry_distance"]
    total = trajectory_data["total_distance"]
    apex = trajectory_data["apex"]
    hangtime = trajectory_data["time_of_flight"]
    desc_angle = trajectory_data["descending_angle"]
    print(f"Delta t: {delta_ms:.2f}")
    print(f"Ball Speed: {ball_speed_mph:.2f} mph")
    print(f"Vertical Launch Angle: {launch_angle:.2f} degrees")
    print(f"Total Spin: {data['TotalSpin']:.2f} rpm")
    print(f"Side Spin: {side_spin_rpm:.2f} rpm")
    print(f"Back Spin: {back_spin_rpm:.2f} rpm")
    print(f"Spin Axis: {spin_axis:.2f} degrees")
    print(f"Carry Distance: {carry:.2f} yd")
    print(f"Total Distance: {total:.2f} yd")
    print(f"Time of Flight: {hangtime:.2f} s")
    print(f"Apex Height: {apex:.2f} ft")
    print(f"Descending Angle: {desc_angle:.2f} degrees")

    # Persist results to SQLite
    try:
        db_path = os.path.join("data", "shots.db")
        init_db(db_path)

        def _json_default(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            return str(obj)

        record = {
            "initial_ts": float(initial_ts) if initial_ts is not None else None,
            "best_ts": float(best_ts) if best_ts is not None else None,
            "delta_t": float(delta_t),
            "speed_mph": float(ball_speed_mph),
            "vla_deg": float(launch_angle),
            "hla_deg": float(0),
            "total_spin_rpm": float(total_spin_rpm),
            "side_spin_rpm": float(side_spin_rpm),
            "back_spin_rpm": float(back_spin_rpm),
            "spin_axis_deg": float(spin_axis),
            "carry_yd": float(carry),
            "total_yd": float(total),
            "apex_ft": float(apex),
            "flight_time_s": float(hangtime),
            "descending_angle_deg": float(desc_angle),
            "initial_img_path": initial_path if "initial_path" in locals() else None,
            "best_img_path": best_path if "best_path" in locals() else None,
            "positions_json": json.dumps(postitions, default=_json_default),
        }
        insert_shot_record(db_path, record)
        print(f"Saved shot record to {db_path}")
    except Exception as e:
        print(f"Warning: failed to save DB record: {e}")


if __name__ == "__main__":
    main()
