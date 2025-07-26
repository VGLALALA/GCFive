#coding=utf-8
import cv2
import numpy as np
import time
import camera.cv_grab_callback as cv_grab_callback  # Import the monitoring module
import queue
import threading
import math
from camera.hittingZoneCalibration import calibrate_hitting_zone_stream
from image_processing.ballDetectionyolo import detect_golfballs  # Import YOLO detection function
from image_processing.ballinZoneCheck import is_point_in_zone  # Import the zone check function
from image_processing.get2Dcoord import get_ball_xz
from spin.GetBallRotation import get_fine_ball_rotation
from spin.spinAxis import calculate_spin_axis
from spin.GetLaunchAngle import calculate_launch_angle
from image_processing.ballSpeedCalculation import calculate_ball_speed
from trajectory_simulation.flightDataCalculation import get_trajectory_metrics
RECALIBRATE_HITTING_ZONE = False
FPS = 1300
DELTA_T = 1/FPS

def main():
    initial_frame, best_match_frame = None, None
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

    # --- Detection Loop ---
    # Capture frames until ball detected or 'q' pressed
    while not ball_detected and (cv2.waitKey(1) & 0xFF) != ord('q'):
        try:
            # Grab a frame from the camera
            frame = cam.grab()

            # Convert to BGR for YOLO detection if it's a grayscale image
            if monoCamera:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                frame_bgr = frame

            # Use YOLO to detect golf balls
            detected_balls = detect_golfballs(frame_bgr, conf=0.9, imgsz=640, display=False)
            ballx, ballz = get_ball_xz(frame_bgr, detected_balls)
            if detected_balls:
                # Take the first detected ball
                center_x, center_y, radius = detected_balls[0]
                print(f"Ball detected at position: ({ballx}, {ballz}) with radius: {radius}")
                detected_circle = (center_x, center_y, radius)
                
                # Check if the detected ball is within the predefined zone
                if is_point_in_zone(ballx, ballz):
                    print("Ball is within the zone.")
                    ball_detected = True
                else:
                    print("Ball is outside the zone.")
                    continue
                
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
                    original_cropped_roi = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).copy()

                # Draw the detected circle
                cv2.circle(frame_bgr, (center_x, center_y), radius, (0, 255, 0), 2)
                cv2.circle(frame_bgr, (center_x, center_y), 2, (0, 0, 255), 3)

                # Display the frame with detection and the cropped ROI
                cv2.imshow("Ball Detection - Press q to exit", frame_bgr)
                cv2.imshow("Detected Ball (Cropped)", original_cropped_roi)

                print("Ball detected! Press any key in a display window to start monitoring.")
                cv2.waitKey(0) # Wait indefinitely for a key press
                cv2.destroyAllWindows() # Close detection windows

                # Break the detection loop
                break

        except Exception as e:
            print(f"Camera grab failed: {e}")
            pass  # Continue loop on errors

    # --- Start Monitoring if ball was detected ---
    if ball_detected and cam:
        print("Starting monitoring...")

        # Create queue and stop event
        frame_queue = queue.Queue(maxsize=10) # Limit queue size
        stop_event = threading.Event()

        # Create and start acquisition thread
        acquire_thread = threading.Thread(target=cv_grab_callback.acquire_frames, args=(cam, frame_queue, stop_event))
        acquire_thread.start()

        # Create and start processing thread
        process_thread = threading.Thread(target=cv_grab_callback.process_frames, args=(cam, detected_circle, original_cropped_roi, frame_queue, stop_event))
        process_thread.start()

        # Main thread loop to keep program alive and handle events
        print("Press 'q' to stop monitoring.")
        while not stop_event.is_set() and (cv2.waitKey(1) & 0xFF) != ord('q'):
            time.sleep(0.01) # Small sleep to prevent busy waiting

        # Signal threads to stop and wait for them to finish
        stop_event.set()
        acquire_thread.join()
        frame_queue.put(None)
        process_thread.join()

        # Retrieve the initial and best match frames from the processing thread
        initial_frame, best_match_frame, initial_idx, best_idx = cv_grab_callback.retriveData()
        delta_idx = best_idx - initial_idx

        print("Monitoring stopped.")

    # --- Release camera and buffer ---
    cv_grab_callback.release_camera_and_buffer(cam)
    print("Camera and buffer released.")

    best_rot_x, best_rot_y, best_rot_z = get_fine_ball_rotation(initial_frame, best_match_frame)
    side_spin_rpm = math.abs((best_rot_x / (DELTA_T * delta_idx)) * (60 / 360))
    back_spin_rpm = math.abs((best_rot_y / (DELTA_T * delta_idx)) * (60 / 360))
    spin_axis = calculate_spin_axis(back_spin_rpm, side_spin_rpm)
    launch_angle = calculate_launch_angle(initial_frame, best_match_frame)
    ball_speed_mph = calculate_ball_speed(initial_frame, best_match_frame, DELTA_T * delta_idx, return_mph=True)
    data = {
        "Speed": ball_speed_mph,
        "VLA": launch_angle,
        "HLA": 0,
        "TotalSpin":  math.hypot(back_spin_rpm, side_spin_rpm),
        "SpinAxis": spin_axis
    }
    trajectory_data, postitions = get_trajectory_metrics(data)
    carry = trajectory_data["carry_distance"]
    total = trajectory_data["total_distance"]
    apex = trajectory_data["apex"]
    hangtime = trajectory_data["time_of_flight"]
    desc_angle = trajectory_data["descending_angle"]
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
if __name__ == "__main__":
    main() 
