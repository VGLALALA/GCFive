#coding=utf-8
import cv2
import numpy as np
import time
import camera.cv_grab_callback as cv_grab_callback  # Import the monitoring module
import queue
import threading
from camera.hittingZoneCalibration import calibrate_hitting_zone_stream
from image_processing.ballDetectionyolo import detect_golfballs  # Import YOLO detection function
from image_processing.ballinZoneCheck import is_point_in_zone  # Import the zone check function

def main():
    # Setup camera using the helper function from cv_grab_callback
    cam = cv_grab_callback.setup_camera_and_buffer()
    if cam is None:
        print("Failed to set up camera.")
        return

    monoCamera = cam.mono

    # Perform hitting zone calibration using the same camera settings
    calibrate_hitting_zone_stream(cam=cam)

    # The global buffer is allocated in setup_camera_and_buffer now
    # pFrameBuffer = cv_grab_callback.pFrameBuffer_global 

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

            # Debug: Print frame info
            print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}, min: {frame.min()}, max: {frame.max()}")
            
            # Convert to BGR for YOLO detection if it's a grayscale image
            if monoCamera:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                frame_bgr = frame

            # Use YOLO to detect golf balls
            detected_balls = detect_golfballs(frame_bgr, conf=0.25, imgsz=640, display=False)

            if detected_balls:
                # Take the first detected ball
                center_x, center_y, radius = detected_balls[0]
                print(f"Ball detected at position: ({center_x}, {center_y}) with radius: {radius}")
                detected_circle = (center_x, center_y, radius)

                # Check if the detected ball is within the predefined zone
                
                if is_point_in_zone(center_x, center_y):
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
                # Use grayscale for ROI if original was grayscale
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

            # Display the frame during detection search
            cv2.imshow("Ball Detection - Press q to exit", frame_bgr)
            
            # Force display update
            cv2.waitKey(1)

        except Exception as e:
            print(f"Camera grab failed: {e}")
            pass  # Continue loop on errors

    # --- Start Monitoring if ball was detected ---
    if ball_detected and cam:
        print("Starting monitoring...") # This print is also in the process_frames thread, can keep either or both

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
            # The processing thread handles its own display updates
            time.sleep(0.01) # Small sleep to prevent busy waiting

        # Signal threads to stop and wait for them to finish
        stop_event.set()
        acquire_thread.join()
        # Put a sentinel value in the queue to signal the processing thread to exit the queue.get() block
        frame_queue.put(None)
        process_thread.join()

        print("Monitoring stopped.")

    # --- Release camera and buffer ---
    cv_grab_callback.release_camera_and_buffer(cam)
    print("Camera and buffer released.")

if __name__ == "__main__":
    main() 
