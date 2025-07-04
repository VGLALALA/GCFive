#coding=utf-8
import cv2
import numpy as np
import mvsdk
import platform
import time
import cv_grab_callback # Import the monitoring module
import queue
import threading

def main_loop():
    # Setup camera and buffer using the helper function from cv_grab_callback
    hCamera, monoCamera = cv_grab_callback.setup_camera_and_buffer()
    if hCamera is None:
        print("Failed to set up camera.")
        return

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
            # Get a frame from camera for detection
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 1250)
            # Process into the global buffer for display/processing
            mvsdk.CameraImageProcess(hCamera, pRawData, cv_grab_callback.pFrameBuffer_global, FrameHead)
            mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)

            if platform.system() == "Windows":
                mvsdk.CameraFlipFrameBuffer(cv_grab_callback.pFrameBuffer_global, FrameHead, 1)
            
            # Convert frame to OpenCV format
            frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(cv_grab_callback.pFrameBuffer_global)
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))
            
            # Convert to grayscale for detection if it's a color image
            if not monoCamera:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)

            # Detect circles using Hough transform
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=50,
                param1=50,
                param2=30,
                minRadius=30,
                maxRadius=50
            )

            if circles is not None:
                circles = np.uint16(np.around(circles))
                # Assuming the largest circle is the ball (or just take the first one)
                i = circles[0, 0] # Take the first detected circle
                
                # Get ball center and radius
                center_x, center_y = i[0], i[1]
                radius = i[2]
                
                print(f"Ball detected at position: ({center_x}, {center_y}) with radius: {radius}")
                detected_circle = (center_x, center_y, radius)
                ball_detected = True

                # Calculate crop coordinates
                crop_size = 100  # Size of the square crop around the ball
                half_crop = crop_size // 2
                x1 = max(0, center_x - half_crop)
                y1 = max(0, center_y - half_crop)
                x2 = min(gray.shape[1], center_x + half_crop)
                y2 = min(gray.shape[0], center_y + half_crop)

                # Crop the region around the ball and make a deep copy
                original_cropped_roi = gray[y1:y2, x1:x2].copy()

                # Draw the detected circle
                cv2.circle(frame, (center_x, center_y), radius, (0, 255, 0), 2)
                cv2.circle(frame, (center_x, center_y), 2, (0, 0, 255), 3)

                # Display the frame with detection and the cropped ROI
                cv2.imshow("Ball Detection - Press q to exit", frame)
                cv2.imshow("Detected Ball (Cropped)", original_cropped_roi)

                print("Ball detected! Press any key in a display window to start monitoring.")
                cv2.waitKey(0) # Wait indefinitely for a key press
                cv2.destroyAllWindows() # Close detection windows

                # Break the detection loop
                break

            # Display the frame during detection search
            cv2.imshow("Ball Detection - Press q to exit", frame)

        except mvsdk.CameraException as e:
            if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message))
            pass # Continue loop on timeout or other camera errors

    # --- Start Monitoring if ball was detected ---
    if ball_detected and hCamera:
        print("Starting monitoring...") # This print is also in the process_frames thread, can keep either or both

        # Create queue and stop event
        frame_queue = queue.Queue(maxsize=10) # Limit queue size
        stop_event = threading.Event()

        # Create and start acquisition thread
        acquire_thread = threading.Thread(target=cv_grab_callback.acquire_frames, args=(hCamera, frame_queue, stop_event))
        acquire_thread.start()

        # Create and start processing thread
        process_thread = threading.Thread(target=cv_grab_callback.process_frames, args=(hCamera, monoCamera, detected_circle, original_cropped_roi, frame_queue, stop_event))
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
    cv_grab_callback.release_camera_and_buffer(hCamera)
    print("Camera and buffer released.")

def main():
    main_loop()

if __name__ == "__main__":
    main() 