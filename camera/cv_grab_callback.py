#coding=utf-8
import cv2
import numpy as np
from camera.MVSCamera import MVSCamera
import time
import queue
import traceback
from image_processing.ballDetectionyolo import detect_golfballs  # Import YOLO detection function
BALL_DIAM_MM = 42.67
GOLF_BALL_RADIUS_MM = 21.335
THRESHOLD_APART_MM = 50.0  # Minimum distance in mm for capture pairing
DESIRED_EXPOSURE_US = 50.0
DESIRED_ANALOG_GAIN = 1000.0
DESIRED_GAMMA = 0.25
FPS_NOMINAL = 1300.0  # Nominal frames per second for trajectory simulation
DEBUG = True
MAX_CAPTURE_FRAMES = 100  # Maximum frames to capture in hitting mode
SETUP_DET_INTERVAL = 0.5  # Interval for detection in setup mode
WAIT_TO_CAPTURE = 1.5  # Time to hold still before entering hitting mode
MOVEMENT_THRESHOLD_MM = 2.0
# Add new global variables
FRAMES_TO_CAPTURE = 10  # Number of frames to capture
TARGET_FPS = 1300  # Target FPS for camera
recorded_frames = []
is_recording = False
INITIAL_IDX, BEST_IDX = None, None
INITIAL_FRAME, BEST_FRAME = None, None  # Corrected typo from INTIAL_FRAME

def setup_camera_and_buffer():
    """Initialize and open the camera using the MVSCamera wrapper."""
    try:
        cam = MVSCamera(
            640, 280, 0, 120,
            DESIRED_EXPOSURE_US,
            DESIRED_ANALOG_GAIN,
            DESIRED_GAMMA,
        )
        cam.open()
        return cam
    except Exception as e:
        print("Exception in setup_camera_and_buffer:", e)
        traceback.print_exc()
        return None

def release_camera_and_buffer(cam):
    """Close the MVSCamera and free any buffers."""
    try:
        if cam:
            cam.close()
    except Exception as e:
        print("Exception in release_camera_and_buffer:", e)
        traceback.print_exc()

def acquire_frames(cam, frame_queue, stop_event):
    """Continuously grab frames from the camera and push them onto a queue."""
    print("Acquisition thread started.")
    try:
        while not stop_event.is_set():
            try:
                frame = cam.grab()
                frame_queue.put(frame)
            except Exception as e:
                print(f"Acquisition thread camera error: {e}")
    except Exception as e:
        print("Exception in acquire_frames:", e)
        traceback.print_exc()
    print("Acquisition thread stopped.")

import math
from camera.focalPointCalibration import load_calibration

def process_frames(cam,
                   detected_circle,
                   original_cropped_roi,
                   frame_queue,
                   stop_event,
                   interactive=False):
    """
    Pull frames, watch ROI for motion, record a burst, then:

     • if interactive: step through newest→oldest with YOLO overlays
     • always: auto‑select the frame whose real‑world ball displacement
               is closest to FRAME_APART_MM

    """
    print("Processing thread started.")
    global recorded_frames, is_recording, INITIAL_FRAME, BEST_FRAME, INITIAL_IDX, BEST_IDX
    monoCamera = cam.mono

    try:
        # 1) Unpack initial detection
        first_x_px, first_y_px, first_r_px = detected_circle

        # 2) Build square ROI around that point for motion detection
        h_roi, w_roi = original_cropped_roi.shape[:2]
        half_crop = 100 // 2
        x1 = max(0, first_x_px - half_crop)
        y1 = max(0, first_y_px - half_crop)
        x2 = min(x1 + w_roi, 640)
        y2 = min(y1 + h_roi, 300)

        movement_threshold = 12
        print("Monitoring movement in the detected area…")

        frame_count = 0
        start_time = time.time()

        # ——— Phase A: wait for motion, then record FRAMES_TO_CAPTURE frames ———
        while not stop_event.is_set() or not frame_queue.empty():
            try:
                frame = frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if frame is None:
                break

            gray = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if (not monoCamera and frame.ndim == 3) else frame)

            # crop + resize to match original_cropped_roi
            x1c = max(0, min(first_x_px - half_crop, gray.shape[1]-1))
            y1c = max(0, min(first_y_px - half_crop, gray.shape[0]-1))
            x2c = max(x1c+1, min(first_x_px + half_crop, gray.shape[1]))
            y2c = max(y1c+1, min(first_y_px + half_crop, gray.shape[0]))
            crop = gray[y1c:y2c, x1c:x2c]
            if crop.shape[:2] != original_cropped_roi.shape[:2] and crop.size:
                try:
                    crop = cv2.resize(crop, (w_roi, h_roi))
                except Exception:
                    continue

            # compare & trigger
            if original_cropped_roi.size and crop.size:
                diff = cv2.absdiff(original_cropped_roi, crop)
                md = np.mean(diff)
                frame_count += 1
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0.0
                print(f"Comparison FPS: {fps:.1f}, Mean Difference: {md:.2f}")

                if md > movement_threshold and not is_recording:
                    print("BALL MOVED → starting recording")
                    is_recording = True
                    recorded_frames = []

                if is_recording:
                    recorded_frames.append(gray.copy())
                    print(f"Recording FPS: {fps:.1f}")
                    if len(recorded_frames) >= FRAMES_TO_CAPTURE:
                        print(f"Recording complete: {len(recorded_frames)} frames")
                        is_recording = False
                        stop_event.set()
                        break

        # ——— Reference real‑world coords from the *first* frame ———
        focal_px, _ = load_calibration()
        if focal_px is None:
            raise RuntimeError("Camera must be calibrated first")

        # estimate first frame depth:
        z0_mm = (GOLF_BALL_RADIUS_MM * focal_px) / first_r_px

        # ——— 2) Auto best‑match pass: find frame ≈ FRAME_APART_MM apart ———
        print("\n=== Auto best‑match for separation ≈", THRESHOLD_APART_MM, "mm ===")
        best_idx, best_score = None, float('inf')

        for idx, frm in enumerate(recorded_frames):
            # prep for YOLO
            if frm.ndim == 2:
                bgr = cv2.cvtColor(frm, cv2.COLOR_GRAY2BGR)
            else:
                bgr = frm
            if bgr.shape[2] != 3:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)

            try:
                circles = detect_golfballs(bgr, conf=0.7, imgsz=640, display=False)
            except Exception as e:
                print(f"YOLO error in frame {idx}:", e)
                continue

            for (cx, cy, rr) in circles:
                # pixel‐space separation:
                pd = math.hypot(cx - first_x_px, cy - first_y_px)
                # convert px → mm (approx horizontal only)
                sep_mm = pd * (z0_mm / focal_px)
                score = abs(sep_mm - THRESHOLD_APART_MM)
                print(f"  Frame {idx}: {sep_mm:.1f} mm (score {score:.1f})")
                if score < best_score:
                    best_score = score
                    best_idx = idx

        if best_idx is None:
            print("No valid detection found at the desired separation.")
        else:
            INITIAL_FRAME, BEST_FRAME = recorded_frames[0], recorded_frames[best_idx]
            INITIAL_IDX, BEST_IDX = 0, best_idx
        print("Processing complete.")

    except Exception as e:
        print("Exception in process_frames:", e)
        traceback.print_exc()
        stop_event.set()
    finally:
        try: cv2.destroyAllWindows()
        except: pass
        print("Processing thread stopped.")

def retriveData():
    return INITIAL_FRAME, BEST_FRAME, INITIAL_IDX, BEST_IDX