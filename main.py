#!/usr/bin/env python3
# golf_launch_monitor.py
"""
Single‑camera golf‑ball launch‑monitor core loop.

Changes vs. previous revision
-----------------------------
* Square‑based out‑of‑frame check
* One‑shot thread spawning in hitting mode
* Safer delta calculation (prev_ballx / ball_width_px guards)
* Deterministic frame‑dict keys (frame # instead of ballx float)
* Hard timeout on hitting‑mode acquisition
"""

import cv2
import numpy as np
import time
from typing import List, Tuple, Optional, Dict

import camera.mvsdk as mvsdk
from image_processing.ballDetection import detect_golfballs
from image_processing.movementDetection import has_ball_moved
from image_processing.launchAngleCalculation import calculate_launch_angle
from spin.GetBallRotation import get_fine_ball_rotation
from trajectory_simulation.flightDataCalculation import get_trajectory_metrics

# ----------------- CONFIG -----------------
# Region of Interest (ROI) dimensions and position
ROI_W, ROI_H            = 640, 280
ROI_X, ROI_Y            =   0, 120

# Ball and capture settings
BALL_DIAM_MM            = 42.67
THRESHOLD_APART_MM      = 80.0      # Minimum distance in mm for capture pairing
DESIRED_EXPOSURE_US     = 50.0
DESIRED_ANALOG_GAIN     = 1000.0
DESIRED_GAMMA           = 0.25
FPS_NOMINAL             = 1300.0    # Nominal frames per second for trajectory simulation
DEBUG                   = True
MAX_CAPTURE_FRAMES      = 100       # Maximum frames to capture in hitting mode
SETUP_DET_INTERVAL      = 0.5       # Interval for detection in setup mode
WAIT_TO_CAPTURE         = 1.5       # Time to hold still before entering hitting mode
MOVEMENT_THRESHOLD_MM   = 2.0       # Movement threshold in mm to reset hold timer

# ----------------- UTILITIES -----------------
def send_to_visualization(*args, **kwargs):
    pass  # Placeholder for sending data to an external visualization system

def ball_fully_visible(frame_shape: Tuple[int, int, int],
                       x: float, y: float, r: float,
                       margin: int = 2) -> bool:
    # Check if the ball is fully visible within the frame with a margin
    h, w = frame_shape[:2]
    x1, y1 = int(x - r) - margin, int(y - r) - margin
    x2, y2 = int(x + r) + margin, int(y + r) + margin
    return 0 <= x1 < x2 < w and 0 <= y1 < y2 < h

def is_ball_out_of_image(frame_shape: Tuple[int, int, int],
                         x: float, y: float, r: float,
                         margin: int = 2) -> bool:
    """
    Determine if any corner of the smallest square enclosing the ball
    is outside the frame. This helps catch cases where part of the ball
    leaves the ROI even if the circle itself still intersects.
    """
    h, w = frame_shape[:2]
    side = 2 * r
    # Calculate the top-left corner of the square
    x0, y0 = x - r - margin, y - r - margin
    # Define all four corners of the square
    corners = [
        (x0,           y0),
        (x0 + side,    y0),
        (x0,           y0 + side),
        (x0 + side,    y0 + side),
    ]
    # Check if any corner is outside the frame
    for cx, cy in corners:
        if not (0 <= cx < w and 0 <= cy < h):
            return True
    return False

def process_metrics(frames: Dict[int, list], delta_t: float) -> None:
    """Compute ball speed, launch metrics, spin etc. from the two stored frames."""
    # Ensure frames are processed in chronological order
    fnums = sorted(frames.keys())
    frame1_data = frames[fnums[0]]
    frame2_data = frames[fnums[1]]

    frame1, _, ball1 = frame1_data
    frame2, _, ball2 = frame2_data

    # Extract ball positions and radius
    (c2x, c2y, r2) = ball1
    (cFx, cFy, _)  = ball2

    # Calculate distance in pixels and convert to mm
    dist_px = np.hypot(cFx - c2x, cFy - c2y)
    dist_mm = (dist_px / (2 * r2)) * BALL_DIAM_MM if r2 > 0 else 0
    # Calculate speed in meters per second
    speed_mps = (dist_mm / 1000.0) / delta_t if delta_t > 0 else 0
    print(f"Ball 1 position and radius: {ball1}")
    print(f"Ball 2 position and radius: {ball2}")
    # Display frames for visual confirmation
    cv2.imshow("Frame 1", frame1)
    cv2.imshow("Frame 2", frame2)
    cv2.waitKey(0)  # Wait for a key press to close the windows
    cv2.destroyAllWindows()
    # Calculate launch angle
    launch_angle = calculate_launch_angle(ball1, ball2)

    # Convert frames to grayscale for spin calculation
    g2   = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gFar = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    # Get fine ball rotation
    rx, ry, rz = get_fine_ball_rotation(g2, gFar)

    # Calculate spin rates in RPM
    side_spin_rpm  = (rx / delta_t) * (60.0 / 360.0) if delta_t > 0 else 0
    back_spin_rpm  = (ry / delta_t) * (60.0 / 360.0) if delta_t > 0 else 0
    # Calculate spin axis in degrees
    spin_axis_deg  = np.degrees(np.arctan2(side_spin_rpm, back_spin_rpm))

    # Prepare data for trajectory metrics calculation
    data = {
        "Speed":     speed_mps,
        "VLA":       launch_angle,
        "HLA":       0.0,
        "TotalSpin": back_spin_rpm,
        "SpinAxis":  spin_axis_deg,
    }
    print(f"Computed data: {data}")

    # Calculate trajectory metrics and positions
    metrics, positions = get_trajectory_metrics(data, delta_t)
    print(f"dt_between_frames: {delta_t:.6f} s")
    print(f"Speed: {speed_mps:.2f} m/s ({speed_mps * 2.23694:.1f} mph)")
    print(f"Launch angle: {launch_angle:.2f}°")
    print(f"Backspin: {back_spin_rpm:.0f} rpm | Side spin: {side_spin_rpm:.0f} rpm")
    print(f"Carry: {metrics['carry_distance']:.2f} yd | "
          f"Total: {metrics['total_distance']:.2f} yd")
    print(f"Flight time: {metrics['time_of_flight']:.2f} s | "
          f"Apex: {metrics['apex']:.2f} ft | "
          f"Desc: {metrics['descending_angle']:.2f}°")

    # Send computed data to visualization system
    send_to_visualization(
        positions, speed_mps, launch_angle,
        back_spin_rpm, side_spin_rpm, spin_axis_deg,
        metrics['carry_distance'], metrics['total_distance'],
        metrics['apex'], metrics['time_of_flight'],
        metrics['descending_angle']
    )

# ----------------- MAIN -----------------
def main() -> None:
    # Enumerate available camera devices
    devs = mvsdk.CameraEnumerateDevice()
    if not devs:
        print("No camera found")
        return

    # Initialize the first available camera
    hCamera = mvsdk.CameraInit(devs[0], -1, -1)
    pFrameBuffer = None

    try:
        # Get camera capabilities and set output format
        cap = mvsdk.CameraGetCapability(hCamera)
        mono = cap.sIspCapacity.bMonoSensor != 0
        fmt  = mvsdk.CAMERA_MEDIA_TYPE_MONO8 if mono else mvsdk.CAMERA_MEDIA_TYPE_BGR8
        mvsdk.CameraSetIspOutFormat(hCamera, fmt)

        # Set camera resolution and ROI
        res = mvsdk.CameraGetImageResolution(hCamera)
        res.iIndex      = 0xFF
        res.iHOffsetFOV = ROI_X
        res.iVOffsetFOV = ROI_Y
        res.iWidthFOV   = ROI_W
        res.iHeightFOV  = ROI_H
        res.iWidth      = ROI_W
        res.iHeight     = ROI_H
        mvsdk.CameraSetImageResolution(hCamera, res)

        # Configure camera exposure, gain, and gamma
        mvsdk.CameraSetTriggerMode(hCamera, 0)
        mvsdk.CameraSetAeState(hCamera, 0)
        mvsdk.CameraSetExposureTime(hCamera, DESIRED_EXPOSURE_US)
        gmin, gmax, _ = mvsdk.CameraGetAnalogGainXRange(hCamera)
        mvsdk.CameraSetAnalogGainX(hCamera, max(gmin, min(DESIRED_ANALOG_GAIN, gmax)))
        gamma_max = cap.sGammaRange.iMax
        mvsdk.CameraSetGamma(hCamera, int(DESIRED_GAMMA * gamma_max))

        # Start the camera feed
        mvsdk.CameraPlay(hCamera)
        print("Starting camera feed…")
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        delta_t_nominal = 1.0 / FPS_NOMINAL

        # ---- state vars ----
        in_setup          = True
        hitting_mode      = False
        last_setup_detect = 0.0
        prev_setup_det: Optional[Tuple[float, float, float]] = None
        ready_time        = 0.0
        ball_width_px     = None  # Set when entering hitting mode

        frames: Dict[int, list] = {}  # Filled by processing

        # ---------- main live‑view loop ----------
        
        while True:
            try:
                # Capture a frame from the camera
                pRaw, head = mvsdk.CameraGetImageBuffer(hCamera, 2000)
                if pFrameBuffer is None:
                    pFrameBuffer = mvsdk.CameraAlignMalloc(head.uBytes, 16)
                mvsdk.CameraImageProcess(hCamera, pRaw, pFrameBuffer, head)
                mvsdk.CameraReleaseImageBuffer(hCamera, pRaw)

                buf = (mvsdk.c_ubyte * head.uBytes).from_address(pFrameBuffer)
                frame = np.frombuffer(buf, dtype=np.uint8).reshape(
                    (head.iHeight, head.iWidth,
                     1 if fmt == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3)
                )
                if fmt == mvsdk.CAMERA_MEDIA_TYPE_MONO8:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                # -------------- setup mode --------------
                if in_setup:
                    now = time.time()
                    # Check if it's time to perform a new detection
                    if now - last_setup_detect >= SETUP_DET_INTERVAL:
                        last_setup_detect = now
                        dets = detect_golfballs(frame, conf=0.25,
                                                imgsz=640, display=False)
                        if not dets:
                            print("No ball detected in setup mode.")
                            prev_setup_det = None
                            ready_time = 0.0
                        else:
                            x, y, r = dets[0]
                            # Ensure the ball is fully visible
                            if not ball_fully_visible(frame.shape, x, y, r):
                                print("Adjust ball position to be fully visible.")
                                prev_setup_det = None
                                ready_time = 0.0
                            else:
                                mm_per_px = BALL_DIAM_MM / (2 * r)
                                if prev_setup_det is None:
                                    prev_setup_det = (x, y, r)
                                    ready_time = now
                                    print(f"Ball detected – hold still for "
                                          f"{WAIT_TO_CAPTURE:.1f}s…")
                                else:
                                    # Calculate movement since last detection
                                    moved_mm = np.hypot(x - prev_setup_det[0],
                                                        y - prev_setup_det[1]) * mm_per_px
                                    if moved_mm >= MOVEMENT_THRESHOLD_MM:
                                        prev_setup_det = (x, y, r)
                                        ready_time = now
                                        print(f"Ball moved {moved_mm:.2f} mm – "
                                              "resetting timer.")
                                    elif now - ready_time >= WAIT_TO_CAPTURE:
                                        print("Ball stable!  Entering hitting mode.")
                                        in_setup      = False
                                        hitting_mode  = True
                                        ball_width_px = r * 2
                else:
                    # Capture and process frames
                    frames.clear()
                    frame_idx = 0
                    prev_ballx: Optional[float] = None
                    
                    while frame_idx < MAX_CAPTURE_FRAMES:
                        try:
                            pRaw, head = mvsdk.CameraGetImageBuffer(hCamera, 2000)
                            mvsdk.CameraImageProcess(hCamera, pRaw, pFrameBuffer, head)
                            mvsdk.CameraReleaseImageBuffer(hCamera, pRaw)

                            buf = (mvsdk.c_ubyte * head.uBytes).from_address(pFrameBuffer)
                            f = np.frombuffer(buf, dtype=np.uint8).reshape(
                                (head.iHeight, head.iWidth,
                                    1 if fmt == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3)
                            )
                            if fmt == mvsdk.CAMERA_MEDIA_TYPE_MONO8:
                                f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)

                            # Process frame
                            dts = detect_golfballs(f, conf=0.5, imgsz=640, display=False)
                            if not dts:
                                print("No golf balls detected in frame.")
                                continue

                            cx, cy, r = dts[0]
                            # Check if the ball is out of image bounds
                            if is_ball_out_of_image(f.shape, cx, cy, r):
                                print("Ball is out of image bounds.")
                                break

                            if ball_width_px is None:
                                print("Ball width in pixels is not set.")
                                continue  # Safety check; shouldn't happen

                            # Calculate movement in mm
                            mm_per_px = BALL_DIAM_MM / ball_width_px
                            if prev_ballx is None:
                                prev_ballx = cx
                            delta_mm = abs(cx - prev_ballx) * mm_per_px
                            print(f"Delta movement in mm: {delta_mm}")

                            # Keep at most 2 frames for processing
                            frames[frame_idx] = [f, frame_idx, (cx, cy, r)]
                            if len(frames) > 2:
                                del frames[min(frames.keys())]

                            # Check if frames are far enough apart
                            if delta_mm >= THRESHOLD_APART_MM:
                                print("Frames are far enough apart, stopping processing.")
                                break  # Done – we have two frames far enough apart

                            frame_idx += 1
                        except mvsdk.CameraException:
                            print("Exception during frame capture.")
                            continue

                        hitting_mode = False
                        in_setup     = True  # Ready for next shot
                        if len(frames) == 2:
                            process_metrics(frames, delta_t_nominal)
                        else:
                            print("Could not obtain two clear frames 8 cm apart.")
                        prev_setup_det = None  # Force a fresh detection cycle

                # Show the live feed
                cv2.imshow("frame", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exiting main loop.")
                    break

            except mvsdk.CameraException as e:
                if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                    print(f"Camera exception occurred: {e}")
                    raise

        if pFrameBuffer:
            mvsdk.CameraAlignFree(pFrameBuffer)

    finally:
        # Clean up resources
        mvsdk.CameraUnInit(hCamera)
        cv2.destroyAllWindows()
        print("Camera uninitialized and windows destroyed.")

if __name__ == "__main__":
    main()
