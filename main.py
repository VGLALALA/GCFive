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
import threading
from queue import Queue, Empty
from typing import List, Tuple, Optional, Dict

import camera.mvsdk as mvsdk
from image_processing.ballDetection import detect_golfballs
from image_processing.movementDetection import has_ball_moved
from image_processing.launchAngleCalculation import calculate_launch_angle
from spin.GetBallRotation import get_fine_ball_rotation
from trajectory_simulation.flightDataCalculation import get_trajectory_metrics

# ----------------- CONFIG -----------------
ROI_W, ROI_H            = 640, 280
ROI_X, ROI_Y            =   0, 120

BALL_DIAM_MM            = 42.67
THRESHOLD_APART_MM      = 80.0      # 8 cm apart requirement for capture pairing
DESIRED_EXPOSURE_US     = 50.0
DESIRED_ANALOG_GAIN     = 1000.0
DESIRED_GAMMA           = 0.25
FPS_NOMINAL             = 1300.0    # for trajectory‑sim Δt
DEBUG                   = True
MAX_CAPTURE_FRAMES      = 600
      # seconds in hitting mode before we bail
SETUP_DET_INTERVAL      = 0.5        # YOLO every 0.5 s in setup
WAIT_TO_CAPTURE         = 1.5        # must hold still for 1.5 s before hitting mode
MOVEMENT_THRESHOLD_MM   = 2.0        # movement ≥2 mm resets hold timer

# ----------------- UTILITIES -----------------
def send_to_visualization(*args, **kwargs):
    pass  # stub for external socket / UI feed


def ball_fully_visible(frame_shape: Tuple[int, int, int],
                       x: float, y: float, r: float,
                       margin: int = 2) -> bool:
    h, w = frame_shape[:2]
    x1, y1 = int(x - r) - margin, int(y - r) - margin
    x2, y2 = int(x + r) + margin, int(y + r) + margin
    return 0 <= x1 < x2 < w and 0 <= y1 < y2 < h


def is_ball_out_of_image(frame_shape: Tuple[int, int, int],
                         x: float, y: float, r: float,
                         margin: int = 2) -> bool:
    """
    Form the smallest *square* that encloses the circle and test whether
    **any** corner is outside the frame.  This catches cases where part of
    the ball leaves the ROI even if the circle itself still intersects.
    """
    h, w = frame_shape[:2]
    side = 2 * r
    # Top‑left corner of the square
    x0, y0 = x - r - margin, y - r - margin
    # All four corners
    corners = [
        (x0,           y0),
        (x0 + side,    y0),
        (x0,           y0 + side),
        (x0 + side,    y0 + side),
    ]
    for cx, cy in corners:
        if not (0 <= cx < w and 0 <= cy < h):
            return True
    return False


def process_metrics(frames: Dict[int, list], delta_t: float) -> None:
    """Compute ball speed, launch metrics, spin etc. from the two stored frames."""
    # Ensure chronological order
    fnums = sorted(frames.keys())
    frame1_data = frames[fnums[0]]
    frame2_data = frames[fnums[1]]

    frame1, _, ball1 = frame1_data
    frame2, _, ball2 = frame2_data

    (c2x, c2y, r2) = ball1
    (cFx, cFy, _)  = ball2

    dist_px = np.hypot(cFx - c2x, cFy - c2y)
    dist_mm = (dist_px / (2 * r2)) * BALL_DIAM_MM if r2 > 0 else 0
    speed_mps = (dist_mm / 1000.0) / delta_t if delta_t > 0 else 0
    print(f"Ball 1 position and radius: {ball1}")
    print(f"Ball 2 position and radius: {ball2}")
    cv2.imshow("Frame 1", frame1)
    cv2.imshow("Frame 2", frame2)
    cv2.waitKey(0)  # Wait for a key press to close the windows
    cv2.destroyAllWindows()
    launch_angle = calculate_launch_angle(ball1, ball2)

    g2   = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gFar = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    rx, ry, rz = get_fine_ball_rotation(g2, gFar)

    side_spin_rpm  = (rx / delta_t) * (60.0 / 360.0) if delta_t > 0 else 0
    back_spin_rpm  = (ry / delta_t) * (60.0 / 360.0) if delta_t > 0 else 0
    spin_axis_deg  = np.degrees(np.arctan2(side_spin_rpm, back_spin_rpm))

    data = {
        "Speed":     speed_mps,
        "VLA":       launch_angle,
        "HLA":       0.0,
        "TotalSpin": back_spin_rpm,
        "SpinAxis":  spin_axis_deg,
    }
    print(f"Computed data: {data}")

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

    send_to_visualization(
        positions, speed_mps, launch_angle,
        back_spin_rpm, side_spin_rpm, spin_axis_deg,
        metrics['carry_distance'], metrics['total_distance'],
        metrics['apex'], metrics['time_of_flight'],
        metrics['descending_angle']
    )


# ----------------- MAIN -----------------
def main() -> None:
    devs = mvsdk.CameraEnumerateDevice()
    if not devs:
        print("No camera found")
        return

    hCamera = mvsdk.CameraInit(devs[0], -1, -1)
    pFrameBuffer = None

    try:
        cap = mvsdk.CameraGetCapability(hCamera)
        mono = cap.sIspCapacity.bMonoSensor != 0
        fmt  = mvsdk.CAMERA_MEDIA_TYPE_MONO8 if mono else mvsdk.CAMERA_MEDIA_TYPE_BGR8
        mvsdk.CameraSetIspOutFormat(hCamera, fmt)

        # ROI
        res = mvsdk.CameraGetImageResolution(hCamera)
        res.iIndex      = 0xFF
        res.iHOffsetFOV = ROI_X
        res.iVOffsetFOV = ROI_Y
        res.iWidthFOV   = ROI_W
        res.iHeightFOV  = ROI_H
        res.iWidth      = ROI_W
        res.iHeight     = ROI_H
        mvsdk.CameraSetImageResolution(hCamera, res)

        # exposure / gain / gamma
        mvsdk.CameraSetTriggerMode(hCamera, 0)
        mvsdk.CameraSetAeState(hCamera, 0)
        mvsdk.CameraSetExposureTime(hCamera, DESIRED_EXPOSURE_US)
        gmin, gmax, _ = mvsdk.CameraGetAnalogGainXRange(hCamera)
        mvsdk.CameraSetAnalogGainX(hCamera, max(gmin, min(DESIRED_ANALOG_GAIN, gmax)))
        gamma_max = cap.sGammaRange.iMax
        mvsdk.CameraSetGamma(hCamera, int(DESIRED_GAMMA * gamma_max))

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
        ball_width_px     = None  # set when we enter hitting‑mode

        # Thread‑shared objects
        frame_queue: Queue = Queue(maxsize=MAX_CAPTURE_FRAMES)
        stop_event         = threading.Event()

        # Because we only want one capture/processor thread pair,
        # we'll create them lazily once we flip to hitting‑mode.
        capture_thread: Optional[threading.Thread]   = None
        processor_thread: Optional[threading.Thread] = None

        frames: Dict[int, list] = {}  # filled by processor thread

        # ---------- worker defs ----------
        def capture_worker() -> None:
            frame_idx = 0
            while (not stop_event.is_set()
                   and frame_idx < MAX_CAPTURE_FRAMES):
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
                    frame_queue.put((f, frame_idx), timeout=0.01)
                    print(f"Captured frame {frame_idx}")
                    frame_idx += 1
                except (mvsdk.CameraException, Full):
                    print("Capture worker encountered an exception or full queue.")
                    continue
            stop_event.set()  # graceful shutdown
            print("Capture worker stopped.")

        def process_worker() -> None:
            nonlocal frames
            prev_ballx: Optional[float] = None

            while not stop_event.is_set() or not frame_queue.empty():
                try:
                    frame, f_idx = frame_queue.get(timeout=0.1)
                    print(f"Processing frame {f_idx}")
                except Empty:
                    print("Process worker waiting for frames.")
                    continue

                dts = detect_golfballs(frame, conf=0.5, imgsz=640, display=False)
                if not dts:
                    print("No golf balls detected in frame.")
                    continue

                cx, cy, r = dts[0]
                if is_ball_out_of_image(frame.shape, cx, cy, r):
                    print("Ball is out of image bounds.")
                    stop_event.set()
                    break

                if ball_width_px is None:
                    print("Ball width in pixels is not set.")
                    continue  # safety; shouldn't happen

                mm_per_px = BALL_DIAM_MM / ball_width_px
                if prev_ballx is None:
                    prev_ballx = cx
                delta_mm = abs(cx - prev_ballx) * mm_per_px
                print(f"Delta movement in mm: {delta_mm}")

                # keep at most 2 frames
                frames[f_idx] = [frame, f_idx, (cx, cy, r)]
                if len(frames) > 2:
                    del frames[min(frames.keys())]

                if delta_mm >= THRESHOLD_APART_MM:
                    print("Frames are far enough apart, stopping processing.")
                    stop_event.set()
                    break  # done – we have two frames far enough apart

        # ---------- main live‑view loop ----------
        
        while True:
            try:
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

                                        # spawn workers
                                        stop_event.clear()
                                        frames.clear()
                                        capture_thread = threading.Thread(
                                            target=capture_worker,
                                            daemon=True,
                                        )
                                        processor_thread = threading.Thread(
                                            target=process_worker,
                                            daemon=True,
                                        )
                                        capture_thread.start()
                                        processor_thread.start()

                # -------------- hitting mode --------------
                elif hitting_mode:

                    if stop_event.is_set():
                        print("Stopping hitting mode.")
                        capture_thread.join()
                        processor_thread.join()
                        hitting_mode = False
                        in_setup     = True  # ready for next shot
                        if len(frames) == 2:
                            process_metrics(frames, delta_t_nominal)
                        else:
                            print("Could not obtain two clear frames 8 cm apart.")
                        prev_setup_det = None  # force a fresh detection cycle

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
        mvsdk.CameraUnInit(hCamera)
        cv2.destroyAllWindows()
        print("Camera uninitialized and windows destroyed.")


if __name__ == "__main__":
    main()
