import cv2
import numpy as np
import time

from image_processing.ballDetection import detect_golfballs
from image_processing.movementDetection import has_ball_moved
from image_processing.launchAngleCalculation import calculate_launch_angle
from spin.GetBallRotation import get_fine_ball_rotation
from trajectory_simulation.flightDataCalculation import get_trajectory_metrics
# from image_processing.ballSpeedCalculation import calculate_ball_speed  # if you prefer

import camera.mvsdk as mvsdk

# ----------------- CONFIG -----------------
ROI_W, ROI_H = 640, 292
ROI_X, ROI_Y = 0,   120

BALL_DIAM_MM          = 42.67
BALLS_FOR_20CM        = 80.0 / BALL_DIAM_MM    # ≈ 4.69 ball diameters
DESIRED_EXPOSURE_US   = 50.0
DESIRED_ANALOG_GAIN   = 1000.0
DESIRED_GAMMA         = 0.25
FPS_NOMINAL           = 1300.0                  # for trajectory sim dt
DEBUG                 = True
MISS_TOLERANCE        = 5
MAX_CAPTURE_FRAMES    = 400

# Throttled print: at most once every 0.5s
_last_print_time = 0.0
def throttled_print(msg: str):
    global _last_print_time
    now = time.time()
    if now - _last_print_time >= 0.5:
        print(msg)
        _last_print_time = now

def send_to_visualization(*args, **kwargs):
    pass  # TODO: implement networking/plotting here

def show_debug_pair(img_a, img_b, text_left="", text_right="", footer=""):
    pair = np.hstack([img_a.copy(), img_b.copy()])
    if text_left:
        cv2.putText(pair, text_left, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    if text_right:
        off = img_a.shape[1] + 10
        cv2.putText(pair, text_right, (off, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    if footer:
        cv2.putText(pair, footer, (10, pair.shape[0]-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.imshow("DEBUG: 2nd vs FAR", pair)
    cv2.waitKey(0)
    cv2.destroyWindow("DEBUG: 2nd vs FAR")

def main():
    # ---- init camera ----
    devs = mvsdk.CameraEnumerateDevice()
    if not devs:
        print("No camera found")
        return

    hCamera = mvsdk.CameraInit(devs[0], -1, -1)
    pFrameBuffer = None

    try:
        cap  = mvsdk.CameraGetCapability(hCamera)
        mono = (cap.sIspCapacity.bMonoSensor != 0)
        fmt  = mvsdk.CAMERA_MEDIA_TYPE_MONO8 if mono else mvsdk.CAMERA_MEDIA_TYPE_BGR8
        mvsdk.CameraSetIspOutFormat(hCamera, fmt)

        # ROI / resolution
        res = mvsdk.CameraGetImageResolution(hCamera)
        res.iIndex       = 0xFF
        res.iHOffsetFOV  = ROI_X
        res.iVOffsetFOV  = ROI_Y
        res.iWidthFOV    = ROI_W
        res.iHeightFOV   = ROI_H
        res.iWidth       = ROI_W
        res.iHeight      = ROI_H
        mvsdk.CameraSetImageResolution(hCamera, res)

        # exposure / gain / gamma
        mvsdk.CameraSetTriggerMode(hCamera, 0)  # free-run
        mvsdk.CameraSetAeState(hCamera, 0)      # manual exposure
        mvsdk.CameraSetExposureTime(hCamera, DESIRED_EXPOSURE_US)

        gmin, gmax, _ = mvsdk.CameraGetAnalogGainXRange(hCamera)
        mvsdk.CameraSetAnalogGainX(hCamera, max(gmin, min(DESIRED_ANALOG_GAIN, gmax)))

        gamma_max = cap.sGammaRange.iMax
        mvsdk.CameraSetGamma(hCamera, int(DESIRED_GAMMA * gamma_max))

        mvsdk.CameraPlay(hCamera)
        print("Starting camera feed...")

        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        delta_t_nominal = 1.0 / FPS_NOMINAL

        # ---- state vars ----
        tracking              = False
        ready                 = False
        stationary_start_time = None
        prev_frame            = None
        prev_det              = None
        bbox                  = None

        while True:
            try:
                # ----- grab a frame -----
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

                # ----- SETUP MODE -----
                if not tracking:
                    dets = detect_golfballs(frame, conf=0.25, imgsz=640, display=False)
                    if not dets:
                        throttled_print("No ball detected. Place ball on left side.")
                    else:
                        x, y, r = dets[0]
                        if x <= frame.shape[1] // 2:
                            bbox                  = (int(x - r), int(y - r), int(x + r), int(y + r))
                            prev_frame            = frame.copy()
                            prev_det              = dets[0]
                            tracking              = True
                            ready                 = False
                            stationary_start_time = time.time()
                            throttled_print("Ball detected. Hold still…")
                        else:
                            throttled_print("Move ball to the left side.")
                else:
                    # ----- TRACKING MODE -----
                    now = time.time()
                    if not ready and now - stationary_start_time >= 1.5:
                        ready = True
                        throttled_print("Ready to hit!")

                    if ready and bbox:
                        x1, y1, x2, y2 = bbox
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    moved, delta = has_ball_moved(prev_frame, frame, bbox)
                    if moved:
                        throttled_print(f"Ball moved (δ={delta:.4f}). Capturing sequence…")

                        frames     = [prev_frame]
                        dets_list  = [prev_det]
                        timestamps = [time.time()]

                        second_idx = None
                        far_idx    = None
                        misses     = 0
                        start_len  = len(frames)  # 1 now
                        grab_count = 0

                        while grab_count < MAX_CAPTURE_FRAMES:
                            grab_count += 1

                            try:
                                pRaw, head = mvsdk.CameraGetImageBuffer(hCamera, 2000)
                                mvsdk.CameraImageProcess(hCamera, pRaw, pFrameBuffer, head)
                                mvsdk.CameraReleaseImageBuffer(hCamera, pRaw)

                                chunk = np.frombuffer(
                                    (mvsdk.c_ubyte * head.uBytes).from_address(pFrameBuffer),
                                    dtype=np.uint8
                                )
                                f = chunk.reshape((head.iHeight, head.iWidth,
                                                   1 if fmt == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))
                                if f.ndim == 2 or f.shape[2] == 1:
                                    f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
                            except mvsdk.CameraException as e:
                                if e.error_code == mvsdk.CAMERA_STATUS_TIME_OUT:
                                    continue
                                else:
                                    raise

                            dts = detect_golfballs(f, conf=0.25, imgsz=640, display=False)
                            if not dts:
                                misses += 1
                                if misses > MISS_TOLERANCE:
                                    if DEBUG:
                                        print("[DEBUG] Too many misses, abort capture.")
                                    break
                                continue
                            else:
                                misses = 0

                            frames.append(f.copy())
                            dets_list.append(dts[0])
                            timestamps.append(time.time())

                            post_impact_count = len(frames) - start_len  # AFTER impact
                            if post_impact_count == 1:
                                if DEBUG:
                                    print("[DEBUG] First post-impact frame stored.")
                                continue
                            elif post_impact_count == 2 and second_idx is None:
                                second_idx = len(frames) - 1
                                if DEBUG:
                                    print(f"[DEBUG] Second post-impact frame idx={second_idx}")
                                continue

                            if second_idx is not None and far_idx is None:
                                # distance in pixels relative to 2nd frame
                                _, _, r2 = dets_list[second_idx]
                                d_px = r2 * 2.0
                                threshold_px = BALLS_FOR_20CM * d_px

                                cx2, cy2, _ = dets_list[second_idx]
                                cx,  cy,  _ = dts[0]
                                dist_px = ((cx - cx2) ** 2 + (cy - cy2) ** 2) ** 0.5

                                if DEBUG and post_impact_count % 2 == 0:
                                    print(f"[DEBUG] dist from 2nd = {dist_px:.1f}px  (need {threshold_px:.1f}px)")

                                if dist_px >= threshold_px:
                                    far_idx = len(frames) - 1
                                    if DEBUG:
                                        print(f"[DEBUG] Far frame idx={far_idx}, dist_px={dist_px:.1f}")
                                    break

                        # ---- compute metrics ----
                        if second_idx is not None and far_idx is not None:
                            t2   = timestamps[second_idx]
                            tFar = timestamps[far_idx]
                            delta_t_new = tFar - t2

                            det2   = dets_list[second_idx]
                            detFar = dets_list[far_idx]

                            c2x, c2y, r2 = det2
                            cFx, cFy, _  = detFar

                            d_px = r2 * 2.0
                            dist_px = ((cFx - c2x) ** 2 + (cFy - c2y) ** 2) ** 0.5
                            dist_mm = (dist_px / d_px) * BALL_DIAM_MM

                            speed_mps = (dist_mm / 1000.0) / delta_t_new
                            launch_angle = calculate_launch_angle(det2, detFar)

                            g2   = cv2.cvtColor(frames[second_idx], cv2.COLOR_BGR2GRAY)
                            gFar = cv2.cvtColor(frames[far_idx],  cv2.COLOR_BGR2GRAY)
                            rx, ry, rz = get_fine_ball_rotation(g2, gFar)

                            side_spin_rpm = (rx / delta_t_new) * (60.0 / 360.0)
                            back_spin_rpm = (ry / delta_t_new) * (60.0 / 360.0)
                            vec           = np.array([rx, ry, rz], dtype=float)
                            total_spin    = np.linalg.norm(vec)
                            spin_axis     = vec / total_spin if total_spin > 0 else vec

                            data = {
                                "Speed":     speed_mps,
                                "VLA":       launch_angle,
                                "HLA":       0.0,
                                "TotalSpin": back_spin_rpm,   # or total_spin if needed
                                "SpinAxis":  spin_axis,
                            }

                            print(f"delta_t_new: {delta_t_new:.6f} s")
                            print(f"Speed: {speed_mps:.2f} m/s  ({speed_mps*2.23694:.1f} mph)")
                            print(f"Launch angle: {launch_angle:.2f}°")
                            print(f"Backspin: {back_spin_rpm:.0f} rpm  Side spin: {side_spin_rpm:.0f} rpm")

                            metrics, positions = get_trajectory_metrics(data, delta_t_nominal)

                            print(f"Carry: {metrics['carry_distance']:.2f} yd")
                            print(f"Total: {metrics['total_distance']:.2f} yd")
                            print(f"Flight time: {metrics['time_of_flight']:.2f} s")
                            print(f"Apex: {metrics['apex']:.2f} ft")
                            print(f"Desc angle: {metrics['descending_angle']:.2f}°")

                            if DEBUG:
                                footer = f"dt={delta_t_new:.6f}s, dist={dist_mm:.1f}mm"
                                show_debug_pair(frames[second_idx], frames[far_idx],
                                                "2nd frame", "far frame", footer)

                            send_to_visualization(positions,
                                                  speed_mps,
                                                  launch_angle,
                                                  back_spin_rpm,
                                                  side_spin_rpm,
                                                  spin_axis,
                                                  metrics['carry_distance'],
                                                  metrics['total_distance'],
                                                  metrics['apex'],
                                                  metrics['time_of_flight'],
                                                  metrics['descending_angle'])
                        else:
                            throttled_print("Could not obtain both required frames.")

                        # reset
                        tracking   = False
                        prev_frame = None
                        prev_det   = None
                        bbox       = None

                # live feed
                cv2.imshow("frame", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except mvsdk.CameraException as e:
                if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                    raise

        # clean up
        if pFrameBuffer:
            mvsdk.CameraAlignFree(pFrameBuffer)

    finally:
        mvsdk.CameraUnInit(hCamera)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
