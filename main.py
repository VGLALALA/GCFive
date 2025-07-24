
import cv2
import numpy as np
import socket
import json
from image_processing.ballDetection import detect_golfballs
from image_processing.movementDetection import has_ball_moved
from image_processing.ballSpeedCalculation import calculate_ball_speed
from image_processing.launchAngleCalculation import calculate_launch_angle
from spin.GetBallRotation import get_fine_ball_rotation
from trajectory_simulation.flightDataCalculation import get_trajectory_metrics


ROI_W, ROI_H = 640, 280
ROI_X, ROI_Y = 0, 200
VIS_HOST = "localhost"
VIS_PORT = 49152
DISTANCE_THRESHOLD_MM = 200  # 20 cm


def send_to_visualization(positions: np.ndarray, speed_mps: float, launch_angle: float, backspin_rpm: float, side_spin_rpm: float, spin_axis: float, carry: float, total: float, apex: float, time_of_flight: float, descending_angle: float):
    pass




import camera.mvsdk as mvsdk
def main():
    devs = mvsdk.CameraEnumerateDevice()
    if not devs:
        print("No camera found")
        return

    hCamera = mvsdk.CameraInit(devs[0], -1, -1)

    try:
        cap = mvsdk.CameraGetCapability(hCamera)
        mono = (cap.sIspCapacity.bMonoSensor != 0)

        fmt = mvsdk.CAMERA_MEDIA_TYPE_MONO8 if mono else mvsdk.CAMERA_MEDIA_TYPE_BGR8
        mvsdk.CameraSetIspOutFormat(hCamera, fmt)

        res = mvsdk.CameraGetImageResolution(hCamera)
        res.iIndex = 0xFF
        res.iHOffsetFOV = ROI_X
        res.iVOffsetFOV = ROI_Y
        res.iWidthFOV = ROI_W
        res.iHeightFOV = ROI_H
        res.iWidth = ROI_W
        res.iHeight = ROI_H
        mvsdk.CameraSetImageResolution(hCamera, res)

        mvsdk.CameraSetTriggerMode(hCamera, 0)
        mvsdk.CameraSetAeState(hCamera, 0)
        mvsdk.CameraSetExposureTime(hCamera, 50)  # 0.5 ms
        mvsdk.CameraPlay(hCamera)

        pFrameBuffer = None
        fps = 1300.0 
        delta_t = 1/fps # Assuming a default FPS as mvsdk does not provide FPS directly
        print("Starting camera feed...")
        tracking = False
        prev_frame = None
        prev_det = None
        bbox = None

        while True:
            try:
                pRaw, head = mvsdk.CameraGetImageBuffer(hCamera, 2000)

                if pFrameBuffer is None:
                    pFrameBuffer = mvsdk.CameraAlignMalloc(head.uBytes, 16)

                mvsdk.CameraImageProcess(hCamera, pRaw, pFrameBuffer, head)
                mvsdk.CameraReleaseImageBuffer(hCamera, pRaw)

                frame_data = (mvsdk.c_ubyte * head.uBytes).from_address(pFrameBuffer)
                frame = np.frombuffer(frame_data, dtype=np.uint8)
                frame = frame.reshape((head.iHeight, head.iWidth, 1 if fmt == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))

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
                    pRaw, head = mvsdk.CameraGetImageBuffer(hCamera, 2000)
                    mvsdk.CameraImageProcess(hCamera, pRaw, pFrameBuffer, head)
                    mvsdk.CameraReleaseImageBuffer(hCamera, pRaw)

                    frame_data = (mvsdk.c_ubyte * head.uBytes).from_address(pFrameBuffer)
                    f = np.frombuffer(frame_data, dtype=np.uint8)
                    f = f.reshape((head.iHeight, head.iWidth, 1 if fmt == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))

                    dets = detect_golfballs(f, conf=0.25, imgsz=640, display=False)
                    if not dets:
                        break
                    frames.append(f.copy())
                    detections.append(dets[0])
                    cx, cy, _ = dets[0]
                    dx = cx - last_center[0]
                    dy = cy - last_center[1]
                    total_distance = (dx**2 + dy**2) ** 0.5 / pixels_per_mm
                    last_center = (cx, cy)

                if len(frames) >= 2:
                    speed = calculate_ball_speed(detections[0], detections[1], fps)
                    angle = calculate_launch_angle(detections[0], detections[1])
                    gray0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
                    gray1 = cv2.cvtColor(frames[1], cv2.COLOR_BGR2GRAY)
                    best_rot_x, best_rot_y, best_rot_z = get_fine_ball_rotation(gray0, gray1)
                    side_spin_rpm = (best_rot_x / delta_t) * (60 / 360)
                    back_spin_rpm = (best_rot_y / delta_t) * (60 / 360)
                    rot_vec = np.array([best_rot_x, best_rot_y, best_rot_z], dtype=float)
                    angle = np.linalg.norm(rot_vec)
                    spin_axis = rot_vec/angle
                    data = {
                        "Speed": speed,
                        "VLA": angle,
                        "HLA": 0.0,
                        "TotalSpin": back_spin_rpm,
                        "SpinAxis": spin_axis,
                    }
                    
                    metrics,positions = get_trajectory_metrics(data,delta_t)
                    carry = metrics["carry_distance"]   
                    total = metrics["total_distance"]
                    apex = metrics["apex"]
                    time_of_flight = metrics["time_of_flight"]
                    descending_angle = metrics["descending_angle"]
                    print(f"Carry Distance: {carry:.2f} yd")
                    print(f"Total Distance: {total:.2f} yd")
                    print(f"Time of Flight: {time_of_flight:.2f} s")
                    print(f"Apex Height: {apex:.2f} ft")
                    print(f"Descending Angle: {descending_angle:.2f} degrees")

                    send_to_visualization(positions, speed, angle, back_spin_rpm, side_spin_rpm, spin_axis, carry, total, apex, time_of_flight, descending_angle)
                    
                else:
                    print("Insufficient frames captured, to compute metrics.")

                tracking = False
                prev_frame = None
                prev_det = None
                bbox = None

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except mvsdk.CameraException as e:
                if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                    raise

        if pFrameBuffer:
            mvsdk.CameraAlignFree(pFrameBuffer)

    finally:
        mvsdk.CameraUnInit(hCamera)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
