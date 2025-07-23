import cv2
import numpy as np
from . import mvsdk
import platform
import time
import threading
import queue
from datetime import datetime

# Global buffer variable
pFrameBuffer_global = 0

def setup_camera_and_buffer():
    global pFrameBuffer_global
    # This function will handle camera init and buffer allocation
    DevList = mvsdk.CameraEnumerateDevice()
    nDev = len(DevList)
    if nDev < 1:
        print("No camera was found!")
        return None, None

    for i, DevInfo in enumerate(DevList):
        print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))
    i = 0 if nDev == 1 else int(input("Select camera: "))
    DevInfo = DevList[i]
    print(DevInfo)

    hCamera = 0
    try:
        hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
    except mvsdk.CameraException as e:
        print("CameraInit Failed({}): {}".format(e.error_code, e.message))
        return None, None

    cap = mvsdk.CameraGetCapability(hCamera)
    monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

    if monoCamera:
        mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
    else:
        mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

    # Set camera parameters for high-speed capture
    mvsdk.CameraSetTriggerMode(hCamera, 0)  # Continuous mode
    mvsdk.CameraSetFrameSpeed(hCamera, 2)   # Highest frame speed
    mvsdk.CameraSetAeState(hCamera, 0)      # Manual exposure
    mvsdk.CameraSetExposureTime(hCamera, 500)  # 1ms exposure for 800 FPS
    mvsdk.CameraSetGain(hCamera, 100, 100, 100)  # Set RGB gains to 100
    mvsdk.CameraSetAntiFlick(hCamera, 0)

    mvsdk.CameraPlay(hCamera)

    # Calculate buffer size using fixed resolution (640x480)
    FrameBufferSize = 640 * 480 * (1 if monoCamera else 3)
    pFrameBuffer_global = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)

    return hCamera, monoCamera

def release_camera_and_buffer(hCamera):
    global pFrameBuffer_global
    if hCamera:
        mvsdk.CameraUnInit(hCamera)
    if pFrameBuffer_global:
        mvsdk.CameraAlignFree(pFrameBuffer_global)
        pFrameBuffer_global = 0

def acquire_frames(hCamera, frame_queue, stop_event):
    print("Acquisition thread started.")
    while not stop_event.is_set():
        try:
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 1250)
            frame_queue.put((pRawData, FrameHead))
        except mvsdk.CameraException as e:
            if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                print(f"Acquisition thread camera error: {e.error_code} - {e.message}")
            pass
    print("Acquisition thread stopped.")

def record_and_display_frames(hCamera, monoCamera, frame_queue, stop_event):
    print("Starting frame recording...")
    frames = []
    circle_xs = []
    circle_ys = []
    radii = []
    frame_count = 0
    start_time = time.time()
    
    while frame_count < 30 and not stop_event.is_set():
        try:
            frame_data_tuple = frame_queue.get(timeout=0.1)
            if frame_data_tuple is None:
                break
            
            pRawData, FrameHead = frame_data_tuple
            mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer_global, FrameHead)
            mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)

            if platform.system() == "Windows":
                mvsdk.CameraFlipFrameBuffer(pFrameBuffer_global, FrameHead, 1)

            frame = np.frombuffer((mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer_global), dtype=np.uint8)
            frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))
            
            if not monoCamera and frame.ndim == 3:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray_frame = frame

            # Detect circle using HoughCircles
            blurred = cv2.GaussianBlur(gray_frame, (9, 9), 2)
            circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40, param1=100, param2=30, minRadius=0, maxRadius=0)
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                x, y, r = circles[0]
                circle_xs.append(x)
                circle_ys.append(y)
                radii.append(r)
            else:
                circle_xs.append(None)
                circle_ys.append(None)
                radii.append(None)

            frames.append(gray_frame.copy())
            frame_count += 1
            print(f"Captured frame {frame_count}/30")
            
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Error capturing frame: {e}")
            break

    end_time = time.time()
    duration = end_time - start_time
    fps = frame_count / duration if duration > 0 else 0
    print(f"\nRecording complete!")
    print(f"Captured {frame_count} frames in {duration:.2f} seconds")
    print(f"Average FPS: {fps:.2f}")

    # Find the frame where the detected circle's x is closest to 50
    min_diff = float('inf')
    best_idx = None
    for idx, x in enumerate(circle_xs):
        if x is not None:
            diff = abs(x - 50)
            if diff < min_diff:
                min_diff = diff
                best_idx = idx

    if best_idx is not None:
        print(f"Displaying frame {best_idx+1} where detected circle x is closest to 50 (x={circle_xs[best_idx]})")
        display_frame = frames[best_idx].copy()
        # Draw the detected circle if available
        if circle_xs[best_idx] is not None and circle_ys[best_idx] is not None and radii[best_idx] is not None:
            cv2.circle(display_frame, (circle_xs[best_idx], circle_ys[best_idx]), radii[best_idx], (0, 255, 0), 2)
            cv2.circle(display_frame, (circle_xs[best_idx], circle_ys[best_idx]), 2, (0, 0, 255), 3)
        cv2.imshow(f"Frame with circle x closest to 50", display_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No circles detected in any frame.")

    return fps

def main():
    # Initialize camera
    hCamera, monoCamera = setup_camera_and_buffer()
    if hCamera is None:
        print("Failed to initialize camera")
        return

    try:
        # Create frame queue and stop event
        frame_queue = queue.Queue()
        stop_event = threading.Event()

        # Start acquisition thread
        acquisition_thread = threading.Thread(
            target=acquire_frames,
            args=(hCamera, frame_queue, stop_event)
        )
        acquisition_thread.start()

        # Record and display frames
        fps = record_and_display_frames(hCamera, monoCamera, frame_queue, stop_event)
        
        # Clean up
        stop_event.set()
        acquisition_thread.join()
        
    finally:
        release_camera_and_buffer(hCamera)

if __name__ == '__main__':
    main() 