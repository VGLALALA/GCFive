import cv2
import numpy as np
import mvsdk
import platform
import time

# Desired settings
WIDTH = 320
HEIGHT = 180  # Crop to 16:9 if possible
TARGET_FPS = 1600
NUM_FRAMES = 1600  # Number of frames to capture for the demo


def main():
    # Enumerate cameras
    DevList = mvsdk.CameraEnumerateDevice()
    nDev = len(DevList)
    if nDev < 1:
        print("No camera was found!")
        return

    for i, DevInfo in enumerate(DevList):
        print(f"{i}: {DevInfo.GetFriendlyName()} {DevInfo.GetPortType()}")
    i = 0 if nDev == 1 else int(input("Select camera: "))
    DevInfo = DevList[i]
    print(DevInfo)

    # Initialize camera
    try:
        hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
    except mvsdk.CameraException as e:
        print(f"CameraInit Failed({e.error_code}): {e.message}")
        return

    cap = mvsdk.CameraGetCapability(hCamera)
    monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

    # Print all supported resolutions
    print("Supported camera resolutions:")
    for idx in range(cap.iImageSizeDesc):
        res = cap.pImageSizeDesc[idx]
        print(f"Mode {idx}: {res.iWidth}x{res.iHeight}")
    # Force use of Mode 1 (320x240)
    image_resolution = cap.pImageSizeDesc[1]
    print(f"Using mode 1: {image_resolution.iWidth}x{image_resolution.iHeight}")
    mvsdk.CameraSetImageResolution(hCamera, image_resolution)

    # Set output format
    if monoCamera:
        mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
    else:
        mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

    # Set high-speed capture parameters
    mvsdk.CameraSetTriggerMode(hCamera, 0)  # Continuous mode
    mvsdk.CameraSetFrameSpeed(hCamera, 2)   # Highest frame speed
    mvsdk.CameraSetAeState(hCamera, 0)      # Manual exposure
    mvsdk.CameraSetExposureTime(hCamera, 0.625)  # Exposure time in ms for 1/1600s
    mvsdk.CameraSetGain(hCamera, 100, 100, 100)
    mvsdk.CameraSetContrast(hCamera, 100)
    mvsdk.CameraSetAntiFlick(hCamera, 0)

    mvsdk.CameraPlay(hCamera)

    print(f"Capturing {NUM_FRAMES} frames at {WIDTH}x{HEIGHT} (cropped if needed)...")
    frames = []
    pFrameBuffer = None
    actual_width = None
    actual_height = None
    start_time = time.time()
    for idx in range(NUM_FRAMES):
        try:
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 1250)
            if pFrameBuffer is None:
                # Allocate buffer based on actual frame size
                actual_width = FrameHead.iWidth
                actual_height = FrameHead.iHeight
                print(f"Actual frame size: {actual_width}x{actual_height}")
                FrameBufferSize = actual_width * actual_height * (1 if monoCamera else 3)
                pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)
            mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
            mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)
            if platform.system() == "Windows":
                mvsdk.CameraFlipFrameBuffer(pFrameBuffer, FrameHead, 1)
            frame = np.frombuffer((mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer), dtype=np.uint8)
            frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))
            # Crop to 320x180 if actual frame is larger
            if actual_width >= WIDTH and actual_height >= HEIGHT:
                frame_cropped = frame[0:HEIGHT, 0:WIDTH]
            else:
                frame_cropped = frame  # Use as is if smaller
            frames.append(frame_cropped.copy())
            if idx < 5:
                cv2.imwrite(f"demo_frame_{idx}.png", frame_cropped)
        except mvsdk.CameraException as e:
            print(f"Frame {idx}: Camera error: {e.error_code} - {e.message}")
            break
    elapsed = time.time() - start_time
    actual_fps = len(frames) / elapsed if elapsed > 0 else 0
    print(f"Captured {len(frames)} frames in {elapsed:.2f} seconds. Actual FPS: {actual_fps:.1f}")

    # Release resources
    mvsdk.CameraUnInit(hCamera)
    if pFrameBuffer is not None:
        mvsdk.CameraAlignFree(pFrameBuffer)
    print("Demo complete. First 5 frames saved as demo_frame_*.png.")

if __name__ == "__main__":
    main() 