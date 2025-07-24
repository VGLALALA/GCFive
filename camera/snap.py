# test_roi_camera.py
import time, platform
import numpy as np
import cv2
import mvsdk

# Region of Interest settings
ROI_W, ROI_H = 640, 280
ROI_X, ROI_Y = 0, 200

# === User-configurable settings ===
# Set exposure time (in microseconds)
DESIRED_EXPOSURE_US = 50.0    # 0.05 ms
# Set analog gain (must be between camera's min and max)
DESIRED_ANALOG_GAIN = 1000.0     # change this value as needed
# Set gamma (0.0 – 1.0)
DESIRED_GAMMA = 0.25         # change this value as needed
# Enable or disable 2D denoise filter
ENABLE_2D_DENOISE = True
# ==================================

def main():
    # Enumerate and initialize camera
    devs = mvsdk.CameraEnumerateDevice()
    if not devs:
        print("No camera found")
        return

    hCamera = mvsdk.CameraInit(devs[0], -1, -1)

    try:
        # Get camera capabilities
        cap = mvsdk.CameraGetCapability(hCamera)

        # --- Manual control: disable AE, set exposure ---
        mvsdk.CameraSetAeState(hCamera, False)
        mvsdk.CameraSetExposureTime(hCamera, DESIRED_EXPOSURE_US)

        # --- Analog gain ---
        gain_min, gain_max, gain_step = mvsdk.CameraGetAnalogGainXRange(hCamera)
        # Clamp desired gain into valid range
        gain_to_set = max(gain_min, min(DESIRED_ANALOG_GAIN, gain_max))
        mvsdk.CameraSetAnalogGainX(hCamera, gain_to_set)

        # --- Gamma ---
        # Scale DESIRED_GAMMA (0.0–1.0) into camera's integer range
        gamma_max = cap.sGammaRange.iMax
        gamma_int = int(DESIRED_GAMMA * gamma_max)
        mvsdk.CameraSetGamma(hCamera, gamma_int)

        # --- 2D Denoise ---
        mvsdk.CameraSetNoiseFilter(hCamera, bool(ENABLE_2D_DENOISE))

        # Set output format based on sensor type
        mono = (cap.sIspCapacity.bMonoSensor != 0)
        fmt = mvsdk.CAMERA_MEDIA_TYPE_MONO8 if mono else mvsdk.CAMERA_MEDIA_TYPE_BGR8
        mvsdk.CameraSetIspOutFormat(hCamera, fmt)

        # Configure ROI resolution
        res = mvsdk.CameraGetImageResolution(hCamera)
        res.iIndex      = 0xFF
        res.iHOffsetFOV = ROI_X
        res.iVOffsetFOV = ROI_Y
        res.iWidthFOV   = ROI_W
        res.iHeightFOV  = ROI_H
        res.iWidth      = ROI_W
        res.iHeight     = ROI_H
        mvsdk.CameraSetImageResolution(hCamera, res)

        # Start streaming
        mvsdk.CameraPlay(hCamera)

        pFrameBuffer = None
        frame_count = 0

        while True:
            try:
                pRaw, head = mvsdk.CameraGetImageBuffer(hCamera, 2000)
                if pFrameBuffer is None:
                    pFrameBuffer = mvsdk.CameraAlignMalloc(head.uBytes, 16)
                mvsdk.CameraImageProcess(hCamera, pRaw, pFrameBuffer, head)
                mvsdk.CameraReleaseImageBuffer(hCamera, pRaw)

                # Convert to OpenCV image
                frame_data = (mvsdk.c_ubyte * head.uBytes).from_address(pFrameBuffer)
                frame = np.frombuffer(frame_data, dtype=np.uint8)
                frame = frame.reshape((head.iHeight, head.iWidth,
                                       1 if fmt == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))

                # Display
                cv2.imshow('Camera Feed', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('\r'):
                    cv2.imwrite(f"data/Images/frame_{frame_count:05d}.jpg", frame)
                    print(f"Saved frame_{frame_count:05d}.jpg")
                    frame_count += 1

            except mvsdk.CameraException as e:
                if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                    raise

        # Free buffer
        if pFrameBuffer:
            mvsdk.CameraAlignFree(pFrameBuffer)

    finally:
        mvsdk.CameraUnInit(hCamera)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
