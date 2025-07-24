# test_roi_camera.py
import time, platform
import numpy as np
import cv2
import mvsdk

ROI_W, ROI_H = 640, 280
ROI_X, ROI_Y = 0, 200

def main():
    devs = mvsdk.CameraEnumerateDevice()
    if not devs:
        print("No camera found")
        return

    hCamera = mvsdk.CameraInit(devs[0], -1, -1)

    try:
        cap  = mvsdk.CameraGetCapability(hCamera)
        mono = (cap.sIspCapacity.bMonoSensor != 0)

        fmt = mvsdk.CAMERA_MEDIA_TYPE_MONO8 if mono else mvsdk.CAMERA_MEDIA_TYPE_BGR8
        mvsdk.CameraSetIspOutFormat(hCamera, fmt)

        res = mvsdk.CameraGetImageResolution(hCamera)
        res.iIndex      = 0xFF
        res.iHOffsetFOV = ROI_X
        res.iVOffsetFOV = ROI_Y
        res.iWidthFOV   = ROI_W
        res.iHeightFOV  = ROI_H
        res.iWidth      = ROI_W
        res.iHeight     = ROI_H
        mvsdk.CameraSetImageResolution(hCamera, res)

        mvsdk.CameraSetTriggerMode(hCamera, 0)
        mvsdk.CameraSetAeState(hCamera, 0)
        mvsdk.CameraSetExposureTime(hCamera, 250)  # 0.5 ms
        print(mvsdk.CameraGetImageResolution(hCamera))
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

                # Convert frame to OpenCV format
                frame_data = (mvsdk.c_ubyte * head.uBytes).from_address(pFrameBuffer)
                frame = np.frombuffer(frame_data, dtype=np.uint8)
                frame = frame.reshape((head.iHeight, head.iWidth, 1 if fmt == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))

                # Display the frame
                cv2.imshow('Camera Feed', frame)

                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('\r'):  # Enter key
                    cv2.imwrite(f"data/Images/frame_{frame_count:05d}.jpg", frame)
                    print(f"Saved frame_{frame_count:05d}.jpg")
                    frame_count += 1

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
