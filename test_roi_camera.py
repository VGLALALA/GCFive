# test_roi_camera.py
import time, platform
import numpy as np
import mvsdk

ROI_W, ROI_H = 640, 300
ROI_X, ROI_Y = 0, 0

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
        mvsdk.CameraSetExposureTime(hCamera, 500)  # 0.5 ms
        print(mvsdk.CameraGetImageResolution(hCamera))
        mvsdk.CameraPlay(hCamera)

        grabbed = 0
        pFrameBuffer = None
        start_time = time.time()

        while grabbed < 100:
            try:
                pRaw, head = mvsdk.CameraGetImageBuffer(hCamera, 2000)

                if pFrameBuffer is None:
                    pFrameBuffer = mvsdk.CameraAlignMalloc(head.uBytes, 16)

                mvsdk.CameraImageProcess(hCamera, pRaw, pFrameBuffer, head)
                mvsdk.CameraReleaseImageBuffer(hCamera, pRaw)

                # Skip any frame processing/saving
                grabbed += 1

            except mvsdk.CameraException as e:
                if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                    raise

        elapsed = time.time() - start_time
        fps = grabbed / elapsed
        print(f"Captured {grabbed} frames in {elapsed:.2f} seconds → avg FPS: {fps:.2f}")

        if pFrameBuffer:
            mvsdk.CameraAlignFree(pFrameBuffer)

    finally:
        mvsdk.CameraUnInit(hCamera)

if __name__ == "__main__":
    main()
